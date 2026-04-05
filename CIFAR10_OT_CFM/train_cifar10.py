# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os
import time

import torch
from absl import app, flags
from torchvision import datasets, transforms
from tqdm import trange
from CIFAR10_OT_CFM.utils_cifar import ema, generate_samples, infiniteloop

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

# ── Our amortised solvers ────────────────────────────────────────────────────
from CIFAR10_OT_CFM.RA_OT_CIFAR import AmortizedRA_OT_CIFAR
from CIFAR10_OT_CFM.OA_OT_CIFAR import AmortizedOA_OT_CIFAR
# ─────────────────────────────────────────────────────────────────────────────

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)

# ── Amortised OT pre-training hyper-parameters ──────────────────────────────
flags.DEFINE_integer("pretrain_M", 50,  help="number of mini-batches for OT pre-training (RA/OA-OT)")
flags.DEFINE_integer("pretrain_L", 100, help="number of random projections for sliced OT (RA/OA-OT)")
flags.DEFINE_integer("pretrain_T", 5000, help="optimisation steps for OA-OT dual objective")
flags.DEFINE_float("pretrain_eps", 0.1, help="Sinkhorn regularisation for RA/OA-OT")
flags.DEFINE_float("pretrain_ridge", 1e-3, help="ridge penalty for RA-OT regression")
flags.DEFINE_float("pretrain_lr", 1e-3, help="Adam learning rate for OA-OT optimisation")
# ─────────────────────────────────────────────────────────────────────────────


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    # DATASETS/DATALOADER
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #       Flow Matching Setup
    #################################

    sigma = 0.0

    # ── Amortised OT models (two-phase protocol) ─────────────────────────────
    amortized_solver = None      # will be set for ra-ot / oa-ot
    pretrain_time    = 0.0

    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)

    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)

    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)

    # ── RA-OT ─────────────────────────────────────────────────────────────────
    elif FLAGS.model == "ra-ot":
        FM = ConditionalFlowMatcher(sigma=sigma)   # same path/flow as I-CFM
        ot_device = "cuda" if use_cuda else "cpu"

        solver = AmortizedRA_OT_CIFAR(
            L=FLAGS.pretrain_L,
            eps=FLAGS.pretrain_eps,
            ridge=FLAGS.pretrain_ridge,
            device=ot_device,
        )

        # ---- helper samplers ------------------------------------------------
        pretrain_loader = torch.utils.data.DataLoader(
            dataset, batch_size=FLAGS.batch_size, shuffle=True,
            num_workers=FLAGS.num_workers, drop_last=True,
        )
        pretrain_looper = infiniteloop(pretrain_loader)

        def target_sampler(B):
            return next(pretrain_looper).to(device)

        def source_sampler(x1):
            return torch.randn_like(x1)
        # ---------------------------------------------------------------------

        print("\n" + "="*60)
        print("  PHASE 1 — RA-OT Pre-training")
        print("="*60)
        solver.pretrain(
            source_sampler=source_sampler,
            target_sampler=target_sampler,
            M=FLAGS.pretrain_M,
            B=FLAGS.batch_size,
        )
        pretrain_time = solver.pretrain_time
        amortized_solver = solver
        print(f"  RA-OT pre-training done in {pretrain_time:.2f}s")
        print("="*60 + "\n")

    # ── OA-OT ─────────────────────────────────────────────────────────────────
    elif FLAGS.model == "oa-ot":
        FM = ConditionalFlowMatcher(sigma=sigma)   # same path/flow as I-CFM
        ot_device = "cuda" if use_cuda else "cpu"

        solver = AmortizedOA_OT_CIFAR(
            L=FLAGS.pretrain_L,
            eps=FLAGS.pretrain_eps,
            lr=FLAGS.pretrain_lr,
            device=ot_device,
        )

        # ---- helper samplers ------------------------------------------------
        pretrain_loader = torch.utils.data.DataLoader(
            dataset, batch_size=FLAGS.batch_size, shuffle=True,
            num_workers=FLAGS.num_workers, drop_last=True,
        )
        pretrain_looper = infiniteloop(pretrain_loader)

        def target_sampler(B):
            return next(pretrain_looper).to(device)

        def source_sampler(x1):
            return torch.randn_like(x1)
        # ---------------------------------------------------------------------

        print("\n" + "="*60)
        print("  PHASE 1 — OA-OT Pre-training")
        print("="*60)
        solver.pretrain(
            source_sampler=source_sampler,
            target_sampler=target_sampler,
            M=FLAGS.pretrain_M,
            B=FLAGS.batch_size,
            T=FLAGS.pretrain_T,
        )
        pretrain_time = solver.pretrain_time
        amortized_solver = solver
        print(f"  OA-OT pre-training done in {pretrain_time:.2f}s")
        print("="*60 + "\n")

    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of "
            "['otcfm', 'icfm', 'fm', 'si', 'ra-ot', 'oa-ot']"
        )

    savedir = FLAGS.output_dir + FLAGS.model + "/"
    os.makedirs(savedir, exist_ok=True)

    #################################
    #   PHASE 2 — U-Net Training
    #################################

    print("\n" + "="*60)
    print(f"  PHASE 2 — U-Net Flow Matching Training  [{FLAGS.model.upper()}]")
    print("="*60)
    training_start = time.time()

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)

            # ── OT coupling ──────────────────────────────────────────────────
            if amortized_solver is not None:
                # RA-OT / OA-OT: replace exact EMD with amortised plan
                x0, x1 = amortized_solver.sample_pairs(x0, x1)
                x0 = x0.to(device)
                x1 = x1.to(device)
            # baseline methods handle coupling inside FM.sample_location_and_conditional_flow
            # ─────────────────────────────────────────────────────────────────

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                generate_samples(net_model, FLAGS.parallel, savedir, step, net_="normal")
                generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    savedir + f"{FLAGS.model}_cifar10_weights_step_{step}.pt",
                )

    training_time = time.time() - training_start
    total_time    = pretrain_time + training_time

    print("\n" + "="*60)
    print(f"  Training complete  [{FLAGS.model.upper()}]")
    print(f"  Pre-training time : {pretrain_time/3600:.4f} h  ({pretrain_time:.1f} s)")
    print(f"  U-Net train time  : {training_time/3600:.4f} h  ({training_time:.1f} s)")
    print(f"  Total time        : {total_time/3600:.4f} h  ({total_time:.1f} s)")
    print("="*60 + "\n")

    # Save timing report
    report_path = savedir + f"{FLAGS.model}_timing_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {FLAGS.model}\n")
        f.write(f"Total steps: {FLAGS.total_steps}\n")
        f.write(f"Batch size: {FLAGS.batch_size}\n")
        if FLAGS.model in ("ra-ot", "oa-ot"):
            f.write(f"Pretrain M: {FLAGS.pretrain_M}\n")
            f.write(f"Pretrain L: {FLAGS.pretrain_L}\n")
            f.write(f"Pretrain eps: {FLAGS.pretrain_eps}\n")
        if FLAGS.model == "ra-ot":
            f.write(f"Ridge lambda: {FLAGS.pretrain_ridge}\n")
        if FLAGS.model == "oa-ot":
            f.write(f"Pretrain T: {FLAGS.pretrain_T}\n")
            f.write(f"Pretrain lr: {FLAGS.pretrain_lr}\n")
        f.write(f"\nPre-training time : {pretrain_time:.2f} s  ({pretrain_time/3600:.4f} h)\n")
        f.write(f"U-Net train time  : {training_time:.2f} s  ({training_time/3600:.4f} h)\n")
        f.write(f"Total time        : {total_time:.2f} s  ({total_time/3600:.4f} h)\n")
    print(f"  Timing report saved to {report_path}")


if __name__ == "__main__":
    app.run(train)
