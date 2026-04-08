import copy
import os
import time
from collections import OrderedDict

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
from CIFAR10_OT_CFM.RA_OT_CIFAR import AmortizedRA_OT_CIFAR
from CIFAR10_OT_CFM.OA_OT_CIFAR import AmortizedOA_OT_CIFAR

# ── Flag definitions ──────────────────────────────────────────────────────────
FLAGS = flags.FLAGS

# Core
flags.DEFINE_string("model", "otcfm",
                    help="Coupling method for fine-tuning: "
                         "fm | icfm | si | otcfm | ra-ot | oa-ot")
flags.DEFINE_string("checkpoint", "",
                    help="Path to pretrained I-CFM .pt checkpoint "
                         "(e.g. ./results/icfm/icfm_cifar10_weights_step_400000.pt)")
flags.DEFINE_string("output_dir", "./results_ft/",
                    help="Root directory for fine-tuned checkpoints")

# UNet architecture (must match the checkpoint)
flags.DEFINE_integer("num_channel", 128, help="Base channel count of UNet")

# Fine-tuning hyper-parameters
flags.DEFINE_float("lr", 2e-4, help="Fine-tune learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="Gradient norm clipping")
flags.DEFINE_integer("finetune_epochs", 2,
                     help="Number of passes through CIFAR-10 train set (1 epoch ≈ 390 steps at bs=128)")
flags.DEFINE_integer("warmup", 200, help="LR warm-up steps")
flags.DEFINE_integer("batch_size", 128, help="Batch size")
flags.DEFINE_integer("num_workers", 4, help="DataLoader workers")
flags.DEFINE_float("ema_decay", 0.9999, help="EMA decay rate")
flags.DEFINE_bool("parallel", False, help="Multi-GPU fine-tuning")
flags.DEFINE_integer("save_step", 0,
                     help="Intermediate checkpoint frequency (0 = only save at end)")

# Amortised OT pre-training hyper-parameters (RA-OT / OA-OT only)
flags.DEFINE_integer("pretrain_M", 50,
                     help="Mini-batches collected for OT pre-training")
flags.DEFINE_integer("pretrain_L", 100,
                     help="Random projections for sliced OT")
flags.DEFINE_integer("pretrain_T", 5000,
                     help="Optimisation steps for OA-OT dual objective")
flags.DEFINE_float("pretrain_eps", 800.0,
                   help="Sinkhorn regularisation ε (applied directly to "
                        "squared-L2 cost in pixel space)")
flags.DEFINE_float("pretrain_ridge", 1e-3,
                   help="Ridge penalty λ for RA-OT regression")
flags.DEFINE_float("pretrain_lr", 1e-3,
                   help="Adam lr for OA-OT dual-objective optimisation")

# ─────────────────────────────────────────────────────────────────────────────
CIFAR10_TRAIN_SIZE = 50_000   # used to derive steps-per-epoch

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step: int) -> float:
    return min(step, FLAGS.warmup) / FLAGS.warmup


# ── Checkpoint loading (compatible with compute_fid.py) ───────────────────────
def _strip_module_prefix(state_dict: dict) -> dict:
    """Remove 'module.' prefix introduced by DataParallel."""
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k[7:] if k.startswith("module.") else k] = v
    return new_sd


def load_pretrained(path: str, net_model: torch.nn.Module,
                    ema_model: torch.nn.Module) -> dict:
    """Load a .pt checkpoint (same format as train_cifar10.py output).

    Both net_model and ema_model are restored in-place.
    Returns the raw checkpoint dict so the caller can inspect 'step' etc.
    """
    assert os.path.isfile(path), \
        f"Checkpoint not found: {path}\n" \
        f"Train the I-CFM baseline first with train_cifar10.py --model icfm"

    print(f"  Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)

    for attr, key in [("net_model", "net_model"), ("ema_model", "ema_model")]:
        sd = ckpt[key]
        model_obj = net_model if attr == "net_model" else ema_model
        try:
            model_obj.load_state_dict(sd)
        except RuntimeError:
            model_obj.load_state_dict(_strip_module_prefix(sd))

    loaded_step = ckpt.get("step", "?")
    print(f"  Checkpoint step: {loaded_step}")
    return ckpt


# ── Main fine-tuning function ─────────────────────────────────────────────────
def finetune(argv):
    assert FLAGS.checkpoint, \
        "You must provide --checkpoint pointing to the pretrained I-CFM .pt file."

    # ── Dataset / dataloader ─────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root="./data", train=True,
                               download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True,
    )
    datalooper = infiniteloop(dataloader)

    # steps_per_epoch: how many full batches fit in one pass over CIFAR-10
    steps_per_epoch = CIFAR10_TRAIN_SIZE // FLAGS.batch_size   # 390 @ bs=128
    total_ft_steps  = FLAGS.finetune_epochs * steps_per_epoch

    print(f"\n{'='*60}")
    print(f"  Fine-tune settings [{FLAGS.model.upper()}]")
    print(f"  Checkpoint    : {FLAGS.checkpoint}")
    print(f"  Epochs        : {FLAGS.finetune_epochs}  ({steps_per_epoch} steps/epoch)")
    print(f"  Total steps   : {total_ft_steps}")
    print(f"  Batch size    : {FLAGS.batch_size}")
    print(f"  LR            : {FLAGS.lr}")
    print(f"{'='*60}\n")

    # ── Model ────────────────────────────────────────────────────────────────
    net_model = UNetModelWrapper(
        dim=(3, 32, 32), num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4, num_head_channels=64,
        attention_resolutions="16", dropout=0.1,
    ).to(device)
    ema_model = copy.deepcopy(net_model)

    print("  Loading pretrained weights …")
    load_pretrained(FLAGS.checkpoint, net_model, ema_model)

    # Fresh optimiser + LR scheduler for fine-tuning
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    if FLAGS.parallel:
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    n_params = sum(p.data.nelement() for p in net_model.parameters())
    print(f"  Model params: {n_params / 1e6:.2f} M")

    # ── Flow Matching / Amortised OT setup ───────────────────────────────────
    sigma              = 0.0
    amortized_solver   = None
    pretrain_time      = 0.0

    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)

    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)

    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)

    # ── RA-OT ─────────────────────────────────────────────────────────────
    elif FLAGS.model == "ra-ot":
        FM = ConditionalFlowMatcher(sigma=sigma)
        ot_device = "cuda" if use_cuda else "cpu"

        solver = AmortizedRA_OT_CIFAR(
            L=FLAGS.pretrain_L, eps=FLAGS.pretrain_eps,
            ridge=FLAGS.pretrain_ridge, device=ot_device,
        )

        pretrain_loader = torch.utils.data.DataLoader(
            dataset, batch_size=FLAGS.batch_size, shuffle=True,
            num_workers=FLAGS.num_workers, drop_last=True,
        )
        pretrain_looper = infiniteloop(pretrain_loader)

        def target_sampler(B):
            return next(pretrain_looper).to(device)

        def source_sampler(x1):
            return torch.randn_like(x1)

        print("\n" + "="*60)
        print("  PHASE 1 — RA-OT Pre-training")
        print("="*60)
        solver.pretrain(source_sampler=source_sampler,
                        target_sampler=target_sampler,
                        M=FLAGS.pretrain_M, B=FLAGS.batch_size)
        pretrain_time    = solver.pretrain_time
        amortized_solver = solver
        print(f"  RA-OT pre-training done in {pretrain_time:.2f}s")
        print("="*60 + "\n")

    # ── OA-OT ─────────────────────────────────────────────────────────────
    elif FLAGS.model == "oa-ot":
        FM = ConditionalFlowMatcher(sigma=sigma)
        ot_device = "cuda" if use_cuda else "cpu"

        solver = AmortizedOA_OT_CIFAR(
            L=FLAGS.pretrain_L, eps=FLAGS.pretrain_eps,
            lr=FLAGS.pretrain_lr, device=ot_device,
        )

        pretrain_loader = torch.utils.data.DataLoader(
            dataset, batch_size=FLAGS.batch_size, shuffle=True,
            num_workers=FLAGS.num_workers, drop_last=True,
        )
        pretrain_looper = infiniteloop(pretrain_loader)

        def target_sampler(B):
            return next(pretrain_looper).to(device)

        def source_sampler(x1):
            return torch.randn_like(x1)

        print("\n" + "="*60)
        print("  PHASE 1 — OA-OT Pre-training")
        print("="*60)
        solver.pretrain(source_sampler=source_sampler,
                        target_sampler=target_sampler,
                        M=FLAGS.pretrain_M, B=FLAGS.batch_size,
                        T=FLAGS.pretrain_T)
        pretrain_time    = solver.pretrain_time
        amortized_solver = solver
        print(f"  OA-OT pre-training done in {pretrain_time:.2f}s")
        print("="*60 + "\n")

    else:
        raise NotImplementedError(
            f"Unknown model '{FLAGS.model}'. "
            "Choices: fm | icfm | si | otcfm | ra-ot | oa-ot"
        )

    # ── Output directory ─────────────────────────────────────────────────────
    # Tag: "ft_<method>" so compute_fid.py is called with --model ft_<method>
    model_tag = f"ft_{FLAGS.model}"
    savedir   = os.path.join(FLAGS.output_dir, model_tag) + "/"
    os.makedirs(savedir, exist_ok=True)

    # ── PHASE 2: Fine-tuning loop ─────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  PHASE 2 — U-Net Fine-tuning  [{FLAGS.model.upper()}]")
    print(f"  Total steps: {total_ft_steps}  "
          f"({FLAGS.finetune_epochs} epoch × {steps_per_epoch} steps)")
    print("="*60)

    ft_start = time.time()

    def _save_checkpoint(step: int):
        ckpt_name = f"{model_tag}_cifar10_weights_step_{step}.pt"
        ckpt_path = savedir + ckpt_name
        generate_samples(net_model, FLAGS.parallel, savedir, step, net_="normal")
        generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")
        torch.save({
            "net_model": net_model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "sched":     sched.state_dict(),
            "optim":     optim.state_dict(),
            "step":      step,
            "ft_model":  FLAGS.model,
            "pretrain_checkpoint": FLAGS.checkpoint,
        }, ckpt_path)
        print(f"  Checkpoint saved → {ckpt_path}")

    with trange(total_ft_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)

            # OT coupling (amortised or exact/independent)
            if amortized_solver is not None:
                x0, x1 = amortized_solver.sample_pairs(x0, x1)
                x0 = x0.to(device)
                x1 = x1.to(device)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt   = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # Current epoch label for progress bar
            cur_epoch = step // steps_per_epoch + 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                epoch=f"{cur_epoch}/{FLAGS.finetune_epochs}",
            )

            # Intermediate saves
            if FLAGS.save_step > 0 and step > 0 and step % FLAGS.save_step == 0:
                _save_checkpoint(step)

    ft_time    = time.time() - ft_start
    total_time = pretrain_time + ft_time

    # Always save final checkpoint
    final_step = total_ft_steps
    _save_checkpoint(final_step)

    # ── Timing report ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  Fine-tuning complete  [{FLAGS.model.upper()}]")
    print(f"  Pre-training time : {pretrain_time/3600:.4f} h  ({pretrain_time:.1f} s)")
    print(f"  Fine-tune time    : {ft_time/3600:.4f} h  ({ft_time:.1f} s)")
    print(f"  Total time        : {total_time/3600:.4f} h  ({total_time:.1f} s)")
    print("="*60 + "\n")

    report_path = savedir + f"{model_tag}_timing_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Method: {FLAGS.model}  (tag: {model_tag})\n")
        f.write(f"Base checkpoint: {FLAGS.checkpoint}\n")
        f.write(f"Finetune epochs: {FLAGS.finetune_epochs}  "
                f"({steps_per_epoch} steps/epoch = {total_ft_steps} total)\n")
        f.write(f"Batch size: {FLAGS.batch_size}\n")
        f.write(f"LR: {FLAGS.lr}\n")
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
        f.write(f"Fine-tune time    : {ft_time:.2f} s  ({ft_time/3600:.4f} h)\n")
        f.write(f"Total time        : {total_time:.2f} s  ({total_time/3600:.4f} h)\n")
        f.write(f"\nFinal checkpoint  : {savedir}{model_tag}_cifar10_weights_step_{final_step}.pt\n")
        f.write(f"\n--- FID evaluation command ---\n")
        f.write(f"python3 compute_fid.py --input_dir {FLAGS.output_dir} "
                f"--model {model_tag} --step {final_step} --integration_method dopri5\n")
    print(f"  Timing report saved → {report_path}")


if __name__ == "__main__":
    app.run(finetune)
