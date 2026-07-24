import argparse
import os
import time
import pickle
from time import localtime, strftime

import numpy as np
import torch
from tqdm import tqdm

from cfg import init_cfg
from Data.dataset_class import MNIST

# Reuse existing, already-validated helpers from eval_grayscale.py verbatim.
from Eval_report.eval_grayscale import (
    build_cost_grid, sample_pairs, pairs_to_loader,
    POOL_SEED, POOL_SIZE, TRAIN_RATIO,
)

from Solvers.Regression_SlicedOT.OT_Regression_Sliced import OT_Regression_Sliced
from Solvers.Objective_SlicedOT.OT_Objective_Sliced import OT_Objective_Sliced
from Solvers.Meta_OT.Meta_OT_gray_scale import OT_Discrete, dual_obj_loss
from Models.ot_models import PotentialMLP


def sinkhorn_warmstart_trace(a, b, C_t, eps, f_inits, g_inits, max_iter, device):
    B = len(f_inits)
    n = len(a)
    dtype = torch.float64

    a_safe = np.clip(a, 1e-300, None); a_safe = a_safe / a_safe.sum()
    b_safe = np.clip(b, 1e-300, None); b_safe = b_safe / b_safe.sum()
    a_t = torch.tensor(a_safe, dtype=dtype, device=device).unsqueeze(0).expand(B, n)
    b_t = torch.tensor(b_safe, dtype=dtype, device=device).unsqueeze(0).expand(B, n)
    log_a = torch.log(a_t)
    log_b = torch.log(b_t)
    log_K = (-C_t / eps).unsqueeze(0)  # (1, n, n), broadcasts over batch

    log_v = torch.zeros(B, n, dtype=dtype, device=device)
    for i in range(B):
        if g_inits[i] is not None:
            log_v[i] = torch.tensor(g_inits[i], dtype=dtype, device=device) / eps
    # NOTE: the initial log_u (from f_init) is mathematically never used —
    # the very first alternating update recomputes log_u from log_v alone
    # (see below), which is standard for alternating Sinkhorn. Only the
    # initial log_v (from g_init) affects the first iteration.

    err_trace = torch.zeros(max_iter, B, dtype=torch.float64, device=device)

    with torch.no_grad():
        for it in range(max_iter):
            M1 = log_K + log_v.unsqueeze(1)          # (B, n, n)
            log_u = log_a - torch.logsumexp(M1, dim=2)
            M2 = log_K + log_u.unsqueeze(2)          # (B, n, n)
            log_v = log_b - torch.logsumexp(M2, dim=1)

            log_P = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)
            P = torch.exp(log_P)
            err_row = (P.sum(dim=2) - a_t).abs().sum(dim=1)
            err_col = (P.sum(dim=1) - b_t).abs().sum(dim=1)
            err_trace[it] = torch.maximum(err_row, err_col)

    return err_trace.cpu().numpy()


# --------------------------------------------------------------------------- #
# Potential extraction per method
# --------------------------------------------------------------------------- #
def get_meta_potentials(a, b, mlp_meta, lf_meta, device):
    a_t = torch.tensor(a, dtype=torch.float64, device=device).unsqueeze(0)
    b_t = torch.tensor(b, dtype=torch.float64, device=device).unsqueeze(0)
    with torch.no_grad():
        f_pred = mlp_meta(a_t, b_t)
        g_sink, f_sink = lf_meta.g_from_f(a_t, b_t, f_pred)
    return f_sink[0].cpu().numpy(), g_sink[0].cpu().numpy()


def make_cfg_proj(solver, seed, gpu, flag_time):
    return argparse.Namespace(seed=seed, flag_time=flag_time, flag_load=None,
                               solver=solver, data_name="MNIST", gpu=gpu)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Saved -> {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M",        type=int,   default=50, help="train pairs, matches Table 1 M=50")
    p.add_argument("--N",        type=int,   default=50, help="test pairs for the warm-start benchmark")
    p.add_argument("--tol",      type=float, default=1e-6, help="L1 marginal-violation convergence threshold")
    p.add_argument("--max_iter", type=int,   default=500,
                    help="cap per init method; cold start may hit this cap "
                         "without converging — that itself is part of the "
                         "evidence for Q2 (warm start converges, cold start doesn't)")
    p.add_argument("--gpu",      type=str,   default="0")
    p.add_argument("--out",      type=str,   default="./results/grayscale_q2_warmstart")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out, exist_ok=True)
    flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())

    C   = build_cost_grid(28)
    eps = 1e-2   # matches eval_grayscale.py's runtime value for GT / RA-OT / OA-OT

    # ------------------------------------------------------ same pool as Table 1
    n_train_pool = int(POOL_SIZE * TRAIN_RATIO)
    print(f"Pre-sampling pool of {POOL_SIZE} pairs (seed={POOL_SEED}) ...")
    pool = sample_pairs(POOL_SIZE, seed=POOL_SEED)
    train_pool = pool[:n_train_pool]
    test_pool  = pool[n_train_pool:]

    assert args.M <= len(train_pool)
    assert args.N <= len(test_pool)
    train_pairs = train_pool[:args.M]
    test_pairs  = test_pool[:args.N]
    print(f"  M={args.M} train pairs | N={args.N} test pairs "
          f"(identical pool/split logic as eval_grayscale.py — first N pairs "
          f"here coincide with the pairs used to produce Table 1's M={args.M} row)")

    dl_train = pairs_to_loader(train_pairs, batch_size=1)

    # --------------------------------------------------------------- train RA-OT
    print("\n[1/3] Training RA-OT ...")
    cfg_r = init_cfg("OT_Regression_Sliced")
    cfg_r["num_bootstrap"] = args.M
    cfg_r["epsilon"]       = eps
    model_reg = OT_Regression_Sliced(
        make_cfg_proj("OT_Regression_Sliced", POOL_SEED, args.gpu, flag_time), cfg_r)
    model_reg.alpha = model_reg._fit(dl_train)
    model_reg.beta  = np.zeros(cfg_r["num_projections"])
    save_model(model_reg, os.path.join(args.out, "regression.pkl"))

    # --------------------------------------------------------------- train OA-OT
    print("\n[2/3] Training OA-OT ...")
    cfg_o = init_cfg("OT_Objective_Sliced")
    cfg_o["num_bootstrap"] = args.M
    cfg_o["epsilon"]       = eps
    model_obj = OT_Objective_Sliced(
        make_cfg_proj("OT_Objective_Sliced", POOL_SEED, args.gpu, flag_time), cfg_o)
    model_obj.alpha = model_obj._fit(dl_train)
    model_obj.beta  = np.zeros(cfg_o["num_projections"])
    save_model(model_obj, os.path.join(args.out, "objective.pkl"))

    # ------------------------------------------------------------- train Meta-OT
    print("\n[3/3] Training Meta-OT ...")
    cfg_meta = init_cfg("OT_Discrete")
    cfg_meta["epsilon"] = eps   # <-- explicit fix, see module docstring
    T_target = 5000
    cfg_meta["epochs"]       = max(1, T_target // args.M)
    cfg_meta["batch_size"]   = 1
    cfg_meta["log_interval"] = max(1, T_target // args.M)

    cfg_proj_meta = make_cfg_proj("OT_Discrete", POOL_SEED, args.gpu, flag_time)
    model_meta = OT_Discrete(cfg_proj_meta, cfg_meta)
    model_meta.OT_D_train(dl_train, None, flad_load_ckp=False)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_meta = PotentialMLP(dim_in=28**2*2, dim_out=28**2,
                             hidden_num=cfg_meta.MLP_hidden_num).to(dev)
    mlp_meta, _, _, _ = model_meta.load_ckp(mlp_meta, None, None, "OT_D-train")
    mlp_meta.eval()
    lf_meta = dual_obj_loss(img_size=28, epsilon=cfg_meta.epsilon, device=dev)

    # ------------------------------------------------------------ warm-start eval
    print(f"\nRunning warm-start Sinkhorn benchmark "
          f"(max_iter={args.max_iter}, device={dev}) ...")

    methods = ["cold_start", "meta_ot", "ra_ot", "oa_ot"]
    labels  = {"cold_start": "Zeros (cold start)", "meta_ot": "Meta-OT",
               "ra_ot": "RA-OT", "oa_ot": "OA-OT"}

    C_t = torch.tensor(C, dtype=torch.float64, device=dev)  # precompute once, reuse every pair
    all_traces = np.zeros((args.N, args.max_iter, len(methods)), dtype=np.float64)

    for idx, (a, b) in enumerate(tqdm(test_pairs, desc="Warm-start benchmark")):
        f_meta, g_meta = get_meta_potentials(a, b, mlp_meta, lf_meta, dev)
        f_reg,  g_reg  = model_reg._predict_potentials(a, b, model_reg.alpha)
        f_obj,  g_obj  = model_obj._predict_potentials(a, b, model_obj.alpha)

        # Batch order must match `methods` above.
        f_inits = [None, f_meta, f_reg, f_obj]
        g_inits = [None, g_meta, g_reg, g_obj]

        err_trace = sinkhorn_warmstart_trace(
            a, b, C_t, eps, f_inits, g_inits, args.max_iter, dev)  # (max_iter, 4)
        all_traces[idx] = err_trace

    # ------------------------------------------------------------------- report
    mean_trace = all_traces.mean(axis=0)   # (max_iter, 4)
    std_trace  = all_traces.std(axis=0)    # (max_iter, 4)

    print(f"\n===== Marginal error at fixed iteration checkpoints "
          f"(mean over N={args.N} test pairs) =====")
    checkpoints = [c for c in [1, 2, 5, 10, 20, 50, 100, 200, args.max_iter] if c <= args.max_iter]
    header = "  {:<14}".format("Init method") + "".join(f"{'iter='+str(c):>14}" for c in checkpoints)
    print(header)
    rows_csv = []
    for m_idx, m in enumerate(methods):
        vals = [mean_trace[c - 1, m_idx] for c in checkpoints]
        print(f"  {m:<14}" + "".join(f"{v:>14.2e}" for v in vals))
        rows_csv.append([m] + vals)

    csv_path = os.path.join(args.out, "q2_warmstart_error_checkpoints.csv")
    with open(csv_path, "w") as f:
        f.write("init_method," + ",".join(f"err_at_iter_{c}" for c in checkpoints) + "\n")
        for row in rows_csv:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"\nCheckpoint table -> {csv_path}")

    np.save(os.path.join(args.out, "all_traces.npy"), all_traces)
    print(f"Raw traces (N={args.N}, max_iter={args.max_iter}, 4 methods) -> "
          f"{os.path.join(args.out, 'all_traces.npy')}")

    # ----------------------------------------------------------------- plotting
    # Reproduces the style of the Meta-OT paper's own Figure 3: error vs.
    # Sinkhorn iteration, mean line + shaded std band, averaged over the test set.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"cold_start": "tab:blue", "meta_ot": "tab:red",
                  "ra_ot": "tab:purple", "oa_ot": "tab:green"}
        x = np.arange(1, args.max_iter + 1)

        plt.figure(figsize=(6.5, 4.5))
        for m_idx, m in enumerate(methods):
            mean_m = mean_trace[:, m_idx]
            std_m  = std_trace[:, m_idx]
            plt.plot(x, mean_m, label=labels[m], color=colors[m])
            plt.fill_between(x, np.clip(mean_m - std_m, 0, None), mean_m + std_m,
                              color=colors[m], alpha=0.2)
        plt.xlabel("Sinkhorn Iterations")
        plt.ylabel("Error")
        plt.title(f"MNIST (M={args.M}, N={args.N}, eps={eps})")
        plt.legend(fontsize=8, title="Initialization")
        plt.tight_layout()
        fig_path = os.path.join(args.out, "q2_convergence_curves.png")
        plt.savefig(fig_path, dpi=150)
        print(f"Figure  -> {fig_path}")

        # Zoomed-in version (first ~50 iters) — the early-iteration gap is
        # usually the most visually convincing part, easy to miss on a full
        # max_iter=500 x-axis.
        zoom = min(50, args.max_iter)
        plt.figure(figsize=(6.5, 4.5))
        for m_idx, m in enumerate(methods):
            mean_m = mean_trace[:zoom, m_idx]
            std_m  = std_trace[:zoom, m_idx]
            plt.plot(x[:zoom], mean_m, label=labels[m], color=colors[m])
            plt.fill_between(x[:zoom], np.clip(mean_m - std_m, 0, None), mean_m + std_m,
                              color=colors[m], alpha=0.2)
        plt.xlabel("Sinkhorn Iterations")
        plt.ylabel("Error")
        plt.title(f"MNIST (M={args.M}, N={args.N}, eps={eps}) — first {zoom} iters")
        plt.legend(fontsize=8, title="Initialization")
        plt.tight_layout()
        fig_zoom_path = os.path.join(args.out, "q2_convergence_curves_zoom.png")
        plt.savefig(fig_zoom_path, dpi=150)
        print(f"Figure  -> {fig_zoom_path}")
    except ImportError:
        print("matplotlib not available — skipped plotting, all_traces.npy still saved.")


if __name__ == "__main__":
    main()
