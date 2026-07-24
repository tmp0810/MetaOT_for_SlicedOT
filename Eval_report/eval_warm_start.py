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
from eval_grayscale import (
    build_cost_grid, sample_pairs, pairs_to_loader,
    POOL_SEED, POOL_SIZE, TRAIN_RATIO,
)

from Solvers.Regression_SlicedOT.OT_Regression_Sliced import OT_Regression_Sliced
from Solvers.Objective_SlicedOT.OT_Objective_Sliced import OT_Objective_Sliced
from Solvers.Meta_OT.Meta_OT_gray_scale import OT_Discrete, dual_obj_loss
from Models.ot_models import PotentialMLP

def sinkhorn_warmstart_batch(a, b, C_t, eps, f_inits, g_inits, tol, max_iter, device):
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

    log_u = torch.zeros(B, n, dtype=dtype, device=device)
    log_v = torch.zeros(B, n, dtype=dtype, device=device)
    for i in range(B):
        if f_inits[i] is not None:
            log_u[i] = torch.tensor(f_inits[i], dtype=dtype, device=device) / eps
        if g_inits[i] is not None:
            log_v[i] = torch.tensor(g_inits[i], dtype=dtype, device=device) / eps

    n_iter_used   = torch.full((B,), max_iter, dtype=torch.long, device=device)
    converged_msk = torch.zeros(B, dtype=torch.bool, device=device)
    err_trace_first = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for it in range(1, max_iter + 1):
            M1 = log_K + log_v.unsqueeze(1)          # (B, n, n)
            log_u = log_a - torch.logsumexp(M1, dim=2)
            M2 = log_K + log_u.unsqueeze(2)          # (B, n, n)
            log_v = log_b - torch.logsumexp(M2, dim=1)

            log_P = log_u.unsqueeze(2) + log_K + log_v.unsqueeze(1)
            P = torch.exp(log_P)
            err_row = (P.sum(dim=2) - a_t).abs().sum(dim=1)
            err_col = (P.sum(dim=1) - b_t).abs().sum(dim=1)
            err = torch.maximum(err_row, err_col)   # (B,)

            err_trace_first.append(err[0].item())

            newly = (~converged_msk) & (err < tol)
            n_iter_used[newly] = it
            converged_msk = converged_msk | newly

            if converged_msk.all():
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_elapsed = time.perf_counter() - t0

    return (n_iter_used.cpu().tolist(), t_elapsed,
            converged_msk.cpu().tolist(), err_trace_first)

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

    print("\n[1/3] Training RA-OT ...")
    cfg_r = init_cfg("OT_Regression_Sliced")
    cfg_r["num_bootstrap"] = args.M
    cfg_r["epsilon"]       = eps
    model_reg = OT_Regression_Sliced(
        make_cfg_proj("OT_Regression_Sliced", POOL_SEED, args.gpu, flag_time), cfg_r)
    model_reg.alpha = model_reg._fit(dl_train)
    model_reg.beta  = np.zeros(cfg_r["num_projections"])
    save_model(model_reg, os.path.join(args.out, "regression.pkl"))

    print("\n[2/3] Training OA-OT ...")
    cfg_o = init_cfg("OT_Objective_Sliced")
    cfg_o["num_bootstrap"] = args.M
    cfg_o["epsilon"]       = eps
    model_obj = OT_Objective_Sliced(
        make_cfg_proj("OT_Objective_Sliced", POOL_SEED, args.gpu, flag_time), cfg_o)
    model_obj.alpha = model_obj._fit(dl_train)
    model_obj.beta  = np.zeros(cfg_o["num_projections"])
    save_model(model_obj, os.path.join(args.out, "objective.pkl"))

    print("\n[3/3] Training Meta-OT ...")
    cfg_meta = init_cfg("OT_Discrete")
    cfg_meta["epsilon"] = eps   
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
          f"(tol={args.tol:.0e}, max_iter={args.max_iter}, device={dev}) ...")

    methods = ["cold_start", "meta_ot", "ra_ot", "oa_ot"]
    stats   = {m: {"n_iter": [], "time": []} for m in methods}
    example_traces = None  # save one pair's full trace (batch element 0) for plotting

    C_t = torch.tensor(C, dtype=torch.float64, device=dev)  # precompute once, reuse every pair

    for idx, (a, b) in enumerate(tqdm(test_pairs, desc="Warm-start benchmark")):
        f_meta, g_meta = get_meta_potentials(a, b, mlp_meta, lf_meta, dev)
        f_reg,  g_reg  = model_reg._predict_potentials(a, b, model_reg.alpha)
        f_obj,  g_obj  = model_obj._predict_potentials(a, b, model_obj.alpha)

        # Batch order must match `methods` above.
        f_inits = [None, f_meta, f_reg, f_obj]
        g_inits = [None, g_meta, g_reg, g_obj]

        n_iter_list, t_elapsed, converged_list, err_trace_first = sinkhorn_warmstart_batch(
            a, b, C_t, eps, f_inits, g_inits, args.tol, args.max_iter, dev)

    
        for m, n_iter, conv in zip(methods, n_iter_list, converged_list):
            stats[m]["n_iter"].append(n_iter)
            stats[m]["time"].append(t_elapsed / len(methods))
            if not conv:
                print(f"  [warn] pair {idx}, {m} did NOT converge within "
                      f"{args.max_iter} iters")

        if idx == 0:
            example_traces = {"cold_start": err_trace_first}
            for m, f_i, g_i in zip(["meta_ot", "ra_ot", "oa_ot"],
                                    [f_meta, f_reg, f_obj], [g_meta, g_reg, g_obj]):
                _, _, _, trace_m = sinkhorn_warmstart_batch(
                    a, b, C_t, eps, [f_i], [g_i], args.tol, args.max_iter, dev)
                example_traces[m] = trace_m

    # ------------------------------------------------------------------- report
    print("\n===== Warm-start Sinkhorn convergence (mean over N=%d test pairs) ====="
          % args.N)
    header = f"  {'Init method':<14} {'Iters (mean±std)':>22} {'Time (ms, mean±std)':>24} {'Speedup vs cold':>18}"
    print(header)
    cold_mean_iter = np.mean(stats["cold_start"]["n_iter"])
    rows = []
    for m in methods:
        it_arr = np.array(stats[m]["n_iter"], dtype=np.float64)
        tm_arr = np.array(stats[m]["time"],   dtype=np.float64) * 1000.0
        speedup = cold_mean_iter / it_arr.mean()
        print(f"  {m:<14} {it_arr.mean():>10.1f} ± {it_arr.std():<8.1f} "
              f"{tm_arr.mean():>12.2f} ± {tm_arr.std():<8.2f} "
              f"{speedup:>16.2f}x")
        rows.append((m, it_arr.mean(), it_arr.std(), tm_arr.mean(), tm_arr.std(), speedup))

    csv_path = os.path.join(args.out, "q2_warmstart_results.csv")
    with open(csv_path, "w") as f:
        f.write("init_method,iters_mean,iters_std,time_ms_mean,time_ms_std,speedup_vs_cold\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"\nResults -> {csv_path}")

    with open(os.path.join(args.out, "example_traces.pkl"), "wb") as f:
        pickle.dump(example_traces, f)

    # ----------------------------------------------------------------- plotting
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4.5))
        labels = {"cold_start": "Cold start (f=g=0)", "meta_ot": "Meta-OT warm start",
                  "ra_ot": "RA-OT warm start", "oa_ot": "OA-OT warm start"}
        for m in methods:
            trace = example_traces[m]
            plt.plot(np.arange(1, len(trace) + 1), trace, label=labels[m])
        plt.axhline(args.tol, color="gray", linestyle=":", label=f"tol={args.tol:.0e}")
        plt.yscale("log")
        plt.xlabel("Sinkhorn iteration")
        plt.ylabel("marginal L1 error (max of row/col)")
        plt.title("Warm-start vs. cold-start Sinkhorn convergence (example pair)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        fig_path = os.path.join(args.out, "q2_convergence_curves.png")
        plt.savefig(fig_path, dpi=150)
        print(f"Figure  -> {fig_path}")
    except ImportError:
        print("matplotlib not available — skipped plotting, .pkl trace still saved.")


if __name__ == "__main__":
    main()
