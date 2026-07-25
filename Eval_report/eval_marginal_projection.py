
"""
W2 final experiment: warm-started Sinkhorn "rounding".
 
Starting from the predicted potentials (f0, g0) returned by
_predict_potentials (no retraining of omega), run K extra alternating
Sinkhorn half-steps:
    g <- _g_from_f(f, b, log_K, eps)
    f <- _f_from_g(g, a, log_K, eps)
using the SAME _g_from_f / _f_from_g already defined on the solver
(these are exactly the update rules that make one marginal exact by
construction -- see W2 discussion), then recover the plan with only a
single global mass-normalization (no hard row/col rescale). As K grows
this is just running more of the reference Sinkhorn algorithm itself,
so both marginals must converge towards the ground truth's (~1e-7) L1
error; the question is how large K needs to be, and what it costs.
 
K=0 reproduces the earlier "mass_norm" variant exactly (sanity check).
 
Only RA-OT and OA-OT are covered here (matches the earlier ablation
scope; no solver code is modified -- this only calls existing methods).
 
Place at: Eval_report/eval_w2_rounding.py
Run:
    python Eval_report/eval_w2_rounding.py --M 50 --N 300 --gpu 0
"""
import argparse
import os
import time
import numpy as np
import torch
import ot
from tqdm import tqdm
from time import localtime, strftime
from torch.utils.data import DataLoader
 
from cfg import init_cfg
from Data.dataset_class import MNIST
 
POOL_SEED = 0
POOL_SIZE = 1000
TRAIN_RATIO = 0.7
K_LIST = [0, 1, 2, 5, 10]
 
 
def build_cost_grid(img_size=28):
    grid = np.array([[j, i]
                     for i in np.linspace(1, 0, num=img_size)
                     for j in np.linspace(0, 1, num=img_size)], dtype=np.float64)
    diff = grid[:, None, :] - grid[None, :, :]
    return np.sum(diff ** 2, axis=-1)
 
 
def sinkhorn_gt(a, b, C, eps, n_iter=800):
    a_s = np.clip(a, 1e-10, None); a_s /= a_s.sum()
    b_s = np.clip(b, 1e-10, None); b_s /= b_s.sum()
    return ot.sinkhorn(a_s, b_s, C, reg=eps, numItermax=n_iter, stopThr=1e-9)
 
 
def marginal_l1(P, a, b):
    err_a = float(np.sum(np.abs(P.sum(axis=1) - a)))
    err_b = float(np.sum(np.abs(P.sum(axis=0) - b)))
    return err_a, err_b
 
 
def sample_pairs(n, seed):
    np.random.seed(seed)
    dataset = MNIST(flag_train=True, cfg_m=argparse.Namespace(datasets_root="../datasets"))
    pairs = []
    for _ in range(n):
        id_a, id_b = np.random.randint(0, len(dataset.data), 2)
        a = dataset.data[id_a].numpy()
        b = dataset.data[id_b].numpy()
        pairs.append((a, b))
    return pairs
 
 
def pairs_to_loader(pairs, batch_size=1):
    data = [(torch.zeros(1), torch.zeros(1),
             torch.tensor(a, dtype=torch.float64),
             torch.tensor(b, dtype=torch.float64))
            for a, b in pairs]
    return DataLoader(data, batch_size=batch_size, shuffle=False)
 
 
def make_cfg_proj(solver, seed, gpu, flag_time):
    return argparse.Namespace(seed=seed, flag_time=flag_time,
                              flag_load=None, solver=solver,
                              data_name="MNIST", gpu=gpu)
 
 
def potentials_to_plan_massnorm(f, g, C, eps):
    """Entropic formula + ONLY a single global mass renormalization
    (no per-row / per-column rescale). Same as the earlier 'mass_norm'
    ablation variant."""
    f_c = f - f.mean()
    g_c = g - g.mean()
    log_P = f_c[:, None] / eps - C / eps + g_c[None, :] / eps
    log_P -= log_P.max()
    P = np.exp(log_P)
    s = P.sum()
    return P / s if s > 0 else P
 
 
def evaluate_sinkhorn_rounding(model, alpha, test_pairs, C, eps, name):
    """For each test pair: start from the predicted (f0, g0), run up to
    max(K_LIST) extra alternating Sinkhorn half-steps, and record
    RMSE / marginal errors / cumulative extra time at every K in K_LIST."""
    log_K = model._precompute_log_K()  # cached, fixed -C/eps, 784x784
    dev = model.device
    out = {k: {"rmse": [], "erra": [], "errb": [], "extra_ms": []} for k in K_LIST}
 
    for a, b in tqdm(test_pairs, desc=f"  {name} Sinkhorn-round", leave=False):
        P_gt = sinkhorn_gt(a, b, C, eps)
        f0, g0 = model._predict_potentials(a, b, alpha)
 
        a_t = torch.tensor(a, dtype=torch.float64, device=dev)
        b_t = torch.tensor(b, dtype=torch.float64, device=dev)
        f_t = torch.tensor(f0, dtype=torch.float64, device=dev)
        g_t = torch.tensor(g0, dtype=torch.float64, device=dev)
 
        t_cum = 0.0
        done = 0
        for k in K_LIST:
            extra = k - done
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(extra):
                    g_t = model._g_from_f(f_t, b_t, log_K, eps)
                    f_t = model._f_from_g(g_t, a_t, log_K, eps)
            t_cum += time.perf_counter() - t0
            done = k
 
            P = potentials_to_plan_massnorm(f_t.cpu().numpy(), g_t.cpu().numpy(), C, eps)
            rmse = float(np.sqrt(np.mean((P - P_gt) ** 2)))
            ea, eb = marginal_l1(P, a, b)
            out[k]["rmse"].append(rmse)
            out[k]["erra"].append(ea)
            out[k]["errb"].append(eb)
            out[k]["extra_ms"].append(t_cum * 1000)
 
    for k in out:
        for m in out[k]:
            out[k][m] = np.array(out[k][m])
    return out
 
 
def print_rounding_table(name, out):
    print(f"\n  -- {name} --")
    print(f"    {'K (extra Sinkhorn steps)':<26} {'RMSE_Plan':>14} "
          f"{'MargErr_a':>13} {'MargErr_b':>13} {'Extra (ms)':>12}")
    print(f"    {'-'*26} {'-'*14} {'-'*13} {'-'*13} {'-'*12}")
    for k in K_LIST:
        r = out[k]
        print(f"    K={k:<24} {r['rmse'].mean():.2e}±{r['rmse'].std():.1e}  "
              f"{r['erra'].mean():>11.3e}  {r['errb'].mean():>11.3e}  "
              f"{r['extra_ms'].mean():>10.3f}")
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=50)
    p.add_argument("--N", type=int, default=300)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--out", type=str, default="./results/w2_rounding")
    return p.parse_args()
 
 
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out, exist_ok=True)
    flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())
 
    C = build_cost_grid(28)
    eps = 1e-2
 
    n_train_pool = int(POOL_SIZE * TRAIN_RATIO)
    n_test_pool = POOL_SIZE - n_train_pool
    assert args.M <= n_train_pool
    assert args.N <= n_test_pool
 
    print(f"\nPre-sampling pool of {POOL_SIZE} pairs (seed={POOL_SEED}) ...")
    pool = sample_pairs(POOL_SIZE, seed=POOL_SEED)
    train_pairs = pool[:n_train_pool][:args.M]
    test_pairs = pool[n_train_pool:][:args.N]
    print(f"  M={args.M} train pairs | N={args.N} test pairs | K in {K_LIST}\n")
 
    dl_train = pairs_to_loader(train_pairs, batch_size=1)
 
    print("[1/2] Training RA-OT ...")
    from Solvers.Regression_SlicedOT.OT_Regression_Sliced import OT_Regression_Sliced
    cfg_r = init_cfg("OT_Regression_Sliced")
    cfg_r["num_bootstrap"] = args.M; cfg_r["epsilon"] = eps
    model_reg = OT_Regression_Sliced(
        make_cfg_proj("OT_Regression_Sliced", POOL_SEED, args.gpu, flag_time), cfg_r)
    alpha_reg = model_reg._fit(dl_train)
 
    print("\n[2/2] Training OA-OT ...")
    from Solvers.Objective_SlicedOT.OT_Objective_Sliced import OT_Objective_Sliced
    cfg_o = init_cfg("OT_Objective_Sliced")
    cfg_o["num_bootstrap"] = args.M; cfg_o["epsilon"] = eps
    model_obj = OT_Objective_Sliced(
        make_cfg_proj("OT_Objective_Sliced", POOL_SEED, args.gpu, flag_time), cfg_o)
    alpha_obj = model_obj._fit(dl_train)
 
    print("\nRunning warm-started Sinkhorn-rounding sweep on the test set ...")
    out_reg = evaluate_sinkhorn_rounding(model_reg, alpha_reg, test_pairs, C, eps, "RA-OT")
    out_obj = evaluate_sinkhorn_rounding(model_obj, alpha_obj, test_pairs, C, eps, "OA-OT")
 
    print(f"\n{'='*88}")
    print(f"  W2: warm-started Sinkhorn rounding, RMSE / marginal error vs K  "
          f"(M={args.M}, N={args.N})")
    print(f"{'='*88}")
    print_rounding_table("RA-OT", out_reg)
    print_rounding_table("OA-OT", out_obj)
    print(f"\n{'='*88}\n")
 
    csv_path = os.path.join(args.out, f"results_M{args.M}.csv")
    with open(csv_path, "w") as f:
        f.write("method,K,rmse_mean,rmse_std,marg_err_a_mean,marg_err_b_mean,extra_ms_mean\n")
        for name, out in [("RA-OT", out_reg), ("OA-OT", out_obj)]:
            for k in K_LIST:
                r = out[k]
                f.write(f"{name},{k},{r['rmse'].mean():.6e},{r['rmse'].std():.6e},"
                        f"{r['erra'].mean():.6e},{r['errb'].mean():.6e},"
                        f"{r['extra_ms'].mean():.4f}\n")
    print(f"Results -> {csv_path}")
 
 
if __name__ == "__main__":
    main()
