"""
W2 follow-up ablation: is the 14-18% marginal violation of RA-OT/OA-OT
coming from the quality of the predicted potentials, or from the extra
row-then-column hard-rescale step in _potentials_to_plan stacking on top
of the (already marginal-consistent-ish) Sinkhorn half-step done inside
_predict_potentials?

We compare 3 ways of turning the SAME predicted (f, g) into a plan:
  1. raw          : P = exp((f + g - C)/eps), no post-processing at all
  2. mass_norm     : raw, then divide by its total sum (single scalar,
                     no per-row/per-col correction)
  3. row_col_rescale (current pipeline): _potentials_to_plan's row-then-
                     column hard rescale (row exact, then col exact,
                     which perturbs row again)

For each variant we report RMSE(plan) and both L1 marginal errors, on
the SAME trained model / SAME test pairs, so any difference is purely
attributable to the post-processing step -- nothing about training or
potential quality changes.

Only RA-OT and OA-OT are relevant here (Meta-OT's recovery formula does
not go through this row/col rescale code path at all).

Place at: Eval_report/eval_w2_ablation.py
Run:
    python Eval_report/eval_w2_ablation.py --M 50 --N 300 --gpu 0
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


# ---------------------------------------------------------------------
# The 3 recovery variants, built directly from a solver's own
# _predict_potentials -- no solver code is modified, this only reuses
# what's already there plus one extra "raw" formula copied verbatim
# from the un-rescaled part of _potentials_to_plan.
# ---------------------------------------------------------------------
def recover_variants(model, a, b, alpha, C, eps):
    f, g = model._predict_potentials(a, b, alpha)

    f_c = f - f.mean()
    g_c = g - g.mean()
    log_P = f_c[:, None] / eps - C / eps + g_c[None, :] / eps
    log_P -= log_P.max()
    P_raw = np.exp(log_P)                      # variant 1: no post-processing

    total = P_raw.sum()
    P_mass = P_raw / total if total > 0 else P_raw  # variant 2: single scalar renorm

    P_final = model._potentials_to_plan(a, b, f, g)  # variant 3: current pipeline

    return {"raw": P_raw, "mass_norm": P_mass, "row_col_rescale (current)": P_final}


def evaluate_variants(model, alpha, test_pairs, C, eps, name):
    stats = {k: {"rmse": [], "erra": [], "errb": []}
             for k in ["raw", "mass_norm", "row_col_rescale (current)"]}
    for a, b in tqdm(test_pairs, desc=f"  {name} variants", leave=False):
        P_gt = sinkhorn_gt(a, b, C, eps)
        variants = recover_variants(model, a, b, alpha, C, eps)
        for k, P in variants.items():
            rmse = float(np.sqrt(np.mean((P - P_gt) ** 2)))
            ea, eb = marginal_l1(P, a, b)
            stats[k]["rmse"].append(rmse)
            stats[k]["erra"].append(ea)
            stats[k]["errb"].append(eb)
    for k in stats:
        for m in stats[k]:
            stats[k][m] = np.array(stats[k][m])
    return stats


def print_ablation_table(name, stats):
    print(f"\n  -- {name} --")
    print(f"    {'Variant':<28} {'RMSE_Plan':>14} {'MargErr_a':>14} {'MargErr_b':>14}")
    print(f"    {'-'*28} {'-'*14} {'-'*14} {'-'*14}")
    for k in ["raw", "mass_norm", "row_col_rescale (current)"]:
        rmse = stats[k]["rmse"]; ea = stats[k]["erra"]; eb = stats[k]["errb"]
        print(f"    {k:<28} {rmse.mean():.2e}±{rmse.std():.1e}  "
              f"{ea.mean():>12.3e}  {eb.mean():>12.3e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=50)
    p.add_argument("--N", type=int, default=300)
    p.add_argument("--gpu", type=str, default="0")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
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
    print(f"  M={args.M} train pairs | N={args.N} test pairs\n")

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

    print("\nRunning 3-variant recovery ablation on the test set ...")
    stats_reg = evaluate_variants(model_reg, alpha_reg, test_pairs, C, eps, "RA-OT")
    stats_obj = evaluate_variants(model_obj, alpha_obj, test_pairs, C, eps, "OA-OT")

    print(f"\n{'='*80}")
    print(f"  W2 ablation: where does the marginal violation come from?  "
          f"(M={args.M}, N={args.N})")
    print(f"{'='*80}")
    print_ablation_table("RA-OT", stats_reg)
    print_ablation_table("OA-OT", stats_obj)
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()