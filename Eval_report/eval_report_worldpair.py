import argparse
import pickle
import os
import time
import numpy as np
import torch
import ot
from tqdm import tqdm
from time import localtime, strftime

from cfg import init_cfg
from Data.world_pair_data import load_world_locations, get_world_pair_dataloader
from Solvers.Regression_SlicedOT.OT_Regression_Sliced_World import (
    OT_Regression_Sliced_World, _sphere_cost)

TRAIN_SEED = 0
TEST_SEED  = 999


def sinkhorn_gt(a, b, C, eps, n_iter=500):
    a_s = np.clip(a, 1e-10, None); a_s /= a_s.sum()
    b_s = np.clip(b, 1e-10, None); b_s /= b_s.sum()
    return ot.sinkhorn(a_s, b_s, C, reg=eps, numItermax=n_iter, stopThr=1e-9)


def sample_pair(n_supply, n_demand, seed, bernoulli_p=0.5):
    rng  = np.random.default_rng(seed)
    mask = rng.binomial(1, bernoulli_p, n_supply).astype(np.float64)
    a    = mask * rng.uniform(0, 1, n_supply)
    if a.sum() < 1e-12: a = np.ones(n_supply, dtype=np.float64)
    a   /= a.sum()
    b    = rng.uniform(0, 1, n_demand).astype(np.float64); b /= b.sum()
    return a, b


def collect_test_pairs(N, n_supply, n_demand, seed=TEST_SEED):
    return [sample_pair(n_supply, n_demand, seed + i) for i in range(N)]


def evaluate(model, test_pairs, C, eps, name):
    rmse_list, time_list = [], []
    for a, b in tqdm(test_pairs, desc=f"  Eval {name}", leave=False):
        P_gt = sinkhorn_gt(a, b, C, eps)
        t0   = time.perf_counter()
        P    = model.predict_plan(a, b)
        time_list.append(time.perf_counter() - t0)
        rmse_list.append(float(np.sqrt(np.mean((P - P_gt)**2))))
    return np.array(rmse_list), np.array(time_list)


def print_table(results, M, N):
    print(f"\n{'='*72}")
    print(f"  World Pair  |  M={M} train pairs  |  N={N} test pairs")
    print(f"{'='*72}")
    print(f"  {'Method':<28} {'RMSE_Plan':>14} {'Train (s)':>12} {'Infer (ms)':>12}")
    print(f"  {'-'*28} {'-'*14} {'-'*12} {'-'*12}")
    for name, rmse_arr, time_arr, t_train in results:
        train_str = f"{t_train:.1f}" if t_train > 0 else "0 (no train)"
        print(f"  {name:<28} "
              f"{rmse_arr.mean():.2e}±{rmse_arr.std():.1e}  "
              f"{train_str:>10}s  "
              f"{time_arr.mean()*1000:.2f}±{time_arr.std()*1000:.2f}ms")
    print(f"{'='*72}\n")



def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Saved → {path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pop_tiff", type=str, required=True)
    p.add_argument("--M",        type=int,   default=50)
    p.add_argument("--N",        type=int,   default=20)
    p.add_argument("--n_supply", type=int,   default=100)
    p.add_argument("--n_demand", type=int,   default=10_000)
    p.add_argument("--gpu",      type=str,   default="0")
    p.add_argument("--out",      type=str,   default="./results/worldpair")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out, exist_ok=True)
    import argparse as _ap

    print("Loading world locations ...")
    supply_sph, supply_euc, demand_sph, demand_euc = load_world_locations(
        args.pop_tiff, n_supply=args.n_supply, n_demand=args.n_demand, seed=TRAIN_SEED)
    C   = _sphere_cost(supply_euc, demand_euc)
    eps = 0.5

    # Fixed test pairs (same for ALL methods)
    test_pairs = collect_test_pairs(args.N, args.n_supply, args.n_demand)
    print(f"Test pairs: {len(test_pairs)} (seed={TEST_SEED})")
    results = []

    def make_cfg_proj(solver, seed):
        return _ap.Namespace(seed=seed,
                             flag_time=strftime("%Y-%m-%d_%H-%M-%S", localtime()),
                             flag_load=None, solver=solver,
                             data_name="world_pair", gpu=args.gpu)

    def make_train_loader(cfg_m, seed):
        return get_world_pair_dataloader(
            n_supply=args.n_supply, n_demand=args.n_demand,
            batch_size=cfg_m["batch_size"],
            supply_bernoulli_p=cfg_m["supply_bernoulli_p"],
            num_pairs=args.M, seed=seed)

    # ── 1. OT Regression Sliced World ─────────────────────────────────────
    print("\n[1/4] OT Regression Sliced World (Method 1) ...")
    cfg_m1 = init_cfg("OT_Regression_Sliced_World")
    cfg_m1["num_bootstrap"] = args.M; cfg_m1["epsilon"] = eps
    cfg_m1["n_supply"] = args.n_supply; cfg_m1["n_demand"] = args.n_demand
    model1 = OT_Regression_Sliced_World(make_cfg_proj("OT_Regression_Sliced_World", TRAIN_SEED),
                                         cfg_m1, supply_euc, demand_euc, supply_sph, demand_sph)
    dl1 = make_train_loader(cfg_m1, TRAIN_SEED)
    t0 = time.perf_counter()
    model1.alpha, model1.beta = model1._fit(dl1)
    t1 = time.perf_counter() - t0
    save_model(model1, os.path.join(args.out, f"M{args.M}", "regression.pkl"))
    rmse1, tinf1 = evaluate(model1, test_pairs, C, eps, "OT_Regression")
    results.append(("OT Regression (M1)", rmse1, tinf1, t1))
    print(f"  Train: {t1:.1f}s  RMSE: {rmse1.mean():.2e}")

    # ── 2. OT Objective Sliced World ──────────────────────────────────────
    print("\n[2/4] OT Objective Sliced World (Method 2) ...")
    from Solvers.Objective_SlicedOT.OT_Objective_Sliced_World import OT_Objective_Sliced_World
    cfg_m2 = init_cfg("OT_Objective_Sliced_World")
    cfg_m2["num_bootstrap"] = args.M; cfg_m2["epsilon"] = eps
    cfg_m2["n_supply"] = args.n_supply; cfg_m2["n_demand"] = args.n_demand
    model2 = OT_Objective_Sliced_World(make_cfg_proj("OT_Objective_Sliced_World", TRAIN_SEED),
                                        cfg_m2, supply_euc, demand_euc, supply_sph, demand_sph)
    dl2 = make_train_loader(cfg_m2, TRAIN_SEED)
    t0 = time.perf_counter()
    model2.alpha, model2.beta = model2._fit(dl2)
    t2 = time.perf_counter() - t0
    save_model(model2, os.path.join(args.out, f"M{args.M}", "objective.pkl"))
    rmse2, tinf2 = evaluate(model2, test_pairs, C, eps, "OT_Objective")
    results.append(("OT Objective (M2)", rmse2, tinf2, t2))
    print(f"  Train: {t2:.1f}s  RMSE: {rmse2.mean():.2e}")

    # ── 3. Meta OT World ──────────────────────────────────────────────────
    print("\n[3/4] Meta OT World (baseline) ...")
    from Solvers.Meta_OT.Meta_OT_World import Meta_OT_World
    cfg_m3 = init_cfg("Meta_OT_World")
    cfg_m3["num_bootstrap"] = args.M; cfg_m3["epsilon"] = eps
    cfg_m3["n_supply"] = args.n_supply; cfg_m3["n_demand"] = args.n_demand
    model3 = Meta_OT_World(make_cfg_proj("Meta_OT_World", TRAIN_SEED),
                            cfg_m3, supply_euc, demand_euc, supply_sph, demand_sph)
    dl3 = make_train_loader(cfg_m3, TRAIN_SEED)
    t0 = time.perf_counter()
    model3.train(dl3)
    t3 = time.perf_counter() - t0
    save_model(model3, os.path.join(args.out, f"M{args.M}", "meta_ot.pkl"))
    rmse3, tinf3 = evaluate(model3, test_pairs, C, eps, "Meta OT")
    results.append(("Meta OT (baseline)", rmse3, tinf3, t3))
    print(f"  Train: {t3:.1f}s  RMSE: {rmse3.mean():.2e}")

    # ── 4. min-SWGG World ─────────────────────────────────────────────────
    print("\n[4/4] min-SWGG World (baseline, no training) ...")
    from Solvers.SWGG.min_SWGG_World import min_SWGG_World
    cfg_m4 = init_cfg("min_SWGG_World")
    cfg_m4["epsilon"] = eps
    cfg_m4["n_supply"] = args.n_supply; cfg_m4["n_demand"] = args.n_demand
    model4 = min_SWGG_World(make_cfg_proj("min_SWGG_World", TRAIN_SEED),
                             cfg_m4, supply_euc, demand_euc, supply_sph, demand_sph)
    rmse4, tinf4 = evaluate(model4, test_pairs, C, eps, "min-SWGG")
    results.append(("min-SWGG (baseline)", rmse4, tinf4, 0.0))
    save_model(model4, os.path.join(args.out, f"M{args.M}", "swgg.pkl"))
    print(f"  RMSE: {rmse4.mean():.2e}")

    print_table(results, args.M, args.N)

    csv_path = os.path.join(args.out, f"results_M{args.M}.csv")
    with open(csv_path, "w") as f:
        f.write("method,rmse_mean,rmse_std,train_s,infer_ms_mean,infer_ms_std\n")
        for name, rmse_arr, time_arr, t_train in results:
            f.write(f"{name},{rmse_arr.mean():.6e},{rmse_arr.std():.6e},"
                    f"{t_train:.2f},{time_arr.mean()*1000:.4f},{time_arr.std()*1000:.4f}\n")
    print(f"Results saved → {csv_path}")


if __name__ == "__main__":
    main()