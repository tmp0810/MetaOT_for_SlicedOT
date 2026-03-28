import argparse
import os
import pickle
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

POOL_SEED   = 0
POOL_SIZE   = 1000
TRAIN_RATIO = 0.7   # 490 train / 210 test


def sinkhorn_gt(a, b, C, eps, n_iter=500):
    a_s = np.clip(a, 1e-10, None); a_s /= a_s.sum()
    b_s = np.clip(b, 1e-10, None); b_s /= b_s.sum()
    return ot.sinkhorn(a_s, b_s, C, reg=eps, numItermax=n_iter, stopThr=1e-9)


def sample_pair(n_supply, n_demand, seed):
    rng = np.random.default_rng(seed)
    mask = rng.binomial(1, 0.5, n_supply).astype(np.float64)
    a = mask * rng.uniform(0, 1, n_supply)
    if a.sum() < 1e-12: a = np.ones(n_supply, dtype=np.float64)
    a /= a.sum()
    b = rng.uniform(0, 1, n_demand).astype(np.float64); b /= b.sum()
    return a, b


def evaluate(predict_fn, test_pairs, C, eps, name):
    # Warmup: eliminate CUDA cold-start outlier from timing
    if test_pairs:
        try: predict_fn(*test_pairs[0])
        except Exception: pass

    rmse_list, time_list = [], []
    for a, b in tqdm(test_pairs, desc=f"  Eval {name}", leave=False):
        P_gt = sinkhorn_gt(a, b, C, eps)
        t0   = time.perf_counter()
        P    = predict_fn(a, b)
        time_list.append(time.perf_counter() - t0)
        rmse_list.append(float(np.sqrt(np.mean((P - P_gt)**2))))
    return np.array(rmse_list), np.array(time_list)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Saved → {path}")


def pairs_to_world_loader(pairs_list, batch_size=1):
    """Wrap list of (a, b) tuples into DataLoader — yields (0,0,a_t,b_t)."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    class _DS(Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, i):
            a, b = self.data[i]
            return (torch.zeros(1), torch.zeros(1),
                    torch.tensor(a, dtype=torch.float64),
                    torch.tensor(b, dtype=torch.float64))
    return DataLoader(_DS(pairs_list), batch_size=batch_size, shuffle=False)


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


def make_cfg_proj(solver, seed, gpu, flag_time):
    import argparse as _ap
    return _ap.Namespace(seed=seed, flag_time=flag_time,
                         flag_load=None, solver=solver,
                         data_name="world_pair", gpu=gpu)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(os.path.join(args.out, f"M{args.M}"), exist_ok=True)
    flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())

    print("Loading world locations ...")
    supply_sph, supply_euc, demand_sph, demand_euc = load_world_locations(
        args.pop_tiff, n_supply=args.n_supply, n_demand=args.n_demand, seed=POOL_SEED)
    C   = _sphere_cost(supply_euc, demand_euc)
    eps = 0.5

    # ── Pre-sample pool ONCE, split 70/30, shared by ALL methods ──────────
    n_train_pool = int(POOL_SIZE * TRAIN_RATIO)   # 490
    n_test_pool  = POOL_SIZE - n_train_pool        # 210

    assert args.M <= n_train_pool, \
        f"M={args.M} exceeds train pool size {n_train_pool}"
    assert args.N <= n_test_pool, \
        f"N={args.N} exceeds test pool size {n_test_pool}"

    print(f"\nPre-sampling pool of {POOL_SIZE} pairs (seed={POOL_SEED}) ...")
    # Mỗi pair dùng seed = POOL_SEED + i để đảm bảo reproducible
    # Train pool: index 0..489   → seed POOL_SEED + 0  .. POOL_SEED + 489
    # Test pool:  index 490..699 → seed POOL_SEED + 490 .. POOL_SEED + 699
    # → không bao giờ overlap về seed
    pool = [sample_pair(args.n_supply, args.n_demand, POOL_SEED + i)
            for i in range(POOL_SIZE)]

    train_pool = pool[:n_train_pool]   # first 490 — train
    test_pool  = pool[n_train_pool:]   # last  210 — test, never touched during train

    train_pairs = train_pool[:args.M]  # first M of train pool (nested)
    test_pairs  = test_pool[:args.N]   # first N of test pool (fixed across all M)

    print(f"  Pool: {POOL_SIZE}  →  train pool: {n_train_pool}  |  test pool: {n_test_pool}")
    print(f"  Using M={args.M} train pairs  |  N={args.N} test pairs")
    print(f"  → All 4 methods will use EXACTLY these same pairs.\n")

    # Save test pairs for plot scripts
    test_pairs_path = os.path.join(args.out, f"M{args.M}", "test_pairs.pkl")
    os.makedirs(os.path.dirname(test_pairs_path), exist_ok=True)
    with open(test_pairs_path, "wb") as _f:
        pickle.dump({"pairs": test_pairs,
                     "supply_euc": supply_euc, "demand_euc": demand_euc,
                     "supply_sph": supply_sph, "demand_sph": demand_sph}, _f)
    print(f"  Test pairs saved → {test_pairs_path}")

    dl_shared = pairs_to_world_loader(train_pairs, batch_size=1)
    print(f"  {len(train_pairs)} train pairs ready.\n")

    results = []

    # # ── 1. OT Regression Sliced World ─────────────────────────────────────
    # print("\n[1/4] OT Regression Sliced World (Method 1) ...")
    # cfg1 = init_cfg("OT_Regression_Sliced_World")
    # cfg1["num_bootstrap"] = args.M; cfg1["epsilon"] = eps
    # cfg1["n_supply"] = args.n_supply; cfg1["n_demand"] = args.n_demand
    # model1 = OT_Regression_Sliced_World(
    #     make_cfg_proj("OT_Regression_Sliced_World", POOL_SEED, args.gpu, flag_time),
    #     cfg1, supply_euc, demand_euc, supply_sph, demand_sph)
    # t0 = time.perf_counter()
    # model1.alpha = model1._fit(dl_shared)
    # model1.beta  = np.zeros_like(model1.alpha)
    # t1 = time.perf_counter() - t0
    # save_model(model1, os.path.join(args.out, f"M{args.M}", "regression.pkl"))
    # rmse1, tinf1 = evaluate(model1.predict_plan, test_pairs, C, eps, "OT_Regression")
    # results.append(("OT Regression (M1)", rmse1, tinf1, t1))
    # print(f"  Train: {t1:.1f}s  RMSE: {rmse1.mean():.2e}")

    # # ── 2. OT Objective Sliced World ──────────────────────────────────────
    # print("\n[2/4] OT Objective Sliced World (Method 2) ...")
    # from Solvers.Objective_SlicedOT.OT_Objective_Sliced_World import OT_Objective_Sliced_World
    # cfg2 = init_cfg("OT_Objective_Sliced_World")
    # cfg2["num_bootstrap"] = args.M; cfg2["epsilon"] = eps
    # cfg2["n_supply"] = args.n_supply; cfg2["n_demand"] = args.n_demand
    # model2 = OT_Objective_Sliced_World(
    #     make_cfg_proj("OT_Objective_Sliced_World", POOL_SEED, args.gpu, flag_time),
    #     cfg2, supply_euc, demand_euc, supply_sph, demand_sph)
    # t0 = time.perf_counter()
    # model2.alpha = model2._fit(dl_shared)
    # model2.beta = np.zeros_like(model2.alpha)
    # t2 = time.perf_counter() - t0
    # save_model(model2, os.path.join(args.out, f"M{args.M}", "objective.pkl"))
    # rmse2, tinf2 = evaluate(model2.predict_plan, test_pairs, C, eps, "OT_Objective")
    # results.append(("OT Objective (M2)", rmse2, tinf2, t2))
    # print(f"  Train: {t2:.1f}s  RMSE: {rmse2.mean():.2e}")

    # # ── 3. Meta OT World ──────────────────────────────────────────────────
    # print("\n[3/4] Meta OT World (baseline) ...")
    # from Solvers.Meta_OT.Meta_OT_World import Meta_OT_World
    # cfg3 = init_cfg("Meta_OT_World")
    # cfg3["epsilon"] = eps; cfg3["n_supply"] = args.n_supply; cfg3["n_demand"] = args.n_demand
    # model3 = Meta_OT_World(
    #     make_cfg_proj("Meta_OT_World", POOL_SEED, args.gpu, flag_time),
    #     cfg3, supply_euc, demand_euc, supply_sph, demand_sph)
    # t0 = time.perf_counter()
    # model3.train(dl_shared)
    # t3 = time.perf_counter() - t0
    # save_model(model3, os.path.join(args.out, f"M{args.M}", "meta_ot.pkl"))
    # rmse3, tinf3 = evaluate(model3.predict_plan, test_pairs, C, eps, "Meta OT")
    # results.append(("Meta OT (baseline)", rmse3, tinf3, t3))
    # print(f"  Train: {t3:.1f}s  RMSE: {rmse3.mean():.2e}")

    # # ── 4. min-SWGG World ─────────────────────────────────────────────────
    # print("\n[4/4] min-SWGG World (baseline, no training) ...")
    # from Solvers.SWGG.min_SWGG_World import min_SWGG_World
    # cfg4 = init_cfg("min_SWGG_World")
    # cfg4["epsilon"] = eps; cfg4["n_supply"] = args.n_supply; cfg4["n_demand"] = args.n_demand
    # model4 = min_SWGG_World(
    #     make_cfg_proj("min_SWGG_World", POOL_SEED, args.gpu, flag_time),
    #     cfg4, supply_euc, demand_euc, supply_sph, demand_sph)
    # save_model(model4, os.path.join(args.out, f"M{args.M}", "swgg.pkl"))
    # rmse4, tinf4 = evaluate(model4.predict_plan, test_pairs, C, eps, "min-SWGG")
    # results.append(("min-SWGG (baseline)", rmse4, tinf4, 0.0))
    # print(f"  RMSE: {rmse4.mean():.2e}")

    # ── 5. Min-STP World ──────────────────────────────────────────────────
    print("\n[5/5] Min-STP World (baseline) ...")
    from Solvers.MinSTP.Min_STP_World import Min_STP_World
    cfg3 = init_cfg("Min_STP_World")
    cfg3["n_supply"] = args.n_supply; cfg3["n_demand"] = args.n_demand
    model3 = Min_STP_World(
        make_cfg_proj("Min_STP_World", POOL_SEED, args.gpu, flag_time),
        cfg3, supply_euc, demand_euc, supply_sph, demand_sph)
    t0 = time.perf_counter()
    model3.train(dl_shared)
    t3 = time.perf_counter() - t0
    save_model(model3, os.path.join(args.out, f"M{args.M}", "min_stp.pkl"))
    rmse3, tinf3 = evaluate(model3.predict_plan, test_pairs, C, eps, "Min-STP")
    results.append(("Min-STP (baseline)", rmse3, tinf3, t3))
    print(f"  Train: {t3:.1f}s  RMSE: {rmse3.mean():.2e}")

    print_table(results, args.M, args.N)

    csv_path = os.path.join(args.out, f"results_M{args.M}.csv")
    with open(csv_path, "w") as f:
        f.write("method,rmse_mean,rmse_std,train_s,infer_ms_mean,infer_ms_std\n")
        for name, rmse_arr, time_arr, t_train in results:
            f.write(f"{name},{rmse_arr.mean():.6e},{rmse_arr.std():.6e},"
                    f"{t_train:.2f},{time_arr.mean()*1000:.4f},{time_arr.std()*1000:.4f}\n")
    print(f"Results → {csv_path}  |  Models → {args.out}/M{args.M}/")


if __name__ == "__main__":
    main()
