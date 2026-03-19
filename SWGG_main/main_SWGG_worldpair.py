import argparse
import os
import pickle
import time
import numpy as np

from cfg import init_cfg
from Data.world_pair_data import load_world_locations, get_world_pair_dataloader
from Solvers.SWGG.min_SWGG_World import min_SWGG_World
from Solvers.Regression_SlicedOT.OT_Regression_Sliced_World import _sphere_cost
from time import localtime, strftime


def parse_args():
    p = argparse.ArgumentParser(
        description="min-SWGG baseline for WorldPair spherical OT")
    p.add_argument("--pop_tiff",      type=str, required=True)
    p.add_argument("--out_dir",       type=str, default="./runs/min_swgg_world")
    p.add_argument("--n_supply",      type=int, default=100)
    p.add_argument("--n_demand",      type=int, default=10_000)
    p.add_argument("--n_projections", type=int, default=None,
                   help="L random directions (default: cfg=200)")
    p.add_argument("--num_samples",   type=int, default=5,
                   help="Test pairs for timing/quality evaluation")
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--gpu",           type=str, default="0")
    return p.parse_args()


def sample_one_pair(n_supply, n_demand, supply_bernoulli_p=0.5, seed=0):
    rng      = np.random.default_rng(seed)
    mask     = rng.binomial(1, supply_bernoulli_p, n_supply).astype(np.float64)
    supply_w = mask * rng.uniform(0.0, 1.0, n_supply)
    if supply_w.sum() < 1e-12:
        supply_w = np.ones(n_supply, dtype=np.float64)
    supply_w /= supply_w.sum()
    demand_w  = rng.uniform(0.0, 1.0, n_demand).astype(np.float64)
    demand_w /= demand_w.sum()
    return supply_w, demand_w


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    cfg_m = init_cfg("min_SWGG_World")
    if args.n_projections is not None:
        cfg_m["n_projections"] = args.n_projections
    cfg_m["n_supply"] = args.n_supply
    cfg_m["n_demand"] = args.n_demand
    cfg_m["gpu"] = int(args.gpu) if args.gpu.isdigit() else 0

    cfg_proj = argparse.Namespace(
        seed      = args.seed,
        flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime()),
        flag_load = None,
        solver    = "min_SWGG_World",
        data_name = "world_pair",
        gpu       = args.gpu,
    )

    print(f"\n{'='*55}")
    print(f"  min-SWGG baseline — WorldPair Spherical OT")
    print(f"  No training: test-time θ* random search")
    print(f"{'='*55}")
    print(f"  n_supply      : {args.n_supply}")
    print(f"  n_demand      : {args.n_demand}")
    print(f"  n_projections : {cfg_m['n_projections']}")
    print(f"  epsilon       : {cfg_m['epsilon']}  (Sinkhorn comparison)")
    print(f"{'='*55}\n")

    # Load fixed locations
    print("Loading world locations ...")
    supply_sph, supply_euc, demand_sph, demand_euc = load_world_locations(
        args.pop_tiff,
        n_supply=args.n_supply,
        n_demand=args.n_demand,
        seed=args.seed,
    )
    print(f"  Supply: {supply_euc.shape}  Demand: {demand_euc.shape}")

    # Build model (no training)
    model = min_SWGG_World(
        cfg_proj   = cfg_proj,
        cfg_m      = cfg_m,
        supply_euc = supply_euc,
        demand_euc = demand_euc,
        supply_sph = supply_sph,
        demand_sph = demand_sph,
    )
    model.train(None)  # no-op

    # Save model
    model_path = os.path.join(args.out_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {model_path}")

    # ── Evaluation: timing + quality vs Sinkhorn ──────────────────────
    import ot as pot
    eps = float(cfg_m["epsilon"])
    C   = _sphere_cost(supply_euc, demand_euc)

    print(f"\nEvaluating on {args.num_samples} test pairs ...")
    times_swgg, times_sink, rmse_list = [], [], []

    for i in range(args.num_samples):
        a, b = sample_one_pair(args.n_supply, args.n_demand,
                               seed=args.seed + i + 100)

        # min-SWGG
        t0     = time.perf_counter()
        P_pred = model.predict_plan(a, b)
        t_swgg = time.perf_counter() - t0
        times_swgg.append(t_swgg)

        # Sinkhorn GT
        a_s = np.clip(a, 1e-10, None); a_s /= a_s.sum()
        b_s = np.clip(b, 1e-10, None); b_s /= b_s.sum()
        t0     = time.perf_counter()
        P_gt   = pot.sinkhorn(a_s, b_s, C, reg=eps, numItermax=500, stopThr=1e-9)
        t_sink = time.perf_counter() - t0
        times_sink.append(t_sink)

        rmse = float(np.sqrt(np.mean((P_pred - P_gt)**2)))
        rmse_list.append(rmse)
        print(f"  [{i+1}/{args.num_samples}] "
              f"RMSE={rmse:.8f}  "
              f"t_swgg={t_swgg:.3f}s  t_sink={t_sink:.3f}s  "
              f"speedup={t_sink/max(t_swgg,1e-9):.1f}x")

    # Summary
    print(f"\n{'='*55}")
    print(f"Summary ({args.num_samples} pairs, L={cfg_m['n_projections']})")
    print(f"{'='*55}")
    print(f"RMSE_Plan    : {np.mean(rmse_list):.8f} ± {np.std(rmse_list):.8f}")
    print(f"min-SWGG     : {np.mean(times_swgg):.3f}s ± {np.std(times_swgg):.3f}s")
    print(f"Sinkhorn     : {np.mean(times_sink):.3f}s ± {np.std(times_sink):.3f}s")
    ts, tw = np.mean(times_sink), np.mean(times_swgg)
    print(f"Speedup      : {ts/max(tw,1e-9):.1f}x")
    print(f"\nDone. Run plot_world_pair_torch.py --model_dir {args.out_dir}")


if __name__ == "__main__":
    main()
