import argparse
import os
import pickle
import time
import numpy as np
from time import localtime, strftime

from cfg import init_cfg
from Data.world_pair_data import load_world_locations, get_world_pair_dataloader
from Solvers.Objective_SlicedOT.OT_Objective_Sliced_World import OT_Objective_Sliced_World
from Solvers.Regression_SlicedOT.OT_Regression_Sliced_World import _sphere_cost


def parse_args():
    p = argparse.ArgumentParser(
        description="Method 2: Objective-based Sliced OT for WorldPair")
    p.add_argument("--pop_tiff",       type=str,   required=True)
    p.add_argument("--out_dir",        type=str,   default="./runs/objective_world")
    p.add_argument("--n_supply",       type=int,   default=100)
    p.add_argument("--n_demand",       type=int,   default=10_000)
    p.add_argument("--num_bootstrap",  type=int,   default=None)
    p.add_argument("--num_train_iter", type=int,   default=None)
    p.add_argument("--num_proj",       type=int,   default=None)
    p.add_argument("--learning_rate",  type=float, default=None)
    p.add_argument("--epsilon",        type=float, default=None)
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--gpu",            type=str,   default="0")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    cfg_m = init_cfg("OT_Objective_Sliced_World")
    if args.num_bootstrap  is not None: cfg_m["num_bootstrap"]  = args.num_bootstrap
    if args.num_train_iter is not None: cfg_m["num_train_iter"] = args.num_train_iter
    if args.num_proj       is not None: cfg_m["num_projections"]= args.num_proj
    if args.learning_rate  is not None: cfg_m["learning_rate"]  = args.learning_rate
    if args.epsilon        is not None: cfg_m["epsilon"]        = args.epsilon
    cfg_m["n_supply"] = args.n_supply
    cfg_m["n_demand"] = args.n_demand
    cfg_m["gpu"] = int(args.gpu) if args.gpu.isdigit() else 0

    cfg_proj = argparse.Namespace(
        seed      = args.seed,
        flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime()),
        flag_load = None,
        solver    = "OT_Objective_Sliced_World",
        data_name = "world_pair",
        gpu       = args.gpu,
    )

    print(f"\n{'='*60}")
    print(f"  Method 2: Objective-based Amortized Sliced OT — WorldPair")
    print(f"  Model: f = Φ_f(a,b) @ α  (α ∈ ℝ^L, global)")
    print(f"  Loss:  -E[dual_obj(Φ_f@α; a,b,c)]  — no GT Sinkhorn")
    print(f"{'='*60}")
    print(f"  n_supply      : {args.n_supply}")
    print(f"  n_demand      : {args.n_demand}")
    print(f"  num_bootstrap : {cfg_m['num_bootstrap']}  (M pair pool)")
    print(f"  num_train_iter: {cfg_m['num_train_iter']}  (T gradient steps)")
    print(f"  num_proj      : {cfg_m['num_projections']}  (L directions)")
    print(f"  learning_rate : {cfg_m['learning_rate']}")
    print(f"  epsilon       : {cfg_m['epsilon']}")
    print(f"{'='*60}\n")

    print("Loading world locations ...")
    supply_sph, supply_euc, demand_sph, demand_euc = load_world_locations(
        args.pop_tiff,
        n_supply=args.n_supply,
        n_demand=args.n_demand,
        seed=args.seed,
    )
    print(f"  Supply: {supply_euc.shape}  Demand: {demand_euc.shape}")

    train_loader = get_world_pair_dataloader(
        n_supply           = args.n_supply,
        n_demand           = args.n_demand,
        batch_size         = cfg_m["batch_size"],
        supply_bernoulli_p = cfg_m["supply_bernoulli_p"],
        num_pairs          = None,
        seed               = args.seed,
    )

    model = OT_Objective_Sliced_World(
        cfg_proj   = cfg_proj,
        cfg_m      = cfg_m,
        supply_euc = supply_euc,
        demand_euc = demand_euc,
        supply_sph = supply_sph,
        demand_sph = demand_sph,
    )

    t0 = time.perf_counter()
    model.train(train_loader)
    t_train = time.perf_counter() - t0
    print(f"\nTraining: {t_train:.1f}s  ({t_train/60:.1f}min)")

    model_path = os.path.join(args.out_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model → {model_path}")

    import ot as pot
    rng  = np.random.default_rng(args.seed + 999)
    mask = rng.binomial(1, 0.5, args.n_supply).astype(np.float64)
    a    = mask * rng.uniform(0, 1, args.n_supply)
    if a.sum() < 1e-12: a = np.ones(args.n_supply, dtype=np.float64)
    a   /= a.sum()
    b    = rng.uniform(0, 1, args.n_demand).astype(np.float64)
    b   /= b.sum()

    t0    = time.perf_counter()
    P_pred = model.predict_plan(a, b)
    t_ours = time.perf_counter() - t0

    C   = _sphere_cost(supply_euc, demand_euc)
    eps = float(cfg_m["epsilon"])
    a_s = np.clip(a, 1e-10, None); a_s /= a_s.sum()
    b_s = np.clip(b, 1e-10, None); b_s /= b_s.sum()
    t0     = time.perf_counter()
    P_gt   = pot.sinkhorn(a_s, b_s, C, reg=eps, numItermax=500, stopThr=1e-9)
    t_sink = time.perf_counter() - t0

    rmse = float(np.sqrt(np.mean((P_pred - P_gt)**2)))
    print(f"\nSanity check:")
    print(f"  Plan: {P_pred.shape}  sum={P_pred.sum():.4f}")
    print(f"  RMSE_Plan={rmse:.8f}")
    print(f"  t_ours={t_ours:.3f}s  t_sink={t_sink:.3f}s  "
          f"speedup={t_sink/max(t_ours,1e-9):.1f}x")
    print(f"\nDone. Run plot_world_pair_torch.py --model_dir {args.out_dir}")


if __name__ == "__main__":
    main()
