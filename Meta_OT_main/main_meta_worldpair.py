import argparse
import os
import pickle
import time

import numpy as np

from cfg import init_cfg
from Data.world_pair_data import load_world_locations, get_world_pair_dataloader
from Solvers.Meta_OT.Meta_OT_World import Meta_OT_World
from time import localtime, strftime


def parse_args():
    p = argparse.ArgumentParser(description="Train Meta-OT (PotentialMLP) for WorldPair")
    p.add_argument("--pop_tiff",       type=str, required=True)
    p.add_argument("--out_dir",        type=str, default="./runs/meta_ot_world")
    p.add_argument("--n_supply",       type=int, default=100)
    p.add_argument("--n_demand",       type=int, default=10_000)
    p.add_argument("--num_train_iter", type=int, default=None)
    p.add_argument("--learning_rate",  type=float, default=None)
    p.add_argument("--batch_size",     type=int, default=None)
    p.add_argument("--n_hidden",       type=int, default=None)
    p.add_argument("--epsilon",        type=float, default=None)
    p.add_argument("--seed",           type=int, default=0)
    p.add_argument("--gpu",            type=str, default="0")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    cfg_m = init_cfg("Meta_OT_World")
    if args.num_train_iter is not None: cfg_m["num_train_iter"] = args.num_train_iter
    if args.learning_rate  is not None: cfg_m["learning_rate"]  = args.learning_rate
    if args.batch_size     is not None: cfg_m["batch_size"]     = args.batch_size
    if args.n_hidden       is not None: cfg_m["n_hidden"]       = args.n_hidden
    if args.epsilon        is not None: cfg_m["epsilon"]        = args.epsilon
    cfg_m["n_supply"] = args.n_supply
    cfg_m["n_demand"] = args.n_demand
    cfg_m["gpu"]      = int(args.gpu) if args.gpu.isdigit() else 0

    cfg_proj = argparse.Namespace(
        seed      = args.seed,
        flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime()),
        flag_load = None,
        solver    = "Meta_OT_World",
        data_name = "world_pair",
        gpu       = args.gpu,
    )

    # print(f"\n{'='*55}")
    # print(f"  Meta-OT (PotentialMLP) — WorldPair")
    # print(f"{'='*55}")
    # print(f"  Architecture   : PotentialMLP(concat(a,b)) → f")
    # print(f"                   g = 1 Sinkhorn step from f")
    # print(f"                   Loss = -Sinkhorn dual objective")
    # print(f"  n_supply       : {args.n_supply}")
    # print(f"  n_demand       : {args.n_demand}")
    # print(f"  n_hidden       : {cfg_m['n_hidden']} x {cfg_m['n_hidden_layer']}")
    # print(f"  num_train_iter : {cfg_m['num_train_iter']}")
    # print(f"  batch_size     : {cfg_m['batch_size']}")
    # print(f"  epsilon        : {cfg_m['epsilon']}")
    # print(f"{'='*55}\n")

    print("Loading world locations...")
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
        num_pairs          = None,   # infinite stream
        seed               = args.seed,
    )

    model = Meta_OT_World(
        cfg_proj   = cfg_proj,
        cfg_m      = cfg_m,
        supply_euc = supply_euc,
        demand_euc = demand_euc,
        supply_sph = supply_sph,
        demand_sph = demand_sph,
    )

    t0 = time.perf_counter()
    model.train(train_loader)
    print(f"\nTraining: {time.perf_counter()-t0:.1f}s")

    model_path = os.path.join(args.out_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model -> {model_path}")

    # Sanity check
    rng    = np.random.default_rng(args.seed + 999)
    mask   = rng.binomial(1, 0.5, args.n_supply).astype(np.float64)
    test_a = (mask * rng.uniform(0, 1, args.n_supply))
    if test_a.sum() < 1e-12: test_a = np.ones(args.n_supply)
    test_a /= test_a.sum()
    test_b  = rng.uniform(0, 1, args.n_demand).astype(np.float64)
    test_b /= test_b.sum()

    t0 = time.perf_counter()
    P  = model.predict_plan(test_a, test_b)
    print(f"Plan: {P.shape}  sum={P.sum():.4f}  nonzero={(P>1e-10).sum()}  "
          f"predict_time={time.perf_counter()-t0:.3f}s")
    print("\nDone. Run plot_world_pair_torch.py to visualise.")


if __name__ == "__main__":
    main()
