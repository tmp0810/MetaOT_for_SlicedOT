import argparse
import os
import pickle

import numpy as np
import torch

from cfg import init_cfg
from world_pair_data import load_world_locations, get_world_pair_dataloader
from OT_Regression_Sliced_World import OT_Regression_Sliced_World


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pop_tiff',   type=str, required=True,
                   help='Path to pop-15min.tif population raster')
    p.add_argument('--out_dir',    type=str, default='./runs/world_pair')
    p.add_argument('--n_supply',   type=int, default=100)
    p.add_argument('--n_demand',   type=int, default=10_000)
    p.add_argument('--seed',       type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────────
    cfg_m = init_cfg("OT_Regression_Sliced_World")

    # cfg_proj: minimal object with log_folder (used by Defense_Train_Base)
    cfg_proj = type('cfg_proj', (), {
        'log_folder': args.out_dir,
        'project':    'world_pair',
    })()

    # ── Load fixed supply / demand locations ─────────────────────────────
    print("Loading world locations from population raster …")
    supply_sph, supply_euc, demand_sph, demand_euc = load_world_locations(
        args.pop_tiff,
        n_supply=args.n_supply,
        n_demand=args.n_demand,
        seed=args.seed,
    )
    print(f"  Supply: {supply_euc.shape}  Demand: {demand_euc.shape}")

    # ── Build dataloaders ─────────────────────────────────────────────────
    M = cfg_m.num_bootstrap
    train_loader = get_world_pair_dataloader(
        n_supply=args.n_supply,
        n_demand=args.n_demand,
        batch_size=cfg_m.batch_size,
        supply_bernoulli_p=0.5,
        num_pairs=M,          # finite: exactly M pairs for training
        seed=args.seed,
    )

    # ── Build model ───────────────────────────────────────────────────────
    model = OT_Regression_Sliced_World(
        cfg_proj=cfg_proj,
        cfg_m=cfg_m,
        supply_euc=supply_euc,
        demand_euc=demand_euc,
        supply_sph=supply_sph,
        demand_sph=demand_sph,
    )

    # ── Train (fit regression weights) ────────────────────────────────────
    print(f"\nFitting on M={M} pairs …")
    model.train(train_loader)

    # ── Save model ────────────────────────────────────────────────────────
    model_path = os.path.join(args.out_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved → {model_path}")
    print(f"alpha: {model.alpha.shape}  beta: {model.beta.shape}")

    # ── Quick sanity check on one test pair ──────────────────────────────
    print("\nSanity check: predict plan for one test pair …")
    rng = np.random.default_rng(args.seed + 999)

    mask     = rng.binomial(1, 0.5, args.n_supply).astype(np.float64)
    test_a   = mask * rng.uniform(0, 1, args.n_supply)
    test_a   = test_a / test_a.sum() if test_a.sum() > 0 else np.ones(args.n_supply) / args.n_supply
    test_b   = rng.uniform(0, 1, args.n_demand).astype(np.float64)
    test_b  /= test_b.sum()

    P = model.predict_plan(test_a, test_b)
    print(f"  Plan shape: {P.shape}  sum={P.sum():.4f}  "
          f"max={P.max():.6f}  nonzero={np.sum(P > 1e-10)}")
    print("\nDone. Run plot_world_pair_torch.py to visualise results.")


if __name__ == '__main__':
    main()
