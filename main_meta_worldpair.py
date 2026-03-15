import argparse
import os
import pickle
from time import localtime, strftime

import numpy as np

from cfg import init_cfg
from Data.world_pair_data import load_world_locations, get_world_pair_dataloader
from Solvers.Meta_OT_World import Meta_OT_World


def parse_args():
    p = argparse.ArgumentParser(description='Train Meta-OT for WorldPair')
    p.add_argument('--pop_tiff',       type=str, required=True)
    p.add_argument('--out_dir',        type=str, default='./runs/meta_ot_world')
    p.add_argument('--n_supply',       type=int, default=100)
    p.add_argument('--n_demand',       type=int, default=10_000)
    p.add_argument('--num_train_iter', type=int, default=None)
    p.add_argument('--pretrain_iter',  type=int, default=None)
    p.add_argument('--batch_size',     type=int, default=None)
    p.add_argument('--learning_rate',  type=float, default=None)
    p.add_argument('--seed',           type=int, default=0)
    p.add_argument('--gpu',            type=str, default='0')
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    cfg_m = init_cfg("Meta_OT_World")
    if args.num_train_iter is not None: cfg_m.num_train_iter = args.num_train_iter
    if args.pretrain_iter  is not None: cfg_m.pretrain_iter  = args.pretrain_iter
    if args.batch_size     is not None: cfg_m.batch_size     = args.batch_size
    if args.learning_rate  is not None: cfg_m.learning_rate  = args.learning_rate

    # Keep n_supply / n_demand consistent with cfg_m
    cfg_m.n_supply = args.n_supply
    cfg_m.n_demand = args.n_demand

    cfg_proj = argparse.Namespace(
        seed      = args.seed,
        flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime()),
        flag_load = None,
        solver    = "Meta_OT_World",
        data_name = "world_pair",
        gpu       = args.gpu,
    )

    print(f"\n{'='*55}")
    print(f"  Meta-OT Baseline — WorldPair")
    print(f"{'='*55}")
    print(f"  n_supply      : {args.n_supply}")
    print(f"  n_demand      : {args.n_demand}")
    print(f"  num_train_iter: {cfg_m.num_train_iter}")
    print(f"  pretrain_iter : {cfg_m.pretrain_iter}")
    print(f"  icnn_hidden   : {cfg_m.icnn_hidden_dim} × {cfg_m.icnn_hidden_num}")
    print(f"  enc_dim       : {cfg_m.enc_dim}")
    print(f"  epsilon (eval): {cfg_m.epsilon}")
    print(f"{'='*55}\n")

    # ── Load fixed supply / demand locations ──────────────────────────
    print("Loading world locations ...")
    supply_sph, supply_euc, demand_sph, demand_euc = load_world_locations(
        args.pop_tiff,
        n_supply = args.n_supply,
        n_demand = args.n_demand,
        seed     = args.seed,
    )
    print(f"  Supply: {supply_euc.shape}  Demand: {demand_euc.shape}")

    # ── DataLoader ────────────────────────────────────────────────────
    train_loader = get_world_pair_dataloader(
        n_supply           = args.n_supply,
        n_demand           = args.n_demand,
        batch_size         = cfg_m.batch_size,
        supply_bernoulli_p = cfg_m.supply_bernoulli_p,
        num_pairs          = cfg_m.num_train_iter * cfg_m.batch_size * 2,
        seed               = args.seed,
    )

    # ── Build + train ──────────────────────────────────────────────────
    model = Meta_OT_World(
        cfg_proj   = cfg_proj,
        cfg_m      = cfg_m,
        supply_euc = supply_euc,
        demand_euc = demand_euc,
        supply_sph = supply_sph,
        demand_sph = demand_sph,
    )
    model.train(train_loader)

    # ── Save ──────────────────────────────────────────────────────────
    model_path = os.path.join(args.out_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved -> {model_path}")

    # ── Sanity check ──────────────────────────────────────────────────
    print("\nSanity check: predict plan for one test pair ...")
    rng    = np.random.default_rng(args.seed + 999)
    mask   = rng.binomial(1, 0.5, args.n_supply).astype(np.float64)
    test_a = mask * rng.uniform(0, 1, args.n_supply)
    if test_a.sum() < 1e-12:
        test_a = np.ones(args.n_supply, dtype=np.float64)
    test_a /= test_a.sum()
    test_b  = rng.uniform(0, 1, args.n_demand).astype(np.float64)
    test_b /= test_b.sum()

    P = model.predict_plan(test_a, test_b)
    print(f"  Plan: shape={P.shape}  sum={P.sum():.4f}  "
          f"max={P.max():.6f}  nonzero={(P > 1e-10).sum()}")
    print("\nDone.")


if __name__ == '__main__':
    main()
