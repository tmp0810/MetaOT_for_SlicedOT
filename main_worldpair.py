import argparse
import os
import pickle
import time

import numpy as np
import torch

from cfg import init_cfg
from Data.world_pair_data import load_world_locations, get_world_pair_dataloader
from Solvers.OT_Regression_Sliced_World import OT_Regression_Sliced_World
from time import localtime, strftime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pop_tiff',  type=str, required=True)
    p.add_argument('--out_dir',   type=str, default='./runs/world_pair')
    p.add_argument('--n_supply',  type=int, default=100)
    p.add_argument('--n_demand',  type=int, default=10_000)
    p.add_argument('--seed',      type=int, default=0)
    p.add_argument('--gpu',       type=str, default='0')
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    cfg_m = init_cfg("OT_Regression_Sliced_World")

    # cfg_proj must satisfy Defense_Train_Base.init_env():
    #   - cfg_proj.seed      : for torch/numpy seeding
    #   - cfg_proj.flag_time : used in log_sub_folder path
    #   - vars(cfg_proj)     : iterated for logging → must be a real __dict__
    # Note: init_env uses hardcoded log_folder="inProc_data", NOT cfg_proj.log_folder.
    # We mirror main.py's argparse namespace exactly.
    cfg_proj = argparse.Namespace(
        seed       = args.seed,
        flag_time  = strftime("%Y-%m-%d_%H-%M-%S", localtime()),
        flag_load  = None,
        solver     = "OT_Regression_Sliced_World",
        data_name  = "world_pair",
        gpu        = args.gpu,
    )

    # ── Load fixed supply / demand locations ──────────────────────────────
    print("Loading world locations from population raster …")
    supply_sph, supply_euc, demand_sph, demand_euc = load_world_locations(
        args.pop_tiff,
        n_supply=args.n_supply,
        n_demand=args.n_demand,
        seed=args.seed,
    )
    print(f"  Supply: {supply_euc.shape}  Demand: {demand_euc.shape}")

    # ── Build dataloader ──────────────────────────────────────────────────
    M = cfg_m.num_bootstrap
    train_loader = get_world_pair_dataloader(
        n_supply           = args.n_supply,
        n_demand           = args.n_demand,
        batch_size         = cfg_m.batch_size,
        supply_bernoulli_p = cfg_m.supply_bernoulli_p,
        num_pairs          = M,
        seed               = args.seed,
    )

    # ── Build model ───────────────────────────────────────────────────────
    model = OT_Regression_Sliced_World(
        cfg_proj   = cfg_proj,
        cfg_m      = cfg_m,
        supply_euc = supply_euc,
        demand_euc = demand_euc,
        supply_sph = supply_sph,
        demand_sph = demand_sph,
    )

    print(f"\nFitting on M={M} pairs …")
     start_train = time.perf_counter()
    
    model.train(train_loader)
    
    # Chốt giờ Training
    end_train = time.perf_counter()
    total_train_time = end_train - start_train
    
    print(f"\n[!] Tổng thời gian Training (Tạo data + Giải Alpha/Beta): {total_train_time:.2f} giây")
    model_path = os.path.join(args.out_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved → {model_path}")
    print(f"alpha/beta also saved → {model.log_sub_folder}/")

    print("\nSanity check: predict plan for one test pair …")
    rng    = np.random.default_rng(args.seed + 999)
    mask   = rng.binomial(1, 0.5, args.n_supply).astype(np.float64)
    test_a = mask * rng.uniform(0, 1, args.n_supply)
    if test_a.sum() < 1e-12:
        test_a = np.ones(args.n_supply, dtype=np.float64)
    test_a /= test_a.sum()
    test_b  = rng.uniform(0, 1, args.n_demand).astype(np.float64)
    test_b /= test_b.sum()

    P_pred = model.predict_plan(test_a, test_b)

    f_gt, g_gt = model._solve_entropic_ot(test_a, test_b)
    P_gt = model._potentials_to_plan(test_a, test_b, f_gt, g_gt)

    rmse_P = float(np.sqrt(np.mean((P_pred - P_gt) ** 2)))

    print(f"  Plan shape: {P_pred.shape}  sum={P_pred.sum():.4f}  "
          f"max={P_pred.max():.6f}  nonzero={np.sum(P_pred > 1e-10)}")
    
    print(f"  RMSE_Plan: {rmse_P:.8f} | plan_sum_gt={P_gt.sum():.4f}  plan_sum_pred={P_pred.sum():.4f}")
    
    print("\nDone. Run plot_world_pair_torch.py to visualise results.")


if __name__ == '__main__':
    main()
