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
 
 
def altschuler_round(F, r, c):
    F = np.clip(F, 0.0, None)
 
    # Step 1-2: row scale (x_i = min(r_i / row_sum_i, 1))
    row_sums = F.sum(axis=1)
    x = np.minimum(r / np.maximum(row_sums, 1e-300), 1.0)
    F_prime = F * x[:, None]
 
    # Step 3-4: col scale (y_j = min(c_j / col_sum_j, 1))
    col_sums = F_prime.sum(axis=0)
    y = np.minimum(c / np.maximum(col_sums, 1e-300), 1.0)
    F_dbl = F_prime * y[None, :]
 
    # Step 5-6: rank-1 correction
    err_r = r - F_dbl.sum(axis=1)   # residual row mass (>= 0)
    err_c = c - F_dbl.sum(axis=0)   # residual col mass (>= 0)
    norm_err_r = np.sum(np.abs(err_r))
    if norm_err_r > 1e-300:
        G = F_dbl + np.outer(err_r, err_c) / norm_err_r
    else:
        G = F_dbl
    return np.clip(G, 0.0, None)
 
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
 
 
def evaluate_with_round(predict_fn, test_pairs, C, eps, name):
    if test_pairs:
        try: predict_fn(*test_pairs[0])
        except Exception: pass
 
    for a, b in tqdm(test_pairs, desc=f"  {name}", leave=False):
        P_gt = sinkhorn_gt(a, b, C, eps)
 
        # --- before rounding ---
        t0 = time.perf_counter()
        P = predict_fn(a, b)
        before["infer_ms"].append((time.perf_counter() - t0) * 1000)
        before["rmse"].append(float(np.sqrt(np.mean((P - P_gt) ** 2))))
        ea, eb = marginal_l1(P, a, b)
        before["erra"].append(ea); before["errb"].append(eb)
 
        # --- after rounding (Altschuler et al. 2017, Alg. 2) ---
        t0 = time.perf_counter()
        P_rounded = altschuler_round(P, a, b)
        after["round_ms"].append((time.perf_counter() - t0) * 1000)
        after["rmse"].append(float(np.sqrt(np.mean((P_rounded - P_gt) ** 2))))
        ea_r, eb_r = marginal_l1(P_rounded, a, b)
        after["erra"].append(ea_r); after["errb"].append(eb_r)
 
    for d in (before, after):
        for k in d:
            d[k] = np.array(d[k])
    return before, after

def print_table(results):
    print(f"\n{'='*110}")
    print(f"  W2: Before vs After Altschuler ROUND  (M=50, N=300, MNIST, eps=1e-2)")
    print(f"{'='*110}")
    print(f"  {'Method':<22} {'Stage':<8} {'RMSE_Plan':>14} {'MargErr_a':>13} "
          f"{'MargErr_b':>13} {'Time (ms)':>12}")
    print(f"  {'-'*22} {'-'*8} {'-'*14} {'-'*13} {'-'*13} {'-'*12}")
    for name, before, after in results:
        print(f"  {name:<22} {'before':<8} "
              f"{before['rmse'].mean():.2e}±{before['rmse'].std():.1e}  "
              f"{before['erra'].mean():>11.3e}  "
              f"{before['errb'].mean():>11.3e}  "
              f"{before['infer_ms'].mean():>10.2f}ms")
        print(f"  {'':22} {'after':<8} "
              f"{after['rmse'].mean():.2e}±{after['rmse'].std():.1e}  "
              f"{after['erra'].mean():>11.3e}  "
              f"{after['errb'].mean():>11.3e}  "
              f"+{after['round_ms'].mean():.3f}ms")
        print(f"  {'-'*22}")
    print(f"{'='*110}\n")
    print("  NOTE: 'after' MargErr_{a,b} should be ~machine epsilon by construction of ROUND.\n")
 
 
# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M",  type=int, default=50)
    p.add_argument("--N",  type=int, default=300)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--out", type=str, default="./results/w2_round")
    return p.parse_args()
 
 
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out, exist_ok=True)
    flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())
 
    C   = build_cost_grid(28)
    eps = 1e-2
 
    n_train_pool = int(POOL_SIZE * TRAIN_RATIO)
    n_test_pool  = POOL_SIZE - n_train_pool
    assert args.M <= n_train_pool
    assert args.N <= n_test_pool
 
    print(f"\nPre-sampling pool of {POOL_SIZE} pairs (seed={POOL_SEED}) ...")
    pool = sample_pairs(POOL_SIZE, seed=POOL_SEED)
    train_pairs = pool[:n_train_pool][:args.M]
    test_pairs  = pool[n_train_pool:][:args.N]
    print(f"  M={args.M} train | N={args.N} test\n")
 
    dl_train = pairs_to_loader(train_pairs, batch_size=1)
    all_results = []
 
    # ---- RA-OT ----
    print("[1/3] Training RA-OT ...")
    from Solvers.Regression_SlicedOT.OT_Regression_Sliced import OT_Regression_Sliced
    cfg_r = init_cfg("OT_Regression_Sliced")
    cfg_r["num_bootstrap"] = args.M; cfg_r["epsilon"] = eps
    model_reg = OT_Regression_Sliced(
        make_cfg_proj("OT_Regression_Sliced", POOL_SEED, args.gpu, flag_time), cfg_r)
    alpha_reg = model_reg._fit(dl_train)
 
    def predict_reg(a, b):
        f, g = model_reg._predict_potentials(a, b, alpha_reg)
        return model_reg._potentials_to_plan(a, b, f, g)
 
    before_r, after_r = evaluate_with_round(predict_reg, test_pairs, C, eps, "RA-OT")
    all_results.append(("RA-OT (ours)", before_r, after_r))
    print(f"  Before: RMSE={before_r['rmse'].mean():.2e}  "
          f"MargErr=({before_r['erra'].mean():.2e}, {before_r['errb'].mean():.2e})")
    print(f"  After:  RMSE={after_r['rmse'].mean():.2e}  "
          f"MargErr=({after_r['erra'].mean():.2e}, {after_r['errb'].mean():.2e})  "
          f"+{after_r['round_ms'].mean():.3f}ms")
 
    # ---- OA-OT ----
    print("\n[2/3] Training OA-OT ...")
    from Solvers.Objective_SlicedOT.OT_Objective_Sliced import OT_Objective_Sliced
    cfg_o = init_cfg("OT_Objective_Sliced")
    cfg_o["num_bootstrap"] = args.M; cfg_o["epsilon"] = eps
    model_obj = OT_Objective_Sliced(
        make_cfg_proj("OT_Objective_Sliced", POOL_SEED, args.gpu, flag_time), cfg_o)
    alpha_obj = model_obj._fit(dl_train)
 
    def predict_obj(a, b):
        f, g = model_obj._predict_potentials(a, b, alpha_obj)
        return model_obj._potentials_to_plan(a, b, f, g)
 
    before_o, after_o = evaluate_with_round(predict_obj, test_pairs, C, eps, "OA-OT")
    all_results.append(("OA-OT (ours)", before_o, after_o))
    print(f"  Before: RMSE={before_o['rmse'].mean():.2e}  "
          f"MargErr=({before_o['erra'].mean():.2e}, {before_o['errb'].mean():.2e})")
    print(f"  After:  RMSE={after_o['rmse'].mean():.2e}  "
          f"MargErr=({after_o['erra'].mean():.2e}, {after_o['errb'].mean():.2e})  "
          f"+{after_o['round_ms'].mean():.3f}ms")
 
    # ---- Meta-OT ----
    print("\n[3/3] Training Meta-OT ...")
    from Solvers.Meta_OT.Meta_OT_gray_scale import OT_Discrete, dual_obj_loss
    from Models.ot_models import PotentialMLP
    cfg_meta = init_cfg("OT_Discrete")
    cfg_meta["epsilon"] = eps
    T_target = 5000
    cfg_meta["epochs"]       = max(1, T_target // args.M)
    cfg_meta["batch_size"]   = 1
    cfg_meta["log_interval"] = max(1, T_target // args.M)
    model_meta = OT_Discrete(
        make_cfg_proj("OT_Discrete", POOL_SEED, args.gpu, flag_time), cfg_meta)
    model_meta.OT_D_train(dl_train, None, flad_load_ckp=False)
 
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_meta = PotentialMLP(dim_in=28**2*2, dim_out=28**2,
                             hidden_num=cfg_meta.MLP_hidden_num).to(dev)
    mlp_meta, _, _, _ = model_meta.load_ckp(mlp_meta, None, None, "OT_D-train")
    mlp_meta.eval()
    lf_meta = dual_obj_loss(img_size=28, epsilon=cfg_meta.epsilon, device=dev)
 
    def predict_meta(a, b):
        a_t = torch.tensor(a, dtype=torch.float64, device=dev).unsqueeze(0)
        b_t = torch.tensor(b, dtype=torch.float64, device=dev).unsqueeze(0)
        with torch.no_grad():
            f = mlp_meta(a_t, b_t)
        return lf_meta.pred_transport(a_t, b_t, f)[0]
 
    before_m, after_m = evaluate_with_round(predict_meta, test_pairs, C, eps, "Meta-OT")
    all_results.append(("Meta-OT (baseline)", before_m, after_m))
    print(f"  Before: RMSE={before_m['rmse'].mean():.2e}  "
          f"MargErr=({before_m['erra'].mean():.2e}, {before_m['errb'].mean():.2e})")
    print(f"  After:  RMSE={after_m['rmse'].mean():.2e}  "
          f"MargErr=({after_m['erra'].mean():.2e}, {after_m['errb'].mean():.2e})  "
          f"+{after_m['round_ms'].mean():.3f}ms")
 
    print_table(all_results)
 
    # ---- CSV ----
    csv_path = os.path.join(args.out, f"results_M{args.M}.csv")
    with open(csv_path, "w") as f:
        f.write("method,stage,rmse_mean,rmse_std,marg_err_a_mean,marg_err_b_mean,time_ms_mean\n")
        for name, before, after in all_results:
            f.write(f"{name},before,{before['rmse'].mean():.6e},{before['rmse'].std():.6e},"
                    f"{before['erra'].mean():.6e},{before['errb'].mean():.6e},"
                    f"{before['infer_ms'].mean():.4f}\n")
            f.write(f"{name},after,{after['rmse'].mean():.6e},{after['rmse'].std():.6e},"
                    f"{after['erra'].mean():.6e},{after['errb'].mean():.6e},"
                    f"{after['round_ms'].mean():.4f}\n")
    print(f"Results -> {csv_path}")
 
 
if __name__ == "__main__":
    main()
