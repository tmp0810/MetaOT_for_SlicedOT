import argparse
import os
import pickle
import time
import numpy as np
import torch
import ot
from tqdm import tqdm
from time import localtime, strftime
from torch.utils.data import DataLoader

from cfg import init_cfg
from Data.dataset_class import MNIST

POOL_SEED  = 0
POOL_SIZE  = 1000
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


# ---------------------------------------------------------------------
# W2: marginal-violation metric.
#
# A predicted plan P is only a valid transport coupling if BOTH
# P.sum(axis=1) == a  and  P.sum(axis=0) == b.  Recovering P from a single
# predicted potential (via one Sinkhorn-style update, as every solver here
# does) enforces at most ONE of the two marginals exactly and leaves the
# other only approximately satisfied. We measure both violations directly
# on the exact P returned by predict_fn -- i.e. the plan actually reported
# / used for RMSE -- with no extra rounding or projection applied anywhere.
# ---------------------------------------------------------------------
def marginal_l1(P, a, b):
    """L1 marginal violation: sum_i |P.sum(axis=1)_i - a_i| and
    sum_j |P.sum(axis=0)_j - b_j|. Same quantity Sinkhorn's own stopThr
    tracks internally to decide convergence."""
    err_a = float(np.sum(np.abs(P.sum(axis=1) - a)))
    err_b = float(np.sum(np.abs(P.sum(axis=0) - b)))
    return err_a, err_b


def evaluate(predict_fn, test_pairs, C, eps, name):
    if test_pairs:
        try: predict_fn(*test_pairs[0])
        except Exception: pass

    rmse_list, time_list = [], []
    erra_list, errb_list = [], []          # marginal L1 error of the PREDICTED plan
    gt_erra_list, gt_errb_list = [], []    # marginal L1 error of the Sinkhorn GT plan (reference)

    for a, b in tqdm(test_pairs, desc=f"  Eval {name}", leave=False):
        P_gt = sinkhorn_gt(a, b, C, eps)
        t0   = time.perf_counter()
        P    = predict_fn(a, b)
        time_list.append(time.perf_counter() - t0)
        rmse_list.append(float(np.sqrt(np.mean((P - P_gt) ** 2))))

        err_a, err_b = marginal_l1(P, a, b)
        erra_list.append(err_a)
        errb_list.append(err_b)

        gt_err_a, gt_err_b = marginal_l1(P_gt, a, b)
        gt_erra_list.append(gt_err_a)
        gt_errb_list.append(gt_err_b)

    return (np.array(rmse_list), np.array(time_list),
            np.array(erra_list), np.array(errb_list),
            np.array(gt_erra_list), np.array(gt_errb_list))


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Saved → {path}")


def pairs_to_loader(pairs, batch_size=1):
    import torch
    data = [(torch.zeros(1), torch.zeros(1),
             torch.tensor(a, dtype=torch.float64),
             torch.tensor(b, dtype=torch.float64))
            for a, b in pairs]
    return DataLoader(data, batch_size=batch_size, shuffle=False)


def print_table(results, M, N):
    print(f"\n{'='*104}")
    print(f"  MNIST Gray Scale  |  M={M} train pairs  |  N={N} test pairs")
    print(f"{'='*104}")
    print(f"  {'Method':<24} {'RMSE_Plan':>14} {'Train (s)':>10} {'Infer (ms)':>12} "
          f"{'MargErr_a(L1)':>14} {'MargErr_b(L1)':>14}")
    print(f"  {'-'*24} {'-'*14} {'-'*10} {'-'*12} {'-'*14} {'-'*14}")
    gt_a_ref, gt_b_ref = None, None
    for name, rmse_arr, time_arr, t_train, erra_arr, errb_arr, gt_erra_arr, gt_errb_arr in results:
        train_str = f"{t_train:.1f}" if t_train > 0 else "0 (no train)"
        print(f"  {name:<24} "
              f"{rmse_arr.mean():.2e}±{rmse_arr.std():.1e}  "
              f"{train_str:>8}s  "
              f"{time_arr.mean()*1000:6.2f}±{time_arr.std()*1000:.2f}ms  "
              f"{erra_arr.mean():>8.2e}      "
              f"{errb_arr.mean():>8.2e}")
        if gt_a_ref is None:
            gt_a_ref, gt_b_ref = gt_erra_arr.mean(), gt_errb_arr.mean()
    print(f"  {'-'*24}")
    print(f"  {'Sinkhorn GT (reference)':<24} {'':>14} {'':>10} {'':>12} "
          f"{gt_a_ref:>8.2e}      {gt_b_ref:>8.2e}   <- ~stopThr, i.e. 'fully converged' baseline")
    print(f"{'='*104}\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M",   type=int, default=50)
    p.add_argument("--N",   type=int, default=20)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--out", type=str, default="./results/grayscale")
    return p.parse_args()


def make_cfg_proj(solver, seed, gpu, flag_time):
    return argparse.Namespace(seed=seed, flag_time=flag_time,
                              flag_load=None, solver=solver,
                              data_name="MNIST", gpu=gpu)

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(os.path.join(args.out, f"M{args.M}"), exist_ok=True)
    flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())

    C   = build_cost_grid(28)
    eps = 1e-2

    n_train_pool = int(POOL_SIZE * TRAIN_RATIO)  
    n_test_pool  = POOL_SIZE - n_train_pool       

    assert args.M <= n_train_pool, \
        f"M={args.M} exceeds train pool size {n_train_pool}"
    assert args.N <= n_test_pool, \
        f"N={args.N} exceeds test pool size {n_test_pool}"

    print(f"\nPre-sampling pool of {POOL_SIZE} pairs (seed={POOL_SEED}) ...")
    pool = sample_pairs(POOL_SIZE, seed=POOL_SEED)

    train_pool = pool[:n_train_pool]   # first 490 — train
    test_pool  = pool[n_train_pool:]   # last  210 — test, never touched during train

    train_pairs = train_pool[:args.M]  # first M of train pool
    test_pairs  = test_pool[:args.N]   # first N of test pool (fixed across all M)

    print(f"  Pool: {POOL_SIZE}  →  train pool: {n_train_pool}  |  test pool: {n_test_pool}")
    print(f"  Using M={args.M} train pairs  |  N={args.N} test pairs")
    print(f"  → All methods will use EXACTLY these same pairs.\n")

    test_pairs_path = os.path.join(args.out, f"M{args.M}", "test_pairs.pkl")
    os.makedirs(os.path.dirname(test_pairs_path), exist_ok=True)
    with open(test_pairs_path, "wb") as _f:
        pickle.dump(test_pairs, _f)
    print(f"  Test pairs saved → {test_pairs_path}")

    dl_train = pairs_to_loader(train_pairs, batch_size=1)
    results  = []

    print("[1/5] OT Regression Sliced (Method 1) ...")
    from Solvers.Regression_SlicedOT.OT_Regression_Sliced import OT_Regression_Sliced

    cfg_r = init_cfg("OT_Regression_Sliced")
    cfg_r["num_bootstrap"] = args.M; cfg_r["epsilon"] = eps
    model_reg = OT_Regression_Sliced(
        make_cfg_proj("OT_Regression_Sliced", POOL_SEED, args.gpu, flag_time), cfg_r)

    t0 = time.perf_counter()
    model_reg.alpha = model_reg._fit(dl_train)
    model_reg.beta  = np.zeros(cfg_r['num_projections'])
    t_reg = time.perf_counter() - t0

    def predict_reg(a, b):
        f, g = model_reg._predict_potentials(a, b, model_reg.alpha)
        return model_reg._potentials_to_plan(a, b, f, g)

    save_model(model_reg, os.path.join(args.out, f"M{args.M}", "regression.pkl"))
    rmse_r, tinf_r, erra_r, errb_r, gta_r, gtb_r = evaluate(predict_reg, test_pairs, C, eps, "OT_Regression")
    results.append(("OT Regression (M1)", rmse_r, tinf_r, t_reg, erra_r, errb_r, gta_r, gtb_r))
    print(f"  Train: {t_reg:.1f}s  RMSE: {rmse_r.mean():.2e}  Infer: {tinf_r.mean()*1000:.2f}ms  "
          f"MargErr(a,b)=({erra_r.mean():.2e}, {errb_r.mean():.2e})")

    print("\n[2/5] OT Objective Sliced (Method 2) ...")
    from Solvers.Objective_SlicedOT.OT_Objective_Sliced import OT_Objective_Sliced

    cfg_o = init_cfg("OT_Objective_Sliced")
    cfg_o["num_bootstrap"] = args.M; cfg_o["epsilon"] = eps
    model_obj = OT_Objective_Sliced(
        make_cfg_proj("OT_Objective_Sliced", POOL_SEED, args.gpu, flag_time), cfg_o)

    t0 = time.perf_counter()
    model_obj.alpha = model_obj._fit(dl_train)   # same dl_train!
    model_obj.beta  = np.zeros(cfg_o["num_projections"])
    t_obj = time.perf_counter() - t0

    def predict_obj(a, b):
        f, g = model_obj._predict_potentials(a, b, model_obj.alpha)
        return model_obj._potentials_to_plan(a, b, f, g)

    save_model(model_obj, os.path.join(args.out, f"M{args.M}", "objective.pkl"))
    rmse_o, tinf_o, erra_o, errb_o, gta_o, gtb_o = evaluate(predict_obj, test_pairs, C, eps, "OT_Objective")
    results.append(("OT Objective (M2)", rmse_o, tinf_o, t_obj, erra_o, errb_o, gta_o, gtb_o))
    print(f"  Train: {t_obj:.1f}s  RMSE: {rmse_o.mean():.2e}  Infer: {tinf_o.mean()*1000:.2f}ms  "
          f"MargErr(a,b)=({erra_o.mean():.2e}, {errb_o.mean():.2e})")

    print("\n[3/5] Meta OT GrayScale (baseline) ...")
    from Solvers.Meta_OT.Meta_OT_gray_scale import OT_Discrete, dual_obj_loss
    from Models.ot_models import PotentialMLP

    cfg_meta = init_cfg("OT_Discrete")
    T_target = 5000
    cfg_meta["epochs"]       = max(1, T_target // args.M)
    cfg_meta["batch_size"]   = 1
    cfg_meta["log_interval"] = max(1, T_target // args.M)  # save 1 checkpoint at end

    cfg_proj_meta = make_cfg_proj("OT_Discrete", POOL_SEED, args.gpu, flag_time)
    model_meta = OT_Discrete(cfg_proj_meta, cfg_meta)

    t0 = time.perf_counter()
    model_meta.OT_D_train(dl_train, None, flad_load_ckp=False)  # same dl_train!
    t_meta = time.perf_counter() - t0

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

    model_meta._eval_mlp = mlp_meta
    model_meta._eval_lf  = lf_meta
    save_model(model_meta, os.path.join(args.out, f"M{args.M}", "meta_ot.pkl"))
    rmse_m, tinf_m, erra_m, errb_m, gta_m, gtb_m = evaluate(predict_meta, test_pairs, C, eps, "Meta OT")
    results.append(("Meta OT (baseline)", rmse_m, tinf_m, t_meta, erra_m, errb_m, gta_m, gtb_m))
    print(f"  Train: {t_meta:.1f}s  RMSE: {rmse_m.mean():.2e}  Infer: {tinf_m.mean()*1000:.2f}ms  "
          f"MargErr(a,b)=({erra_m.mean():.2e}, {errb_m.mean():.2e})")

    print("\n[4/5] min-SWGG GrayScale (baseline, no training) ...")
    from Solvers.SWGG.min_SWGG_GrayScale import min_SWGG_GrayScale

    cfg_swgg = init_cfg("min_SWGG_GrayScale")
    cfg_swgg["epsilon"] = eps
    model_swgg = min_SWGG_GrayScale(
        make_cfg_proj("min_SWGG_GrayScale", POOL_SEED, args.gpu, flag_time), cfg_swgg)

    save_model(model_swgg, os.path.join(args.out, f"M{args.M}", "swgg.pkl"))
    rmse_s, tinf_s, erra_s, errb_s, gta_s, gtb_s = evaluate(model_swgg.predict_plan, test_pairs, C, eps, "min-SWGG")
    results.append(("min-SWGG (baseline)", rmse_s, tinf_s, 0.0, erra_s, errb_s, gta_s, gtb_s))
    print(f"  RMSE: {rmse_s.mean():.2e}  Infer: {tinf_s.mean()*1000:.2f}ms  "
          f"MargErr(a,b)=({erra_s.mean():.2e}, {errb_s.mean():.2e})")

    
    print("\n[5/5] Min-STP GrayScale (amortized baseline) ...")
    from Solvers.MinSTP.Min_STP_GrayScale import Min_STP_GrayScale
 
    cfg_stp = init_cfg("Min_STP_GrayScale")
    cfg_stp["epsilon"]        = eps
    cfg_stp["num_train_iter"] = 5000   # same compute budget as Method 2 / Meta-OT
    model_stp = Min_STP_GrayScale(
        make_cfg_proj("Min_STP_GrayScale", POOL_SEED, args.gpu, flag_time), cfg_stp)
 
    t0 = time.perf_counter()
    model_stp.train(dl_train)
    t_stp = time.perf_counter() - t0
 
    save_model(model_stp, os.path.join(args.out, f"M{args.M}", "min_stp.pkl"))
    rmse_stp, tinf_stp, erra_stp, errb_stp, gta_stp, gtb_stp = evaluate(model_stp.predict_plan, test_pairs, C, eps, "Min-STP")
    results.append(("Min-STP (baseline)", rmse_stp, tinf_stp, t_stp, erra_stp, errb_stp, gta_stp, gtb_stp))
    print(f"  Train: {t_stp:.1f}s  RMSE: {rmse_stp.mean():.2e}  Infer: {tinf_stp.mean()*1000:.2f}ms  "
          f"MargErr(a,b)=({erra_stp.mean():.2e}, {errb_stp.mean():.2e})")
 
    print_table(results, args.M, args.N)
 
    csv_path = os.path.join(args.out, f"results_M{args.M}.csv")
    with open(csv_path, "w") as f:
        f.write("method,rmse_mean,rmse_std,train_s,infer_ms_mean,infer_ms_std,"
                 "marg_err_a_mean,marg_err_a_std,marg_err_b_mean,marg_err_b_std,"
                 "gt_marg_err_a_mean,gt_marg_err_b_mean\n")
        for name, rmse_arr, time_arr, t_train, erra_arr, errb_arr, gta_arr, gtb_arr in results:
            f.write(f"{name},{rmse_arr.mean():.6e},{rmse_arr.std():.6e},"
                    f"{t_train:.2f},{time_arr.mean()*1000:.4f},{time_arr.std()*1000:.4f},"
                    f"{erra_arr.mean():.6e},{erra_arr.std():.6e},"
                    f"{errb_arr.mean():.6e},{errb_arr.std():.6e},"
                    f"{gta_arr.mean():.6e},{gtb_arr.mean():.6e}\n")
    print(f"Results → {csv_path}  |  Models → {args.out}/M{args.M}/")


if __name__ == "__main__":
    main()
