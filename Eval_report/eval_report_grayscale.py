import argparse
import pickle
import os
import time
import pickle
import numpy as np
import torch
import ot
from tqdm import tqdm

from cfg import init_cfg
from Data.pre_data import pre_data


TRAIN_SEED = 0
TEST_SEED  = 999


def sinkhorn_gt(a, b, C, eps, n_iter=800):
    a_s = np.clip(a, 1e-10, None); a_s /= a_s.sum()
    b_s = np.clip(b, 1e-10, None); b_s /= b_s.sum()
    return ot.sinkhorn(a_s, b_s, C, reg=eps, numItermax=n_iter, stopThr=1e-9)


def build_cost_grid(img_size=28):
    grid = np.array([[j, i]
                     for i in np.linspace(1, 0, num=img_size)
                     for j in np.linspace(0, 1, num=img_size)], dtype=np.float64)
    diff = grid[:, None, :] - grid[None, :, :]
    return np.sum(diff ** 2, axis=-1)   # (784, 784)

def collect_pairs(dataloader, n, desc="pairs"):
    pairs = []
    pbar  = tqdm(total=n, desc=desc, leave=False)
    for _, _, x_a, x_b in dataloader:
        for a, b in zip(x_a.numpy(), x_b.numpy()):
            if len(pairs) >= n:
                break
            pairs.append((a, b))
            pbar.update(1)
        if len(pairs) >= n:
            break
    pbar.close()
    return pairs

def evaluate(model, test_pairs, C, eps, model_name):
    rmse_list, time_list = [], []
    for a, b in tqdm(test_pairs, desc=f"  Eval {model_name}", leave=False):
        P_gt = sinkhorn_gt(a, b, C, eps)

        t0 = time.perf_counter()
        P  = model.predict_plan(a, b)
        t_inf = time.perf_counter() - t0

        rmse = float(np.sqrt(np.mean((P - P_gt) ** 2)))
        rmse_list.append(rmse)
        time_list.append(t_inf)
    return np.array(rmse_list), np.array(time_list)


def print_table(results, M, N):
    print(f"\n{'='*72}")
    print(f"  MNIST Gray Scale  |  M={M} train pairs  |  N={N} test pairs")
    print(f"{'='*72}")
    print(f"  {'Method':<25} {'RMSE_Plan':>14} {'Train (s)':>12} {'Infer (ms)':>12}")
    print(f"  {'-'*25} {'-'*14} {'-'*12} {'-'*12}")
    for name, rmse_arr, time_arr, t_train in results:
        rmse_mean = rmse_arr.mean()
        rmse_std  = rmse_arr.std()
        inf_mean  = time_arr.mean() * 1000   # ms
        inf_std   = time_arr.std()  * 1000
        train_str = f"{t_train:.1f}" if t_train > 0 else "0 (no train)"
        print(f"  {name:<25} "
              f"{rmse_mean:.2e}±{rmse_std:.1e}  "
              f"{train_str:>10}s  "
              f"{inf_mean:.2f}±{inf_std:.2f}ms")
    print(f"{'='*72}\n")

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Saved → {path}")

def load_model(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M",     type=int,   default=50,  help="Train pairs")
    p.add_argument("--N",     type=int,   default=20,  help="Test pairs")
    p.add_argument("--gpu",   type=str,   default="0")
    p.add_argument("--out",   type=str,   default="./results/grayscale")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out, exist_ok=True)

    C   = build_cost_grid(28)
    eps = 1e-2
    results = []

    # ── 1. OT Regression Sliced ───────────────────────────────────────────
    print("\n[1/4] OT Regression Sliced (Method 1) ...")
    import argparse as _ap
    from time import localtime, strftime
    from Solvers.Regression_SlicedOT.OT_Regression_Sliced import OT_Regression_Sliced

    cfg_m    = init_cfg("OT_Regression_Sliced")
    cfg_m["num_bootstrap"]  = args.M
    cfg_m["epsilon"]        = eps
    cfg_proj = _ap.Namespace(seed=TRAIN_SEED,
                              flag_time=strftime("%Y-%m-%d_%H-%M-%S", localtime()),
                              flag_load=None, solver="OT_Regression_Sliced",
                              data_name="MNIST", gpu=args.gpu)

    [dl_train, _, _], _ = pre_data("MNIST", cfg_proj, cfg_m)
    model_reg = OT_Regression_Sliced(cfg_proj, cfg_m)

    t0 = time.perf_counter()
    model_reg.alpha, model_reg.beta = model_reg._fit(dl_train)
    t_train_reg = time.perf_counter() - t0
    print(f"  Train time: {t_train_reg:.1f}s")
    save_model(model_reg, os.path.join(args.out, f"M{args.M}", "regression.pkl"))

    cfg_proj_test = _ap.Namespace(seed=TEST_SEED,
                                   flag_time=strftime("%Y-%m-%d_%H-%M-%S", localtime()),
                                   flag_load=None, solver="OT_Regression_Sliced",
                                   data_name="MNIST", gpu=args.gpu)
    cfg_m_test = init_cfg("OT_Regression_Sliced")
    cfg_m_test["num_bootstrap"] = args.N
    cfg_m_test["epsilon"] = eps
    [dl_test, _, _], _ = pre_data("MNIST", cfg_proj_test, cfg_m_test)
    test_pairs = collect_pairs(dl_test, args.N, "  Test pairs")

    rmse_r, time_r = evaluate(model_reg, test_pairs, C, eps, "OT_Regression")
    results.append(("OT Regression (M1)", rmse_r, time_r, t_train_reg))
    print(f"  RMSE: {rmse_r.mean():.2e}  Infer: {time_r.mean()*1000:.2f}ms")

    # ── 2. OT Objective Sliced ────────────────────────────────────────────
    print("\n[2/4] OT Objective Sliced (Method 2) ...")
    from Solvers.Objective_SlicedOT.OT_Objective_Sliced import OT_Objective_Sliced

    cfg_m2 = init_cfg("OT_Objective_Sliced")
    cfg_m2["num_bootstrap"]  = args.M
    cfg_m2["epsilon"]        = eps
    cfg_proj2 = _ap.Namespace(seed=TRAIN_SEED,
                               flag_time=strftime("%Y-%m-%d_%H-%M-%S", localtime()),
                               flag_load=None, solver="OT_Objective_Sliced",
                               data_name="MNIST", gpu=args.gpu)
    [dl_train2, _, _], _ = pre_data("MNIST", cfg_proj2, cfg_m2)
    model_obj = OT_Objective_Sliced(cfg_proj2, cfg_m2)

    t0 = time.perf_counter()
    model_obj.alpha = model_obj._fit(dl_train2)
    model_obj.beta  = np.zeros(cfg_m2["num_projections"])
    t_train_obj = time.perf_counter() - t0
    print(f"  Train time: {t_train_obj:.1f}s")
    save_model(model_obj, os.path.join(args.out, f"M{args.M}", "objective.pkl"))

    rmse_o, time_o = evaluate(model_obj, test_pairs, C, eps, "OT_Objective")
    results.append(("OT Objective (M2)", rmse_o, time_o, t_train_obj))
    print(f"  RMSE: {rmse_o.mean():.2e}  Infer: {time_o.mean()*1000:.2f}ms")

    # ── 3. Meta OT (discrete MLP) ─────────────────────────────────────────
    print("\n[3/4] Meta OT (discrete MLP baseline) ...")
    from Solvers.Meta_OT.Meta_OT_gray_scale import OT_Discrete

    cfg_m3 = init_cfg("OT_Discrete")
    cfg_m3["epochs"]   = max(1, int(args.M / 8))   # scale epochs with M
    cfg_proj3 = _ap.Namespace(seed=TRAIN_SEED,
                               flag_time=strftime("%Y-%m-%d_%H-%M-%S", localtime()),
                               flag_load=None, solver="OT_Discrete",
                               data_name="MNIST", gpu=args.gpu)
    [dl_train3, _, dl_test3], _ = pre_data("MNIST", cfg_proj3, cfg_m3)
    model_meta = OT_Discrete(cfg_proj3, cfg_m3)
    loss_func  = model_meta.__class__   # need dual_obj_loss

    # Train Meta OT and capture time
    t0 = time.perf_counter()
    model_meta.OT_D_train(dl_train3, None, flad_load_ckp=False)
    t_train_meta = time.perf_counter() - t0
    print(f"  Train time: {t_train_meta:.1f}s")
    save_model(model_meta, os.path.join(args.out, f"M{args.M}", "meta_ot.pkl"))

    # For inference: load trained MLP and predict_plan
    from Solvers.Meta_OT.Meta_OT_gray_scale import dual_obj_loss, Cal_P
    from Models.ot_models import PotentialMLP
    mlp_meta = PotentialMLP(dim_in=28**2*2, dim_out=28**2,
                             hidden_num=cfg_m3.MLP_hidden_num).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    mlp_meta, _, _, _ = model_meta.load_ckp(mlp_meta, None, None, "OT_D-train")
    mlp_meta.eval()
    lf = dual_obj_loss(img_size=28, epsilon=cfg_m3.epsilon,
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def predict_meta(a, b):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        a_t = torch.tensor(a, dtype=torch.float64, device=dev).unsqueeze(0)
        b_t = torch.tensor(b, dtype=torch.float64, device=dev).unsqueeze(0)
        with torch.no_grad():
            f = mlp_meta(a_t, b_t)
        P = lf.pred_transport(a_t, b_t, f)
        return P[0]

    rmse_m, time_m = [], []
    for a, b in tqdm(test_pairs, desc="  Eval Meta OT", leave=False):
        P_gt = sinkhorn_gt(a, b, C, cfg_m3.epsilon)
        t0   = time.perf_counter()
        P    = predict_meta(a, b)
        time_m.append(time.perf_counter() - t0)
        rmse_m.append(float(np.sqrt(np.mean((P - P_gt)**2))))
    rmse_m = np.array(rmse_m); time_m = np.array(time_m)
    results.append(("Meta OT (baseline)", rmse_m, time_m, t_train_meta))
    print(f"  RMSE: {rmse_m.mean():.2e}  Infer: {time_m.mean()*1000:.2f}ms")

    # ── 4. min-SWGG ───────────────────────────────────────────────────────
    print("\n[4/4] min-SWGG (baseline, no training) ...")
    from Solvers.SWGG.min_SWGG_GrayScale import min_SWGG_GrayScale

    cfg_m4 = init_cfg("min_SWGG_GrayScale")
    cfg_m4["epsilon"] = eps
    cfg_proj4 = _ap.Namespace(seed=TRAIN_SEED,
                               flag_time=strftime("%Y-%m-%d_%H-%M-%S", localtime()),
                               flag_load=None, solver="min_SWGG_GrayScale",
                               data_name="MNIST", gpu=args.gpu)
    model_swgg = min_SWGG_GrayScale(cfg_proj4, cfg_m4)

    rmse_s, time_s = evaluate(model_swgg, test_pairs, C, eps, "min-SWGG")
    results.append(("min-SWGG (baseline)", rmse_s, time_s, 0.0))
    save_model(model_swgg, os.path.join(args.out, f"M{args.M}", "swgg.pkl"))
    print(f"  RMSE: {rmse_s.mean():.2e}  Infer: {time_s.mean()*1000:.2f}ms")

    # ── Print table ───────────────────────────────────────────────────────
    print_table(results, args.M, args.N)

    # Save to CSV
    csv_path = os.path.join(args.out, f"results_M{args.M}.csv")
    with open(csv_path, "w") as f:
        f.write("method,rmse_mean,rmse_std,train_s,infer_ms_mean,infer_ms_std\n")
        for name, rmse_arr, time_arr, t_train in results:
            f.write(f"{name},{rmse_arr.mean():.6e},{rmse_arr.std():.6e},"
                    f"{t_train:.2f},{time_arr.mean()*1000:.4f},{time_arr.std()*1000:.4f}\n")
    print(f"Results saved → {csv_path}")


if __name__ == "__main__":
    main()