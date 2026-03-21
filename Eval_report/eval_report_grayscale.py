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

TRAIN_SEED = 0
TEST_SEED  = 999


# ── helpers ───────────────────────────────────────────────────────────────────

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
    """Sample n MNIST pairs reproducibly. Returns list of (a, b) numpy arrays."""
    np.random.seed(seed)
    dataset = MNIST(flag_train=True, cfg_m=argparse.Namespace(datasets_root="../datasets"))
    pairs = []
    for _ in range(n):
        id_a, id_b = np.random.randint(0, len(dataset.data), 2)
        a = dataset.data[id_a].numpy()
        b = dataset.data[id_b].numpy()
        pairs.append((a, b))
    return pairs


def evaluate(predict_fn, test_pairs, C, eps, name):
    """predict_fn(a, b) → P numpy array."""
    rmse_list, time_list = [], []
    for a, b in tqdm(test_pairs, desc=f"  Eval {name}", leave=False):
        P_gt = sinkhorn_gt(a, b, C, eps)
        t0   = time.perf_counter()
        P    = predict_fn(a, b)
        time_list.append(time.perf_counter() - t0)
        rmse_list.append(float(np.sqrt(np.mean((P - P_gt) ** 2))))
    return np.array(rmse_list), np.array(time_list)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Saved → {path}")


def pairs_to_loader(pairs, batch_size=1):
    """Wrap a list of (a,b) pairs into a DataLoader-like iterable."""
    # yields (_, _, a_batch, b_batch) to match existing _fit() interface
    import torch
    data = [(torch.zeros(1), torch.zeros(1),
             torch.tensor(a, dtype=torch.float64),
             torch.tensor(b, dtype=torch.float64))
            for a, b in pairs]
    return DataLoader(data, batch_size=batch_size, shuffle=False)


def print_table(results, M, N):
    print(f"\n{'='*72}")
    print(f"  MNIST Gray Scale  |  M={M} train pairs  |  N={N} test pairs")
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
    p.add_argument("--M",   type=int, default=50)
    p.add_argument("--N",   type=int, default=20)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--out", type=str, default="./results/grayscale")
    return p.parse_args()


def make_cfg_proj(solver, seed, gpu, flag_time):
    return argparse.Namespace(seed=seed, flag_time=flag_time,
                              flag_load=None, solver=solver,
                              data_name="MNIST", gpu=gpu)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(os.path.join(args.out, f"M{args.M}"), exist_ok=True)
    flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())

    C   = build_cost_grid(28)
    eps = 1e-2

    # ── Pre-sample pairs ONCE — shared by ALL methods ─────────────────────
    print(f"\nSampling {args.M} train pairs (seed={TRAIN_SEED}) ...")
    train_pairs = sample_pairs(args.M, seed=TRAIN_SEED)
    print(f"Sampling {args.N} test  pairs (seed={TEST_SEED}) ...")
    test_pairs  = sample_pairs(args.N, seed=TEST_SEED)
    print(f"  → All 4 methods will use EXACTLY these same pairs.\n")

    dl_train = pairs_to_loader(train_pairs, batch_size=1)
    results  = []

    # ── 1. OT Regression Sliced ───────────────────────────────────────────
    print("[1/4] OT Regression Sliced (Method 1) ...")
    from Solvers.Regression_SlicedOT.OT_Regression_Sliced import OT_Regression_Sliced

    cfg_r = init_cfg("OT_Regression_Sliced")
    cfg_r["num_bootstrap"] = args.M; cfg_r["epsilon"] = eps
    model_reg = OT_Regression_Sliced(
        make_cfg_proj("OT_Regression_Sliced", TRAIN_SEED, args.gpu, flag_time), cfg_r)

    t0 = time.perf_counter()
    model_reg.alpha, model_reg.beta = model_reg._fit(dl_train)
    t_reg = time.perf_counter() - t0

    def predict_reg(a, b):
        f, g = model_reg._predict_potentials(a, b, model_reg.alpha, model_reg.beta)
        return model_reg._potentials_to_plan(a, b, f, g)

    save_model(model_reg, os.path.join(args.out, f"M{args.M}", "regression.pkl"))
    rmse_r, tinf_r = evaluate(predict_reg, test_pairs, C, eps, "OT_Regression")
    results.append(("OT Regression (M1)", rmse_r, tinf_r, t_reg))
    print(f"  Train: {t_reg:.1f}s  RMSE: {rmse_r.mean():.2e}  Infer: {tinf_r.mean()*1000:.2f}ms")

    # ── 2. OT Objective Sliced ────────────────────────────────────────────
    print("\n[2/4] OT Objective Sliced (Method 2) ...")
    from Solvers.Objective_SlicedOT.OT_Objective_Sliced import OT_Objective_Sliced

    cfg_o = init_cfg("OT_Objective_Sliced")
    cfg_o["num_bootstrap"] = args.M; cfg_o["epsilon"] = eps
    model_obj = OT_Objective_Sliced(
        make_cfg_proj("OT_Objective_Sliced", TRAIN_SEED, args.gpu, flag_time), cfg_o)

    t0 = time.perf_counter()
    model_obj.alpha = model_obj._fit(dl_train)   # same dl_train!
    model_obj.beta  = np.zeros(cfg_o["num_projections"])
    t_obj = time.perf_counter() - t0

    def predict_obj(a, b):
        f, g = model_obj._predict_potentials(a, b, model_obj.alpha)
        return model_obj._potentials_to_plan(a, b, f, g)

    save_model(model_obj, os.path.join(args.out, f"M{args.M}", "objective.pkl"))
    rmse_o, tinf_o = evaluate(predict_obj, test_pairs, C, eps, "OT_Objective")
    results.append(("OT Objective (M2)", rmse_o, tinf_o, t_obj))
    print(f"  Train: {t_obj:.1f}s  RMSE: {rmse_o.mean():.2e}  Infer: {tinf_o.mean()*1000:.2f}ms")

    # ── 3. Meta OT (OT_Discrete) ──────────────────────────────────────────
    print("\n[3/4] Meta OT GrayScale (baseline) ...")
    from Solvers.Meta_OT.Meta_OT_gray_scale import OT_Discrete, dual_obj_loss
    from Models.ot_models import PotentialMLP

    cfg_meta = init_cfg("OT_Discrete")
    # Data budget = M pairs (same as Regression/Objective).
    # Compute budget = free: Meta OT needs many epochs to converge from random init.
    # dl_train has exactly M pairs → epochs=500 means 500×M gradient steps,
    # same way Objective runs num_train_iter=5000 steps on the M-pair pool.
    cfg_meta["epochs"] = 500
    cfg_meta["batch_size"] = 1
    cfg_meta["log_interval"] = 500

    cfg_proj_meta = make_cfg_proj("OT_Discrete", TRAIN_SEED, args.gpu, flag_time)
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
    rmse_m, tinf_m = evaluate(predict_meta, test_pairs, C, eps, "Meta OT")
    results.append(("Meta OT (baseline)", rmse_m, tinf_m, t_meta))
    print(f"  Train: {t_meta:.1f}s  RMSE: {rmse_m.mean():.2e}  Infer: {tinf_m.mean()*1000:.2f}ms")

    # ── 4. min-SWGG ───────────────────────────────────────────────────────
    print("\n[4/4] min-SWGG GrayScale (baseline, no training) ...")
    from Solvers.SWGG.min_SWGG_GrayScale import min_SWGG_GrayScale

    cfg_swgg = init_cfg("min_SWGG_GrayScale")
    cfg_swgg["epsilon"] = eps
    model_swgg = min_SWGG_GrayScale(
        make_cfg_proj("min_SWGG_GrayScale", TRAIN_SEED, args.gpu, flag_time), cfg_swgg)

    save_model(model_swgg, os.path.join(args.out, f"M{args.M}", "swgg.pkl"))
    rmse_s, tinf_s = evaluate(model_swgg.predict_plan, test_pairs, C, eps, "min-SWGG")
    results.append(("min-SWGG (baseline)", rmse_s, tinf_s, 0.0))
    print(f"  RMSE: {rmse_s.mean():.2e}  Infer: {tinf_s.mean()*1000:.2f}ms")

    # ── Print table & save CSV ─────────────────────────────────────────────
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
