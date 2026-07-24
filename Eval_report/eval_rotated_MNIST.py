import argparse
import os
import pickle
import time
import numpy as np
import torch
import ot
from tqdm import tqdm
from time import localtime, strftime

from cfg import init_cfg
from Data.dataset_class import MNIST
from Data.rotate_utils import ROTATION_ANGLES, make_rotated_pairs

POOL_SEED = 0
POOL_SIZE = 1000
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
    from torch.utils.data import DataLoader
    data = [(torch.zeros(1), torch.zeros(1),
             torch.tensor(a, dtype=torch.float64),
             torch.tensor(b, dtype=torch.float64))
            for a, b in pairs]
    return DataLoader(data, batch_size=batch_size, shuffle=False)


def evaluate(predict_fn, test_pairs, C, eps, name):
    if test_pairs:
        try: predict_fn(*test_pairs[0])
        except Exception: pass
    rmse_list, erra_list, errb_list = [], [], []
    for a, b in tqdm(test_pairs, desc=f"    {name}", leave=False):
        P_gt = sinkhorn_gt(a, b, C, eps)
        P = predict_fn(a, b)
        rmse_list.append(float(np.sqrt(np.mean((P - P_gt) ** 2))))
        ea, eb = marginal_l1(P, a, b)
        erra_list.append(ea); errb_list.append(eb)
    return np.array(rmse_list), np.array(erra_list), np.array(errb_list)


def make_cfg_proj(solver, seed, gpu, flag_time):
    return argparse.Namespace(seed=seed, flag_time=flag_time,
                              flag_load=None, solver=solver,
                              data_name="MNIST", gpu=gpu)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=50)
    p.add_argument("--N", type=int, default=300)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--out", type=str, default="./results/grayscale_rotated")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(os.path.join(args.out, f"M{args.M}"), exist_ok=True)
    flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())

    C = build_cost_grid(28)
    eps = 1e-2

    n_train_pool = int(POOL_SIZE * TRAIN_RATIO)
    n_test_pool = POOL_SIZE - n_train_pool
    assert args.M <= n_train_pool
    assert args.N <= n_test_pool

    print(f"\nPre-sampling pool of {POOL_SIZE} pairs (seed={POOL_SEED}) ...")
    pool = sample_pairs(POOL_SIZE, seed=POOL_SEED)
    train_pairs = pool[:n_train_pool][:args.M]
    test_pairs_clean = pool[n_train_pool:][:args.N]
    print(f"  M={args.M} train pairs (UNROTATED) | N={args.N} test pairs, "
          f"rotated at test time only: angles={ROTATION_ANGLES}\n")

    dl_train = pairs_to_loader(train_pairs, batch_size=1)

    # ---------------- Train all 3 methods ONCE on clean (unrotated) data ----------------
    print("[1/3] Training RA-OT on clean data ...")
    from Solvers.Regression_SlicedOT.OT_Regression_Sliced import OT_Regression_Sliced
    cfg_r = init_cfg("OT_Regression_Sliced")
    cfg_r["num_bootstrap"] = args.M; cfg_r["epsilon"] = eps
    model_reg = OT_Regression_Sliced(
        make_cfg_proj("OT_Regression_Sliced", POOL_SEED, args.gpu, flag_time), cfg_r)
    model_reg.alpha = model_reg._fit(dl_train)

    def predict_reg(a, b):
        f, g = model_reg._predict_potentials(a, b, model_reg.alpha)
        return model_reg._potentials_to_plan(a, b, f, g)

    print("\n[2/3] Training OA-OT on clean data ...")
    from Solvers.Objective_SlicedOT.OT_Objective_Sliced import OT_Objective_Sliced
    cfg_o = init_cfg("OT_Objective_Sliced")
    cfg_o["num_bootstrap"] = args.M; cfg_o["epsilon"] = eps
    model_obj = OT_Objective_Sliced(
        make_cfg_proj("OT_Objective_Sliced", POOL_SEED, args.gpu, flag_time), cfg_o)
    model_obj.alpha = model_obj._fit(dl_train)

    def predict_obj(a, b):
        f, g = model_obj._predict_potentials(a, b, model_obj.alpha)
        return model_obj._potentials_to_plan(a, b, f, g)

    print("\n[3/3] Training Meta-OT on clean data ...")
    from Solvers.Meta_OT.Meta_OT_gray_scale import OT_Discrete, dual_obj_loss
    from Models.ot_models import PotentialMLP
    cfg_meta = init_cfg("OT_Discrete")
    T_target = 5000
    cfg_meta["epochs"] = max(1, T_target // args.M)
    cfg_meta["batch_size"] = 1
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

    methods = [
        ("RA-OT", predict_reg),
        ("OA-OT", predict_obj),
        ("Meta-OT", predict_meta),
    ]

    # ---------------- Evaluate every method at every rotation angle (no retraining) ----------------
    all_results = {name: {} for name, _ in methods}
    print("\nEvaluating under distribution shift (Rotated MNIST) ...")
    for angle in ROTATION_ANGLES:
        test_pairs_rot = make_rotated_pairs(test_pairs_clean, angle, img_size=28)
        print(f"\n  -- rotation = {angle} deg --")
        for name, predict_fn in methods:
            rmse, erra, errb = evaluate(predict_fn, test_pairs_rot, C, eps, name)
            all_results[name][angle] = (rmse.mean(), rmse.std(), erra.mean(), errb.mean())
            print(f"    {name:<10} RMSE={rmse.mean():.3e}±{rmse.std():.1e}  "
                  f"MargErr(a,b)=({erra.mean():.2e}, {errb.mean():.2e})")

    # ---------------- Print degradation table ----------------
    print(f"\n{'='*90}")
    print(f"  Q3: Robustness under distribution shift (Rotated MNIST)  |  M={args.M}, N={args.N}")
    print(f"{'='*90}")
    header = f"  {'Method':<10}" + "".join(f"{f'{a} deg':>13}" for a in ROTATION_ANGLES)
    print(header)
    for name, _ in methods:
        row = f"  {name:<10}"
        for angle in ROTATION_ANGLES:
            rmse_mean = all_results[name][angle][0]
            row += f"{rmse_mean:>13.2e}"
        print(row)
    print(f"{'='*90}")
    print("  (RMSE_Plan vs rotation angle; 0 deg column reproduces the original, no-shift result)\n")

    # ---------------- Save CSV ----------------
    csv_path = os.path.join(args.out, f"results_M{args.M}.csv")
    with open(csv_path, "w") as f:
        f.write("method,angle_deg,rmse_mean,rmse_std,marg_err_a_mean,marg_err_b_mean\n")
        for name, _ in methods:
            for angle in ROTATION_ANGLES:
                rmse_mean, rmse_std, erra_mean, errb_mean = all_results[name][angle]
                f.write(f"{name},{angle},{rmse_mean:.6e},{rmse_std:.6e},"
                        f"{erra_mean:.6e},{errb_mean:.6e}\n")
    print(f"Results -> {csv_path}")


if __name__ == "__main__":
    main()
