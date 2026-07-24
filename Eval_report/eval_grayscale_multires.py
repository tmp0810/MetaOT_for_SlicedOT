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
from Data.multires_utils import RESOLUTIONS, make_grid, build_cost_cross, resize_prob, infer_res

POOL_SEED = 0
POOL_SIZE = 1000
TRAIN_RATIO = 0.7
CANONICAL_SIZE = 28  # Meta-OT's fixed operating resolution


def sample_pairs_multires(n, seed, resolutions=RESOLUTIONS, datasets_root="../datasets"):
    """Same sampling logic as eval_grayscale.sample_pairs, but each of the
    two images in a pair is independently resized to a resolution drawn
    from `resolutions` (so both same-resolution and cross-resolution
    pairs occur)."""
    np.random.seed(seed)
    dataset = MNIST(flag_train=True, cfg_m=argparse.Namespace(datasets_root=datasets_root))
    pairs, res_log = [], []
    for _ in range(n):
        id_a, id_b = np.random.randint(0, len(dataset.data), 2)
        ra, rb = np.random.choice(resolutions, size=2)
        a28 = dataset.data[id_a].numpy()
        b28 = dataset.data[id_b].numpy()
        a = resize_prob(a28, 28, int(ra))
        b = resize_prob(b28, 28, int(rb))
        pairs.append((a, b))
        res_log.append((int(ra), int(rb)))
    return pairs, res_log


def pairs_to_loader(pairs, batch_size=1):
    """Identical to eval_grayscale.pairs_to_loader. batch_size MUST stay 1:
    pairs have variable, mismatched (n_a, n_b) across different iterations,
    which only works if every batch holds a single pair."""
    data = [(torch.zeros(1), torch.zeros(1),
             torch.tensor(a, dtype=torch.float64),
             torch.tensor(b, dtype=torch.float64))
            for a, b in pairs]
    return DataLoader(data, batch_size=batch_size, shuffle=False)


def to_canonical_pairs(pairs):
    """Resample every (a, b) pair to the canonical 28x28 grid, for Meta-OT."""
    out = []
    for a, b in pairs:
        a_c = resize_prob(a, infer_res(a), CANONICAL_SIZE)
        b_c = resize_prob(b, infer_res(b), CANONICAL_SIZE)
        out.append((a_c, b_c))
    return out


# ---------------------------------------------------------------------
# Ground truth / evaluation
# ---------------------------------------------------------------------
_COST_CACHE = {}


def cost_for(ra, rb):
    key = (ra, rb)
    if key not in _COST_CACHE:
        _COST_CACHE[key] = build_cost_cross(make_grid(ra), make_grid(rb))
    return _COST_CACHE[key]


def sinkhorn_gt(a, b, C, eps, n_iter=800):
    a_s = np.clip(a, 1e-10, None); a_s /= a_s.sum()
    b_s = np.clip(b, 1e-10, None); b_s /= b_s.sum()
    return ot.sinkhorn(a_s, b_s, C, reg=eps, numItermax=n_iter, stopThr=1e-9)


def evaluate_native(predict_fn, test_pairs, eps, name):
    """Ground truth computed at each pair's own native (n_a, m_b)."""
    if test_pairs:
        try: predict_fn(*test_pairs[0])
        except Exception: pass

    rmse_list, time_list = [], []
    for a, b in tqdm(test_pairs, desc=f"  Eval {name}", leave=False):
        ra, rb = infer_res(a), infer_res(b)
        C = cost_for(ra, rb)
        P_gt = sinkhorn_gt(a, b, C, eps)
        t0 = time.perf_counter()
        P = predict_fn(a, b)
        time_list.append(time.perf_counter() - t0)
        rmse_list.append(float(np.sqrt(np.mean((P - P_gt) ** 2))))
    return np.array(rmse_list), np.array(time_list)


def evaluate_canonical(predict_fn, test_pairs, eps, name):
    """For Meta-OT: test_pairs are already resampled to canonical 28x28;
    ground truth is Sinkhorn on that same canonical pair."""
    C = cost_for(CANONICAL_SIZE, CANONICAL_SIZE)
    if test_pairs:
        try: predict_fn(*test_pairs[0])
        except Exception: pass

    rmse_list, time_list = [], []
    for a, b in tqdm(test_pairs, desc=f"  Eval {name}", leave=False):
        P_gt = sinkhorn_gt(a, b, C, eps)
        t0 = time.perf_counter()
        P = predict_fn(a, b)
        time_list.append(time.perf_counter() - t0)
        rmse_list.append(float(np.sqrt(np.mean((P - P_gt) ** 2))))
    return np.array(rmse_list), np.array(time_list)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Saved -> {path}")


def print_table(results, M, N):
    print(f"\n{'='*88}")
    print(f"  MNIST Multi-Resolution (W1)  |  M={M} train pairs  |  N={N} test pairs")
    print(f"  resolutions = {RESOLUTIONS}  (n in {{784, 400, 196}})")
    print(f"{'='*88}")
    print(f"  {'Method':<24} {'RMSE_Plan':>14} {'Train (s)':>12} {'Infer (ms)':>12} {'#Params':>10}")
    print(f"  {'-'*24} {'-'*14} {'-'*12} {'-'*12} {'-'*10}")
    for name, rmse_arr, time_arr, t_train, n_params in results:
        train_str = f"{t_train:.1f}" if t_train > 0 else "0 (no train)"
        print(f"  {name:<24} "
              f"{rmse_arr.mean():.2e}+-{rmse_arr.std():.1e}  "
              f"{train_str:>10}s  "
              f"{time_arr.mean()*1000:6.2f}+-{time_arr.std()*1000:.2f}ms  "
              f"{n_params:>10}")
    print(f"{'='*88}\n")


def resolution_breakdown(rmse_arr, res_log):
    """Split RMSE into same-resolution vs cross-resolution pairs."""
    res_log = np.array(res_log)
    same_mask = res_log[:, 0] == res_log[:, 1]
    out = {}
    if same_mask.any():
        out["same-res"] = (rmse_arr[same_mask].mean(), rmse_arr[same_mask].std(), int(same_mask.sum()))
    if (~same_mask).any():
        out["cross-res"] = (rmse_arr[~same_mask].mean(), rmse_arr[~same_mask].std(), int((~same_mask).sum()))
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=50)
    p.add_argument("--N", type=int, default=300)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--out", type=str, default="./results/grayscale_multires")
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

    eps = 1e-2  # same value as eval_grayscale.py

    n_train_pool = int(POOL_SIZE * TRAIN_RATIO)
    n_test_pool = POOL_SIZE - n_train_pool
    assert args.M <= n_train_pool, f"M={args.M} exceeds train pool size {n_train_pool}"
    assert args.N <= n_test_pool, f"N={args.N} exceeds test pool size {n_test_pool}"

    print(f"\nPre-sampling pool of {POOL_SIZE} multi-res pairs (seed={POOL_SEED}) ...")
    pool, res_log_pool = sample_pairs_multires(POOL_SIZE, seed=POOL_SEED)

    train_pairs = pool[:n_train_pool][:args.M]
    test_pairs = pool[n_train_pool:][:args.N]
    test_res_log = res_log_pool[n_train_pool:][:args.N]

    n_same = sum(1 for ra, rb in test_res_log if ra == rb)
    print(f"  Using M={args.M} train pairs | N={args.N} test pairs "
          f"({n_same} same-res, {args.N - n_same} cross-res)\n")

    test_pairs_path = os.path.join(args.out, f"M{args.M}", "test_pairs.pkl")
    with open(test_pairs_path, "wb") as f:
        pickle.dump({"pairs": test_pairs, "res_log": test_res_log}, f)
    print(f"  Test pairs saved -> {test_pairs_path}")

    dl_train = pairs_to_loader(train_pairs, batch_size=1)
    results = []

    # ---------------- [1/3] RA-OT MultiRes ----------------
    print("[1/3] RA-OT Multi-Res ...")
    from Solvers.Regression_SlicedOT.OT_Regression_Sliced_MultiRes import OT_Regression_Sliced_MultiRes

    cfg_r = init_cfg("OT_Regression_Sliced")
    cfg_r["num_bootstrap"] = args.M
    cfg_r["epsilon"] = eps
    model_reg = OT_Regression_Sliced_MultiRes(
        make_cfg_proj("OT_Regression_Sliced_MultiRes", POOL_SEED, args.gpu, flag_time), cfg_r)

    t0 = time.perf_counter()
    model_reg.alpha = model_reg._fit(dl_train)
    t_reg = time.perf_counter() - t0

    def predict_reg(a, b):
        f, g = model_reg._predict_potentials(a, b, model_reg.alpha)
        return model_reg._potentials_to_plan(a, b, f, g)

    save_model(model_reg, os.path.join(args.out, f"M{args.M}", "regression_multires.pkl"))
    rmse_r, tinf_r = evaluate_native(predict_reg, test_pairs, eps, "RA-OT")
    results.append(("RA-OT (multi-res)", rmse_r, tinf_r, t_reg, int(model_reg.alpha.size)))
    print(f"  Train: {t_reg:.1f}s  RMSE: {rmse_r.mean():.2e}  Infer: {tinf_r.mean()*1000:.2f}ms")
    print(f"  Breakdown: {resolution_breakdown(rmse_r, test_res_log)}")

    # ---------------- [2/3] OA-OT MultiRes ----------------
    print("\n[2/3] OA-OT Multi-Res ...")
    from Solvers.Objective_SlicedOT.OT_Objective_Sliced_MultiRes import OT_Objective_Sliced_MultiRes

    cfg_o = init_cfg("OT_Objective_Sliced")
    cfg_o["num_bootstrap"] = args.M
    cfg_o["epsilon"] = eps
    model_obj = OT_Objective_Sliced_MultiRes(
        make_cfg_proj("OT_Objective_Sliced_MultiRes", POOL_SEED, args.gpu, flag_time), cfg_o)

    t0 = time.perf_counter()
    model_obj.alpha = model_obj._fit(dl_train)
    t_obj = time.perf_counter() - t0

    def predict_obj(a, b):
        f, g = model_obj._predict_potentials(a, b, model_obj.alpha)
        return model_obj._potentials_to_plan(a, b, f, g)

    save_model(model_obj, os.path.join(args.out, f"M{args.M}", "objective_multires.pkl"))
    rmse_o, tinf_o = evaluate_native(predict_obj, test_pairs, eps, "OA-OT")
    results.append(("OA-OT (multi-res)", rmse_o, tinf_o, t_obj, int(model_obj.alpha.size)))
    print(f"  Train: {t_obj:.1f}s  RMSE: {rmse_o.mean():.2e}  Infer: {tinf_o.mean()*1000:.2f}ms")
    print(f"  Breakdown: {resolution_breakdown(rmse_o, test_res_log)}")

    # ---------------- [3/3] Meta-OT (canonical-resampled, unmodified core) ----------------
    print("\n[3/3] Meta OT (baseline, resampled to canonical 28x28) ...")
    from Solvers.Meta_OT.Meta_OT_gray_scale import OT_Discrete, dual_obj_loss
    from Models.ot_models import PotentialMLP

    train_pairs_canon = to_canonical_pairs(train_pairs)
    test_pairs_canon = to_canonical_pairs(test_pairs)
    dl_train_meta = pairs_to_loader(train_pairs_canon, batch_size=1)

    cfg_meta = init_cfg("OT_Discrete")
    cfg_meta["epsilon"] = eps
    T_target = 5000
    cfg_meta["epochs"] = max(1, T_target // args.M)
    cfg_meta["batch_size"] = 1
    cfg_meta["log_interval"] = max(1, T_target // args.M)

    cfg_proj_meta = make_cfg_proj("OT_Discrete", POOL_SEED, args.gpu, flag_time)
    model_meta = OT_Discrete(cfg_proj_meta, cfg_meta)

    t0 = time.perf_counter()
    model_meta.OT_D_train(dl_train_meta, None, flad_load_ckp=False)
    t_meta = time.perf_counter() - t0

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_meta = PotentialMLP(dim_in=28**2*2, dim_out=28**2,
                             hidden_num=cfg_meta.MLP_hidden_num).to(dev)
    mlp_meta, _, _, _ = model_meta.load_ckp(mlp_meta, None, None, "OT_D-train")
    mlp_meta.eval()
    lf_meta = dual_obj_loss(img_size=28, epsilon=cfg_meta.epsilon, device=dev)
    n_params_meta = sum(p.numel() for p in mlp_meta.parameters())

    def predict_meta(a, b):
        a_t = torch.tensor(a, dtype=torch.float64, device=dev).unsqueeze(0)
        b_t = torch.tensor(b, dtype=torch.float64, device=dev).unsqueeze(0)
        with torch.no_grad():
            f = mlp_meta(a_t, b_t)
        return lf_meta.pred_transport(a_t, b_t, f)[0]

    save_model((mlp_meta.state_dict(),), os.path.join(args.out, f"M{args.M}", "meta_ot_multires.pkl"))
    rmse_m, tinf_m = evaluate_canonical(predict_meta, test_pairs_canon, eps, "Meta OT")
    results.append(("Meta-OT (canonical)", rmse_m, tinf_m, t_meta, n_params_meta))
    print(f"  Train: {t_meta:.1f}s  RMSE: {rmse_m.mean():.2e}  Infer: {tinf_m.mean()*1000:.2f}ms  "
          f"#Params: {n_params_meta}")
    print("  NOTE: Meta-OT cannot operate on the native n_a x m_b grids at all "
          "(fixed 784-dim MLP) - every pair had to be resampled to canonical "
          "28x28 first, and RMSE above is against the Sinkhorn ground truth of "
          "that resampled surrogate problem, not the original one.")

    print_table(results, args.M, args.N)

    csv_path = os.path.join(args.out, f"results_M{args.M}.csv")
    with open(csv_path, "w") as f:
        f.write("method,rmse_mean,rmse_std,train_s,infer_ms_mean,infer_ms_std,n_params\n")
        for name, rmse_arr, time_arr, t_train, n_params in results:
            f.write(f"{name},{rmse_arr.mean():.6e},{rmse_arr.std():.6e},"
                    f"{t_train:.2f},{time_arr.mean()*1000:.4f},{time_arr.std()*1000:.4f},{n_params}\n")
    print(f"Results -> {csv_path}  |  Models -> {args.out}/M{args.M}/")


if __name__ == "__main__":
    main()
