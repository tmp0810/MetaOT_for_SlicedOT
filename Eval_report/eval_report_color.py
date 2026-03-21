import argparse
import os
import pickle
import glob
import time
import numpy as np
import ot
from tqdm import tqdm
from time import localtime, strftime

from cfg import init_cfg
from Data.color_transfer_data import load_and_quantize, get_color_transfer_dataloader

TRAIN_SEED = 0
TEST_SEED  = 999


def sinkhorn_gt(a, b, C, eps, n_iter=1000):
    a_s = np.clip(a, 1e-10, None); a_s /= a_s.sum()
    b_s = np.clip(b, 1e-10, None); b_s /= b_s.sum()
    return ot.sinkhorn(a_s, b_s, C, reg=eps, numItermax=n_iter, stopThr=1e-9)


def build_test_pairs(image_paths, N, n_clusters, seed=TEST_SEED):
    rng   = np.random.default_rng(seed)
    cache = {}
    pairs = []
    pbar  = tqdm(total=N, desc="  Test pairs", leave=False)
    attempts = 0
    while len(pairs) < N and attempts < N * 20:
        attempts += 1
        i, j = rng.choice(len(image_paths), size=2, replace=False)
        pi, pj = image_paths[i], image_paths[j]
        if pi not in cache:
            w, c, _, _ = load_and_quantize(pi, n_clusters, seed=0)
            cache[pi] = (w, c)
        if pj not in cache:
            w, c, _, _ = load_and_quantize(pj, n_clusters, seed=0)
            cache[pj] = (w, c)
        sw, sc = cache[pi]
        tw, tc = cache[pj]
        pairs.append((sw, sc, tw, tc))
        pbar.update(1)
    pbar.close()
    return pairs[:N]


def evaluate_color(predict_fn, test_pairs, eps, name):
    """predict_fn(a, b, src_c, tgt_c) → P"""
    from Solvers.Regression_SlicedOT.OT_Regression_Sliced_Color import OT_Regression_Sliced_Color
    rmse_list, time_list = [], []
    for sw, sc, tw, tc in tqdm(test_pairs, desc=f"  Eval {name}", leave=False):
        diff = sc[:, None, :] - tc[None, :, :]
        C    = np.sum(diff**2, axis=-1)
        P_gt = sinkhorn_gt(sw, tw, C, eps)
        t0   = time.perf_counter()
        P    = predict_fn(sw, tw, sc, tc)
        time_list.append(time.perf_counter() - t0)
        rmse_list.append(float(np.sqrt(np.mean((P - P_gt)**2))))
    return np.array(rmse_list), np.array(time_list)



def pairs_to_color_loader(pairs_list, batch_size=1):
    """Wrap list of (sw,sc,tw,tc) into DataLoader yielding batched tensors."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    class _DS(Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, i):
            sw, sc, tw, tc = self.data[i]
            return (torch.tensor(sw, dtype=torch.float64),
                    torch.tensor(sc, dtype=torch.float64),
                    torch.tensor(tw, dtype=torch.float64),
                    torch.tensor(tc, dtype=torch.float64))
    return DataLoader(_DS(pairs_list), batch_size=batch_size, shuffle=False)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Saved → {path}")


def print_table(results, M, N):
    print(f"\n{'='*72}")
    print(f"  Color Transfer  |  M={M} train pairs  |  N={N} test pairs")
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
    p.add_argument("--data_dir",   type=str, required=True)
    p.add_argument("--M",          type=int, default=50)
    p.add_argument("--N",          type=int, default=20)
    p.add_argument("--n_clusters", type=int, default=500)
    p.add_argument("--gpu",        type=str, default="0")
    p.add_argument("--out",        type=str, default="./results/color")
    return p.parse_args()


def make_cfg_proj(solver, seed, gpu, flag_time):
    import argparse as _ap
    return _ap.Namespace(seed=seed, flag_time=flag_time,
                         flag_load=None, solver=solver,
                         data_name="color_transfer", gpu=gpu)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(os.path.join(args.out, f"M{args.M}"), exist_ok=True)
    flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())

    exts = ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"]
    image_paths = sorted(sum(
        [glob.glob(os.path.join(args.data_dir, e)) for e in exts], []))
    assert len(image_paths) >= 2
    print(f"Found {len(image_paths)} images.")
    eps = 0.5

    # Fixed test pairs — same for ALL methods
    print("Building test pairs ...")
    test_pairs = build_test_pairs(image_paths, args.N, args.n_clusters)
    print(f"  {len(test_pairs)} test pairs ready (seed={TEST_SEED})")

    # Save test pairs for plot scripts
    test_pairs_path = os.path.join(args.out, f"M{args.M}", "test_pairs.pkl")
    os.makedirs(os.path.dirname(test_pairs_path), exist_ok=True)
    with open(test_pairs_path, "wb") as _f:
        pickle.dump(test_pairs, _f)
    print(f"  Test pairs saved → {test_pairs_path}")
    results = []

    def make_train_loader(cfg_m):
        return get_color_transfer_dataloader(
            image_dir=args.data_dir, n_clusters=args.n_clusters,
            batch_size=cfg_m["batch_size"], seed=TRAIN_SEED,
            max_pairs=args.M * 4, num_workers=0)

    # ── Pre-sample M train pairs ONCE — shared by ALL methods ──────────────
    print(f"Pre-sampling M={args.M} train pairs ...")
    _tmp_loader = make_train_loader(init_cfg("OT_Regression_Sliced_Color"))
    train_pairs_list = []
    _pbar = tqdm(total=args.M, desc="  Sampling train pairs", leave=False)
    for _sw, _sc, _tw, _tc in _tmp_loader:
        for _i in range(_sw.shape[0]):
            if len(train_pairs_list) >= args.M: break
            train_pairs_list.append((
                _sw[_i].numpy(), _sc[_i].numpy(),
                _tw[_i].numpy(), _tc[_i].numpy()))
            _pbar.update(1)
        if len(train_pairs_list) >= args.M: break
    _pbar.close()
    print(f"  {len(train_pairs_list)} train pairs sampled (seed={TRAIN_SEED})")
    dl_shared = pairs_to_color_loader(train_pairs_list, batch_size=1)

    # ── 1. OT Regression Sliced Color ─────────────────────────────────────
    print("\n[1/4] OT Regression Sliced Color (Method 1) ...")
    from Solvers.Regression_SlicedOT.OT_Regression_Sliced_Color import OT_Regression_Sliced_Color
    cfg1 = init_cfg("OT_Regression_Sliced_Color")
    cfg1["num_bootstrap"] = args.M; cfg1["n_clusters"] = args.n_clusters; cfg1["epsilon"] = eps
    model1 = OT_Regression_Sliced_Color(
        make_cfg_proj("OT_Regression_Sliced_Color", TRAIN_SEED, args.gpu, flag_time), cfg1)
    t0 = time.perf_counter()
    model1.alpha, model1.beta = model1._fit(dl_shared)
    t1 = time.perf_counter() - t0
    save_model(model1, os.path.join(args.out, f"M{args.M}", "regression.pkl"))
    # predict_plan(a, b, x_src, x_tgt) ✓
    rmse1, tinf1 = evaluate_color(model1.predict_plan, test_pairs, eps, "OT_Regression")
    results.append(("OT Regression (M1)", rmse1, tinf1, t1))
    print(f"  Train: {t1:.1f}s  RMSE: {rmse1.mean():.2e}")

    # ── 2. OT Objective Sliced Color ──────────────────────────────────────
    print("\n[2/4] OT Objective Sliced Color (Method 2) ...")
    from Solvers.Objective_SlicedOT.OT_Objective_Sliced_Color import OT_Objective_Sliced_Color
    cfg2 = init_cfg("OT_Objective_Sliced_Color")
    cfg2["num_bootstrap"] = args.M; cfg2["n_clusters"] = args.n_clusters; cfg2["epsilon"] = eps
    model2 = OT_Objective_Sliced_Color(
        make_cfg_proj("OT_Objective_Sliced_Color", TRAIN_SEED, args.gpu, flag_time), cfg2)
    t0 = time.perf_counter()
    model2.alpha = model2._fit(dl_shared)
    t2 = time.perf_counter() - t0
    save_model(model2, os.path.join(args.out, f"M{args.M}", "objective.pkl"))
    # predict_plan(a, b, src_c, tgt_c) ✓
    rmse2, tinf2 = evaluate_color(model2.predict_plan, test_pairs, eps, "OT_Objective")
    results.append(("OT Objective (M2)", rmse2, tinf2, t2))
    print(f"  Train: {t2:.1f}s  RMSE: {rmse2.mean():.2e}")

    # ── 3. Meta OT Color ──────────────────────────────────────────────────
    print("\n[3/4] Meta OT Color Discrete (baseline) ...")
    from Solvers.Meta_OT.Meta_OT_Color import Meta_OT_Color
    cfg3 = init_cfg("Meta_OT_Color")
    cfg3["n_clusters"] = args.n_clusters; cfg3["epsilon"] = eps
    model3 = Meta_OT_Color(
        make_cfg_proj("Meta_OT_Color", TRAIN_SEED, args.gpu, flag_time), cfg3)
    t0 = time.perf_counter()
    model3.train(dl_shared)
    t3 = time.perf_counter() - t0
    save_model(model3, os.path.join(args.out, f"M{args.M}", "meta_ot.pkl"))
    # predict_plan(a, b, src_c, tgt_c) ✓
    rmse3, tinf3 = evaluate_color(model3.predict_plan, test_pairs, eps, "Meta OT")
    results.append(("Meta OT (baseline)", rmse3, tinf3, t3))
    print(f"  Train: {t3:.1f}s  RMSE: {rmse3.mean():.2e}")

    # ── 4. min-SWGG Color ─────────────────────────────────────────────────
    print("\n[4/4] min-SWGG Color (baseline, no training) ...")
    from Solvers.SWGG.min_SWGG_Color import min_SWGG_Color
    cfg4 = init_cfg("min_SWGG_Color")
    cfg4["n_clusters"] = args.n_clusters; cfg4["epsilon"] = eps
    model4 = min_SWGG_Color(
        make_cfg_proj("min_SWGG_Color", TRAIN_SEED, args.gpu, flag_time), cfg4)
    save_model(model4, os.path.join(args.out, f"M{args.M}", "swgg.pkl"))
    # predict_plan(a, b, src_c, tgt_c) ✓
    rmse4, tinf4 = evaluate_color(model4.predict_plan, test_pairs, eps, "min-SWGG")
    results.append(("min-SWGG (baseline)", rmse4, tinf4, 0.0))
    print(f"  RMSE: {rmse4.mean():.2e}")

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
