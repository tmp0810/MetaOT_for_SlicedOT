import argparse
import os
import pickle
import glob
import time
import numpy as np
import ot
from tqdm import tqdm
from time import localtime, strftime
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset
import torch

from cfg import init_cfg
from Data.color_transfer_data import get_color_transfer_dataloader

TRAIN_SEED = 0
TEST_SEED  = 999
EPS        = 0.005   # RGB space [0,1]^3: small distances → need small eps for sharp plans


def quantize_image(img_path, n_clusters=500, seed=0):
    img     = np.array(Image.open(img_path).convert("RGB"))
    H, W, _ = img.shape
    X       = img.reshape(-1, 3).astype(np.float32) / 255.0
    km = MiniBatchKMeans(n_clusters=n_clusters, n_init=4,
                         batch_size=min(4096, H*W), random_state=seed)
    km.fit(X)
    centroids = km.cluster_centers_.astype(np.float64)
    labels    = km.labels_.astype(np.int64)
    counts    = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    weights   = counts / counts.sum()
    return img, weights, centroids, labels


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
            img, w, c, lbl = quantize_image(pi, n_clusters, seed=0)
            cache[pi] = (img, w, c, lbl)
        if pj not in cache:
            img, w, c, lbl = quantize_image(pj, n_clusters, seed=0)
            cache[pj] = (img, w, c, lbl)
        src_img, sw, sc, sl = cache[pi]
        tgt_img, tw, tc, _  = cache[pj]
        pairs.append((sw, sc, sl, src_img, tw, tc, tgt_img, pi, pj))
        pbar.update(1)
    pbar.close()
    return pairs[:N]


def build_train_loader(image_paths, M, n_clusters, seed=TRAIN_SEED):
    """Pre-sample M train pairs — shared by ALL methods."""
    rng   = np.random.default_rng(seed)
    cache = {}
    pairs = []
    pbar  = tqdm(total=M, desc="  Train pairs", leave=False)
    attempts = 0
    while len(pairs) < M and attempts < M * 20:
        attempts += 1
        i, j = rng.choice(len(image_paths), size=2, replace=False)
        pi, pj = image_paths[i], image_paths[j]
        if pi not in cache:
            _, w, c, _ = quantize_image(pi, n_clusters, seed=0)
            cache[pi] = (w, c)
        if pj not in cache:
            _, w, c, _ = quantize_image(pj, n_clusters, seed=0)
            cache[pj] = (w, c)
        sw, sc = cache[pi]
        tw, tc = cache[pj]
        pairs.append((sw, sc, tw, tc))
        pbar.update(1)
    pbar.close()

    class _DS(Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, k):
            sw, sc, tw, tc = self.data[k]
            return (torch.tensor(sw, dtype=torch.float64),
                    torch.tensor(sc, dtype=torch.float64),
                    torch.tensor(tw, dtype=torch.float64),
                    torch.tensor(tc, dtype=torch.float64))
    return DataLoader(_DS(pairs), batch_size=1, shuffle=False)


# ── evaluation ────────────────────────────────────────────────────────────────

def sinkhorn_gt(a, b, C, eps, n_iter=1000):
    a_s = np.clip(a, 1e-10, None); a_s /= a_s.sum()
    b_s = np.clip(b, 1e-10, None); b_s /= b_s.sum()
    return ot.sinkhorn(a_s, b_s, C, reg=eps, numItermax=n_iter, stopThr=1e-9)


def evaluate_color(predict_fn, test_pairs, eps, name):
    if test_pairs:
        try:
            sw, sc, _sl, _si, tw, tc, _ti, *_ = test_pairs[0]
            predict_fn(sw, tw, sc, tc)
        except Exception: pass
 
    rmse_list, time_list = [], []
    for sw, sc, _sl, _si, tw, tc, _ti, *_ in tqdm(test_pairs,
                                                    desc=f"  Eval {name}", leave=False):
        diff = sc[:, None, :] - tc[None, :, :]
        C    = np.sum(diff**2, axis=-1)
        P_gt = sinkhorn_gt(sw, tw, C, eps)
        t0   = time.perf_counter()
        P    = predict_fn(sw, tw, sc, tc)
        time_list.append(time.perf_counter() - t0)
        rmse_list.append(float(np.sqrt(np.mean((P - P_gt)**2))))
    return np.array(rmse_list), np.array(time_list)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Saved → {path}")


def print_table(results, M, N):
    print(f"\n{'='*72}")
    print(f"  Color Transfer  |  M={M} train pairs  |  N={N} test pairs  |  eps={EPS}")
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
    return argparse.Namespace(seed=seed, flag_time=flag_time,
                              flag_load=None, solver=solver,
                              data_name="color_transfer", gpu=gpu)


# ── main ──────────────────────────────────────────────────────────────────────

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

    # ── Build test + train pairs ONCE ─────────────────────────────────────
    print(f"\nBuilding {args.N} test  pairs (seed={TEST_SEED}) ...")
    test_pairs = build_test_pairs(image_paths, args.N, args.n_clusters)
    test_pairs_path = os.path.join(args.out, f"M{args.M}", "test_pairs.pkl")
    with open(test_pairs_path, "wb") as f:
        pickle.dump(test_pairs, f)
    print(f"  {len(test_pairs)} test pairs saved → {test_pairs_path}")

    print(f"\nBuilding {args.M} train pairs (seed={TRAIN_SEED}) ...")
    dl_shared = build_train_loader(image_paths, args.M, args.n_clusters)
    print(f"  {args.M} train pairs ready — shared by all methods\n")

    results = []

    # ── 1. OT Regression Sliced Color ─────────────────────────────────────
    print("[1/4] OT Regression Sliced Color (Method 1) ...")
    from Solvers.Regression_SlicedOT.OT_Regression_Sliced_Color import OT_Regression_Sliced_Color
    cfg1 = init_cfg("OT_Regression_Sliced_Color")
    cfg1["num_bootstrap"] = args.M; cfg1["n_clusters"] = args.n_clusters
    cfg1["epsilon"] = EPS
    model1 = OT_Regression_Sliced_Color(
        make_cfg_proj("OT_Regression_Sliced_Color", TRAIN_SEED, args.gpu, flag_time), cfg1)
    t0 = time.perf_counter()
    model1.alpha, model1.beta = model1._fit(dl_shared)
    t1 = time.perf_counter() - t0
    save_model(model1, os.path.join(args.out, f"M{args.M}", "regression.pkl"))
    rmse1, tinf1 = evaluate_color(model1.predict_plan, test_pairs, EPS, "OT_Regression")
    results.append(("OT Regression (M1)", rmse1, tinf1, t1))
    print(f"  Train: {t1:.1f}s  RMSE: {rmse1.mean():.2e}")

    # ── 2. OT Objective Sliced Color ──────────────────────────────────────
    print("\n[2/4] OT Objective Sliced Color (Method 2) ...")
    from Solvers.Objective_SlicedOT.OT_Objective_Sliced_Color import OT_Objective_Sliced_Color
    cfg2 = init_cfg("OT_Objective_Sliced_Color")
    cfg2["num_bootstrap"] = args.M; cfg2["n_clusters"] = args.n_clusters
    cfg2["epsilon"] = EPS
    model2 = OT_Objective_Sliced_Color(
        make_cfg_proj("OT_Objective_Sliced_Color", TRAIN_SEED, args.gpu, flag_time), cfg2)
    t0 = time.perf_counter()
    model2.alpha = model2._fit(dl_shared)
    t2 = time.perf_counter() - t0
    save_model(model2, os.path.join(args.out, f"M{args.M}", "objective.pkl"))
    rmse2, tinf2 = evaluate_color(model2.predict_plan, test_pairs, EPS, "OT_Objective")
    results.append(("OT Objective (M2)", rmse2, tinf2, t2))
    print(f"  Train: {t2:.1f}s  RMSE: {rmse2.mean():.2e}")

    # ── 3. Meta OT Color ──────────────────────────────────────────────────
    print("\n[3/4] Meta OT Color Discrete (baseline) ...")
    from Solvers.Meta_OT.Meta_OT_Color import Meta_OT_Color
    cfg3 = init_cfg("Meta_OT_Color")
    cfg3["n_clusters"] = args.n_clusters; cfg3["epsilon"] = EPS
    model3 = Meta_OT_Color(
        make_cfg_proj("Meta_OT_Color", TRAIN_SEED, args.gpu, flag_time), cfg3)
    t0 = time.perf_counter()
    model3.train(dl_shared)
    t3 = time.perf_counter() - t0
    save_model(model3, os.path.join(args.out, f"M{args.M}", "meta_ot.pkl"))
    rmse3, tinf3 = evaluate_color(model3.predict_plan, test_pairs, EPS, "Meta OT")
    results.append(("Meta OT (baseline)", rmse3, tinf3, t3))
    print(f"  Train: {t3:.1f}s  RMSE: {rmse3.mean():.2e}")

    # ── 4. min-SWGG Color ─────────────────────────────────────────────────
    print("\n[4/4] min-SWGG Color (baseline, no training) ...")
    from Solvers.SWGG.min_SWGG_Color import min_SWGG_Color
    cfg4 = init_cfg("min_SWGG_Color")
    cfg4["n_clusters"] = args.n_clusters; cfg4["epsilon"] = EPS
    model4 = min_SWGG_Color(
        make_cfg_proj("min_SWGG_Color", TRAIN_SEED, args.gpu, flag_time), cfg4)
    save_model(model4, os.path.join(args.out, f"M{args.M}", "swgg.pkl"))
    rmse4, tinf4 = evaluate_color(model4.predict_plan, test_pairs, EPS, "min-SWGG")
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
