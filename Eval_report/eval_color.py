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

POOL_SEED   = 0
POOL_SIZE   = 1000
TRAIN_RATIO = 0.7   # 490 train / 210 test
EPS         = 0.005  # RGB space [0,1]^3: small distances → need small eps for sharp plans


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


def build_pool(image_paths, n_clusters, pool_size, seed):
    """
    Gen toàn bộ unordered pairs C(N_img, 2), shuffle với seed cố định,
    lấy pool_size pairs đầu. Cache quantization để tránh re-compute.
    Trả về list of (sw, sc, sl, src_img, tw, tc, tgt_img, pi, pj).
    """
    # Tạo tất cả unordered pairs (i < j)
    n = len(image_paths)
    all_pairs_idx = [(i, j) for i in range(n) for j in range(i+1, n)]

    # Shuffle với seed cố định
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(all_pairs_idx))
    all_pairs_idx = [all_pairs_idx[k] for k in order]

    # Lấy pool_size pairs đầu (sau shuffle)
    needed = min(pool_size, len(all_pairs_idx))
    assert needed == pool_size, (
        f"Không đủ pairs: cần {pool_size} nhưng chỉ có C({n},2)={len(all_pairs_idx)} pairs. "
        f"Cần ít nhất {int(np.ceil((1 + np.sqrt(1 + 8*pool_size)) / 2))} ảnh."
    )
    selected = all_pairs_idx[:pool_size]

    # Quantize từng ảnh cần dùng (cache)
    needed_imgs = set()
    for i, j in selected:
        needed_imgs.add(i); needed_imgs.add(j)

    cache = {}
    print(f"  Quantizing {len(needed_imgs)} images ...")
    for idx in tqdm(sorted(needed_imgs), desc="  Quantize", leave=False):
        img, w, c, lbl = quantize_image(image_paths[idx], n_clusters, seed=0)
        cache[idx] = (img, w, c, lbl)

    # Build pool list với đầy đủ thông tin
    pool = []
    for i, j in selected:
        src_img, sw, sc, sl = cache[i]
        tgt_img, tw, tc, _  = cache[j]
        pi, pj = image_paths[i], image_paths[j]
        pool.append((sw, sc, sl, src_img, tw, tc, tgt_img, pi, pj))

    return pool


def pool_to_train_loader(train_pool):
    """
    Chuyển train pool (list of full tuples) thành DataLoader.
    Chỉ dùng (sw, sc, tw, tc) — bỏ labels và ảnh gốc không cần cho training.
    """
    class _DS(Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, k):
            sw, sc, _sl, _si, tw, tc, _ti, _pi, _pj = self.data[k]
            return (torch.tensor(sw, dtype=torch.float64),
                    torch.tensor(sc, dtype=torch.float64),
                    torch.tensor(tw, dtype=torch.float64),
                    torch.tensor(tc, dtype=torch.float64))
    return DataLoader(_DS(train_pool), batch_size=1, shuffle=False)


# ── evaluation ────────────────────────────────────────────────────────────────

def sinkhorn_gt(a, b, C, eps, n_iter=1000):
    a_s = np.clip(a, 1e-10, None); a_s /= a_s.sum()
    b_s = np.clip(b, 1e-10, None); b_s /= b_s.sum()
    return ot.sinkhorn(a_s, b_s, C, reg=eps, numItermax=n_iter, stopThr=1e-9)


def evaluate_color(predict_fn, test_pairs, eps, name):
    # Warmup: eliminate CUDA cold-start outlier from timing
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

    # ── Build pool ONCE, split 70/30, shared by ALL methods ───────────────
    n_train_pool = int(POOL_SIZE * TRAIN_RATIO)   # 490
    n_test_pool  = POOL_SIZE - n_train_pool        # 210

    assert args.M <= n_train_pool, \
        f"M={args.M} exceeds train pool size {n_train_pool}"
    assert args.N <= n_test_pool, \
        f"N={args.N} exceeds test pool size {n_test_pool}"

    print(f"\nBuilding pool of {POOL_SIZE} pairs (seed={POOL_SEED}) ...")
    print(f"  Unordered pairs from {len(image_paths)} images: "
          f"C({len(image_paths)},2) = {len(image_paths)*(len(image_paths)-1)//2} total")
    pool = build_pool(image_paths, args.n_clusters, POOL_SIZE, POOL_SEED)

    train_pool = pool[:n_train_pool]   # first 490 — train
    test_pool  = pool[n_train_pool:]   # last  210 — test, image pairs never in train

    train_pairs = train_pool[:args.M]  # first M of train pool (nested)
    test_pairs  = test_pool[:args.N]   # first N of test pool (fixed across all M)

    print(f"  Pool: {POOL_SIZE}  →  train pool: {n_train_pool}  |  test pool: {n_test_pool}")
    print(f"  Using M={args.M} train pairs  |  N={args.N} test pairs")
    print(f"  → All 4 methods will use EXACTLY these same pairs.\n")

    # Save test pairs for plot scripts (format giữ nguyên như cũ)
    test_pairs_path = os.path.join(args.out, f"M{args.M}", "test_pairs.pkl")
    with open(test_pairs_path, "wb") as f:
        pickle.dump(test_pairs, f)
    print(f"  {len(test_pairs)} test pairs saved → {test_pairs_path}")

    dl_shared = pool_to_train_loader(train_pairs)
    print(f"  {len(train_pairs)} train pairs ready — shared by all methods\n")

    results = []

    # # ── 1. OT Regression Sliced Color ─────────────────────────────────────
    # print("[1/4] OT Regression Sliced Color (Method 1) ...")
    # from Solvers.Regression_SlicedOT.OT_Regression_Sliced_Color import OT_Regression_Sliced_Color
    # cfg1 = init_cfg("OT_Regression_Sliced_Color")
    # cfg1["num_bootstrap"] = args.M; cfg1["n_clusters"] = args.n_clusters
    # cfg1["epsilon"] = EPS
    # model1 = OT_Regression_Sliced_Color(
    #     make_cfg_proj("OT_Regression_Sliced_Color", POOL_SEED, args.gpu, flag_time), cfg1)
    # t0 = time.perf_counter()
    # model1.alpha = model1._fit(dl_shared)
    # model1.beta  = np.zeros_like(model1.alpha)
    # t1 = time.perf_counter() - t0
    # save_model(model1, os.path.join(args.out, f"M{args.M}", "regression.pkl"))
    # rmse1, tinf1 = evaluate_color(model1.predict_plan, test_pairs, EPS, "OT_Regression")
    # results.append(("OT Regression (M1)", rmse1, tinf1, t1))
    # print(f"  Train: {t1:.1f}s  RMSE: {rmse1.mean():.2e}")

    # # ── 2. OT Objective Sliced Color ──────────────────────────────────────
    # print("\n[2/4] OT Objective Sliced Color (Method 2) ...")
    # from Solvers.Objective_SlicedOT.OT_Objective_Sliced_Color import OT_Objective_Sliced_Color
    # cfg2 = init_cfg("OT_Objective_Sliced_Color")
    # cfg2["num_bootstrap"] = args.M; cfg2["n_clusters"] = args.n_clusters
    # cfg2["epsilon"] = EPS
    # model2 = OT_Objective_Sliced_Color(
    #     make_cfg_proj("OT_Objective_Sliced_Color", POOL_SEED, args.gpu, flag_time), cfg2)
    # t0 = time.perf_counter()
    # model2.alpha = model2._fit(dl_shared)
    # t2 = time.perf_counter() - t0
    # save_model(model2, os.path.join(args.out, f"M{args.M}", "objective.pkl"))
    # rmse2, tinf2 = evaluate_color(model2.predict_plan, test_pairs, EPS, "OT_Objective")
    # results.append(("OT Objective (M2)", rmse2, tinf2, t2))
    # print(f"  Train: {t2:.1f}s  RMSE: {rmse2.mean():.2e}")

    # # ── 3. Meta OT Color ──────────────────────────────────────────────────
    # print("\n[3/4] Meta OT Color Discrete (baseline) ...")
    # from Solvers.Meta_OT.Meta_OT_Color import Meta_OT_Color
    # cfg3 = init_cfg("Meta_OT_Color")
    # cfg3["n_clusters"] = args.n_clusters; cfg3["epsilon"] = EPS
    # model3 = Meta_OT_Color(
    #     make_cfg_proj("Meta_OT_Color", POOL_SEED, args.gpu, flag_time), cfg3)
    # t0 = time.perf_counter()
    # model3.train(dl_shared)
    # t3 = time.perf_counter() - t0
    # save_model(model3, os.path.join(args.out, f"M{args.M}", "meta_ot.pkl"))
    # rmse3, tinf3 = evaluate_color(model3.predict_plan, test_pairs, EPS, "Meta OT")
    # results.append(("Meta OT (baseline)", rmse3, tinf3, t3))
    # print(f"  Train: {t3:.1f}s  RMSE: {rmse3.mean():.2e}")

    # # ── 4. min-SWGG Color ─────────────────────────────────────────────────
    # print("\n[4/4] min-SWGG Color (baseline, no training) ...")
    # from Solvers.SWGG.min_SWGG_Color import min_SWGG_Color
    # cfg4 = init_cfg("min_SWGG_Color")
    # cfg4["n_clusters"] = args.n_clusters; cfg4["epsilon"] = EPS
    # model4 = min_SWGG_Color(
    #     make_cfg_proj("min_SWGG_Color", POOL_SEED, args.gpu, flag_time), cfg4)
    # save_model(model4, os.path.join(args.out, f"M{args.M}", "swgg.pkl"))
    # rmse4, tinf4 = evaluate_color(model4.predict_plan, test_pairs, EPS, "min-SWGG")
    # results.append(("min-SWGG (baseline)", rmse4, tinf4, 0.0))
    # print(f"  RMSE: {rmse4.mean():.2e}")

    # ── 5. Min-STP Color ──────────────────────────────────────────────────
    print("\n[5/5] Min-STP Color (baseline) ...")
    from Solvers.MinSTP.Min_STP_Color import Min_STP_Color
    cfg3 = init_cfg("Min_STP_Color")
    cfg3["n_clusters"] = args.n_clusters
    model3 = Min_STP_Color(
        make_cfg_proj("Min_STP_Color", POOL_SEED, args.gpu, flag_time), cfg3)
    t0 = time.perf_counter()
    model3.train(dl_shared)
    t3 = time.perf_counter() - t0
    save_model(model3, os.path.join(args.out, f"M{args.M}", "min_stp.pkl"))
    rmse3, tinf3 = evaluate_color(model3.predict_plan, test_pairs, EPS, "Min-STP")
    results.append(("Min-STP (baseline)", rmse3, tinf3, t3))
    print(f"  Train: {t3:.1f}s  RMSE: {rmse3.mean():.2e}")

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
