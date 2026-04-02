import argparse
import os
import pickle
import shutil
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ot
from PIL import Image

plt.style.use("bmh")



def apply_color_transfer(
    src_img:     np.ndarray,   # (H, W, 3) uint8  original source image
    src_labels:  np.ndarray,   # (H*W,)    int64  KMeans label per pixel
    tgt_centroids: np.ndarray, # (K, 3)    float64 target cluster centers [0,1]
    P:           np.ndarray,   # (K_src, K_tgt) transport plan
) -> np.ndarray:
    row_sums      = P.sum(axis=1, keepdims=True).clip(1e-12, None)
    P_row         = P / row_sums                            # row-stochastic
    new_centroids = P_row @ tgt_centroids                   # (K_src, 3)
    new_pixels    = new_centroids[src_labels]               # (H*W, 3)
    return (np.clip(new_pixels, 0.0, 1.0) * 255).astype(np.uint8).reshape(src_img.shape)


def save_image(arr: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)
    print(f"    Saved → {path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot color transfer results in Meta OT paper format "
                    "(one standalone PNG per method per pair).")
    p.add_argument("--result_dir",  type=str, required=True,
                   help="M{N} dir, e.g. ./results/color/M50")
    p.add_argument("--idx",         type=str, default="all",
                   help="Test pair index (int) or 'all'")
    p.add_argument("--no_baseline", action="store_true",
                   help="Skip Sinkhorn GT computation")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load test pairs ───────────────────────────────────────────────────
    with open(os.path.join(args.result_dir, "test_pairs.pkl"), "rb") as f:
        test_pairs = pickle.load(f)
    print(f"Loaded {len(test_pairs)} test pairs.")

    # ── Load trained models ────────────────────────────────────────────────
    def load_pkl(name):
        with open(os.path.join(args.result_dir, name), "rb") as f:
            return pickle.load(f)

    # model_reg  = load_pkl("regression.pkl")
    # model_obj  = load_pkl("objective.pkl")
    model_meta = load_pkl("meta_ot.pkl")
    # model_swgg = load_pkl("swgg.pkl")
    # model_stp  = load_pkl("min_stp.pkl")

    eps = float(model_reg.cfg_m.epsilon)
    print(f"eps = {eps}")
    methods = [
        # ("OT_Regression", model_reg.predict_plan),
        # ("OT_Objective",  model_obj.predict_plan),
        ("Meta_OT",       model_meta.predict_plan),
        # ("min_SWGG",      model_swgg.predict_plan),
        # ("Min_STP",       model_stp.predict_plan),
    ]

    # test_pairs format: (sw, sc, sl, src_img, tw, tc, tgt_img, src_path, tgt_path)
    indices = range(len(test_pairs)) if args.idx == "all" else [int(args.idx)]

    for idx in indices:
        sw, sc, sl, src_img, tw, tc, tgt_img, src_path, tgt_path = test_pairs[idx]
        print(f"\n=== Pair {idx}: "
              f"{os.path.basename(src_path)} → {os.path.basename(tgt_path)} ===")

        pair_dir = os.path.join(args.result_dir, "plots", f"pair_{idx:02d}")
        os.makedirs(pair_dir, exist_ok=True)

        # ── Source image ──────────────────────────────────────────────────
        save_image(src_img, os.path.join(pair_dir, "Source.png"))

        # ── Target image ──────────────────────────────────────────────────
        save_image(tgt_img, os.path.join(pair_dir, "Target.png"))

        # ── Precompute cost matrix (shared by Sinkhorn + all methods) ─────
        diff = sc[:, None, :] - tc[None, :, :]      # (K_src, K_tgt, 3)
        C    = np.sum(diff ** 2, axis=-1)            # (K_src, K_tgt)

        # ── Sinkhorn ground truth ─────────────────────────────────────────
        if not args.no_baseline:
            print(f"  [Sinkhorn_GT] computing ...")
            a_s = np.clip(sw, 1e-10, None); a_s /= a_s.sum()
            b_s = np.clip(tw, 1e-10, None); b_s /= b_s.sum()
            t0     = time.time()
            P_sink = ot.sinkhorn(a_s, b_s, C, reg=eps,
                                 numItermax=1000, stopThr=1e-9, log=False)
            t_sink = time.time() - t0
            print(f"  [Sinkhorn_GT] done in {t_sink:.3f}s")
            img_sink = apply_color_transfer(src_img, sl, tc, P_sink)
            save_image(img_sink, os.path.join(pair_dir, "Sinkhorn_GT.png"))

        # ── Each method → one standalone PNG ─────────────────────────────
        for name, predict_fn in methods:
            print(f"  [{name}] computing ...")
            try:
                t0 = time.time()
                P  = predict_fn(sw, tw, sc, tc)
                print(f"  [{name}] done in {time.time()-t0:.3f}s")
            except Exception as e:
                print(f"  [{name}] ERROR: {e}")
                continue

            img_method = apply_color_transfer(src_img, sl, tc, P)
            save_image(img_method, os.path.join(pair_dir, f"{name}.png"))

    print(f"\nDone.  Output → {args.result_dir}/plots/")


if __name__ == "__main__":
    main()
