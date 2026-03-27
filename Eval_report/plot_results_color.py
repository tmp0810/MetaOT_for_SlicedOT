import argparse
import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import ot
plt.style.use("bmh")


def apply_color_transfer(src_img, src_labels, tgt_centroids, P):
    row_sums      = P.sum(axis=1, keepdims=True).clip(1e-12, None)
    P_row         = P / row_sums
    new_centroids = P_row @ tgt_centroids
    new_pixels    = new_centroids[src_labels]
    return (np.clip(new_pixels, 0, 1) * 255).astype(np.uint8).reshape(src_img.shape)


def solve_sinkhorn_baseline(a, b, C, reg, num_iter=1000):
    return ot.sinkhorn(a, b, C, reg=reg, numItermax=num_iter,
                       stopThr=1e-9, log=False)


def plot_pair(idx, src_img, src_labels, src_w, src_c,
              tgt_img, tgt_w, tgt_c,
              methods, out_dir, eps, run_baseline=True):
    """Same layout as eval_color_transfer.py:eval_pair."""
    pair_dir = os.path.join(out_dir, "plots", f"pair_{idx:02d}")
    os.makedirs(pair_dir, exist_ok=True)

    C = np.sum((src_c[:, None, :] - tgt_c[None, :, :])**2, axis=-1)

    P_sink = None
    if run_baseline:
        t0     = time.time()
        P_sink = solve_sinkhorn_baseline(src_w, tgt_w, C, reg=eps)
        t_sink = time.time() - t0
        img_sink = apply_color_transfer(src_img, src_labels, tgt_c, P_sink)
        print(f"  [{idx}] Sinkhorn_GT: {t_sink:.3f}s")

    method_results = []
    for name, predict_fn in methods:
        t0   = time.time()
        P    = predict_fn(src_w, tgt_w, src_c, tgt_c)
        t    = time.time() - t0
        img  = apply_color_transfer(src_img, src_labels, tgt_c, P)
        rmse = float(np.sqrt(np.mean((P - P_sink)**2))) if P_sink is not None else None
        method_results.append((name, img, t, rmse))
        print(f"  [{idx}] {name}: {t:.3f}s" +
              (f"  RMSE={rmse:.2e}" if rmse is not None else ""))

    n_cols = 2 + len(methods) + (1 if run_baseline else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5))

    axes[0].imshow(src_img)
    axes[0].set_title("Source", fontsize=8)

    for col, (name, img, t, rmse) in enumerate(method_results, start=1):
        axes[col].imshow(img)
        title = f"{name}\n({t:.3f}s)"
        if rmse is not None:
            title += f"\nRMSE={rmse:.2e}"
        axes[col].set_title(title, fontsize=8)

    last = 1 + len(methods)
    if run_baseline:
        axes[last].imshow(img_sink)
        axes[last].set_title(f"Sinkhorn GT (eps={eps})\n({t_sink:.3f}s)", fontsize=8)
        last += 1

    axes[last].imshow(tgt_img)
    axes[last].set_title("Target", fontsize=8)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Color Transfer — Test Pair {idx}", fontsize=9, y=1.01)
    fig.tight_layout()
    out_path = os.path.join(pair_dir, "comparison.png")
    fig.savefig(out_path, dpi=100, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  [{idx}] Saved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir",  type=str, required=True)
    p.add_argument("--idx",         type=str, default="all")
    p.add_argument("--no_baseline", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    with open(os.path.join(args.result_dir, "test_pairs.pkl"), "rb") as f:
        test_pairs = pickle.load(f)
    print(f"Loaded {len(test_pairs)} test pairs.")

    def load_pkl(name):
        with open(os.path.join(args.result_dir, name), "rb") as f:
            return pickle.load(f)

    model_reg  = load_pkl("regression.pkl")
    model_obj  = load_pkl("objective.pkl")
    model_meta = load_pkl("meta_ot.pkl")
    model_swgg = load_pkl("swgg.pkl")
    model_stp  = load_pkl("min_stp.pkl")
    eps        = float(model_reg.cfg_m.epsilon)   # read from model, not hardcoded

    print(f"eps = {eps}")

    methods = [
        ("OT_Regression", model_reg.predict_plan),
        ("OT_Objective",  model_obj.predict_plan),
        ("Meta_OT",       model_meta.predict_plan),
        ("min_SWGG",      model_swgg.predict_plan),
        ("Min-STP",       model_stp.predict_plan),
    ]

    # test_pairs format: (src_w, src_c, src_labels, src_img, tgt_w, tgt_c, tgt_img, src_path, tgt_path)
    indices = range(len(test_pairs)) if args.idx == "all" else [int(args.idx)]

    for idx in indices:
        sw, sc, sl, src_img, tw, tc, tgt_img, src_path, tgt_path = test_pairs[idx]
        print(f"\nPair {idx}: {os.path.basename(src_path)} → {os.path.basename(tgt_path)}")
        plot_pair(idx, src_img, sl, sw, sc,
                  tgt_img, tw, tc,
                  methods, args.result_dir,
                  eps=eps, run_baseline=not args.no_baseline)

    print(f"\nDone. Output → {args.result_dir}/plots/")


if __name__ == "__main__":
    main()
