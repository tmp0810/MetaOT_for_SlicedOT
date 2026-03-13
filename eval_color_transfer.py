import argparse
import os
import glob
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ot
from sklearn.cluster import MiniBatchKMeans

plt.style.use('bmh')


def quantize_for_eval(img_path: str, n_clusters: int = 500, seed: int = 0):
    img     = np.array(Image.open(img_path).convert('RGB'))
    H, W, _ = img.shape
    X       = img.reshape(-1, 3).astype(np.float32) / 255.0

    km = MiniBatchKMeans(
        n_clusters=n_clusters, n_init=4,
        batch_size=min(4096, H * W), random_state=seed,
    )
    km.fit(X)

    centroids = km.cluster_centers_.astype(np.float64)
    labels    = km.labels_.astype(np.int64)
    counts    = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    weights   = counts / counts.sum()
    return img, weights, centroids, labels



def apply_color_transfer(
    src_img:      np.ndarray,
    src_labels:   np.ndarray,
    tgt_centroids: np.ndarray,
    P:            np.ndarray,
) -> np.ndarray:
    row_sums      = P.sum(axis=1, keepdims=True).clip(1e-12, None)
    P_row         = P / row_sums                          # row-stochastic
    new_centroids = P_row @ tgt_centroids                 # (n_src, 3) in [0,1]
    new_pixels    = new_centroids[src_labels]             # (H*W, 3)
    return (np.clip(new_pixels, 0, 1) * 255).astype(np.uint8).reshape(src_img.shape)


def solve_sinkhorn_baseline(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    reg: float = 0.5,
    num_iter: int = 1000,
) -> np.ndarray:
    return ot.sinkhorn(
        a, b, C, reg=reg,
        numItermax=num_iter, stopThr=1e-9, log=False,
    )



def eval_pair(
    model,
    src_path: str,
    tgt_path: str,
    out_path: str,
    n_clusters:   int  = 500,
    run_baseline: bool = True,
    seed:         int  = 0,
):

    # ── Quantize (full-res for src labels, smaller ok for tgt centroid fitting)
    src_img, src_w, src_c, src_labels = quantize_for_eval(src_path, n_clusters, seed)
    tgt_img, tgt_w, tgt_c, _         = quantize_for_eval(tgt_path, n_clusters, seed)

    C   = model._compute_cost(src_c, tgt_c)   # (n_src, n_tgt)
    eps = float(getattr(model.cfg_m, 'epsilon', 0.5))

    timings = {}

    # ── OT Regression (our method) ────────────────────────────────────────
    t0    = time.time()
    P_reg = model.predict_plan(src_w, tgt_w, src_c, tgt_c)
    timings['regression'] = time.time() - t0
    img_reg = apply_color_transfer(src_img, src_labels, tgt_c, P_reg)

    # ── Sinkhorn baseline ─────────────────────────────────────────────────
    if run_baseline:
        t0      = time.time()
        P_sink  = solve_sinkhorn_baseline(src_w, tgt_w, C, reg=eps)
        timings['sinkhorn'] = time.time() - t0
        img_sink = apply_color_transfer(src_img, src_labels, tgt_c, P_sink)

        rmse_P = float(np.sqrt(np.mean((P_reg - P_sink) ** 2)))
        timings['rmse_P'] = rmse_P
        
        print(f"  RMSE_Plan: {rmse_P:.8f} | sum_gt={P_sink.sum():.4f} sum_pred={P_reg.sum():.4f}")

    # if run_baseline:
    #     t0      = time.time()
        

    #     f_sink, g_sink = model._solve_entropic_ot(src_w, tgt_w, C)
        
    #     # Dùng hàm biến điện thế thành Plan (Nhớ truyền đủ a, b, f, g, C như bạn đã sửa)
    #     P_sink         = model._potentials_to_plan(src_w, tgt_w, f_sink, g_sink, C)
        
    #     timings['sinkhorn'] = time.time() - t0
    #     img_sink = apply_color_transfer(src_img, src_labels, tgt_c, P_sink)

    # ── Plot ──────────────────────────────────────────────────────────────
    n_cols = 4 if run_baseline else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5))

    axes[0].imshow(src_img)
    axes[0].set_title(
        f'Source\n{os.path.basename(src_path)[:28]}', fontsize=8)

    axes[1].imshow(img_reg)
    axes[1].set_title(
        f'OT Regression\n({timings["regression"]:.2f}s)', fontsize=8)

    if run_baseline:
        axes[2].imshow(img_sink)
        axes[2].set_title(
            f'Sinkhorn (eps={eps})\n({timings["sinkhorn"]:.2f}s)', fontsize=8)
        axes[3].imshow(tgt_img)
        axes[3].set_title(
            f'Target\n{os.path.basename(tgt_path)[:28]}', fontsize=8)
    else:
        axes[2].imshow(tgt_img)
        axes[2].set_title(
            f'Target\n{os.path.basename(tgt_path)[:28]}', fontsize=8)

    for ax in axes:
        ax.axis('off')

    fig.suptitle(
        f'{os.path.basename(src_path)} → {os.path.basename(tgt_path)}',
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"  Saved -> {out_path}")
    return timings



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  type=str, required=True,
                        help='Path to model.pkl from main_color_transfer.py')
    parser.add_argument('--data_dir',    type=str, default=None,
                        help='Image folder (random pairs sampled from here)')
    parser.add_argument('--src',         type=str, default=None,
                        help='Specific source image (overrides --data_dir)')
    parser.add_argument('--tgt',         type=str, default=None,
                        help='Specific target image (overrides --data_dir)')
    parser.add_argument('--out_dir',     type=str, default='./results/color_transfer')
    parser.add_argument('--n_clusters',  type=int, default=500,
                        help='KMeans clusters for quantization')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of random test pairs (ignored if --src/--tgt given)')
    parser.add_argument('--seed',        type=int, default=0)
    parser.add_argument('--no_baseline', action='store_true',
                        help='Skip Sinkhorn baseline (faster evaluation)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path} ...")
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)

    all_timings = {'regression': [], 'sinkhorn': []}

    # ── Single specific pair mode ──────────────────────────────────────────
    if args.src and args.tgt:
        print(f"\nEvaluating: {os.path.basename(args.src)} -> {os.path.basename(args.tgt)}")
        out_path = os.path.join(args.out_dir, 'transfer_specific.png')
        timings  = eval_pair(
            model, args.src, args.tgt, out_path,
            n_clusters   = args.n_clusters,
            run_baseline = not args.no_baseline,
            seed         = args.seed,
        )
        print(f"  OT Regression: {timings['regression']:.3f}s")
        if 'sinkhorn' in timings:
            print(f"  Sinkhorn:      {timings['sinkhorn']:.3f}s")
            print(f"  Speedup:       {timings['sinkhorn']/timings['regression']:.1f}x")
        return

    # ── Random pairs from data_dir ────────────────────────────────────────
    assert args.data_dir is not None, "Provide --data_dir or --src/--tgt"
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(args.data_dir, ext)))
    image_paths = sorted(image_paths)
    assert len(image_paths) >= 2, f"Need >= 2 images in {args.data_dir}"
    print(f"Found {len(image_paths)} images.")

    rng = np.random.default_rng(args.seed)

    for i in range(args.num_samples):
        src_idx, tgt_idx = rng.choice(len(image_paths), size=2, replace=False)
        src_path = image_paths[src_idx]
        tgt_path = image_paths[tgt_idx]
        print(f"\nSample {i+1}/{args.num_samples}: "
              f"{os.path.basename(src_path)} -> {os.path.basename(tgt_path)}")

        out_path = os.path.join(args.out_dir, f'transfer_{i:04d}.png')
        try:
            timings = eval_pair(
                model, src_path, tgt_path, out_path,
                n_clusters   = args.n_clusters,
                run_baseline = not args.no_baseline,
                seed         = args.seed + i,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        all_timings['regression'].append(timings['regression'])
        if 'sinkhorn' in timings:
            all_timings['sinkhorn'].append(timings['sinkhorn'])

    # ── Summary ───────────────────────────────────────────────────────────
    if all_timings['regression']:
        r = np.array(all_timings['regression'])
        print(f"\n{'='*50}")
        print(f"Timing summary  (n_clusters={args.n_clusters})")
        print(f"{'='*50}")
        print(f"OT Regression : {r.mean():.3f}s +/- {r.std():.3f}s")
        if all_timings['sinkhorn']:
            s = np.array(all_timings['sinkhorn'])
            print(f"Sinkhorn      : {s.mean():.3f}s +/- {s.std():.3f}s")
            print(f"Speedup       : {s.mean()/r.mean():.1f}x")
        print(f"\nAll results saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
