import argparse
import os
import pickle
import time
import glob
import numpy as np
from time import localtime, strftime

from cfg import init_cfg
from Solvers.SWGG.min_SWGG_Color import min_SWGG_Color


def parse_args():
    p = argparse.ArgumentParser(
        description="min-SWGG baseline for color transfer")
    p.add_argument("--data_dir",      type=str, required=True,
                   help="Directory of painting images")
    p.add_argument("--out_dir",       type=str,
                   default="./runs/min_swgg_color")
    p.add_argument("--n_clusters",    type=int, default=None,
                   help="KMeans clusters per image (default: cfg=500)")
    p.add_argument("--n_projections", type=int, default=None,
                   help="L random directions (default: cfg=200)")
    p.add_argument("--num_samples",   type=int, default=10,
                   help="Test pairs for timing/quality evaluation")
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--gpu",           type=str, default="0")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    cfg_m = init_cfg("min_SWGG_Color")
    if args.n_clusters    is not None: cfg_m["n_clusters"]    = args.n_clusters
    if args.n_projections is not None: cfg_m["n_projections"] = args.n_projections
    cfg_m["gpu"] = int(args.gpu) if args.gpu.isdigit() else 0

    cfg_proj = argparse.Namespace(
        seed      = args.seed,
        flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime()),
        flag_load = None,
        solver    = "min_SWGG_Color",
        data_name = "color_transfer",
        gpu       = args.gpu,
    )

    # Discover images
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(args.data_dir, ext)))
    image_paths = sorted(image_paths)
    assert len(image_paths) >= 2, f"Need >= 2 images in {args.data_dir}"

    print(f"\n{'='*55}")
    print(f"  min-SWGG baseline — Color Transfer")
    print(f"  No training: test-time θ* random search per pair")
    print(f"{'='*55}")
    print(f"  data_dir      : {args.data_dir}  ({len(image_paths)} images)")
    print(f"  n_clusters    : {cfg_m['n_clusters']}")
    print(f"  n_projections : {cfg_m['n_projections']}")
    print(f"  epsilon       : {cfg_m['epsilon']}  (Sinkhorn comparison)")
    print(f"{'='*55}\n")

    # Build model (no training)
    model = min_SWGG_Color(cfg_proj=cfg_proj, cfg_m=cfg_m)
    model.train(None)  # no-op

    # Save model (lightweight — only cfg, no weights)
    model_path = os.path.join(args.out_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {model_path}")

    # ── Quick sanity check + timing ───────────────────────────────────
    from Data.color_transfer_data import load_and_quantize
    import ot as pot

    print(f"\nEvaluating on {args.num_samples} random pairs ...")
    rng = np.random.default_rng(args.seed)
    times_swgg, times_sink, rmse_list = [], [], []
    eps = float(cfg_m["epsilon"])

    for i in range(args.num_samples):
        idx_s, idx_t = rng.choice(len(image_paths), size=2, replace=False)
        src_path = image_paths[idx_s]
        tgt_path = image_paths[idx_t]

        # Quantize
        src_w, src_c, _, _ = load_and_quantize(src_path, cfg_m["n_clusters"])
        tgt_w, tgt_c, _, _ = load_and_quantize(tgt_path, cfg_m["n_clusters"])

        # min-SWGG
        t0      = time.perf_counter()
        P_pred  = model.predict_plan(src_w, tgt_w, src_c, tgt_c)
        t_swgg  = time.perf_counter() - t0
        times_swgg.append(t_swgg)

        # Sinkhorn GT
        C       = model._compute_cost(src_c, tgt_c)
        a_s     = np.clip(src_w, 1e-10, None); a_s /= a_s.sum()
        b_s     = np.clip(tgt_w, 1e-10, None); b_s /= b_s.sum()
        t0      = time.perf_counter()
        P_gt    = pot.sinkhorn(a_s, b_s, C, reg=eps, numItermax=1000, stopThr=1e-9)
        t_sink  = time.perf_counter() - t0
        times_sink.append(t_sink)

        rmse    = float(np.sqrt(np.mean((P_pred - P_gt)**2)))
        rmse_list.append(rmse)

        import os as _os
        print(f"  [{i+1}/{args.num_samples}] "
              f"{_os.path.basename(src_path)[:20]} → "
              f"{_os.path.basename(tgt_path)[:20]}  "
              f"RMSE={rmse:.8f}  "
              f"t_swgg={t_swgg:.3f}s  t_sink={t_sink:.3f}s  "
              f"speedup={t_sink/max(t_swgg,1e-9):.1f}x")

    # Summary
    print(f"\n{'='*55}")
    print(f"Summary ({args.num_samples} pairs, L={cfg_m['n_projections']})")
    print(f"{'='*55}")
    print(f"RMSE_Plan    : {np.mean(rmse_list):.8f} ± {np.std(rmse_list):.8f}")
    print(f"min-SWGG     : {np.mean(times_swgg):.3f}s ± {np.std(times_swgg):.3f}s")
    print(f"Sinkhorn     : {np.mean(times_sink):.3f}s ± {np.std(times_sink):.3f}s")
    print(f"Speedup      : {np.mean(times_sink)/max(np.mean(times_swgg),1e-9):.1f}x")
    print(f"\nDone. Run eval_color_transfer.py to evaluate and plot.")
    print(f"  python eval_color_transfer.py --model_path {model_path} \\")
    print(f"      --data_dir {args.data_dir} --num_samples 10")


if __name__ == "__main__":
    main()
