import argparse
import os
import pickle
import numpy as np
import torch
import ot
import matplotlib
matplotlib.use("Agg")       
import matplotlib.pyplot as plt

plt.style.use("bmh")

def crop_hist_col(hist: np.ndarray) -> np.ndarray:
    col_sum = hist.sum(axis=0)
    nonzero = np.where(col_sum > 0)[0]
    if len(nonzero) == 0:
        return hist          # all-zero image — return as-is
    return hist[:, nonzero[0]: nonzero[-1] + 1]


def build_cost_grid(img_size: int = 28) -> np.ndarray:
    grid = np.array(
        [[j, i]
         for i in np.linspace(1, 0, num=img_size)
         for j in np.linspace(0, 1, num=img_size)],
        dtype=np.float64,
    )
    diff = grid[:, None, :] - grid[None, :, :]
    return np.sum(diff ** 2, axis=-1)


def get_hist(
    t: float,
    P_flat: np.ndarray,
    img_size: int = 28,
    batch_size: int = 50_000,
    num_estimation_iter: int = 20,
    rng: np.random.Generator = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)

    n_pixels = img_size * img_size
    grid = np.array(
        [[j, i]
         for i in np.linspace(1, 0, num=img_size)
         for j in np.linspace(0, 1, num=img_size)],
    )

    hist_acc = np.zeros((img_size, img_size), dtype=np.float64)
    for _ in range(num_estimation_iter):
        map_samples = rng.choice(
            len(P_flat), size=batch_size, p=P_flat)
        a_samples = grid[map_samples // n_pixels]
        b_samples = grid[map_samples  % n_pixels]
        proj = (1.0 - t) * a_samples + t * b_samples
        h, _, _ = np.histogram2d(
            proj[:, 1], proj[:, 0],
            bins=np.linspace(0.0, 1.0, num=img_size + 1),
        )
        hist_acc += h

    hist_acc = np.flipud(hist_acc)
    hist_acc /= max(hist_acc.sum(), 1e-12)

    thresh = np.quantile(hist_acc, 0.9)
    if thresh > 0:
        hist_acc = np.clip(hist_acc, 0, thresh)
    if hist_acc.max() > 0:
        hist_acc /= hist_acc.max()
    return hist_acc


def interp_plan(
    P: np.ndarray,
    num_interp: int = 8,
    img_size: int = 28,
    batch_size: int = 50_000,
    num_estimation_iter: int = 20,
) -> np.ndarray:
    P_flat = P.flatten()
    P_flat = np.clip(P_flat, 0.0, None)
    s = P_flat.sum()
    if s < 1e-12:
        P_flat = np.ones_like(P_flat) / len(P_flat)
    else:
        P_flat /= s

    rng = np.random.default_rng(0)
    ts = np.linspace(0.0, 1.0, num=num_interp)

    frames = []
    for t in ts:
        h = get_hist(t, P_flat, img_size, batch_size, num_estimation_iter, rng)
        h = crop_hist_col(h)
        frames.append(h)

    # Pad all frames to the same height × width before hstack
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    padded = []
    for f in frames:
        ph = max_h - f.shape[0]
        pw = max_w - f.shape[1]
        padded.append(np.pad(f, ((0, ph), (0, pw)), constant_values=0.0))

    return np.hstack(padded)   # (H, num_interp * W_frame)


def save_strip(strip: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, strip, cmap="Blues")
    print(f"    Saved → {path}")


def sinkhorn_gt(a, b, C, eps, n_iter=800):
    a_s = np.clip(a, 1e-10, None); a_s /= a_s.sum()
    b_s = np.clip(b, 1e-10, None); b_s /= b_s.sum()
    return ot.sinkhorn(a_s, b_s, C, reg=eps, numItermax=n_iter, stopThr=1e-9)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot MNIST interpolation strips in Meta OT paper format.")
    p.add_argument("--result_dir", type=str, required=True,
                   help="Path to M{N} dir, e.g. ./results/grayscale/M50")
    p.add_argument("--idx",        type=str, default="0",
                   help="Test pair index (int) or 'all'")
    p.add_argument("--num_interp", type=int, default=8,
                   help="Number of interpolation frames (default 8)")
    p.add_argument("--num_iter",   type=int, default=20,
                   help="Monte Carlo estimation iterations per frame (default 20)")
    p.add_argument("--batch_size", type=int, default=50_000,
                   help="Samples per Monte Carlo iteration (default 50000)")
    p.add_argument("--gpu",        type=str, default="0")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    img_size = 28
    eps      = 1e-2
    C        = build_cost_grid(img_size)

    test_pairs = load_pkl(os.path.join(args.result_dir, "test_pairs.pkl"))
    print(f"Loaded {len(test_pairs)} test pairs from {args.result_dir}")

    model_reg  = load_pkl(os.path.join(args.result_dir, "regression.pkl"))
    model_obj  = load_pkl(os.path.join(args.result_dir, "objective.pkl"))
    model_meta = load_pkl(os.path.join(args.result_dir, "meta_ot.pkl"))
    model_swgg = load_pkl(os.path.join(args.result_dir, "swgg.pkl"))
    model_stp  = load_pkl(os.path.join(args.result_dir, "min_stp.pkl"))

    mlp     = model_meta._eval_mlp
    lf_meta = model_meta._eval_lf
    mlp.eval()
    dev = next(mlp.parameters()).device

    def predict_reg(a, b):
        f, g = model_reg._predict_potentials(a, b, model_reg.alpha)
        return model_reg._potentials_to_plan(a, b, f, g)

    def predict_obj(a, b):
        f, g = model_obj._predict_potentials(a, b, model_obj.alpha)
        return model_obj._potentials_to_plan(a, b, f, g)

    def predict_meta(a, b):
        a_t = torch.tensor(a, dtype=torch.float64, device=dev).unsqueeze(0)
        b_t = torch.tensor(b, dtype=torch.float64, device=dev).unsqueeze(0)
        with torch.no_grad():
            f = mlp(a_t, b_t)
        return lf_meta.pred_transport(a_t, b_t, f)[0]

    def predict_gt(a, b):
        return sinkhorn_gt(a, b, C, eps)

    methods = [
        ("Sinkhorn_GT",    predict_gt),
        ("OT_Regression",  predict_reg),
        ("OT_Objective",   predict_obj),
        ("Meta_OT",        predict_meta),
        ("min_SWGG",       model_swgg.predict_plan),
        ("Min_STP",        model_stp.predict_plan),
    ]

    indices = range(len(test_pairs)) if args.idx == "all" else [int(args.idx)]

    for idx in indices:
        a, b = test_pairs[idx]
        pair_dir = os.path.join(args.result_dir, "plots", f"pair_{idx:02d}")
        os.makedirs(pair_dir, exist_ok=True)
        print(f"\n=== Pair {idx} ===")

        for name, predict_fn in methods:
            print(f"  [{name}] computing transport plan ...")
            try:
                P = predict_fn(a, b)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

            print(f"  [{name}] building interpolation strip ({args.num_interp} frames) ...")
            strip = interp_plan(
                P,
                num_interp=args.num_interp,
                img_size=img_size,
                batch_size=args.batch_size,
                num_estimation_iter=args.num_iter,
            )

            out_path = os.path.join(pair_dir, f"{name}.png")
            save_strip(strip, out_path)

    print(f"\nDone.  Output → {args.result_dir}/plots/")


if __name__ == "__main__":
    main()
