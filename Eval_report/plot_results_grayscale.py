import argparse
import os
import pickle
import numpy as np
import torch

from Solvers.Regression_SlicedOT.OT_Regression_Sliced import OT_Regression_Sliced
from Solvers.Meta_OT.Meta_OT_gray_scale import dual_obj_loss
from Models.ot_models import PotentialMLP
from Utils import utils


def build_cost_grid(img_size=28):
    grid = np.array([[j, i]
                     for i in np.linspace(1, 0, num=img_size)
                     for j in np.linspace(0, 1, num=img_size)], dtype=np.float64)
    diff = grid[:, None, :] - grid[None, :, :]
    return np.sum(diff ** 2, axis=-1)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_pair(idx, a, b, methods, out_dir, img_size=28, eps=1e-2):
    """Plot interpolation for one test pair across all methods + Sinkhorn GT."""
    import ot
    pair_dir = os.path.join(out_dir, "plots", f"pair_{idx:02d}")
    os.makedirs(pair_dir, exist_ok=True)

    C = build_cost_grid(img_size)

    # Sinkhorn ground truth first
    a_s = np.clip(a, 1e-10, None); a_s /= a_s.sum()
    b_s = np.clip(b, 1e-10, None); b_s /= b_s.sum()
    P_gt  = ot.sinkhorn(a_s, b_s, C, reg=eps, numItermax=800, stopThr=1e-9)
    imgs_gt = OT_Regression_Sliced.interp(
        P_gt, num_inter=11, batch_size=50_000, img_size=img_size)
    utils.save_r(imgs_gt, torch.tensor(a), torch.tensor(b),
                 path=pair_dir, title="Sinkhorn_GT")
    print(f"  [{idx}] Sinkhorn_GT → {pair_dir}/")

    for name, predict_fn in methods:
        P = predict_fn(a, b)
        imgs = OT_Regression_Sliced.interp(
            P, num_inter=11, batch_size=50_000, img_size=img_size)
        utils.save_r(imgs, torch.tensor(a), torch.tensor(b),
                     path=pair_dir, title=name)
        print(f"  [{idx}] {name} → {pair_dir}/{name}_*.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", type=str, required=True,
                   help="Path to M{N} dir, e.g. ./results/grayscale/M50")
    p.add_argument("--idx",        type=str, default="0",
                   help="Test pair index to plot, or 'all'")
    p.add_argument("--gpu",        type=str, default="0")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 28
    eps = 1e-2

    # ── Load test pairs ───────────────────────────────────────────────────
    test_pairs = load_pkl(os.path.join(args.result_dir, "test_pairs.pkl"))
    print(f"Loaded {len(test_pairs)} test pairs from {args.result_dir}")

    # ── Load models ───────────────────────────────────────────────────────
    model_reg  = load_pkl(os.path.join(args.result_dir, "regression.pkl"))
    model_obj  = load_pkl(os.path.join(args.result_dir, "objective.pkl"))
    model_meta = load_pkl(os.path.join(args.result_dir, "meta_ot.pkl"))
    model_swgg = load_pkl(os.path.join(args.result_dir, "swgg.pkl"))
    model_stp  = load_pkl(os.path.join(args.result_dir, "min_stp.pkl"))

    # # Rebuild Meta OT inference fn (MLP stored separately in log dir)
    # mlp = PotentialMLP(dim_in=img_size**2*2, dim_out=img_size**2,
    #                    hidden_num=model_meta.cfg_m.MLP_hidden_num).to(dev)
    # mlp, _, _, _ = model_meta.load_ckp(mlp, None, None, "OT_D-train")
    # mlp.eval()
    # lf = dual_obj_loss(img_size=img_size, epsilon=model_meta.cfg_m.epsilon, device=dev)

    # Đoạn mới: Lấy thẳng MLP đã được lưu kèm trong file pickle
    mlp = model_meta._eval_mlp
    mlp.eval()
    lf_meta = model_meta._eval_lf
    dev = next(mlp.parameters()).device # Lấy device hiện tại của model

    # ── Define predict functions ──────────────────────────────────────────
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

        methods = [
        ("OT_Regression", predict_reg),
        ("OT_Objective",  predict_obj),
        ("Meta_OT",       predict_meta),
        ("min_SWGG",      model_swgg.predict_plan),
        ("Min_STP",       model_stp.predict_plan),
    ]

    # ── Plot ──────────────────────────────────────────────────────────────
    indices = range(len(test_pairs)) if args.idx == "all" else [int(args.idx)]
    for idx in indices:
        a, b = test_pairs[idx]
        print(f"\nPlotting pair {idx} ...")
        plot_pair(idx, a, b, methods, args.result_dir)

    print(f"\nDone. Output → {args.result_dir}/plots/")


if __name__ == "__main__":
    main()
