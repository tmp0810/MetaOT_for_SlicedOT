import argparse
import os
import pickle
import time
import numpy as np
from time import localtime, strftime

from cfg import init_cfg
from Data.color_transfer_data import get_color_transfer_dataloader
from Solvers.Meta_OT_Color import Meta_OT_Color


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Meta-OT Discrete (Sinkhorn dual) for color transfer")
    p.add_argument("--data_dir",      type=str, required=True)
    p.add_argument("--out_dir",       type=str, default="./runs/meta_ot_color")
    p.add_argument("--n_clusters",    type=int, default=None)
    p.add_argument("--num_train_iter",type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--enc_dim",       type=int, default=None)
    p.add_argument("--batch_size",    type=int, default=None)
    p.add_argument("--epsilon",       type=float, default=None)
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--gpu",           type=str, default="0")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    cfg_m = init_cfg("Meta_OT_Color")
    if args.n_clusters     is not None: cfg_m["n_clusters"]     = args.n_clusters
    if args.num_train_iter is not None: cfg_m["num_train_iter"] = args.num_train_iter
    if args.learning_rate  is not None: cfg_m["learning_rate"]  = args.learning_rate
    if args.enc_dim        is not None: cfg_m["enc_dim"]        = args.enc_dim
    if args.batch_size     is not None: cfg_m["batch_size"]     = args.batch_size
    if args.epsilon        is not None: cfg_m["epsilon"]        = args.epsilon
    cfg_m["gpu"] = int(args.gpu) if args.gpu.isdigit() else 0

    cfg_proj = argparse.Namespace(
        seed      = args.seed,
        flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime()),
        flag_load = None,
        solver    = "Meta_OT_Color",
        data_name = "color_transfer",
        gpu       = args.gpu,
    )

    n_clusters = cfg_m["n_clusters"]
    print(f"\n{'='*60}")
    print(f"  Meta-OT Color Transfer — DISCRETE (Sinkhorn dual)")
    print(f"  Architecture: DeepSets → f, g = 1 Sinkhorn step from f")
    print(f"  Loss: -Sinkhorn dual objective (faithful to JAX train.py)")
    print(f"{'='*60}")
    print(f"  data_dir       : {args.data_dir}")
    print(f"  n_clusters     : {n_clusters}")
    print(f"  enc_dim        : {cfg_m['enc_dim']}")
    print(f"  head_hidden    : {cfg_m['head_hidden']}")
    print(f"  num_train_iter : {cfg_m['num_train_iter']}")
    print(f"  batch_size     : {cfg_m['batch_size']}")
    print(f"  learning_rate  : {cfg_m['learning_rate']}")
    print(f"  epsilon        : {cfg_m['epsilon']}")
    print(f"{'='*60}\n")

    # Dataloader — same as OT_Regression_Sliced_Color
    print("Quantizing images (cached after first run) ...")
    train_loader = get_color_transfer_dataloader(
        image_dir   = args.data_dir,
        n_clusters  = n_clusters,
        batch_size  = cfg_m["batch_size"],
        seed        = args.seed,
        num_workers = 0,
    )
    print(f"  Dataset: {len(train_loader.dataset)} pairs")

    model = Meta_OT_Color(cfg_proj=cfg_proj, cfg_m=cfg_m)

    t0 = time.perf_counter()
    model.train(train_loader)
    t_train = time.perf_counter() - t0
    print(f"\nTraining: {t_train:.1f}s  ({t_train/3600:.2f}h)")

    model_path = os.path.join(args.out_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model → {model_path}")

    # Sanity check
    print("\nSanity check: predict_plan on one test pair ...")
    ds = train_loader.dataset
    src_w, src_c, tgt_w, tgt_c = ds[0]
    a  = src_w.numpy(); sc = src_c.numpy()
    b  = tgt_w.numpy(); tc = tgt_c.numpy()
    t0 = time.perf_counter()
    P  = model.predict_plan(a, b, sc, tc)
    print(f"  Plan: {P.shape}  sum={P.sum():.4f}  "
          f"marginal_a_err={np.abs(P.sum(1)-a).max():.6f}  "
          f"marginal_b_err={np.abs(P.sum(0)-b).max():.6f}  "
          f"time={time.perf_counter()-t0:.3f}s")
    print("\nDone. Run eval_color_transfer.py to evaluate.")


if __name__ == "__main__":
    main()
