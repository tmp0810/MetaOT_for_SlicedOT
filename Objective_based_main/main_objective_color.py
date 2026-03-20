import argparse
import os
import pickle
import time
import numpy as np
from time import localtime, strftime

from cfg import init_cfg
from Data.color_transfer_data import get_color_transfer_dataloader
from Solvers.Objective_SlicedOT.OT_Objective_Sliced_Color import OT_Objective_Sliced_Color


def parse_args():
    p = argparse.ArgumentParser(
        description="Method 2: Objective-based Sliced OT for Color Transfer")
    p.add_argument("--data_dir",       type=str,   required=True)
    p.add_argument("--out_dir",        type=str,   default="./runs/objective_color")
    p.add_argument("--n_clusters",     type=int,   default=None)
    p.add_argument("--num_bootstrap",  type=int,   default=None)
    p.add_argument("--num_train_iter", type=int,   default=None)
    p.add_argument("--num_proj",       type=int,   default=None)
    p.add_argument("--learning_rate",  type=float, default=None)
    p.add_argument("--epsilon",        type=float, default=None)
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--gpu",            type=str,   default="0")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    cfg_m = init_cfg("OT_Objective_Sliced_Color")
    if args.n_clusters     is not None: cfg_m["n_clusters"]     = args.n_clusters
    if args.num_bootstrap  is not None: cfg_m["num_bootstrap"]  = args.num_bootstrap
    if args.num_train_iter is not None: cfg_m["num_train_iter"] = args.num_train_iter
    if args.num_proj       is not None: cfg_m["num_projections"]= args.num_proj
    if args.learning_rate  is not None: cfg_m["learning_rate"]  = args.learning_rate
    if args.epsilon        is not None: cfg_m["epsilon"]        = args.epsilon
    cfg_m["gpu"] = int(args.gpu) if args.gpu.isdigit() else 0

    cfg_proj = argparse.Namespace(
        seed      = args.seed,
        flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime()),
        flag_load = None,
        solver    = "OT_Objective_Sliced_Color",
        data_name = "color_transfer",
        gpu       = args.gpu,
    )

    print(f"\n{'='*60}")
    print(f"  Method 2: Objective-based Amortized Sliced OT — Color")
    print(f"  Model: f = Φ_f(a,b,src_c,tgt_c) @ α  (α ∈ ℝ^L, global)")
    print(f"  Loss:  -E[dual_obj(Φ_f@α; a,b,log_K)]  — no GT Sinkhorn")
    print(f"  Note:  log_K per-pair (centroids change each pair)")
    print(f"{'='*60}")
    print(f"  data_dir       : {args.data_dir}")
    print(f"  n_clusters     : {cfg_m['n_clusters']}")
    print(f"  num_bootstrap  : {cfg_m['num_bootstrap']}  (M pair pool)")
    print(f"  num_train_iter : {cfg_m['num_train_iter']}  (T gradient steps)")
    print(f"  num_proj       : {cfg_m['num_projections']}  (L directions)")
    print(f"  learning_rate  : {cfg_m['learning_rate']}")
    print(f"  epsilon        : {cfg_m['epsilon']}")
    print(f"{'='*60}\n")

    print("Quantizing images (cached after first run) ...")
    train_loader = get_color_transfer_dataloader(
        image_dir   = args.data_dir,
        n_clusters  = cfg_m["n_clusters"],
        batch_size  = cfg_m["batch_size"],
        seed        = args.seed,
        num_workers = 0,
    )
    print(f"  Dataset: {len(train_loader.dataset)} pairs")

    model = OT_Objective_Sliced_Color(cfg_proj=cfg_proj, cfg_m=cfg_m)

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
          f"time={time.perf_counter()-t0:.3f}s")
    print("\nDone. Run eval_color_transfer.py to evaluate.")
    print(f"  python eval_color_transfer.py --model_path {model_path} \\")
    print(f"      --data_dir {args.data_dir} --num_samples 10")


if __name__ == "__main__":
    main()
