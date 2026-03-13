import argparse
import os
import pickle
import numpy as np
from time import localtime, strftime

from cfg import init_cfg
from Data.color_transfer_data import get_color_transfer_dataloader
from Solvers.OT_Regression_Sliced_Color import OT_Regression_Sliced_Color


def parse_args():
    p = argparse.ArgumentParser(description='Train OT Regression Sliced for color transfer')
    p.add_argument('--data_dir',      type=str, required=True,
                   help='Folder of painting images (JPG/PNG)')
    p.add_argument('--out_dir',       type=str, default='./runs/color_transfer')
    p.add_argument('--n_clusters',    type=int, default=None,
                   help='KMeans clusters per image (default: cfg value = 500)')
    p.add_argument('--num_bootstrap', type=int, default=None,
                   help='Override cfg_m.num_bootstrap (number of training pairs)')
    p.add_argument('--num_proj',      type=int, default=None,
                   help='Override cfg_m.num_projections')
    p.add_argument('--ridge',         type=float, default=None,
                   help='Override cfg_m.ridge')
    p.add_argument('--epsilon',       type=float, default=None,
                   help='Override cfg_m.epsilon (OT regularisation)')
    p.add_argument('--seed',          type=int, default=0)
    p.add_argument('--gpu',           type=str, default='0')
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    cfg_m = init_cfg("OT_Regression_Sliced_Color")
    if args.n_clusters    is not None: cfg_m.n_clusters    = args.n_clusters
    if args.num_bootstrap is not None: cfg_m.num_bootstrap = args.num_bootstrap
    if args.num_proj      is not None: cfg_m.num_projections = args.num_proj
    if args.ridge         is not None: cfg_m.ridge         = args.ridge
    if args.epsilon       is not None: cfg_m.epsilon       = args.epsilon

    cfg_proj = argparse.Namespace(
        seed      = args.seed,
        flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime()),
        flag_load = None,
        solver    = "OT_Regression_Sliced_Color",
        data_name = "color_transfer",
        gpu       = args.gpu,
    )

    print(f"\n{'='*55}")
    print(f"  OT Regression Sliced — Color Transfer")
    print(f"{'='*55}")
    print(f"  data_dir     : {args.data_dir}")
    print(f"  out_dir      : {args.out_dir}")
    print(f"  n_clusters   : {cfg_m.n_clusters}")
    print(f"  num_bootstrap: {cfg_m.num_bootstrap}")
    print(f"  num_proj     : {cfg_m.num_projections}")
    print(f"  epsilon      : {cfg_m.epsilon}")
    print(f"  ridge        : {cfg_m.ridge}")
    print(f"{'='*55}\n")

    print(f"Loading images from: {args.data_dir}")
    train_loader = get_color_transfer_dataloader(
        args.data_dir,
        n_clusters  = cfg_m.n_clusters,
        batch_size  = cfg_m.batch_size,
        seed        = args.seed,
        # Request more pairs than num_bootstrap so we have buffer for any skips
        max_pairs   = cfg_m.num_bootstrap * 4,
    )
    n_images = len(train_loader.dataset.image_paths)
    n_pairs  = len(train_loader.dataset)
    print(f"  {n_images} images  →  {n_pairs} ordered pairs available "
          f"(using {cfg_m.num_bootstrap})\n")

    model = OT_Regression_Sliced_Color(cfg_proj, cfg_m)

    print(f"Fitting ridge regression on M={cfg_m.num_bootstrap} pairs ...")
    model.train(train_loader)

    model_path = os.path.join(args.out_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved -> {model_path}")
    print(f"alpha/beta    -> {model.log_sub_folder}/")

    print("\nSanity check: predict plan for one test pair ...")
    dataset = train_loader.dataset
    rng     = np.random.default_rng(args.seed + 999)
    i, j    = rng.choice(len(dataset.image_paths), size=2, replace=False)
    sw, sc  = dataset._cache[dataset.image_paths[i]]
    tw, tc  = dataset._cache[dataset.image_paths[j]]

    P = model.predict_plan(sw, tw, sc, tc)
    print(f"  Plan: shape={P.shape}  sum={P.sum():.4f}  "
          f"max={P.max():.6f}  nonzero={(P > 1e-10).sum()}")
    print("\nDone.  Run eval_color_transfer.py to visualise results.")


if __name__ == '__main__':
    main()
