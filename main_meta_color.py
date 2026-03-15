import argparse
import os
import pickle
from time import localtime, strftime

from cfg import init_cfg
from Data.color_transfer_data import get_color_transfer_dataloader
from Solvers.Meta_OT_Color import Meta_OT_Color


def parse_args():
    p = argparse.ArgumentParser(description='Train Meta-OT for color transfer')
    p.add_argument('--data_dir',       type=str, required=True)
    p.add_argument('--out_dir',        type=str, default='./runs/meta_ot_color')
    p.add_argument('--n_clusters',     type=int, default=None)
    p.add_argument('--num_train_iter', type=int, default=None)
    p.add_argument('--pretrain_iter',  type=int, default=None)
    p.add_argument('--batch_size',     type=int, default=None)
    p.add_argument('--learning_rate',  type=float, default=None)
    p.add_argument('--seed',           type=int, default=0)
    p.add_argument('--gpu',            type=str, default='0')
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    cfg_m = init_cfg("Meta_OT_Color")
    if args.n_clusters     is not None: cfg_m.n_clusters     = args.n_clusters
    if args.num_train_iter is not None: cfg_m.num_train_iter = args.num_train_iter
    if args.pretrain_iter  is not None: cfg_m.pretrain_iter  = args.pretrain_iter
    if args.batch_size     is not None: cfg_m.batch_size     = args.batch_size
    if args.learning_rate  is not None: cfg_m.learning_rate  = args.learning_rate

    cfg_proj = argparse.Namespace(
        seed      = args.seed,
        flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime()),
        flag_load = None,
        solver    = "Meta_OT_Color",
        data_name = "color_transfer",
        gpu       = args.gpu,
    )

    print(f"\n{'='*55}")
    print(f"  Meta-OT Baseline — Color Transfer")
    print(f"{'='*55}")
    print(f"  data_dir      : {args.data_dir}")
    print(f"  n_clusters    : {cfg_m.n_clusters}")
    print(f"  num_train_iter: {cfg_m.num_train_iter}")
    print(f"  pretrain_iter : {cfg_m.pretrain_iter}")
    print(f"  batch_size    : {cfg_m.batch_size}")
    print(f"  icnn_hidden   : {cfg_m.icnn_hidden_dim} × {cfg_m.icnn_hidden_num}")
    print(f"  enc_dim       : {cfg_m.enc_dim}")
    print(f"  epsilon (eval): {cfg_m.epsilon}")
    print(f"{'='*55}\n")

    # Build dataloader (infinite-style, shuffle=True)
    train_loader = get_color_transfer_dataloader(
        args.data_dir,
        n_clusters = cfg_m.n_clusters,
        batch_size = cfg_m.batch_size,
        seed       = args.seed,
    )
    n_images = len(train_loader.dataset.image_paths)
    print(f"  {n_images} images  → {len(train_loader.dataset)} ordered pairs\n")

    model = Meta_OT_Color(cfg_proj, cfg_m)
    model.train(train_loader)

    # Save
    model_path = os.path.join(args.out_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved -> {model_path}")
    print("Run eval_color_transfer.py with --model_path to visualise results.")


if __name__ == '__main__':
    main()
