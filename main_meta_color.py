import argparse
import os
import pickle
import time
import numpy as np
from time import localtime, strftime

from cfg import init_cfg
from Data.color_meta_data import get_image_paths
from Solvers.Meta_OT_Color import Meta_OT_Color
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Meta-OT (W2GN dual + cycle) for color transfer")
    p.add_argument("--data_dir",          type=str, required=True)
    p.add_argument("--out_dir",           type=str, default="./runs/meta_ot_color")
    p.add_argument("--num_train_iter",    type=int, default=None)
    p.add_argument("--lr",                type=float, default=None)
    p.add_argument("--meta_batch_size",   type=int, default=None)
    p.add_argument("--inner_batch_size",  type=int, default=None)
    p.add_argument("--num_pretrain_iter", type=int, default=None)
    p.add_argument("--cycle_loss_weight", type=float, default=None)
    p.add_argument("--num_rgb_sample",    type=int, default=None)
    p.add_argument("--seed",              type=int, default=0)
    p.add_argument("--gpu",              type=str, default="0")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    cfg_m = init_cfg("Meta_OT_Color")
    if args.num_train_iter    is not None: cfg_m["num_train_iter"]    = args.num_train_iter
    if args.lr                is not None: cfg_m["lr"]                = args.lr
    if args.meta_batch_size   is not None: cfg_m["meta_batch_size"]   = args.meta_batch_size
    if args.inner_batch_size  is not None: cfg_m["inner_batch_size"]  = args.inner_batch_size
    if args.num_pretrain_iter is not None: cfg_m["num_pretrain_iter"] = args.num_pretrain_iter
    if args.cycle_loss_weight is not None: cfg_m["cycle_loss_weight"] = args.cycle_loss_weight
    if args.num_rgb_sample    is not None: cfg_m["num_rgb_sample"]    = args.num_rgb_sample
    cfg_m["gpu"] = int(args.gpu) if args.gpu.isdigit() else 0

    cfg_proj = argparse.Namespace(
        seed      = args.seed,
        flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime()),
        flag_load = None,
        solver    = "Meta_OT_Color",
        data_name = "color_transfer",
        gpu       = args.gpu,
    )

    image_paths = get_image_paths(args.data_dir)
    print(f"\n{'='*55}")
    print(f"  Meta-OT Color Transfer (JAX port)")
    print(f"  Architecture: ICNN dim_hidden={cfg_m['dim_hidden']}  ({653} params)")
    print(f"  MetaICNN: ResNet18 x2 → MLP → D_flat, Dc_flat")
    print(f"  Loss: W2GN dual + cycle (λ={cfg_m['cycle_loss_weight']})")
    print(f"{'='*55}")
    print(f"  data_dir        : {args.data_dir}  ({len(image_paths)} images)")
    print(f"  num_train_iter  : {cfg_m['num_train_iter']}")
    print(f"  num_pretrain    : {cfg_m['num_pretrain_iter']}")
    print(f"  lr              : {cfg_m['lr']}")
    print(f"  meta_batch_size : {cfg_m['meta_batch_size']}")
    print(f"  inner_batch_size: {cfg_m['inner_batch_size']}")
    print(f"{'='*55}\n")
    assert len(image_paths) >= 2

    model = Meta_OT_Color(cfg_proj=cfg_proj, cfg_m=cfg_m)

    t0 = time.perf_counter()
    model.train(args.data_dir)
    t_train = time.perf_counter() - t0
    print(f"\nTraining: {t_train:.1f}s  ({t_train/3600:.2f}h)")

    model_path = os.path.join(args.out_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model → {model_path}")

    # Sanity check
    if len(image_paths) >= 2:
        print("\nSanity check: apply_map ...")
        src = np.array(Image.open(image_paths[0]).convert("RGB"))
        tgt = np.array(Image.open(image_paths[1]).convert("RGB"))
        t0  = time.perf_counter()
        out = model.apply_map(src, tgt)
        print(f"  {src.shape} → {out.shape}  time={time.perf_counter()-t0:.3f}s")

    print("\nDone. Run eval_color_transfer.py to evaluate.")


if __name__ == "__main__":
    main()
