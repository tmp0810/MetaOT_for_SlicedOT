import os
import argparse
from Data.pre_data import pre_data
from Solvers.SWGG.min_SWGG_GrayScale import min_SWGG_GrayScale
from time import localtime, strftime
from cfg import init_cfg


def main(cfg_proj, cfg_m):
    model = min_SWGG_GrayScale(cfg_proj, cfg_m)
    [dataloader_train, dataloader_valid, dataloader_test], _ = pre_data(
        cfg_proj.data_name, cfg_proj, cfg_m)
    model.train(dataloader_train, dataloader_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="min-SWGG baseline for MNIST gray scale OT")
    parser.add_argument("--gpu",          type=str,  default="0",  required=False)
    parser.add_argument("--seed",         type=int,  default=1,    required=False)
    parser.add_argument("--data_name",    type=str,  default="MNIST", required=False)
    parser.add_argument("--n_projections",type=int,  default=None, required=False,
                        help="Number of random 1-D directions (default: cfg value=200)")
    parser.add_argument("--flag_time",    type=str,
                        default=strftime("%Y-%m-%d_%H-%M-%S", localtime()),
                        required=False)
    parser.add_argument("--flag_load",    type=str,  default=None, required=False)
    cfg_proj = parser.parse_args()
    cfg_proj.solver = "min_SWGG_GrayScale"

    cfg_m = init_cfg("min_SWGG_GrayScale")
    if cfg_proj.n_projections is not None:
        cfg_m["n_projections"] = cfg_proj.n_projections

    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % cfg_proj.gpu

    print(f"\n{'='*55}")
    print(f"  min-SWGG baseline — MNIST Gray Scale")
    print(f"  No training: test-time θ* search only")
    print(f"{'='*55}")
    print(f"  data_name     : {cfg_proj.data_name}")
    print(f"  n_projections : {cfg_m['n_projections']}")
    print(f"  epsilon       : {cfg_m['epsilon']}  (for Sinkhorn comparison)")
    print(f"{'='*55}\n")

    main(cfg_proj, cfg_m)