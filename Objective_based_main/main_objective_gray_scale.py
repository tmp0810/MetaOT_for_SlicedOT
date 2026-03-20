import os
import argparse
from Data.pre_data import pre_data
from Solvers.Objective_SlicedOT.OT_Objective_Sliced import OT_Objective_Sliced
from time import localtime, strftime
from cfg import init_cfg


def main(cfg_proj, cfg_m):
    model = OT_Objective_Sliced(cfg_proj, cfg_m)
    [dataloader_train, dataloader_valid, dataloader_test], _ = pre_data(
        cfg_proj.data_name, cfg_proj, cfg_m)
    model.train(dataloader_train, dataloader_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Method 2: Objective-based Sliced OT for MNIST")
    parser.add_argument("--gpu",          type=str,   default="0",  required=False)
    parser.add_argument("--seed",         type=int,   default=1,    required=False)
    parser.add_argument("--data_name",    type=str,   default="MNIST", required=False)
    parser.add_argument("--num_bootstrap",type=int,   default=None, required=False,
                        help="M: number of training pairs (default: cfg=50)")
    parser.add_argument("--num_proj",     type=int,   default=None, required=False,
                        help="L: number of projection directions (default: cfg=100)")
    parser.add_argument("--learning_rate",type=float, default=None, required=False)
    parser.add_argument("--epsilon",      type=float, default=None, required=False)
    parser.add_argument("--flag_time",    type=str,
                        default=strftime("%Y-%m-%d_%H-%M-%S", localtime()),
                        required=False)
    parser.add_argument("--flag_load",    type=str,   default=None, required=False)
    cfg_proj = parser.parse_args()
    cfg_proj.solver = "OT_Objective_Sliced"

    cfg_m = init_cfg("OT_Objective_Sliced")
    if cfg_proj.num_bootstrap is not None: cfg_m["num_bootstrap"]  = cfg_proj.num_bootstrap
    if cfg_proj.num_proj      is not None: cfg_m["num_projections"]= cfg_proj.num_proj
    if cfg_proj.learning_rate is not None: cfg_m["learning_rate"]  = cfg_proj.learning_rate
    if cfg_proj.epsilon       is not None: cfg_m["epsilon"]        = cfg_proj.epsilon

    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % cfg_proj.gpu

    print(f"\n{'='*60}")
    print(f"  Method 2: Objective-based Amortized Sliced OT")
    print(f"  Model: f = Φ_f(a,b) @ α  (α ∈ ℝ^L, global)")
    print(f"  Loss:  -E[dual_obj(Φ_f@α; a,b,c)]  — no GT Sinkhorn")
    print(f"{'='*60}")
    print(f"  data_name     : {cfg_proj.data_name}")
    print(f"  num_bootstrap : {cfg_m['num_bootstrap']}  (M training pairs)")
    print(f"  num_proj      : {cfg_m['num_projections']}  (L sliced directions)")
    print(f"  learning_rate : {cfg_m['learning_rate']}")
    print(f"  epsilon       : {cfg_m['epsilon']}")
    print(f"{'='*60}\n")

    main(cfg_proj, cfg_m)