import os
from Utils.utils import dotdict

def init_cfg(n_solver):
    cfg_m = dotdict()
    cfg_m.insert("Note", None)
    cfg_m.insert("datasets_root", '../datasets')

    if n_solver == "OT_Discrete":
        cfg_m.insert("epochs", 10)
        cfg_m.insert("learning_rate_init", 1e-3)
        cfg_m.insert("log_interval", 100)
        cfg_m.insert("batch_size", 1024)
        cfg_m.insert("valid_rate", 0.0)
        cfg_m.insert("img_size", 28)
        cfg_m.insert("epsilon", 1e-2)
        cfg_m.insert("MLP_hidden_num", 3)

    elif n_solver == "OT_Regression_Sliced":
        cfg_m.insert("num_bootstrap", 10)
        cfg_m.insert("num_projections", 100)
        cfg_m.insert("ridge", 1e-3)
        cfg_m.insert("img_size", 28)
        cfg_m.insert("epsilon", 0.1)
        cfg_m.insert("batch_size", 64)
        cfg_m.insert("valid_rate", 0.0)
        cfg_m.insert("log_interval", 1)
        cfg_m.insert("sinkhorn_iters", 500)
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    elif n_solver == "OT_Regression_Sliced_World":
        cfg_m.insert("num_bootstrap", 50)       # M: training pairs
        cfg_m.insert("num_projections", 500)     # L: stereographic 1-D directions
        cfg_m.insert("ridge", 0.05)
        cfg_m.insert("n_supply", 100)            # number of supply locations
        cfg_m.insert("n_demand", 10_000)         # number of demand locations
        cfg_m.insert("supply_bernoulli_p", 0.5)  # sparsity of supply weights
        cfg_m.insert("epsilon", 0.5)
        cfg_m.insert("sinkhorn_iters", 500)
        cfg_m.insert("batch_size", 1)            # 1 pair per batch (large n_demand)
        cfg_m.insert("valid_rate", 0.0)
        cfg_m.insert("log_interval", 1)
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    elif n_solver == "OT_Regression_Sliced_Color":
        cfg_m.insert("num_bootstrap", 100)       # M: training pairs
        cfg_m.insert("num_projections", 200)     # L: 1-D projection dirs in R^3
        cfg_m.insert("ridge", 1e-3)
        cfg_m.insert("n_clusters", 500)          # KMeans clusters per image
        cfg_m.insert("img_size", 0)              # unused, kept for interface compat
        cfg_m.insert("epsilon", 0.5)
        cfg_m.insert("sinkhorn_iters", 500)
        cfg_m.insert("batch_size", 1)            # 1 pair per batch
        cfg_m.insert("valid_rate", 0.0)
        cfg_m.insert("log_interval", 1)
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    return cfg_m


