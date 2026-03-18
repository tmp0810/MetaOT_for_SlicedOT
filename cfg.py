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

    elif n_solver == "Meta_OT_Color":
        cfg_m.insert("dim_hidden",          [128])  # single hidden layer of 128 units
        cfg_m.insert("quad_rank",              3)   # quadratic term rank
        cfg_m.insert("bottleneck_size",      512)   # ResNet18 output × 2 → concat
        cfg_m.insert("fc_num_hidden_units",  512)
        cfg_m.insert("fc_num_hidden_layers",   2)
        cfg_m.insert("num_train_iter",     50000)
        cfg_m.insert("lr",                  1e-4)   # main training LR
        cfg_m.insert("meta_batch_size",        4)   # B pairs per gradient step
        cfg_m.insert("inner_batch_size",     256)   # N pixels per pair
        cfg_m.insert("cycle_loss_weight",    0.1)   # λ_cyc (JAX default)
        cfg_m.insert("l2_penalty",          1e-5)   # λ_l2
        cfg_m.insert("max_grad_norm",        1.0)
        cfg_m.insert("num_pretrain_iter",  10000)
        cfg_m.insert("pretrain_lr",         1e-3)
        cfg_m.insert("pretrain_loss_threshold", 1e-3)
        cfg_m.insert("num_rgb_sample",      None)   # None = use all pixels
        cfg_m.insert("epsilon",              0.5)
        cfg_m.insert("n_clusters",           500)   # for discrete Sinkhorn baseline
        cfg_m.insert("log_interval",        1000)
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)
 
    elif n_solver == "Meta_OT_World":
        cfg_m.insert("icnn_hidden_dim",   32)
        cfg_m.insert("icnn_hidden_num",   2)
        cfg_m.insert("enc_dim",           128)
        cfg_m.insert("meta_hidden_dim",   256)
        cfg_m.insert("num_train_iter",    5000)
        cfg_m.insert("pretrain_iter",     500)
        cfg_m.insert("learning_rate",     1e-3)
        cfg_m.insert("cycle_loss_weight", 10.0)
        cfg_m.insert("n_inner_samples",   256)
        cfg_m.insert("batch_size",        1)
        cfg_m.insert("n_supply",          100)
        cfg_m.insert("n_demand",          10_000)
        cfg_m.insert("supply_bernoulli_p",0.5)
        cfg_m.insert("epsilon",           0.5)
        cfg_m.insert("sinkhorn_iters",    0)
        cfg_m.insert("log_interval",      200)
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    elif n_solver == "OT_Regression_Sliced":
        cfg_m.insert("num_bootstrap", 50)
        cfg_m.insert("num_projections", 100)
        cfg_m.insert("ridge", 1e-3)
        cfg_m.insert("img_size", 28)
        cfg_m.insert("epsilon", 0.1)
        cfg_m.insert("batch_size", 64)
        cfg_m.insert("valid_rate", 0.0)
        cfg_m.insert("log_interval", 1)
        cfg_m.insert("sinkhorn_iters", 800)
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    elif n_solver == "OT_Regression_Sliced_World":
        cfg_m.insert("num_bootstrap", 50)       # M: training pairs
        cfg_m.insert("num_projections", 100)     # L: stereographic 1-D directions
        cfg_m.insert("ridge", 0.05)
        cfg_m.insert("n_supply", 100)            # number of supply locations
        cfg_m.insert("n_demand", 10_000)         # number of demand locations
        cfg_m.insert("supply_bernoulli_p", 0.5)  # sparsity of supply weights
        cfg_m.insert("epsilon", 0.5)
        cfg_m.insert("sinkhorn_iters", 800)
        cfg_m.insert("batch_size", 1)            # 1 pair per batch (large n_demand)
        cfg_m.insert("valid_rate", 0.0)
        cfg_m.insert("log_interval", 1)
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    elif n_solver == "OT_Regression_Sliced_Color":
        cfg_m.insert("num_bootstrap", 50)       # M: training pairs
        cfg_m.insert("num_projections", 100)     # L: 1-D projection dirs in R^3
        cfg_m.insert("ridge", 1e-3)
        cfg_m.insert("n_clusters", 500)          # KMeans clusters per image
        cfg_m.insert("img_size", 0)              # unused, kept for interface compat
        cfg_m.insert("epsilon", 0.005)
        cfg_m.insert("sinkhorn_iters", 800)
        cfg_m.insert("batch_size", 1)            # 1 pair per batch
        cfg_m.insert("valid_rate", 0.0)
        cfg_m.insert("log_interval", 1)
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    elif n_solver == "Meta_OT_World":
        cfg_m.insert("n_hidden",         512)   # hidden units per layer
        cfg_m.insert("n_hidden_layer",     3)   # number of hidden layers
        cfg_m.insert("num_train_iter",  5000)
        cfg_m.insert("learning_rate",   1e-3)
        cfg_m.insert("max_grad_norm",    1.0)
        cfg_m.insert("batch_size",         8)   # batch of pairs per step
        cfg_m.insert("n_supply",         100)
        cfg_m.insert("n_demand",       10_000)
        cfg_m.insert("supply_bernoulli_p", 0.5)
        cfg_m.insert("epsilon",          0.5)   # safe for arccos in [0,pi]
        cfg_m.insert("log_interval",     100)
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    return cfg_m


