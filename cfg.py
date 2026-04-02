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
        cfg_m.insert("epsilon", 0.1)
        cfg_m.insert("MLP_hidden_num", 3)

    elif n_solver == "Meta_OT_Color":
        cfg_m.insert("n_clusters",    500)   # KMeans clusters per image
        cfg_m.insert("enc_dim",       256)   # DeepSets encoder output dim
        cfg_m.insert("head_hidden",   512)   # f_head hidden dim
        cfg_m.insert("num_train_iter", 5000)
        cfg_m.insert("learning_rate",  1e-3)
        cfg_m.insert("max_grad_norm",  1.0)
        cfg_m.insert("batch_size",     4)    # pairs per gradient step
        cfg_m.insert("log_interval",   5000) # ban đầu là 100
        cfg_m.insert("epsilon",        0.005)
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
        cfg_m.insert("sinkhorn_iters", 1500)
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
        cfg_m.insert("sinkhorn_iters", 1500)
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
        cfg_m.insert("sinkhorn_iters", 1500)
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
        cfg_m.insert("log_interval",     5000)
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    elif n_solver == "min_SWGG_GrayScale":
        cfg_m.insert("img_size",       28)
        cfg_m.insert("n_projections", 200)   
        cfg_m.insert("epsilon",        0.1)   
        cfg_m.insert("batch_size",      64)
        cfg_m.insert("valid_rate",     0.0)
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    elif n_solver == "min_SWGG_World":
        cfg_m.insert("n_projections", 200)   
        cfg_m.insert("n_supply",      100)
        cfg_m.insert("n_demand",    10_000)
        cfg_m.insert("supply_bernoulli_p", 0.5)
        cfg_m.insert("epsilon",       0.5)  
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    elif n_solver == "min_SWGG_Color":
        cfg_m.insert("n_projections", 200)   
        cfg_m.insert("n_clusters",    500)    
        cfg_m.insert("epsilon",       0.005)    
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    elif n_solver == "OT_Objective_Sliced":
        cfg_m.insert("num_bootstrap",    50)    
        cfg_m.insert("num_train_iter", 5000)   
        cfg_m.insert("num_projections", 100) 
        cfg_m.insert("img_size",         28)
        cfg_m.insert("epsilon",        0.1)    # same as Meta OT paper MNIST
        cfg_m.insert("learning_rate",  1e-3)    # Adam lr for α
        cfg_m.insert("max_grad_norm",   1.0)   
        cfg_m.insert("batch_size",       64)    # dataloader batch size
        cfg_m.insert("valid_rate",       0.0)
        cfg_m.insert("log_interval",   5000)
        cfg_m.insert("sinkhorn_iters", 1500)     # for _evaluate GT comparison only
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    elif n_solver == "OT_Objective_Sliced_World":
        cfg_m.insert("num_bootstrap",    50)    # M: pair pool size
        cfg_m.insert("num_train_iter", 5000)    # T: total gradient steps
        cfg_m.insert("num_projections", 100)    # L: stereographic directions
        cfg_m.insert("n_supply",        100)
        cfg_m.insert("n_demand",     10_000)
        cfg_m.insert("supply_bernoulli_p", 0.5)
        cfg_m.insert("epsilon",         0.5)    # same as OT_Regression_Sliced_World
        cfg_m.insert("learning_rate",  1e-3)
        cfg_m.insert("max_grad_norm",   1.0)
        cfg_m.insert("batch_size",        1)
        cfg_m.insert("valid_rate",       0.0)
        cfg_m.insert("log_interval",    5000)
        cfg_m.insert("sinkhorn_iters",  500)    # for sanity check only
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    elif n_solver == "OT_Objective_Sliced_Color":
        cfg_m.insert("num_bootstrap",    50)    
        cfg_m.insert("num_train_iter", 5000)  
        cfg_m.insert("num_projections", 100)    
        cfg_m.insert("n_clusters",      500)    
        cfg_m.insert("img_size",          0)    
        cfg_m.insert("epsilon",         0.005)   
        cfg_m.insert("learning_rate",  1e-3)
        cfg_m.insert("max_grad_norm",   1.0)
        cfg_m.insert("batch_size",        1)   
        cfg_m.insert("valid_rate",       0.0)
        cfg_m.insert("log_interval",    5000)
        cfg_m.insert("sinkhorn_iters",  1500)   
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    elif n_solver == "Min_STP_GrayScale":
        cfg_m.insert("img_size",       28)
        cfg_m.insert("context_dim",    32)
        cfg_m.insert("hidden_dim",    128)
        cfg_m.insert("num_train_iter", 5000)
        cfg_m.insert("learning_rate",  1e-3)
        cfg_m.insert("alpha",          0.05)
        cfg_m.insert("max_grad_norm",  1.0)
        cfg_m.insert("log_interval",   100)
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    elif n_solver == "Min_STP_World":
        cfg_m.insert("context_dim",    32)
        cfg_m.insert("hidden_dim",    128)
        cfg_m.insert("num_train_iter", 5000)
        cfg_m.insert("learning_rate",  1e-3)
        cfg_m.insert("alpha",          0.05)
        cfg_m.insert("max_grad_norm",  1.0)
        cfg_m.insert("log_interval",   100)
        cfg_m.insert("n_supply",       100)
        cfg_m.insert("n_demand",    10_000)
        cfg_m.insert("epsilon",        0.5)
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)

    elif n_solver == "Min_STP_Color":
        cfg_m.insert("enc_dim",        32)
        cfg_m.insert("hidden_dim",     64)
        cfg_m.insert("n_clusters",    500)
        cfg_m.insert("num_train_iter", 5000)
        cfg_m.insert("learning_rate",  1e-3)
        cfg_m.insert("alpha",          0.05)
        cfg_m.insert("max_grad_norm",  1.0)
        cfg_m.insert("log_interval",   100)
        cfg_m.insert("epsilon",        0.005)
        cfg_m.insert("device", "cuda")
        cfg_m.insert("gpu", 0)
        
    return cfg_m


