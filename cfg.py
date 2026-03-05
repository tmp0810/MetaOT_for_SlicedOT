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
        # ── Regression / amortisation parameters ──────────────────────────
        # M: number of bootstrap pairs used to fit α and β.
        #    More pairs → better generalisation, but slower fitting.
        cfg_m.insert("num_bootstrap", 10)

        # L: number of random 1-D projection directions (θ_1, …, θ_L).
        #    Controls the richness of the feature representation.
        #    Typical range: 100 – 1000.
        cfg_m.insert("num_projections", 200)

        # Tikhonov regularisation λ for the simplex-LS problems.
        #    Set to 0.0 for pure simplex projection; > 0 helps when columns
        #    of the feature matrix are nearly collinear.
        cfg_m.insert("ridge", 1e-3)

        # ── OT / image parameters (mirror OT_Discrete for fair comparison) ─
        cfg_m.insert("img_size", 28)          # pixel side length
        cfg_m.insert("epsilon", 0.1)          # entropic regularisation ε — must be large enough so K=exp(-C/ε) has no zero entries (ε=0.01 causes 57% of K to be ~0 → Sinkhorn diverges)
        cfg_m.insert("batch_size", 64)        # dataloader batch size
        cfg_m.insert("valid_rate", 0.0)       # no validation split needed
        cfg_m.insert("log_interval", 1)       # not used (no epochs)
        cfg_m.insert("sinkhorn_iters", 100)  # Algorithm 1 iterations

    return cfg_m
