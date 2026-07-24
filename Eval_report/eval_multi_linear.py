import argparse
import os
import time
import pickle
from time import localtime, strftime

import numpy as np
import torch

from cfg import init_cfg
from Data.dataset_class import MNIST

# Reuse existing, already-validated helpers from eval_grayscale.py verbatim.
from Eval_report.eval_grayscale import build_cost_grid, sinkhorn_gt, pairs_to_loader, evaluate

from Solvers.Regression_SlicedOT.OT_Regression_Sliced import OT_Regression_Sliced


GROUP_LOW  = set(range(0, 5))   # digits 0-4
GROUP_HIGH = set(range(5, 10))  # digits 5-9

POOL_SEED = 0


def get_targets_array(dataset):
    t = dataset.targets
    if hasattr(t, "numpy"):
        t = t.numpy()
    return np.asarray(t)


def sample_pairs_by_group(dataset, targets, n_pairs, group_digits, seed):
    rng = np.random.RandomState(seed)
    idx_pool = np.where(np.isin(targets, list(group_digits)))[0]
    assert len(idx_pool) > 0, f"No samples found for digit group {group_digits}"
    pairs = []
    for _ in range(n_pairs):
        id_a, id_b = rng.choice(idx_pool, size=2, replace=True)
        a = dataset.data[id_a].numpy()
        b = dataset.data[id_b].numpy()
        pairs.append((a, b))
    return pairs


def make_cfg_proj(seed, gpu, tag):
    return argparse.Namespace(
        seed=seed, flag_time=f"w3het_{tag}", flag_load=None,
        solver="OT_Regression_Sliced", data_name="MNIST", gpu=gpu,
    )


def fit_regression_model(cfg_r, cfg_proj, dl_train, shared_proj_matrix=None):
    model = OT_Regression_Sliced(cfg_proj, cfg_r)
    if shared_proj_matrix is not None:
        # Force identical slicing directions theta_1..theta_L across all
        # three fits, so the resulting omega vectors live in the same basis
        # and are directly comparable (required for cosine similarity to
        # mean anything).
        model.projection_matrix = shared_proj_matrix
    t0 = time.perf_counter()
    alpha = model._fit(dl_train)
    t_fit = time.perf_counter() - t0
    model.alpha = alpha
    model.beta = np.zeros(cfg_r["num_projections"])
    return model, alpha, t_fit


def make_predict_fn(model, alpha):
    def predict(a, b):
        f, g = model._predict_potentials(a, b, alpha)
        return model._potentials_to_plan(a, b, f, g)
    return predict


def cosine_sim(u, v):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Saved -> {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M_group", type=int, default=25,
                    help="training pairs per group (global model sees 2x this)")
    p.add_argument("--N_group", type=int, default=20,
                    help="held-out test pairs per group (from MNIST test split)")
    p.add_argument("--epsilon", type=float, default=1e-2,
                    help="entropic reg, matches eval_grayscale.py runtime value")
    p.add_argument("--num_projections", type=int, default=100)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=POOL_SEED)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--out", type=str, default="./results/grayscale_w3_heterogeneous")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out, exist_ok=True)
    flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())

    C = build_cost_grid(28)
    eps = args.epsilon

    # ---------------------------------------------------------------- data
    print("Loading MNIST train/test splits ...")
    ds_proj_cfg = argparse.Namespace(datasets_root="../datasets")
    dataset_train = MNIST(flag_train=True,  cfg_m=ds_proj_cfg)
    dataset_test  = MNIST(flag_train=False, cfg_m=ds_proj_cfg)
    targets_train = get_targets_array(dataset_train)
    targets_test  = get_targets_array(dataset_test)

    print(f"Sampling {args.M_group} train pairs per group (seed={args.seed}) ...")
    pairs_A = sample_pairs_by_group(dataset_train, targets_train, args.M_group,
                                     GROUP_LOW,  seed=args.seed + 1)
    pairs_B = sample_pairs_by_group(dataset_train, targets_train, args.M_group,
                                     GROUP_HIGH, seed=args.seed + 2)
    # Global pool = simple union (50/50 heterogeneous mixture), shuffled.
    rng = np.random.RandomState(args.seed + 3)
    pairs_global = pairs_A + pairs_B
    rng.shuffle(pairs_global)

    print(f"Sampling {args.N_group} test pairs per group from MNIST test split ...")
    test_A = sample_pairs_by_group(dataset_test, targets_test, args.N_group,
                                    GROUP_LOW,  seed=args.seed + 10)
    test_B = sample_pairs_by_group(dataset_test, targets_test, args.N_group,
                                    GROUP_HIGH, seed=args.seed + 11)
    test_combined = test_A + test_B

    with open(os.path.join(args.out, "test_pairs.pkl"), "wb") as f:
        pickle.dump({"A": test_A, "B": test_B}, f)

    dl_A      = pairs_to_loader(pairs_A,      batch_size=1)
    dl_B      = pairs_to_loader(pairs_B,      batch_size=1)
    dl_global = pairs_to_loader(pairs_global, batch_size=1)

    # ------------------------------------------------------------- fitting
    def fresh_cfg_r(num_bootstrap):
        # IMPORTANT: build a brand-new dotdict via init_cfg() for each fit,
        # instead of dict(cfg_r) — dict(...) silently downgrades the
        # dotdict to a plain dict and drops its .insert()/attribute-style
        # API, which Defense_Train_Base.init_env() relies on.
        c = init_cfg("OT_Regression_Sliced")
        c["epsilon"]         = eps
        c["ridge"]           = args.ridge
        c["num_projections"] = args.num_projections
        c["num_bootstrap"]   = num_bootstrap
        return c

    print("\n[1/3] Fitting omega_A  (Group A only, digits 0-4) ...")
    model_A, alpha_A, t_A = fit_regression_model(
        fresh_cfg_r(args.M_group), make_cfg_proj(args.seed, args.gpu, "A"),
        dl_A, shared_proj_matrix=None)
    shared_proj = model_A.projection_matrix  # lock this in for B and global

    print("\n[2/3] Fitting omega_B  (Group B only, digits 5-9) ...")
    model_B, alpha_B, t_B = fit_regression_model(
        fresh_cfg_r(args.M_group), make_cfg_proj(args.seed, args.gpu, "B"),
        dl_B, shared_proj_matrix=shared_proj)

    print("\n[3/3] Fitting omega_global  (mixed pool, 2x M_group pairs) ...")
    model_G, alpha_G, t_G = fit_regression_model(
        fresh_cfg_r(2 * args.M_group), make_cfg_proj(args.seed, args.gpu, "global"),
        dl_global, shared_proj_matrix=shared_proj)

    save_model(model_A, os.path.join(args.out, "model_omega_A.pkl"))
    save_model(model_B, os.path.join(args.out, "model_omega_B.pkl"))
    save_model(model_G, os.path.join(args.out, "model_omega_global.pkl"))
    np.save(os.path.join(args.out, "omega_A.npy"), alpha_A)
    np.save(os.path.join(args.out, "omega_B.npy"), alpha_B)
    np.save(os.path.join(args.out, "omega_global.npy"), alpha_G)

    # ----------------------------------------------------------- evaluation
    def eval_and_report(tag, model, alpha, test_pairs):
        predict_fn = make_predict_fn(model, alpha)
        rmse_arr, tinf_arr = evaluate(predict_fn, test_pairs, C, eps, tag)
        print(f"  {tag:<28} RMSE = {rmse_arr.mean():.4e} ± {rmse_arr.std():.2e}  "
              f"(N={len(test_pairs)})")
        return rmse_arr, tinf_arr

    print("\n===== Evaluation =====")
    rows = []
    r, _ = eval_and_report("global -> test_A",        model_G, alpha_G, test_A);        rows.append(("global -> A", r))
    r, _ = eval_and_report("global -> test_B",        model_G, alpha_G, test_B);        rows.append(("global -> B", r))
    r, _ = eval_and_report("global -> test_combined",  model_G, alpha_G, test_combined); rows.append(("global -> A+B", r))
    r, _ = eval_and_report("oracle_A -> test_A",       model_A, alpha_A, test_A);        rows.append(("oracle_A -> A", r))
    r, _ = eval_and_report("oracle_B -> test_B",       model_B, alpha_B, test_B);        rows.append(("oracle_B -> B", r))
    # Bonus mismatched checks — cheap, makes the failure mode explicit.
    r, _ = eval_and_report("oracle_A -> test_B (mismatched)", model_A, alpha_A, test_B); rows.append(("oracle_A -> B (mismatch)", r))
    r, _ = eval_and_report("oracle_B -> test_A (mismatched)", model_B, alpha_B, test_A); rows.append(("oracle_B -> A (mismatch)", r))

    # ----------------------------------------------------- omega comparison
    cos_AB = cosine_sim(alpha_A, alpha_B)
    cos_AG = cosine_sim(alpha_A, alpha_G)
    cos_BG = cosine_sim(alpha_B, alpha_G)
    print("\n===== Coefficient (omega) comparison =====")
    print(f"  cos(omega_A, omega_B)      = {cos_AB:.4f}")
    print(f"  cos(omega_A, omega_global) = {cos_AG:.4f}")
    print(f"  cos(omega_B, omega_global) = {cos_BG:.4f}")

    # ----------------------------------------------------------- CSV output
    csv_path = os.path.join(args.out, "w3_heterogeneous_results.csv")
    with open(csv_path, "w") as f:
        f.write("setting,rmse_mean,rmse_std,n_test\n")
        for name, rmse_arr in rows:
            f.write(f"{name},{rmse_arr.mean():.6e},{rmse_arr.std():.6e},{len(rmse_arr)}\n")
        f.write(f"\ncos_omegaA_omegaB,{cos_AB:.6f}\n")
        f.write(f"cos_omegaA_omegaglobal,{cos_AG:.6f}\n")
        f.write(f"cos_omegaB_omegaglobal,{cos_BG:.6f}\n")
    print(f"\nResults -> {csv_path}")

    # ----------------------------------------------------------- plotting
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        L = len(alpha_A)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        axes[0].plot(np.arange(L), alpha_A, label="omega_A (digits 0-4)", alpha=0.8)
        axes[0].plot(np.arange(L), alpha_B, label="omega_B (digits 5-9)", alpha=0.8)
        axes[0].plot(np.arange(L), alpha_G, label="omega_global", alpha=0.8, linestyle="--", color="black")
        axes[0].set_xlabel("projection index l")
        axes[0].set_ylabel("coefficient value")
        axes[0].set_title("Fitted coefficients across L=%d projections" % L)
        axes[0].legend(fontsize=8)

        axes[1].scatter(alpha_A, alpha_B, s=10, alpha=0.6)
        lims = [min(alpha_A.min(), alpha_B.min()), max(alpha_A.max(), alpha_B.max())]
        axes[1].plot(lims, lims, color="gray", linestyle=":", linewidth=1)
        axes[1].set_xlabel("omega_A")
        axes[1].set_ylabel("omega_B")
        axes[1].set_title(f"omega_A vs omega_B  (cosine sim = {cos_AB:.3f})")

        plt.tight_layout()
        fig_path = os.path.join(args.out, "w3_omega_comparison.png")
        plt.savefig(fig_path, dpi=150)
        print(f"Figure  -> {fig_path}")
    except ImportError:
        print("matplotlib not available — skipped plotting, .npy files are still saved.")


if __name__ == "__main__":
    main()
