import argparse
import os
import time
import numpy as np
import torch
import ot
from tqdm import tqdm
from time import localtime, strftime
from torch.utils.data import DataLoader

from cfg import init_cfg
from Data.dataset_class import MNIST

POOL_SEED = 0
POOL_SIZE = 1000
TRAIN_RATIO = 0.7
N_GRID = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]  # for the printed cost table


def build_cost_grid(img_size=28):
    grid = np.array([[j, i]
                     for i in np.linspace(1, 0, num=img_size)
                     for j in np.linspace(0, 1, num=img_size)], dtype=np.float64)
    diff = grid[:, None, :] - grid[None, :, :]
    return np.sum(diff ** 2, axis=-1)


def sample_pairs(n, seed):
    np.random.seed(seed)
    dataset = MNIST(flag_train=True, cfg_m=argparse.Namespace(datasets_root="../datasets"))
    pairs = []
    for _ in range(n):
        id_a, id_b = np.random.randint(0, len(dataset.data), 2)
        a = dataset.data[id_a].numpy()
        b = dataset.data[id_b].numpy()
        pairs.append((a, b))
    return pairs


def pairs_to_loader(pairs, batch_size=1):
    data = [(torch.zeros(1), torch.zeros(1),
             torch.tensor(a, dtype=torch.float64),
             torch.tensor(b, dtype=torch.float64))
            for a, b in pairs]
    return DataLoader(data, batch_size=batch_size, shuffle=False)


def make_cfg_proj(solver, seed, gpu, flag_time):
    return argparse.Namespace(seed=seed, flag_time=flag_time,
                              flag_load=None, solver=solver,
                              data_name="MNIST", gpu=gpu)


def time_scratch_solve(test_pairs, C, eps, n_iter=800):
    """Time solving each OT problem completely from scratch (the
    non-amortized baseline every predicted plan is compared against
    throughout this repo)."""
    times = []
    for a, b in tqdm(test_pairs, desc="  scratch Sinkhorn", leave=False):
        a_s = np.clip(a, 1e-10, None); a_s /= a_s.sum()
        b_s = np.clip(b, 1e-10, None); b_s /= b_s.sum()
        t0 = time.perf_counter()
        ot.sinkhorn(a_s, b_s, C, reg=eps, numItermax=n_iter, stopThr=1e-9)
        times.append(time.perf_counter() - t0)
    return np.array(times)


def time_infer(predict_fn, test_pairs):
    if test_pairs:
        try: predict_fn(*test_pairs[0])
        except Exception: pass
    times = []
    for a, b in tqdm(test_pairs, desc="  infer", leave=False):
        t0 = time.perf_counter()
        predict_fn(a, b)
        times.append(time.perf_counter() - t0)
    return np.array(times)


def cost(T_train, t_infer, N):
    return T_train + N * t_infer


def breakeven_vs_scratch(T_train, t_infer, t_scratch):
    if t_scratch <= t_infer:
        return None  # amortized method is never cheaper per-pair either; no crossover
    return T_train / (t_scratch - t_infer)


def breakeven_pair(T_i, t_i, T_j, t_j):
    if t_i == t_j:
        return None
    return (T_j - T_i) / (t_i - t_j)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=50)
    p.add_argument("--N", type=int, default=300)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--out", type=str, default="./results/breakeven")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out, exist_ok=True)
    flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime())

    C = build_cost_grid(28)
    eps = 1e-2

    n_train_pool = int(POOL_SIZE * TRAIN_RATIO)
    n_test_pool = POOL_SIZE - n_train_pool
    assert args.M <= n_train_pool
    assert args.N <= n_test_pool

    print(f"\nPre-sampling pool of {POOL_SIZE} pairs (seed={POOL_SEED}) ...")
    pool = sample_pairs(POOL_SIZE, seed=POOL_SEED)
    train_pairs = pool[:n_train_pool][:args.M]
    test_pairs = pool[n_train_pool:][:args.N]
    print(f"  M={args.M} train | N={args.N} test\n")

    dl_train = pairs_to_loader(train_pairs, batch_size=1)
    stats = {}  # name -> (T_train, t_infer_mean)

    # ---- scratch baseline (no amortization at all) ----
    print("[0/3] Timing solve-from-scratch Sinkhorn (non-amortized baseline) ...")
    t_scratch_arr = time_scratch_solve(test_pairs, C, eps)
    t_scratch = float(t_scratch_arr.mean())
    print(f"  t_scratch = {t_scratch*1000:.3f} ms/pair (mean over N={args.N})")

    # ---- RA-OT ----
    print("\n[1/3] RA-OT ...")
    from Solvers.Regression_SlicedOT.OT_Regression_Sliced import OT_Regression_Sliced
    cfg_r = init_cfg("OT_Regression_Sliced")
    cfg_r["num_bootstrap"] = args.M; cfg_r["epsilon"] = eps
    model_reg = OT_Regression_Sliced(
        make_cfg_proj("OT_Regression_Sliced", POOL_SEED, args.gpu, flag_time), cfg_r)
    t0 = time.perf_counter()
    alpha_reg = model_reg._fit(dl_train)
    T_reg = time.perf_counter() - t0

    def predict_reg(a, b):
        f, g = model_reg._predict_potentials(a, b, alpha_reg)
        return model_reg._potentials_to_plan(a, b, f, g)

    t_reg = float(time_infer(predict_reg, test_pairs).mean())
    stats["RA-OT"] = (T_reg, t_reg)
    print(f"  T_train={T_reg:.2f}s  t_infer={t_reg*1000:.3f}ms/pair")

    # ---- OA-OT ----
    print("\n[2/3] OA-OT ...")
    from Solvers.Objective_SlicedOT.OT_Objective_Sliced import OT_Objective_Sliced
    cfg_o = init_cfg("OT_Objective_Sliced")
    cfg_o["num_bootstrap"] = args.M; cfg_o["epsilon"] = eps
    model_obj = OT_Objective_Sliced(
        make_cfg_proj("OT_Objective_Sliced", POOL_SEED, args.gpu, flag_time), cfg_o)
    t0 = time.perf_counter()
    alpha_obj = model_obj._fit(dl_train)
    T_obj = time.perf_counter() - t0

    def predict_obj(a, b):
        f, g = model_obj._predict_potentials(a, b, alpha_obj)
        return model_obj._potentials_to_plan(a, b, f, g)

    t_obj = float(time_infer(predict_obj, test_pairs).mean())
    stats["OA-OT"] = (T_obj, t_obj)
    print(f"  T_train={T_obj:.2f}s  t_infer={t_obj*1000:.3f}ms/pair")

    # ---- Meta-OT ----
    print("\n[3/3] Meta-OT ...")
    from Solvers.Meta_OT.Meta_OT_gray_scale import OT_Discrete, dual_obj_loss
    from Models.ot_models import PotentialMLP
    cfg_meta = init_cfg("OT_Discrete")
    cfg_meta["epsilon"] = eps
    T_target = 5000
    cfg_meta["epochs"] = max(1, T_target // args.M)
    cfg_meta["batch_size"] = 1
    cfg_meta["log_interval"] = max(1, T_target // args.M)
    model_meta = OT_Discrete(
        make_cfg_proj("OT_Discrete", POOL_SEED, args.gpu, flag_time), cfg_meta)
    t0 = time.perf_counter()
    model_meta.OT_D_train(dl_train, None, flad_load_ckp=False)
    T_meta = time.perf_counter() - t0

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_meta = PotentialMLP(dim_in=28**2*2, dim_out=28**2,
                             hidden_num=cfg_meta.MLP_hidden_num).to(dev)
    mlp_meta, _, _, _ = model_meta.load_ckp(mlp_meta, None, None, "OT_D-train")
    mlp_meta.eval()
    lf_meta = dual_obj_loss(img_size=28, epsilon=cfg_meta.epsilon, device=dev)

    def predict_meta(a, b):
        a_t = torch.tensor(a, dtype=torch.float64, device=dev).unsqueeze(0)
        b_t = torch.tensor(b, dtype=torch.float64, device=dev).unsqueeze(0)
        with torch.no_grad():
            f = mlp_meta(a_t, b_t)
        return lf_meta.pred_transport(a_t, b_t, f)[0]

    t_meta = float(time_infer(predict_meta, test_pairs).mean())
    stats["Meta-OT"] = (T_meta, t_meta)
    print(f"  T_train={T_meta:.2f}s  t_infer={t_meta*1000:.3f}ms/pair")

    # ---------------- Break-even vs scratch ----------------
    print(f"\n{'='*78}")
    print(f"  Break-even N*: amortized method becomes cheaper than solving from scratch")
    print(f"  (scratch: {t_scratch*1000:.3f} ms/pair, numItermax=800, stopThr=1e-9)")
    print(f"{'='*78}")
    for name, (T_train, t_infer) in stats.items():
        Nstar = breakeven_vs_scratch(T_train, t_infer, t_scratch)
        if Nstar is None:
            print(f"  {name:<10}: NEVER cheaper per-pair than scratch "
                  f"(t_infer={t_infer*1000:.3f}ms >= t_scratch={t_scratch*1000:.3f}ms)")
        else:
            print(f"  {name:<10}: N* = {Nstar:8.1f} pairs "
                  f"(T_train={T_train:.2f}s, t_infer={t_infer*1000:.3f}ms)")

    # ---------------- Pairwise break-even among amortized methods ----------------
    print(f"\n{'-'*78}")
    print("  Break-even N* between amortized methods (crossover of total cost)")
    print(f"{'-'*78}")
    names = list(stats.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            Ti, ti = stats[ni]
            Tj, tj = stats[nj]
            Nstar = breakeven_pair(Ti, ti, Tj, tj)
            cheaper_small_N = ni if Ti < Tj else nj
            cheaper_large_N = nj if Ti < Tj else ni
            if Nstar is None or Nstar < 0:
                print(f"  {ni} vs {nj}: no crossover in the relevant range "
                      f"({ni} {'always' if (Ti<Tj)==(ti<tj) else 'depends'} cheaper)")
            else:
                print(f"  {ni} vs {nj}: N* = {Nstar:8.1f}  "
                      f"({cheaper_small_N} cheaper for N<N*, {cheaper_large_N} cheaper for N>N*)")

    # ---------------- Printed cost table over a grid of N ----------------
    print(f"\n{'='*100}")
    print(f"  Total cost (seconds) = T_train + N * t_infer   vs   N * t_scratch")
    print(f"{'='*100}")
    header = f"  {'N':>8}" + "".join(f"{n:>12}" for n in names) + f"{'scratch':>12}"
    print(header)
    for Ntest in N_GRID:
        row = f"  {Ntest:>8}"
        for name in names:
            T_train, t_infer = stats[name]
            row += f"{cost(T_train, t_infer, Ntest):>12.2f}"
        row += f"{Ntest * t_scratch:>12.2f}"
        print(row)
    print(f"{'='*100}\n")

    # ---------------- Save CSV ----------------
    csv_path = os.path.join(args.out, f"breakeven_M{args.M}.csv")
    with open(csv_path, "w") as f:
        f.write("N," + ",".join(names) + ",scratch\n")
        for Ntest in N_GRID:
            row = [str(Ntest)]
            for name in names:
                T_train, t_infer = stats[name]
                row.append(f"{cost(T_train, t_infer, Ntest):.4f}")
            row.append(f"{Ntest * t_scratch:.4f}")
            f.write(",".join(row) + "\n")
    print(f"Cost table -> {csv_path}")

    # ---------------- Optional: matplotlib figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Ns = np.array(N_GRID, dtype=np.float64)
        plt.figure(figsize=(7, 5))
        for name in names:
            T_train, t_infer = stats[name]
            plt.plot(Ns, cost(T_train, t_infer, Ns), marker="o", label=name)
        plt.plot(Ns, Ns * t_scratch, marker="o", linestyle="--", color="black", label="scratch (no amortization)")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of test OT problems (N)")
        plt.ylabel("Total cost (s) = T_train + N * t_infer")
        plt.title(f"Break-even analysis (M={args.M} training pairs)")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        fig_path = os.path.join(args.out, f"breakeven_M{args.M}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Figure -> {fig_path}")
    except ImportError:
        print("matplotlib not available -- skipping figure, CSV table above still has all numbers.")


if __name__ == "__main__":
    main()
