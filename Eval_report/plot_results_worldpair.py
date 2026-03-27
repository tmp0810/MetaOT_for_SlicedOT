import argparse
import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import ot
plt.style.use('bmh')


def great_circle_path(p0: np.ndarray, p1: np.ndarray, n_pts: int = 300):
    t   = np.linspace(0.0, 1.0, n_pts)[:, None]
    arc = (1.0 - t) * p0[None, :] + t * p1[None, :]
    nrm = np.linalg.norm(arc, axis=1, keepdims=True).clip(1e-12)
    return arc / nrm


def euclidean_to_spherical(xyz: np.ndarray):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    phi   = np.arctan2(y, x)
    theta = np.arccos(z.clip(-1.0, 1.0))
    return np.stack([phi, theta], axis=1)


def plot_transport(a, b, P, supply_euc, demand_euc, supply_sph,
                   landmask, title, out_path):
    """Identical to plot_world_pair_torch.py:plot_transport."""
    T = P.argmax(axis=0)
    demand_to_supply_euc = supply_euc[T]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.style.library["bmh"]["axes.prop_cycle"].by_key()["color"]

    ax.imshow(landmask, cmap="gray_r",
              extent=[-np.pi, np.pi, 0, np.pi],
              alpha=0.15, aspect="auto")

    active = a > 1e-10
    ax.scatter(supply_sph[active, 0], supply_sph[active, 1],
               s=6, color="k", zorder=10, label="supply")

    n_demand = len(b)
    # max_arcs = min(n_demand, 2000)
    # rng_plot = np.random.default_rng(42)
    # arc_idxs = rng_plot.choice(n_demand, size=max_arcs, replace=False)

    # for j in arc_idxs:
    #     arc_euc = great_circle_path(demand_euc[j], demand_to_supply_euc[j], n_pts=100)
    #     arc_sph = euclidean_to_spherical(arc_euc)
    #     diff    = np.abs(np.diff(arc_sph[:, 0]))
    #     arc_sph[1:][diff > 0.1] = np.nan
    #     ax.plot(arc_sph[:, 0], arc_sph[:, 1],
    #             color=colors[0], alpha=0.06, linewidth=0.8)

    # ax.set_title(title, fontsize=10)
    # ax.set_xticks([]); ax.set_yticks([])
    # ax.grid(False)
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    # fig.tight_layout()
    # fig.savefig(out_path, transparent=True, bbox_inches="tight")
    # plt.close(fig)
    # print(f"  Saved → {out_path}")

    for j in range(n_demand):
        # 1. Tăng n_pts lên 1000 để đường cong vút mượt mà như bài gốc
        arc_euc = great_circle_path(demand_euc[j], demand_to_supply_euc[j], n_pts=1000)
        arc_sph = euclidean_to_spherical(arc_euc)
        
        # 2. Xóa nét vắt ngang Dateline (bản gốc dùng np.linalg.norm)
        n = np.linalg.norm(arc_sph[:-1] - arc_sph[1:], axis=1)
        arc_sph[1:][n > 0.1] = np.nan

        # 3. Tăng alpha=0.1 và linewidth=1 y hệt bài gốc
        ax.plot(
            arc_sph[:, 0], arc_sph[:, 1],
            color=colors[0],  # Màu xanh dương bmh default
            alpha=0.1, 
            linewidth=1,
        )

    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def load_landmask(pop_tiff):
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(pop_tiff) as src:
        _nodata = src.nodata
        _out_h  = max(1, src.height // 4)
        _out_w  = max(1, src.width  // 4)
        raw = src.read(1, out_shape=(_out_h, _out_w),
                       resampling=Resampling.average).astype(np.float64)
    if _nodata is not None:
        raw[np.isclose(raw, _nodata, rtol=1e-3)] = 0.0
    raw[~np.isfinite(raw)] = 0.0
    raw[raw < 0] = 0.0
    return (raw > 0).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", type=str, required=True,
                   help="M{N} dir from eval_report_worldpair, e.g. ./results/worldpair/M50")
    p.add_argument("--pop_tiff",   type=str, required=True)
    p.add_argument("--idx",        type=str, default="all",
                   help="Test pair index to plot, or 'all'")
    p.add_argument("--no_baseline", action="store_true",
                   help="Skip Sinkhorn GT")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load test pairs + locations ───────────────────────────────────────
    with open(os.path.join(args.result_dir, "test_pairs.pkl"), "rb") as f:
        data = pickle.load(f)
    test_pairs = data["pairs"]
    supply_euc = data["supply_euc"]
    demand_euc = data["demand_euc"]
    supply_sph = data["supply_sph"]
    print(f"Loaded {len(test_pairs)} test pairs.")

    # ── Load models ───────────────────────────────────────────────────────
    def load_pkl(name):
        with open(os.path.join(args.result_dir, name), "rb") as f:
            return pickle.load(f)

    model_reg  = load_pkl("regression.pkl")
    model_obj  = load_pkl("objective.pkl")
    model_meta = load_pkl("meta_ot.pkl")
    model_swgg = load_pkl("swgg.pkl")
    model_stp  = load_pkl("min_stp.pkl")

    methods = [
        ("OT_Regression", model_reg.predict_plan),
        ("OT_Objective",  model_obj.predict_plan),
        ("Meta_OT",       model_meta.predict_plan),
        ("min_SWGG",      model_swgg.predict_plan),
        ("Min_STP",       model_stp.predict_plan),
    ]

    # ── Load landmask + cost ──────────────────────────────────────────────
    landmask = load_landmask(args.pop_tiff)
    from Solvers.Regression_SlicedOT.OT_Regression_Sliced_World import _sphere_cost
    C   = _sphere_cost(supply_euc, demand_euc)
    eps = float(model_reg.cfg_m.epsilon)

    # ── Plot ──────────────────────────────────────────────────────────────
    indices = range(len(test_pairs)) if args.idx == "all" else [int(args.idx)]

    for idx in indices:
        a, b = test_pairs[idx]
        print(f"\nPair {idx} ...")
        pair_dir = os.path.join(args.result_dir, "plots", f"pair_{idx:02d}")
        os.makedirs(pair_dir, exist_ok=True)

        # Sinkhorn GT
        if not args.no_baseline:
            t0   = time.time()
            P_gt = ot.sinkhorn(a, b, C, reg=eps, numItermax=1000, stopThr=1e-9)
            t_gt = time.time() - t0
            plot_transport(a, b, P_gt, supply_euc, demand_euc, supply_sph,
                           landmask,
                           title=f"Sinkhorn GT (eps={eps})  {t_gt:.2f}s",
                           out_path=os.path.join(pair_dir, "Sinkhorn_GT.pdf"))

        # 4 methods
        for name, predict_fn in methods:
            t0 = time.time()
            P  = predict_fn(a, b)
            t  = time.time() - t0
            plot_transport(a, b, P, supply_euc, demand_euc, supply_sph,
                           landmask,
                           title=f"{name}  ({t:.3f}s)",
                           out_path=os.path.join(pair_dir, f"{name}.pdf"))

    print(f"\nDone. Output → {args.result_dir}/plots/")


if __name__ == "__main__":
    main()
