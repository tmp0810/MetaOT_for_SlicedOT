import argparse
import os
import pickle
import shutil
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ot

plt.style.use("bmh")


def euclidean_to_spherical(xyz: np.ndarray) -> np.ndarray:
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    phi   = np.arctan2(y, x)
    theta = np.arccos(z.clip(-1.0, 1.0))
    return np.stack([phi, theta], axis=1)


def geodesic_arc(p0: np.ndarray, p1: np.ndarray, n_pts: int = 1000) -> np.ndarray:
    t   = np.linspace(0.0, 1.0, n_pts)[:, None]   # (n_pts, 1)
    arc = t * p0[None, :] + (1.0 - t) * p1[None, :]
    nrm = np.linalg.norm(arc, axis=1, keepdims=True).clip(1e-12)
    arc = arc / nrm
    sph = euclidean_to_spherical(arc)              # (n_pts, 2)
    # Remove dateline discontinuities (paper: norm of diff > 0.1)
    jumps = np.linalg.norm(sph[:-1] - sph[1:], axis=1)
    sph[1:][jumps > 0.1] = np.nan
    return sph


def plot_transport(
    a:           np.ndarray,   # (n_supply,)  supply weights
    b:           np.ndarray,   # (n_demand,)  demand weights
    P:           np.ndarray,   # (n_supply, n_demand) transport plan
    supply_euc:  np.ndarray,   # (n_supply, 3)
    demand_euc:  np.ndarray,   # (n_demand, 3)
    supply_sph:  np.ndarray,   # (n_supply, 2)  spherical coords
    landmask:    np.ndarray,   # 2-D binary land mask
    out_path:    str,
):
    T = P.argmax(axis=0)                       
    demand_to_supply_euc = supply_euc[T]        

    colors = plt.style.library["bmh"]["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(6, 4))     

    ax.imshow(
        landmask, cmap="gray_r",
        extent=[-np.pi, np.pi, 0, np.pi],
        alpha=0.15,
    )

    active = a > 1e-10
    ax.scatter(
        supply_sph[active, 0], supply_sph[active, 1],
        s=4., color="k", zorder=10,             # ← paper: s=4.
    )

    n_demand = len(b)
    for j in range(n_demand):
        sph = geodesic_arc(
            demand_euc[j],
            demand_to_supply_euc[j],
            n_pts=1000,
        )
        ax.plot(
            sph[:, 0], sph[:, 1],
            color=colors[0],
            alpha=0.1,           # ← paper: alpha=.1
            linewidth=1,         # ← paper: linewidth=1
        )
        
    fig.tight_layout()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["top"].set_visible(False)        
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # NO ax.set_title() — paper has none

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, transparent=True)   
    plt.close(fig)

    if shutil.which("pdfcrop"):
        os.system(f'pdfcrop "{out_path}" "{out_path}"')

    print(f"  Saved → {out_path}")

def load_landmask(pop_tiff: str) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(pop_tiff) as src:
        nodata = src.nodata
        out_h  = max(1, src.height // 4)
        out_w  = max(1, src.width  // 4)
        raw = src.read(
            1, out_shape=(out_h, out_w),
            resampling=Resampling.average,
        ).astype(np.float64)

    if nodata is not None:
        raw[np.isclose(raw, nodata, rtol=1e-3)] = 0.0
    raw[~np.isfinite(raw)] = 0.0
    raw[raw < 0] = 0.0
    return (raw > 0).astype(np.float32)

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot world-pair transport in Meta OT paper format.")
    p.add_argument("--result_dir",  type=str, required=True,
                   help="M{N} dir, e.g. ./results/worldpair/M50")
    p.add_argument("--pop_tiff",    type=str, required=True,
                   help="Population .tiff for land mask")
    p.add_argument("--idx",         type=str, default="all",
                   help="Test pair index (int) or 'all'")
    p.add_argument("--no_baseline", action="store_true",
                   help="Skip Sinkhorn GT computation")
    return p.parse_args()

def main():
    args = parse_args()
    with open(os.path.join(args.result_dir, "test_pairs.pkl"), "rb") as f:
        data = pickle.load(f)
    test_pairs = data["pairs"]
    supply_euc = data["supply_euc"]   # (n_supply, 3)
    demand_euc = data["demand_euc"]   # (n_demand, 3)
    supply_sph = data["supply_sph"]   # (n_supply, 2)
    print(f"Loaded {len(test_pairs)} test pairs.")

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

    landmask = load_landmask(args.pop_tiff)
    from Solvers.Regression_SlicedOT.OT_Regression_Sliced_World import _sphere_cost
    C   = _sphere_cost(supply_euc, demand_euc)
    eps = float(model_reg.cfg_m.epsilon)

    indices = range(len(test_pairs)) if args.idx == "all" else [int(args.idx)]

    for idx in indices:
        a, b = test_pairs[idx]
        print(f"\n=== Pair {idx} ===")
        pair_dir = os.path.join(args.result_dir, "plots", f"pair_{idx:02d}")
        os.makedirs(pair_dir, exist_ok=True)

        if not args.no_baseline:
            print("  [Sinkhorn_GT] computing ...")
            a_s = np.clip(a, 1e-10, None); a_s /= a_s.sum()
            b_s = np.clip(b, 1e-10, None); b_s /= b_s.sum()
            t0   = time.time()
            P_gt = ot.sinkhorn(a_s, b_s, C, reg=eps,
                               numItermax=1000, stopThr=1e-9)
            print(f"  [Sinkhorn_GT] done in {time.time()-t0:.2f}s")
            plot_transport(
                a, b, P_gt, supply_euc, demand_euc, supply_sph,
                landmask,
                out_path=os.path.join(pair_dir, "Sinkhorn_GT.pdf"),
            )

        for name, predict_fn in methods:
            print(f"  [{name}] computing ...")
            try:
                t0 = time.time()
                P  = predict_fn(a, b)
                print(f"  [{name}] done in {time.time()-t0:.3f}s")
            except Exception as e:
                print(f"  [{name}] ERROR: {e}")
                continue
            plot_transport(
                a, b, P, supply_euc, demand_euc, supply_sph,
                landmask,
                out_path=os.path.join(pair_dir, f"{name}.pdf"),
            )

    print(f"\nDone.  Output → {args.result_dir}/plots/")


if __name__ == "__main__":
    main()
