import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')

from Data.world_pair_data import load_world_locations, WorldPairDataset

def sample_one_pair(n_supply: int, n_demand: int,
                    supply_bernoulli_p: float = 0.5,
                    seed: int = 0):
    rng = np.random.default_rng(seed)
    mask     = rng.binomial(1, supply_bernoulli_p, n_supply).astype(np.float64)
    supply_w = mask * rng.uniform(0.0, 1.0, n_supply)
    if supply_w.sum() < 1e-12:
        supply_w = np.ones(n_supply, dtype=np.float64)
    supply_w /= supply_w.sum()

    demand_w  = rng.uniform(0.0, 1.0, n_demand).astype(np.float64)
    demand_w /= demand_w.sum()
    return supply_w, demand_w


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

def plot_transport(
    a: np.ndarray,
    b: np.ndarray,
    P: np.ndarray,
    supply_euc: np.ndarray,
    demand_euc: np.ndarray,
    supply_sph: np.ndarray,
    landmask: np.ndarray,
    title: str,
    out_path: str,
):
    T = P.argmax(axis=0)
    demand_to_supply_euc = supply_euc[T]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.style.library['bmh']['axes.prop_cycle'].by_key()['color']

    ax.imshow(
        landmask, cmap='gray_r',
        extent=[-np.pi, np.pi, 0, np.pi],
        alpha=0.15, aspect='auto',
    )

    active = a > 1e-10
    ax.scatter(
        supply_sph[active, 0], supply_sph[active, 1],
        s=6, color='k', zorder=10, label='supply',
    )

    n_demand   = len(b)
    max_arcs   = min(n_demand, 2000)
    rng_plot   = np.random.default_rng(42)
    arc_idxs   = rng_plot.choice(n_demand, size=max_arcs, replace=False)

    for j in arc_idxs:
        arc_euc = great_circle_path(demand_euc[j], demand_to_supply_euc[j], n_pts=100)
        arc_sph = euclidean_to_spherical(arc_euc)
        diff = np.abs(np.diff(arc_sph[:, 0]))
        arc_sph[1:][diff > 0.1] = np.nan

        ax.plot(
            arc_sph[:, 0], arc_sph[:, 1],
            color=colors[0], alpha=0.06, linewidth=0.8,
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


def solve_ot_pot(a, b, C, reg=0.1, num_iter=1000):
    import ot
    P = ot.sinkhorn(a, b, C, reg=reg, numItermax=num_iter, stopThr=1e-9)
    return P


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',   type=str, required=True,
                        help='Directory with model.pkl (any WorldPair solver)')
    parser.add_argument('--pop_tiff',    type=str, required=True,
                        help='Path to pop-15min.tif population raster')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--n_supply',    type=int, default=100)
    parser.add_argument('--n_demand',    type=int, default=10_000)
    parser.add_argument('--seed',        type=int, default=0)
    parser.add_argument('--out_dir',     type=str, default='./world_plots')
    parser.add_argument('--no_baseline', action='store_true',
                        help='Skip Sinkhorn baseline')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load locations ────────────────────────────────────────────────
    print("Loading world locations ...")
    supply_sph, supply_euc, demand_sph, demand_euc = load_world_locations(
        args.pop_tiff,
        n_supply=args.n_supply,
        n_demand=args.n_demand,
        seed=args.seed,
    )

    # Landmask for background — downsample to avoid OOM
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(args.pop_tiff) as _src:
        _nodata = _src.nodata
        _out_h  = max(1, _src.height // 4)
        _out_w  = max(1, _src.width  // 4)
        _raw = _src.read(1, out_shape=(_out_h, _out_w),
                         resampling=Resampling.average).astype(np.float64)
    if _nodata is not None:
        _raw[np.isclose(_raw, _nodata, rtol=1e-3)] = 0.0
    _raw[~np.isfinite(_raw)] = 0.0
    _raw[_raw < 0] = 0.0
    landmask = (_raw > 0).astype(np.float32)

    # ── Load trained model ────────────────────────────────────────────
    print(f"Loading model from {args.model_dir} ...")
    model_pkl = os.path.join(args.model_dir, 'model.pkl')
    if os.path.exists(model_pkl):
        with open(model_pkl, 'rb') as f:
            model = pickle.load(f)
    else:
        # Fallback: alpha/beta .npy → rebuild OT_Regression_Sliced_World only
        # (Meta_OT_World cannot be rebuilt from .npy — it needs meta_net.pt)
        import argparse as _ap
        from time import localtime, strftime
        from cfg import init_cfg
        from Solvers.OT_Regression_Sliced_World import OT_Regression_Sliced_World

        cfg_m    = init_cfg("OT_Regression_Sliced_World")
        cfg_proj = _ap.Namespace(
            seed      = args.seed,
            flag_time = strftime("%Y-%m-%d_%H-%M-%S", localtime()),
            flag_load = None,
            solver    = "OT_Regression_Sliced_World",
            data_name = "world_pair",
            gpu       = "0",
        )
        model = OT_Regression_Sliced_World(
            cfg_proj, cfg_m, supply_euc, demand_euc, supply_sph, demand_sph)
        model.alpha = np.load(os.path.join(args.model_dir, 'alpha.npy'))
        model.beta  = np.load(os.path.join(args.model_dir, 'beta.npy'))

    model_label = type(model).__name__
    print(f"  Model : {model_label}")

    # Precompute cost matrix (for Sinkhorn baseline)
    from Solvers.OT_Regression_Sliced_World import _sphere_cost
    C   = _sphere_cost(supply_euc, demand_euc)
    eps = float(getattr(model.cfg_m, 'epsilon', 0.5))

    # ── Plot each sample ──────────────────────────────────────────────
    import time
    times_model, times_sinkhorn = [], []

    for i in range(args.num_samples):
        print(f"\nSample {i+1}/{args.num_samples}")
        a, b = sample_one_pair(args.n_supply, args.n_demand, seed=args.seed + i)

        # Both OT_Regression_Sliced_World and Meta_OT_World share the same
        # predict_plan(a, b) interface — no dispatch needed
        t0      = time.time()
        P_pred  = model.predict_plan(a, b)
        t_model = time.time() - t0
        times_model.append(t_model)

        plot_transport(
            a, b, P_pred,
            supply_euc, demand_euc, supply_sph, landmask,
            title=f'{model_label} — sample {i}  ({t_model:.2f}s)',
            out_path=os.path.join(args.out_dir, f'model_{i}.pdf'),
        )

        if not args.no_baseline:
            print("  Running Sinkhorn baseline ...")
            t0     = time.time()
            P_gt   = solve_ot_pot(a, b, C, reg=eps)
            t_sink = time.time() - t0
            times_sinkhorn.append(t_sink)
            plot_transport(
                a, b, P_gt,
                supply_euc, demand_euc, supply_sph, landmask,
                title=f'Sinkhorn (eps={eps}) — sample {i}  ({t_sink:.2f}s)',
                out_path=os.path.join(args.out_dir, f'sinkhorn_{i}.pdf'),
            )

    # ── Timing summary ────────────────────────────────────────────────
    tm = np.array(times_model)
    print(f"\n{'='*50}")
    print(f"Timing summary ({args.num_samples} samples)")
    print(f"{'='*50}")
    print(f"{model_label:35s}: {tm.mean():.3f}s +/- {tm.std():.3f}s")
    if times_sinkhorn:
        ts = np.array(times_sinkhorn)
        print(f"{'Sinkhorn baseline':35s}: {ts.mean():.3f}s +/- {ts.std():.3f}s")
        print(f"{'Speedup':35s}: {ts.mean()/tm.mean():.1f}x")
    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
