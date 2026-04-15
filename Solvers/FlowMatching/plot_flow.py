import os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


METHODS  = ["I-CFM", "OT-CFM", "RA-OT-FM", "OA-OT-FM"]
DATASETS = ["8gaussians", "moons", "scurve"]
DS_LABELS = {"8gaussians": "8-Gaussians", "moons": "Moons", "scurve": "S-Curve"}
METHOD_COLORS = {
    "I-CFM":     "#888888",
    "OT-CFM":    "#d62728",
    "RA-OT-FM":  "#1f77b4",
    "OA-OT-FM":  "#2ca02c",
}


def plot_trajectories(result_dir, out_dir, n_show=500):
    for ds in DATASETS:
        ds_dir = os.path.join(out_dir, "trajectories", ds)
        os.makedirs(ds_dir, exist_ok=True)

        for method in METHODS:
            fpath = os.path.join(result_dir, f"traj_{ds}_{method}.pt")
            if not os.path.exists(fpath):
                print(f"  [skip] no data: traj_{ds}_{method}.pt")
                continue

            traj = torch.load(fpath, map_location="cpu").numpy()  # (T, N, 2)
            T_steps, N, _ = traj.shape
            ns = min(n_show, N)

            fig, ax = plt.subplots(figsize=(4.5, 4.5))

            # flow lines
            ax.scatter(traj[:, :ns, 0].ravel(), traj[:, :ns, 1].ravel(),
                       s=0.15, alpha=0.12, c="olive", rasterized=True)
            # source
            ax.scatter(traj[0, :ns, 0], traj[0, :ns, 1],
                       s=8, alpha=0.6, c="black", zorder=3)
            # generated
            ax.scatter(traj[-1, :ns, 0], traj[-1, :ns, 1],
                       s=6, alpha=0.8, c=METHOD_COLORS[method], zorder=4)

            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            plt.tight_layout(pad=0.2)

            safe_method = method.replace("/", "-")
            save_path = os.path.join(ds_dir, f"{safe_method}.png")
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  → {save_path}")


def plot_metrics(result_dir, out_dir):
    results_path = os.path.join(result_dir, "results.json")
    if not os.path.exists(results_path):
        print("results.json not found — skipping metrics plot.")
        return

    with open(results_path) as f:
        results = json.load(f)

    fig, axes = plt.subplots(len(DATASETS), 3, figsize=(14, 4 * len(DATASETS)))
    if len(DATASETS) == 1:
        axes = axes[np.newaxis, :]

    metric_keys = [
        ("W2",  "W₂ Distance ↓"),
        ("NPE", "NPE ↓"),
        ("train_time", "Training Time (s) ↓"),
    ]

    for i, ds in enumerate(DATASETS):
        if ds not in results:
            continue
        for k, (key, ylabel) in enumerate(metric_keys):
            ax = axes[i, k]
            vals  = []
            colors = []
            labels = []
            for m in METHODS:
                if m in results[ds]:
                    v = results[ds][m][key]
                    if key == "train_time" and "pretrain_time" in results[ds][m]:
                        v += results[ds][m]["pretrain_time"]
                    vals.append(v)
                    colors.append(METHOD_COLORS[m])
                    labels.append(m)

            bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_ylabel(ylabel, fontsize=10)
            if i == 0:
                ax.set_title(ylabel.split("↓")[0].strip(), fontsize=12, fontweight="bold")
            if k == 0:
                ax.set_ylabel(f"{DS_LABELS[ds]}\n{ylabel}", fontsize=10)

            # value labels
            for bar, v in zip(bars, vals):
                fmt = f"{v:.3f}" if v < 10 else f"{v:.1f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt, ha="center", va="bottom", fontsize=8)

            ax.tick_params(axis="x", rotation=25, labelsize=9)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "metrics_comparison.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")



def plot_distributions(result_dir, out_dir, n_show=2000):
    from eval_flow import TARGET_SAMPLERS

    for ds in DATASETS:
        target = TARGET_SAMPLERS[ds](5000).numpy()
        ds_dir = os.path.join(out_dir, "distributions", ds)
        os.makedirs(ds_dir, exist_ok=True)

        for method in METHODS:
            fpath = os.path.join(result_dir, f"traj_{ds}_{method}.pt")
            if not os.path.exists(fpath):
                print(f"  [skip] no data: traj_{ds}_{method}.pt")
                continue

            traj = torch.load(fpath, map_location="cpu").numpy()
            gen  = traj[-1][:n_show]

            fig, ax = plt.subplots(figsize=(4.5, 4.5))

            ax.scatter(target[:n_show, 0], target[:n_show, 1],
                       s=4, alpha=0.25, c="gray")
            ax.scatter(gen[:, 0], gen[:, 1],
                       s=4, alpha=0.5, c=METHOD_COLORS[method])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            plt.tight_layout(pad=0.2)

            safe_method = method.replace("/", "-")
            save_path = os.path.join(ds_dir, f"{safe_method}.png")
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="./results_flow")
    parser.add_argument("--out_dir",    type=str, default="./results_flow/plots")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Plotting trajectories …")
    plot_trajectories(args.result_dir, args.out_dir)

    print("Plotting metrics …")
    plot_metrics(args.result_dir, args.out_dir)

    print("Plotting distributions …")
    plot_distributions(args.result_dir, args.out_dir)

    print("✓ All plots saved.")


if __name__ == "__main__":
    main()
