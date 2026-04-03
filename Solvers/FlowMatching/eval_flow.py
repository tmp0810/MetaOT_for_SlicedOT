import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import ot
from tqdm import tqdm
from sklearn.datasets import make_moons, make_s_curve

from torchdyn.core import NeuralODE
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from Solvers.FlowMatching.RA_OT_flow import AmortizedRA_OT
from Solvers.FlowMatching.OA_OT_flow import AmortizedOA_OT


import math
from torchdyn.datasets import generate_moons

def sample_gaussian(n):
    return torch.randn(n, 2)

def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data

def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()

def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.2)
    return x0 * 3 - 1

def sample_scurve(n):
    data, _ = make_s_curve(n_samples=n, noise=0.1)
    data = data[:, [0, 2]]   # keep x,z only
    return torch.tensor(data, dtype=torch.float32)

TARGET_SAMPLERS = {
    "8gaussians": sample_8gaussians,
    "moons":      sample_moons,
    "scurve":     sample_scurve,
}


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GradModel(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, x):
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.action(x)), x, create_graph=True)[0]
        return grad[:, :-1]


class ODEWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, **kwargs):
        # t may be 0-dim scalar, (1,), or (N,) depending on torchdyn solver step
        t_scalar = t.reshape(-1)[0]   # always a scalar value
        t_vec = t_scalar.expand(x.shape[0]).unsqueeze(-1)  # (N, 1)
        return self.model(torch.cat([x, t_vec], dim=-1))


def pair_independent(x0, x1):
    return x0, x1


def pair_exact_ot(x0, x1):
    B = x0.shape[0]
    C = torch.cdist(x0, x1).pow(2).cpu().numpy()
    a, b = ot.unif(B), ot.unif(B)
    P = ot.emd(a, b, C)
    P_flat = P.ravel()
    P_flat = P_flat / (P_flat.sum() + 1e-30)
    idx = np.random.choice(B * B, size=B, p=P_flat, replace=True)
    return x0[idx // B], x1[idx % B]



def train_flow(method_name, pair_fn, target_sampler,
               n_steps=20000, batch_size=512, lr=1e-3, sigma=0.1, device="cpu"):
    # Use original models.py MLP
    model = MLP(dim=2, time_varying=True).to(device)
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    FM   = ConditionalFlowMatcher(sigma=sigma)

    t0 = time.time()
    for step in tqdm(range(n_steps), desc=f"FM-{method_name}"):
        x0 = sample_gaussian(batch_size).to(device)
        x1 = target_sampler(batch_size).to(device)

        # pair via OT
        px0, px1 = pair_fn(x0, x1)
        t, xt, ut = FM.sample_location_and_conditional_flow(px0, px1)

        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt - ut) ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()

    train_time = time.time() - t0
    print(f"[{method_name}]  train_time={train_time:.2f}s  final_loss={loss.item():.4f}")
    return model, train_time


@torch.no_grad()
def generate_samples(model, n=10000, n_steps=100, device="cpu"):
    # node = NeuralODE(
    #     ODEWrapper(model), solver="dopri5",
    #     sensitivity="adjoint", atol=1e-4, rtol=1e-4,
    # ).to(device)

    node = NeuralODE(model_wrapper, solver='rk4', sensitivity='adjoint')
    x0 = sample_gaussian(n).to(device)
    #t_span = torch.linspace(0, 1, n_steps, device=device)
    t_span = torch.linspace(0, 1, 101).to(device)
    traj = node.trajectory(x0, t_span=t_span)        # (T, N, 2)
    return traj                                        # full trajectories


def compute_w2(generated, target_test):
    n = min(len(generated), len(target_test))
    g = generated[:n].cpu().numpy()
    t = target_test[:n].cpu().numpy()
    C = ot.dist(g, t, metric='sqeuclidean')
    a, b = ot.unif(n), ot.unif(n)
    w2_sq = ot.emd2(a, b, C)
    return float(np.sqrt(max(w2_sq, 0.0)))


def compute_npe(model, n=2000, n_steps=100, device="cpu"):
    node = NeuralODE(
        ODEWrapper(model), solver="dopri5",
        sensitivity="adjoint", atol=1e-4, rtol=1e-4,
    ).to(device)
    x0 = sample_gaussian(n).to(device)
    t_span = torch.linspace(0, 1, n_steps, device=device)

    with torch.no_grad():
        traj = node.trajectory(x0, t_span=t_span)   # (T, N, 2)

    dt = 1.0 / (n_steps - 1)
    pe = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(n_steps):
            t_val = t_span[i].expand(n, 1)
            v = model(torch.cat([traj[i], t_val], dim=-1))
            pe += (v ** 2).sum(dim=-1).mean().item() * dt

    x1_gen = traj[-1]
    C = ot.dist(x0.cpu().numpy(), x1_gen.cpu().numpy(), metric='sqeuclidean')
    a, b = ot.unif(n), ot.unif(n)
    w2_sq = float(ot.emd2(a, b, C))
    w2_sq = max(w2_sq, 1e-12)

    npe = abs(pe - w2_sq) / w2_sq
    return npe, pe, w2_sq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["8gaussians", "moons", "scurve"])
    parser.add_argument("--n_steps",  type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--L",        type=int, default=100)
    parser.add_argument("--eps",      type=float, default=0.1)
    parser.add_argument("--M_pretrain", type=int, default=50)
    parser.add_argument("--T_pretrain", type=int, default=5000)
    parser.add_argument("--sigma",    type=float, default=0.1)
    parser.add_argument("--device",   type=str, default="cpu")
    parser.add_argument("--outdir",   type=str, default="./results_flow")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dev = torch.device(args.device)
    results = {}

    for ds_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: Gaussian → {ds_name}")
        print(f"{'='*60}")
        target_sampler = TARGET_SAMPLERS[ds_name]
        target_test = target_sampler(10000).to(dev)
        results[ds_name] = {}

        # ---------- I-CFM ----------
        model_icfm, t_icfm = train_flow(
            "I-CFM", pair_independent, target_sampler,
            n_steps=args.n_steps, batch_size=args.batch_size,
            sigma=args.sigma, device=dev,
        )
        traj_icfm = generate_samples(model_icfm, device=dev)
        w2_icfm  = compute_w2(traj_icfm[-1], target_test)
        npe_icfm, _, _ = compute_npe(model_icfm, device=dev)
        results[ds_name]["I-CFM"] = dict(W2=w2_icfm, NPE=npe_icfm,
                                         train_time=t_icfm, pretrain_time=0.0)
        print(f"[I-CFM]   W2={w2_icfm:.4f}  NPE={npe_icfm:.4f}  time={t_icfm:.1f}s")

        # ---------- OT-CFM ----------
        model_ot, t_ot = train_flow(
            "OT-CFM", pair_exact_ot, target_sampler,
            n_steps=args.n_steps, batch_size=args.batch_size,
            sigma=args.sigma, device=dev,
        )
        traj_ot = generate_samples(model_ot, device=dev)
        w2_ot  = compute_w2(traj_ot[-1], target_test)
        npe_ot, _, _ = compute_npe(model_ot, device=dev)
        results[ds_name]["OT-CFM"] = dict(W2=w2_ot, NPE=npe_ot,
                                           train_time=t_ot, pretrain_time=0.0)
        print(f"[OT-CFM]  W2={w2_ot:.4f}  NPE={npe_ot:.4f}  time={t_ot:.1f}s")

        # ---------- RA-OT-FM ----------
        ra_ot = AmortizedRA_OT(L=args.L, eps=args.eps, ridge=1e-3, device=dev)
        ra_ot.pretrain(sample_gaussian, target_sampler,
                       M=args.M_pretrain, B=args.batch_size)

        def pair_ra_ot(x0, x1):
            return ra_ot.sample_pairs(x0, x1)

        model_ra, t_ra = train_flow(
            "RA-OT-FM", pair_ra_ot, target_sampler,
            n_steps=args.n_steps, batch_size=args.batch_size,
            sigma=args.sigma, device=dev,
        )
        traj_ra = generate_samples(model_ra, device=dev)
        w2_ra  = compute_w2(traj_ra[-1], target_test)
        npe_ra, _, _ = compute_npe(model_ra, device=dev)
        results[ds_name]["RA-OT-FM"] = dict(
            W2=w2_ra, NPE=npe_ra,
            train_time=t_ra, pretrain_time=ra_ot.pretrain_time)
        print(f"[RA-OT]   W2={w2_ra:.4f}  NPE={npe_ra:.4f}  "
              f"time={t_ra:.1f}s  pretrain={ra_ot.pretrain_time:.1f}s")

        # ---------- OA-OT-FM ----------
        oa_ot = AmortizedOA_OT(L=args.L, eps=args.eps, lr=1e-3, device=dev)
        oa_ot.pretrain(sample_gaussian, target_sampler,
                       M=args.M_pretrain, B=args.batch_size,
                       T=args.T_pretrain)

        def pair_oa_ot(x0, x1):
            return oa_ot.sample_pairs(x0, x1)

        model_oa, t_oa = train_flow(
            "OA-OT-FM", pair_oa_ot, target_sampler,
            n_steps=args.n_steps, batch_size=args.batch_size,
            sigma=args.sigma, device=dev,
        )
        traj_oa = generate_samples(model_oa, device=dev)
        w2_oa  = compute_w2(traj_oa[-1], target_test)
        npe_oa, _, _ = compute_npe(model_oa, device=dev)
        results[ds_name]["OA-OT-FM"] = dict(
            W2=w2_oa, NPE=npe_oa,
            train_time=t_oa, pretrain_time=oa_ot.pretrain_time)
        print(f"[OA-OT]   W2={w2_oa:.4f}  NPE={npe_oa:.4f}  "
              f"time={t_oa:.1f}s  pretrain={oa_ot.pretrain_time:.1f}s")

        # ---------- save trajectories for plotting ----------
        for name, traj in [("I-CFM", traj_icfm), ("OT-CFM", traj_ot),
                           ("RA-OT-FM", traj_ra), ("OA-OT-FM", traj_oa)]:
            torch.save(traj.cpu(),
                       os.path.join(args.outdir, f"traj_{ds_name}_{name}.pt"))

    # ---- save summary ----
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {args.outdir}/results.json")

    # ---- print table ----
    print(f"\n{'Method':<14} {'Dataset':<14} {'W2':>8} {'NPE':>10} "
          f"{'Train(s)':>10} {'Pretrain(s)':>12}")
    print("-" * 70)
    for ds in results:
        for m in results[ds]:
            r = results[ds][m]
            print(f"{m:<14} {ds:<14} {r['W2']:8.4f} {r['NPE']:10.4f} "
                  f"{r['train_time']:10.1f} {r['pretrain_time']:12.1f}")


if __name__ == "__main__":
    main()
