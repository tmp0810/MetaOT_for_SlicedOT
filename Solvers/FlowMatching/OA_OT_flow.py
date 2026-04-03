#!/usr/bin/env python3
"""
OA_OT_flow.py — Objective-based Amortized OT for Flow Matching.

Phase 1: Optimise weight vector α via the Kantorovich dual objective
         (no ground-truth Sinkhorn needed).
Phase 2: At each FM step, predict OT plan with the learned α.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
import time
from tqdm import tqdm

from regression_OT_utils import (
    generate_uniform_unit_sphere_projections,
    emd1D_dual,
)


class AmortizedOA_OT:
    def __init__(self, L=100, eps=0.1, lr=1e-3, device="cpu"):
        self.L = L
        self.eps = eps
        self.lr = lr
        self.device = device
        self.proj_dirs = generate_uniform_unit_sphere_projections(
            dim=2, num_projections=L, dtype=torch.float64, device=device
        )
        self.alpha = None
        self.pretrain_time = 0.0

    # ------------------------------------------------------------------
    def _compute_sliced_potentials(self, x0, x1):
        B = x0.shape[0]
        proj_x0 = x0 @ self.proj_dirs.T
        proj_x1 = x1 @ self.proj_dirs.T
        a = torch.full((B,), 1.0 / B, dtype=torch.float64, device=self.device)
        b = torch.full((B,), 1.0 / B, dtype=torch.float64, device=self.device)
        f_grad, _, _ = emd1D_dual(
            proj_x0.T, proj_x1.T,
            u_weights=a, v_weights=b,
            p=2, require_sort=True,
        )
        Phi = f_grad.T
        Phi = Phi - Phi.mean(dim=0, keepdim=True)
        return Phi

    # ------------------------------------------------------------------
    @staticmethod
    def _g_from_f(f, b, log_K, eps):
        log_b = torch.log(b.clamp(1e-300))
        M     = log_K + f.unsqueeze(1) / eps
        m     = M.max(dim=0, keepdim=True).values
        lse   = (M - m).exp().sum(dim=0).log() + m.squeeze(0)
        return eps * (log_b - lse)

    @staticmethod
    def _f_from_g(g, a, log_K, eps):
        log_a = torch.log(a.clamp(1e-300))
        M     = log_K + g.unsqueeze(0) / eps
        m     = M.max(dim=1, keepdim=True).values
        lse   = (M - m).exp().sum(dim=1).log() + m.squeeze(1)
        return eps * (log_a - lse)

    def _dual_objective(self, a, b, f, log_K, eps):
        g = self._g_from_f(f, b, log_K, eps)
        # KL-style dual
        M_fa = log_K + g.unsqueeze(0) / eps
        m    = M_fa.max(dim=1, keepdim=True).values
        fa   = eps * ((M_fa - m).exp().sum(1).log() + m.squeeze(1))

        M_gb = log_K + f.unsqueeze(1) / eps
        m    = M_gb.max(dim=0, keepdim=True).values
        gb   = eps * ((M_gb - m).exp().sum(0).log() + m.squeeze(0))

        div_a = (a * (f - fa)).sum()
        div_b = (b * (g - gb)).sum()

        log_P     = f.unsqueeze(1) / eps + g.unsqueeze(0) / eps + log_K
        lp_max    = log_P.detach().max()
        total_sum = (log_P - lp_max).exp().sum() * lp_max.exp()
        return div_a + div_b + eps * (1.0 - total_sum)

    # ------------------------------------------------------------------
    def pretrain(self, source_sampler, target_sampler,
                 M=50, B=512, T=5000):
        print(f"[OA-OT] Pre-training  M={M}  B={B}  T={T}  L={self.L}  eps={self.eps}")
        dev = self.device

        # ---- collect training data ----
        pool_Phi, pool_a, pool_b, pool_logK = [], [], [], []
        for _ in tqdm(range(M), desc="OA-OT collect"):
            x0 = source_sampler(B).to(dtype=torch.float64, device=dev)
            x1 = target_sampler(B).to(dtype=torch.float64, device=dev)
            Phi = self._compute_sliced_potentials(x0, x1)
            C   = torch.cdist(x0, x1).pow(2)
            pool_Phi.append(Phi)
            pool_a.append(torch.full((B,), 1.0/B, dtype=torch.float64, device=dev))
            pool_b.append(torch.full((B,), 1.0/B, dtype=torch.float64, device=dev))
            pool_logK.append(-C / self.eps)

        # ---- optimise α ----
        alpha = nn.Parameter(torch.zeros(self.L, dtype=torch.float64, device=dev))
        opt   = torch.optim.Adam([alpha], lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T, eta_min=self.lr*0.01)

        rng = np.random.default_rng(42)
        t0  = time.time()
        pbar = tqdm(total=T, desc="OA-OT optimise")
        loss_ema = None

        for step in range(T):
            idx    = int(rng.integers(0, M))
            f_pred = pool_Phi[idx] @ alpha
            loss   = -self._dual_objective(
                pool_a[idx], pool_b[idx], f_pred, pool_logK[idx], self.eps
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([alpha], 1.0)
            opt.step()
            sched.step()

            lv = loss.item()
            loss_ema = lv if loss_ema is None else 0.95 * loss_ema + 0.05 * lv
            pbar.update(1)
            if (step + 1) % 1000 == 0:
                pbar.set_description(f"OA-OT  dual={-lv:.4e}  ema={loss_ema:.4e}")
        pbar.close()

        self.alpha = alpha.detach().cpu().numpy()
        self.pretrain_time = time.time() - t0
        print(f"[OA-OT] Done in {self.pretrain_time:.2f}s")
        return self.alpha

    # ------------------------------------------------------------------
    def predict_plan(self, x0, x1):
        """Predict OT plan for a minibatch (same logic as RA-OT)."""
        assert self.alpha is not None, "Call pretrain() first."
        B = x0.shape[0]
        x0 = x0.to(dtype=torch.float64, device=self.device)
        x1 = x1.to(dtype=torch.float64, device=self.device)

        C = torch.cdist(x0, x1).pow(2).cpu().numpy()
        Phi = self._compute_sliced_potentials(x0, x1).cpu().numpy()
        f = (Phi @ self.alpha)
        f = f - f.mean()

        log_K = -C / self.eps
        M_f   = f[:, None] / self.eps + log_K
        m_col = M_f.max(axis=0, keepdims=True)
        g = self.eps * (np.log(1.0 / B)
                        - (np.log(np.exp(M_f - m_col).sum(axis=0)) + m_col.squeeze()))

        log_P = f[:, None] / self.eps + g[None, :] / self.eps + log_K
        log_P -= log_P.max()
        P = np.exp(log_P)
        a_unif = 1.0 / B
        P *= (a_unif / (P.sum(axis=1, keepdims=True) + 1e-30))
        P *= (a_unif / (P.sum(axis=0, keepdims=True) + 1e-30))
        return np.clip(P, 0.0, None)

    # ------------------------------------------------------------------
    def sample_pairs(self, x0, x1):
        B = x0.shape[0]
        P = self.predict_plan(x0, x1)
        P_flat = P.ravel()
        P_flat = P_flat / (P_flat.sum() + 1e-30)
        idx = np.random.choice(B * B, size=B, p=P_flat, replace=True)
        return x0[idx // B], x1[idx % B]
