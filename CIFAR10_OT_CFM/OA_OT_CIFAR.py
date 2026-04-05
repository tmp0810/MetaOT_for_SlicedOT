import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from regression_OT_utils import (
    generate_uniform_unit_sphere_projections,
    emd1D_dual,
)

class AmortizedOA_OT_CIFAR:
    def __init__(self, L: int = 100, eps: float = 0.1,
                 lr: float = 1e-3, device: str = "cpu"):
        self.L = L
        self.eps = eps       
        self.lr = lr
        self.device = device
        self.dim = 3 * 32 * 32          

        self.proj_dirs = generate_uniform_unit_sphere_projections(
            dim=self.dim,
            num_projections=L,
            dtype=torch.float64,
            device=device,
        )  # shape (L, D)

        self.alpha = None
        self.pretrain_time = 0.0

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1).to(dtype=torch.float64, device=self.device)

    def _compute_sliced_potentials(self, x0_flat: torch.Tensor,
                                   x1_flat: torch.Tensor) -> torch.Tensor:
        B = x0_flat.shape[0]
        # NOTE: emd1D_dual is numpy/scipy-based → requires CPU tensors.
        # Matrix multiply on GPU (fast), then move to CPU before emd1D_dual.
        proj_x0 = (x0_flat @ self.proj_dirs.T).T.cpu()   # (L, B) on CPU
        proj_x1 = (x1_flat @ self.proj_dirs.T).T.cpu()   # (L, B) on CPU

        a = torch.full((B,), 1.0 / B, dtype=torch.float64, device="cpu")
        b = torch.full((B,), 1.0 / B, dtype=torch.float64, device="cpu")

        f_grad, _, _ = emd1D_dual(
            proj_x0, proj_x1,
            u_weights=a, v_weights=b,
            p=2, require_sort=True,
        )  # f_grad: (L, B) on CPU

        Phi = f_grad.T                                  # (B, L) on CPU
        Phi = Phi - Phi.mean(dim=0, keepdim=True)
        return Phi   # always on CPU

    @staticmethod
    def _g_from_f(f: torch.Tensor, b: torch.Tensor,
                  log_K: torch.Tensor, eps: float) -> torch.Tensor:
        log_b = torch.log(b.clamp(1e-300))
        M = log_K + f.unsqueeze(1) / eps           # (B, B)
        m = M.max(dim=0, keepdim=True).values
        lse = (M - m).exp().sum(dim=0).log() + m.squeeze(0)
        return eps * (log_b - lse)

    @staticmethod
    def _f_from_g(g: torch.Tensor, a: torch.Tensor,
                  log_K: torch.Tensor, eps: float) -> torch.Tensor:
        log_a = torch.log(a.clamp(1e-300))
        M = log_K + g.unsqueeze(0) / eps           # (B, B)
        m = M.max(dim=1, keepdim=True).values
        lse = (M - m).exp().sum(dim=1).log() + m.squeeze(1)
        return eps * (log_a - lse)

    def _dual_objective(self, a: torch.Tensor, b: torch.Tensor,
                        f: torch.Tensor, log_K: torch.Tensor,
                        eps: float) -> torch.Tensor:
        g = self._g_from_f(f, b, log_K, eps)

        M_fa = log_K + g.unsqueeze(0) / eps
        m = M_fa.max(dim=1, keepdim=True).values
        fa = eps * ((M_fa - m).exp().sum(1).log() + m.squeeze(1))

        M_gb = log_K + f.unsqueeze(1) / eps
        m = M_gb.max(dim=0, keepdim=True).values
        gb = eps * ((M_gb - m).exp().sum(0).log() + m.squeeze(0))

        div_a = (a * (f - fa)).sum()
        div_b = (b * (g - gb)).sum()

        log_P = f.unsqueeze(1) / eps + g.unsqueeze(0) / eps + log_K
        lp_max = log_P.detach().max()
        total_sum = (log_P - lp_max).exp().sum() * lp_max.exp()

        return div_a + div_b + eps * (1.0 - total_sum)

    def pretrain(self, source_sampler, target_sampler,
                 M: int = 50, B: int = 128, T: int = 5000):
        print(f"[OA-OT CIFAR] Pre-training  M={M}  B={B}  T={T}  "
              f"L={self.L}  adaptive_eps=median(C)/log(B)  dim={self.dim}")
        dev = self.device

        # ---- collect training pool ----
        pool_Phi, pool_a, pool_b, pool_logK, pool_eps = [], [], [], [], []

        for _ in tqdm(range(M), desc="OA-OT collect"):
            x1 = target_sampler(B)
            x0 = source_sampler(x1)

            x0_flat = self._flatten(x0)
            x1_flat = self._flatten(x1)

            Phi = self._compute_sliced_potentials(x0_flat, x1_flat)  # (B, L) on CPU
            C = torch.cdist(x0_flat, x1_flat).pow(2)                  # (B, B) on device

            c_med = float(torch.median(C).item())
            eps_i = c_med / np.log(B)

            pool_Phi.append(Phi.to(device=dev))          
            pool_logK.append((-C / eps_i).to(dev))       
            pool_eps.append(eps_i)
            pool_a.append(torch.full((B,), 1.0 / B, dtype=torch.float64, device=dev))
            pool_b.append(torch.full((B,), 1.0 / B, dtype=torch.float64, device=dev))

        # ---- optimise alpha ----
        alpha = nn.Parameter(torch.zeros(self.L, dtype=torch.float64, device=dev))
        opt = torch.optim.Adam([alpha], lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T, eta_min=self.lr * 0.01)

        rng = np.random.default_rng(42)
        t0 = time.time()
        pbar = tqdm(total=T, desc="OA-OT optimise")
        loss_ema = None

        for step in range(T):
            idx = int(rng.integers(0, M))
            f_pred = pool_Phi[idx] @ alpha                  # (B,)
            loss = -self._dual_objective(
                pool_a[idx], pool_b[idx],
                f_pred, pool_logK[idx], pool_eps[idx],      # per-batch adaptive eps
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
                pbar.set_description(
                    f"OA-OT  dual={-lv:.4e}  ema={loss_ema:.4e}")
        pbar.close()

        self.alpha = alpha.detach().cpu().numpy()
        print(f"[OA-OT CIFAR]   |  "
              f"alpha norm={np.linalg.norm(self.alpha):.6f}")
        self.pretrain_time = time.time() - t0
        print(f"[OA-OT CIFAR] Pre-training total: {self.pretrain_time:.2f}s")
        return self.alpha

    def predict_plan(self, x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
        assert self.alpha is not None, "Call pretrain() first."
        B = x0.shape[0]

        x0_flat = self._flatten(x0)
        x1_flat = self._flatten(x1)

        # Amortised prediction (Phi on CPU)
        Phi = self._compute_sliced_potentials(x0_flat, x1_flat).numpy()
        f = Phi @ self.alpha
        f = f - f.mean()

        C = torch.cdist(x0_flat, x1_flat).pow(2).cpu().numpy()

        # Adaptive eps: same formula as pretrain
        c_med = float(np.median(C))
        eps = c_med / np.log(B)

        log_K = -C / eps
        log_a = np.log(1.0 / B)
        log_b = np.log(1.0 / B)
        log_f = f / eps

        # 1 Sinkhorn refinement step
        M_f = log_f[:, None] + log_K
        m = M_f.max(axis=0, keepdims=True)
        log_g = log_b - (np.log(np.exp(M_f - m).sum(axis=0)) + m.squeeze())
        M_g = log_g[None, :] + log_K
        m = M_g.max(axis=1, keepdims=True)
        log_f = log_a - (np.log(np.exp(M_g - m).sum(axis=1)) + m.squeeze())

        log_P = log_f[:, None] + log_g[None, :] + log_K
        log_P -= log_P.max()
        P = np.clip(np.exp(log_P), 0.0, None)
        P /= P.sum(axis=1, keepdims=True) + 1e-30
        P /= P.sum(axis=0, keepdims=True) + 1e-30
        return P

    def sample_pairs(self, x0: torch.Tensor,
                     x1: torch.Tensor):
        B = x0.shape[0]
        P = self.predict_plan(x0, x1)
        P_flat = P.ravel()
        P_flat = P_flat / (P_flat.sum() + 1e-30)
        idx = np.random.choice(B * B, size=B, p=P_flat, replace=True)
        return x0[idx // B], x1[idx % B]
