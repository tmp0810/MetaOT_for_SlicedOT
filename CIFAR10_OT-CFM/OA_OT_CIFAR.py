"""
OA-OT (Objective-Amortized Optimal Transport) for CIFAR-10.

Phase 1 — Pre-training:
    For M mini-batches of (x0~N(0,I), x1~CIFAR-10):
        1. Flatten images to (B, D) with D = 3*32*32 = 3072.
        2. Compute sliced Wasserstein feature matrix Phi (B, L).
        3. Collect pool of (Phi, a, b, log_K) triplets.
    Then optimise a shared linear weight alpha over T gradient steps by
    *directly maximising* the Sinkhorn dual objective:
        L(alpha) = <a, f> + <b, g(f)> - eps*(sum_P - 1)
    where f = Phi @ alpha and g is derived via the Sinkhorn C-transform.
    No Sinkhorn labels are needed — fully self-supervised.

Phase 2 — Fast inference (identical to RA-OT):
    f_pred = Phi @ alpha  ->  1 Sinkhorn step  ->  plan P  ->  sample pairs.
"""

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
    """
    OA-OT amortized solver for CIFAR-10 via dual-objective optimisation.

    Parameters
    ----------
    L   : int   — number of random 1-D projection directions.
    eps : float — Sinkhorn regularisation (entropic OT parameter).
    lr  : float — Adam learning rate for alpha.
    device : str — torch device.
    """

    def __init__(self, L: int = 100, eps: float = 0.1,
                 lr: float = 1e-3, device: str = "cpu"):
        self.L = L
        self.eps = eps
        self.lr = lr
        self.device = device
        self.dim = 3 * 32 * 32          # CIFAR-10 flattened dimension

        # Fixed random projection directions on the unit sphere in R^{dim}
        self.proj_dirs = generate_uniform_unit_sphere_projections(
            dim=self.dim,
            num_projections=L,
            dtype=torch.float64,
            device=device,
        )  # shape (L, D)

        self.alpha = None
        self.pretrain_time = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, D) in float64 on self.device."""
        return x.reshape(x.shape[0], -1).to(dtype=torch.float64, device=self.device)

    def _compute_sliced_potentials(self, x0_flat: torch.Tensor,
                                   x1_flat: torch.Tensor) -> torch.Tensor:
        """
        Compute mean-centred sliced Wasserstein feature matrix Phi (B, L).
        """
        B = x0_flat.shape[0]
        proj_x0 = (x0_flat @ self.proj_dirs.T).T   # (L, B)
        proj_x1 = (x1_flat @ self.proj_dirs.T).T   # (L, B)

        a = torch.full((B,), 1.0 / B, dtype=torch.float64, device=self.device)
        b = torch.full((B,), 1.0 / B, dtype=torch.float64, device=self.device)

        f_grad, _, _ = emd1D_dual(
            proj_x0, proj_x1,
            u_weights=a, v_weights=b,
            p=2, require_sort=True,
        )  # f_grad: (L, B)

        Phi = f_grad.T                                  # (B, L)
        Phi = Phi - Phi.mean(dim=0, keepdim=True)
        return Phi

    # ------------------------------------------------------------------
    # Sinkhorn C-transforms  (log-domain, numerically stable)
    # ------------------------------------------------------------------

    @staticmethod
    def _g_from_f(f: torch.Tensor, b: torch.Tensor,
                  log_K: torch.Tensor, eps: float) -> torch.Tensor:
        """Compute g = C-transform of f: g_j = eps*(log b_j - LSE_i(f_i/eps + log K_ij))"""
        log_b = torch.log(b.clamp(1e-300))
        M = log_K + f.unsqueeze(1) / eps           # (B, B)
        m = M.max(dim=0, keepdim=True).values
        lse = (M - m).exp().sum(dim=0).log() + m.squeeze(0)
        return eps * (log_b - lse)

    @staticmethod
    def _f_from_g(g: torch.Tensor, a: torch.Tensor,
                  log_K: torch.Tensor, eps: float) -> torch.Tensor:
        """Compute f = C-transform of g."""
        log_a = torch.log(a.clamp(1e-300))
        M = log_K + g.unsqueeze(0) / eps           # (B, B)
        m = M.max(dim=1, keepdim=True).values
        lse = (M - m).exp().sum(dim=1).log() + m.squeeze(1)
        return eps * (log_a - lse)

    def _dual_objective(self, a: torch.Tensor, b: torch.Tensor,
                        f: torch.Tensor, log_K: torch.Tensor,
                        eps: float) -> torch.Tensor:
        """
        Sinkhorn KL dual objective (to be maximised):
            D(f) = <a, f-fa> + <b, g-gb> + eps*(1 - sum P)
        where fa = C-transform of g(f), gb = C-transform of f.
        """
        g = self._g_from_f(f, b, log_K, eps)

        # Compute fa  (C-transform of g back to x0 side)
        M_fa = log_K + g.unsqueeze(0) / eps
        m = M_fa.max(dim=1, keepdim=True).values
        fa = eps * ((M_fa - m).exp().sum(1).log() + m.squeeze(1))

        # Compute gb  (C-transform of f to x1 side)
        M_gb = log_K + f.unsqueeze(1) / eps
        m = M_gb.max(dim=0, keepdim=True).values
        gb = eps * ((M_gb - m).exp().sum(0).log() + m.squeeze(0))

        div_a = (a * (f - fa)).sum()
        div_b = (b * (g - gb)).sum()

        # Penalty term (keep sum of P close to 1)
        log_P = f.unsqueeze(1) / eps + g.unsqueeze(0) / eps + log_K
        lp_max = log_P.detach().max()
        total_sum = (log_P - lp_max).exp().sum() * lp_max.exp()

        return div_a + div_b + eps * (1.0 - total_sum)

    # ------------------------------------------------------------------
    # Phase 1: pre-training
    # ------------------------------------------------------------------

    def pretrain(self, source_sampler, target_sampler,
                 M: int = 50, B: int = 128, T: int = 5000):
        """
        Collect M mini-batches and optimise alpha via the OT dual objective.

        Parameters
        ----------
        source_sampler : callable(x1) -> Tensor (B, 3, 32, 32)
            Returns Gaussian noise shaped like x1  (e.g. torch.randn_like).
        target_sampler : callable(B) -> Tensor (B, 3, 32, 32)
            Returns real CIFAR-10 images (normalised).
        M  : int — pool size (mini-batches).
        B  : int — mini-batch size.
        T  : int — gradient optimisation steps.
        """
        print(f"[OA-OT CIFAR] Pre-training  M={M}  B={B}  T={T}  "
              f"L={self.L}  eps={self.eps}  dim={self.dim}")
        dev = self.device

        # ---- collect training pool ----
        pool_Phi, pool_a, pool_b, pool_logK = [], [], [], []

        for _ in tqdm(range(M), desc="OA-OT collect"):
            x1 = target_sampler(B)
            x0 = source_sampler(x1)

            x0_flat = self._flatten(x0)
            x1_flat = self._flatten(x1)

            Phi = self._compute_sliced_potentials(x0_flat, x1_flat)  # (B, L)
            C = torch.cdist(x0_flat, x1_flat).pow(2)                  # (B, B)

            pool_Phi.append(Phi)
            pool_logK.append(-C / self.eps)
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
                f_pred, pool_logK[idx], self.eps,
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
        self.pretrain_time = time.time() - t0
        print(f"[OA-OT CIFAR] Pre-training total: {self.pretrain_time:.2f}s")
        return self.alpha

    # ------------------------------------------------------------------
    # Phase 2: fast inference
    # ------------------------------------------------------------------

    def predict_plan(self, x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
        """
        Predict the OT transport plan for a mini-batch.

        Parameters
        ----------
        x0 : Tensor (B, 3, 32, 32)
        x1 : Tensor (B, 3, 32, 32)

        Returns
        -------
        P  : np.ndarray (B, B)
        """
        assert self.alpha is not None, "Call pretrain() first."
        B = x0.shape[0]

        x0_flat = self._flatten(x0)
        x1_flat = self._flatten(x1)

        # Amortised prediction
        Phi = self._compute_sliced_potentials(x0_flat, x1_flat).cpu().numpy()
        f = Phi @ self.alpha
        f = f - f.mean()

        C = torch.cdist(x0_flat, x1_flat).pow(2).cpu().numpy()
        log_K = -C / self.eps
        log_a = np.log(1.0 / B)
        log_b = np.log(1.0 / B)
        log_f = f / self.eps

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
        """
        Sample B paired (x0_i, x1_j) from the predicted OT plan.

        Returns tensors with the *original* shape (B, 3, 32, 32).
        """
        B = x0.shape[0]
        P = self.predict_plan(x0, x1)
        P_flat = P.ravel()
        P_flat = P_flat / (P_flat.sum() + 1e-30)
        idx = np.random.choice(B * B, size=B, p=P_flat, replace=True)
        return x0[idx // B], x1[idx % B]
