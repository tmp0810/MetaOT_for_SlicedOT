"""
RA-OT (Regression-Amortized Optimal Transport) for CIFAR-10.

Phase 1 — Pre-training:
    For M mini-batches of (x0~N(0,I), x1~CIFAR-10):
        1. Flatten images to (B, D) with D = 3*32*32 = 3072.
        2. Compute sliced Wasserstein potentials Phi (B, L) via emd1D_dual
           along L random directions on the unit sphere in R^D.
        3. Solve exact Sinkhorn to obtain the ground-truth potential f_gt (B,).
        4. Stack all (Phi, f_gt) pairs and fit a ridge regression:
               alpha = argmin_alpha ||Phi @ alpha - f_gt||^2 + lambda*||alpha||^2

Phase 2 — Fast inference (called at *every* U-Net training step):
    Given a new (x0, x1) mini-batch:
        1. Compute Phi -> f_pred = Phi @ alpha   (O(B*L) — extremely fast)
        2. Perform 1 Sinkhorn iteration to refine dual potentials (f, g).
        3. Build transport plan P = exp((f+g-C)/eps).
        4. Sample B paired indices (i,j) ~ P   => return x0[i], x1[j].

All heavy-lifting (OT on 3072-d data) is replaced by L cheap 1-D EMDs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch
import ot
from tqdm import tqdm

from regression_OT_utils import (
    generate_uniform_unit_sphere_projections,
    emd1D_dual,
)


def _ridge_regression(X, y, ridge=0.0):
    """Closed-form ridge regression: alpha = (X^T X + lam I)^{-1} X^T y."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    H = X.T @ X
    if ridge > 0:
        H = H + ridge * np.eye(H.shape[0])
    return np.linalg.solve(H, X.T @ y)


class AmortizedRA_OT_CIFAR:
    """
    RA-OT amortized solver for CIFAR-10.

    Parameters
    ----------
    L       : int   — number of random 1-D projection directions.
    eps     : float — Sinkhorn regularisation (used for both labelling & inference).
    ridge   : float — L2 penalty for ridge regression.
    device  : str   — torch device for sliced-potential computation.
    """

    def __init__(self, L: int = 100, eps: float = 0.1,
                 ridge: float = 1e-3, device: str = "cpu"):
        self.L = L
        self.eps = eps
        self.ridge = ridge
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
        """(B, C, H, W) -> (B, D) in float64."""
        return x.reshape(x.shape[0], -1).to(dtype=torch.float64, device=self.device)

    def _compute_sliced_potentials(self, x0_flat: torch.Tensor,
                                   x1_flat: torch.Tensor) -> torch.Tensor:
        """
        Compute sliced Wasserstein dual potentials as feature matrix Phi.

        Parameters
        ----------
        x0_flat, x1_flat : (B, D) float64 tensors (already on self.device)

        Returns
        -------
        Phi : (B, L) float64  — mean-centred per-direction potentials for x0.
        """
        B = x0_flat.shape[0]
        # Project: (B, D) x (D, L) -> (B, L) -> transpose -> (L, B)
        proj_x0 = (x0_flat @ self.proj_dirs.T).T   # (L, B)
        proj_x1 = (x1_flat @ self.proj_dirs.T).T   # (L, B)

        a = torch.full((B,), 1.0 / B, dtype=torch.float64, device=self.device)
        b = torch.full((B,), 1.0 / B, dtype=torch.float64, device=self.device)

        f_grad, _, _ = emd1D_dual(
            proj_x0, proj_x1,
            u_weights=a, v_weights=b,
            p=2, require_sort=True,
        )  # f_grad: (L, B)

        Phi = f_grad.T                                          # (B, L)
        Phi = Phi - Phi.mean(dim=0, keepdim=True)              # mean-centre cols
        return Phi

    def _sinkhorn_potential(self, x0_flat: torch.Tensor,
                            x1_flat: torch.Tensor):
        """
        Solve Sinkhorn on the flattened pair to get ground-truth f potential.
        Returns (f_gt: np.ndarray (B,), eps_used: float).
        """
        B = x0_flat.shape[0]
        C = torch.cdist(x0_flat, x1_flat).pow(2).cpu().numpy()
        a, b = ot.unif(B), ot.unif(B)
        _, log_d = ot.sinkhorn(
            a, b, C, reg=self.eps,
            numItermax=1000, stopThr=1e-9, log=True,
        )
        if "log_u" in log_d:
            f = self.eps * log_d["log_u"]
        else:
            u = log_d.get("u", np.ones(B))
            f = self.eps * np.log(np.clip(u, 1e-50, None))
        f = f - f.mean()
        return f, self.eps

    # ------------------------------------------------------------------
    # Phase 1: pre-training
    # ------------------------------------------------------------------

    def pretrain(self, source_sampler, target_sampler, M: int = 50, B: int = 128):
        """
        Collect M (x0, x1) mini-batches and fit ridge regression.

        Parameters
        ----------
        source_sampler : callable(B) -> Tensor (B, C, H, W) or (B, D)
            Samples from N(0, I).  For CIFAR, pass torch.randn_like(x1).
        target_sampler : callable(B) -> Tensor (B, C, H, W)
            Samples real CIFAR-10 images (normalised to [-1, 1]).
        M  : int — number of mini-batches to collect.
        B  : int — mini-batch size.
        """
        print(f"[RA-OT CIFAR] Pre-training  M={M}  B={B}  L={self.L}  "
              f"eps={self.eps}  ridge={self.ridge}  dim={self.dim}")

        Phi_list, y_list = [], []
        t0 = time.time()

        for _ in tqdm(range(M), desc="RA-OT collect+label"):
            x1 = target_sampler(B)          # (B, 3, 32, 32)
            x0 = source_sampler(x1)         # (B, 3, 32, 32)  e.g. randn_like

            x0_flat = self._flatten(x0)     # (B, D)
            x1_flat = self._flatten(x1)     # (B, D)

            Phi = self._compute_sliced_potentials(x0_flat, x1_flat).cpu().numpy()
            f_gt, _ = self._sinkhorn_potential(x0_flat, x1_flat)

            Phi_list.append(Phi)
            y_list.append(f_gt)

        Phi_all = np.vstack(Phi_list)   # (M*B, L)
        y_all = np.concatenate(y_list)  # (M*B,)

        print(f"[RA-OT CIFAR] Phi_all: {Phi_all.shape}  |  "
              f"y range: [{y_all.min():.4f}, {y_all.max():.4f}]")
        print("[RA-OT CIFAR] Solving ridge regression ...")

        t_reg = time.time()
        self.alpha = _ridge_regression(Phi_all, y_all, self.ridge)
        print(f"[RA-OT CIFAR] Ridge done in {time.time()-t_reg:.2f}s  |  "
              f"alpha norm={np.linalg.norm(self.alpha):.4f}")

        self.pretrain_time = time.time() - t0
        print(f"[RA-OT CIFAR] Pre-training total: {self.pretrain_time:.2f}s")
        return self.alpha

    # ------------------------------------------------------------------
    # Phase 2: fast inference
    # ------------------------------------------------------------------

    def predict_plan(self, x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
        """
        Predict the OT transport plan for a mini-batch.

        Parameters
        ----------
        x0 : Tensor (B, 3, 32, 32) — noise samples
        x1 : Tensor (B, 3, 32, 32) — CIFAR-10 images

        Returns
        -------
        P  : np.ndarray (B, B) — row/column-normalised transport plan.
        """
        assert self.alpha is not None, "Call pretrain() first."
        B = x0.shape[0]

        x0_flat = self._flatten(x0)     # (B, D)
        x1_flat = self._flatten(x1)     # (B, D)

        # Amortised potential prediction
        Phi = self._compute_sliced_potentials(x0_flat, x1_flat).cpu().numpy()
        f = Phi @ self.alpha            # (B,)
        f = f - f.mean()

        # Cost & log-kernel
        C = torch.cdist(x0_flat, x1_flat).pow(2).cpu().numpy()
        log_K = -C / self.eps
        log_a = np.log(1.0 / B)
        log_b = np.log(1.0 / B)
        log_f = f / self.eps

        # 1 Sinkhorn iteration (warm-start from predicted f)
        M_f = log_f[:, None] + log_K                     # (B, B)
        m = M_f.max(axis=0, keepdims=True)
        log_g = log_b - (np.log(np.exp(M_f - m).sum(axis=0)) + m.squeeze())
        M_g = log_g[None, :] + log_K                     # (B, B)
        m = M_g.max(axis=1, keepdims=True)
        log_f = log_a - (np.log(np.exp(M_g - m).sum(axis=1)) + m.squeeze())

        log_P = log_f[:, None] + log_g[None, :] + log_K
        log_P -= log_P.max()
        P = np.exp(log_P)
        P = np.clip(P, 0.0, None)
        # Normalise rows then cols once
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
