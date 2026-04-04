import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import time
from tqdm import tqdm
import ot

from regression_OT_utils import (
    generate_uniform_unit_sphere_projections,
    emd1D_dual,
)


def _ridge_regression(X, y, ridge=0.0):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    H   = X.T @ X
    Xty = X.T @ y
    if ridge > 0:
        H = H + ridge * np.eye(H.shape[0])
    return np.linalg.solve(H, Xty)


class AmortizedRA_OT:
    def __init__(self, L=100, eps=2, ridge=1e-3, device="cpu"):
        self.L = L
        self.eps = eps
        self.ridge = ridge
        self.device = device
        self.proj_dirs = generate_uniform_unit_sphere_projections(
            dim=2, num_projections=L, dtype=torch.float64, device=device
        )
        self.alpha = None
        self.pretrain_time = 0.0

    
    def _compute_sliced_potentials(self, x0, x1):
        B = x0.shape[0]
        proj_x0 = x0 @ self.proj_dirs.T       # (B, L)
        proj_x1 = x1 @ self.proj_dirs.T       # (B, L)

        a = torch.full((B,), 1.0 / B, dtype=torch.float64, device=self.device)
        b = torch.full((B,), 1.0 / B, dtype=torch.float64, device=self.device)

        f_grad, _, _ = emd1D_dual(
            proj_x0.T, proj_x1.T,              # (L, B) each
            u_weights=a, v_weights=b,
            p=2, require_sort=True,
        )
        Phi = f_grad.T                          # (B, L)
        Phi = Phi - Phi.mean(dim=0, keepdim=True)
        return Phi

    
    def _solve_sinkhorn_potential(self, x0, x1):
        B = x0.shape[0]
        C = torch.cdist(x0, x1).pow(2).cpu().numpy()
        a, b = ot.unif(B), ot.unif(B)
        eps=2
        _, log_d = ot.sinkhorn(
            a, b, C, reg=eps,
            numItermax=1000, stopThr=1e-9, log=True,
        )
        if 'log_u' in log_d:
            f = eps * log_d['log_u']
        else:
            f = eps * np.log(np.clip(log_d.get('u', np.ones(B)), 1e-50, None))
        f = f - f.mean()
        return f, eps

    
    def pretrain(self, source_sampler, target_sampler, M=50, B=512):
        print(f"[RA-OT] Pre-training  M={M}  B={B}  L={self.L}  eps={self.eps}")
        Phi_list, y_list = [], []
        t0 = time.time()

        for _ in tqdm(range(M), desc="RA-OT pretrain"):
            x0 = source_sampler(B).to(dtype=torch.float64, device=self.device)
            x1 = target_sampler(B).to(dtype=torch.float64, device=self.device)
            Phi = self._compute_sliced_potentials(x0, x1).cpu().numpy()
            f_gt, eps_used = self._solve_sinkhorn_potential(x0, x1)
            Phi_list.append(Phi)
            y_list.append(f_gt)
            # record eps scale for inference
            if not hasattr(self, '_eps_scale'):
                self._eps_scale = eps_used

        Phi_all = np.vstack(Phi_list)
        y_all   = np.concatenate(y_list)
        self.alpha = _ridge_regression(Phi_all, y_all, self.ridge)
        self.pretrain_time = time.time() - t0
        print(f"[RA-OT] Done in {self.pretrain_time:.2f}s")
        return self.alpha

    
    def predict_plan(self, x0, x1):
        assert self.alpha is not None, "Call pretrain() first."
        B = x0.shape[0]
        x0 = x0.to(dtype=torch.float64, device=self.device)
        x1 = x1.to(dtype=torch.float64, device=self.device)

        C = torch.cdist(x0, x1).pow(2).cpu().numpy()
        eps=2

        Phi = self._compute_sliced_potentials(x0, x1).cpu().numpy()
        f = (Phi @ self.alpha)
        f = f - f.mean()
        log_K = -C / eps
        log_a = np.log(1.0 / B)
        log_b = np.log(1.0 / B)
        log_f = f / eps
        for _ in range(1):
            # g_j update
            M_f = log_f[:, None] + log_K              # (B, B)
            m   = M_f.max(axis=0, keepdims=True)
            log_g = log_b - (np.log(np.exp(M_f - m).sum(axis=0)) + m.squeeze())
            # f_i update
            M_g = log_g[None, :] + log_K              # (B, B)
            m   = M_g.max(axis=1, keepdims=True)
            log_f = log_a - (np.log(np.exp(M_g - m).sum(axis=1)) + m.squeeze())

        log_P = log_f[:, None] + log_g[None, :] + log_K
        log_P -= log_P.max()
        P = np.exp(log_P)
        P = np.clip(P, 0.0, None)
        # Normalise rows then cols once
        P /= P.sum(axis=1, keepdims=True) + 1e-30
        P /= P.sum(axis=0, keepdims=True) + 1e-30
        return P

    
    def sample_pairs(self, x0, x1):
        B = x0.shape[0]
        P = self.predict_plan(x0, x1)
        P_flat = P.ravel()
        P_flat = P_flat / (P_flat.sum() + 1e-30)
        idx = np.random.choice(B * B, size=B, p=P_flat, replace=True)
        return x0[idx // B], x1[idx % B]
