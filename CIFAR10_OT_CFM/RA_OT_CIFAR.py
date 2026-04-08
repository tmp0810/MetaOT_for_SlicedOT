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
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    H = X.T @ X
    if ridge > 0:
        H = H + ridge * np.eye(H.shape[0])
    return np.linalg.solve(H, X.T @ y)


class AmortizedRA_OT_CIFAR:
    def __init__(self, L: int = 100, eps: float = 800.0,
                 ridge: float = 1e-3, device: str = "cpu"):
        self.L = L
        self.eps = eps
        self.ridge = ridge
        self.device = device
        self.dim = 3 * 32 * 32          # CIFAR-10 flattened dimension

        self.proj_dirs = generate_uniform_unit_sphere_projections(
            dim=self.dim,
            num_projections=L,
            dtype=torch.float64,
            device=device,
        )  # shape (L, D)

        self.alpha = None
        self.pretrain_time = 0.0
                     
    def _flatten_f32(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, D) float32 on self.device (fast inference)."""
        return x.reshape(x.shape[0], -1).to(dtype=torch.float32, device=self.device)

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, D) in float64 (kept for pretrain)."""
        return x.reshape(x.shape[0], -1).to(dtype=torch.float64, device=self.device)

    def _compute_sliced_potentials(self, x0_flat: torch.Tensor,
                                   x1_flat: torch.Tensor) -> torch.Tensor:
        """Compute sliced OT dual potentials Phi (B, L).
        x0_flat / x1_flat: float64 on self.device (used during pretrain).
        emd1D_dual is CPU-only → we move projected coords to CPU first.
        """
        B = x0_flat.shape[0]

        proj_x0 = (x0_flat @ self.proj_dirs.T).T.cpu()   # (L, B) on CPU
        proj_x1 = (x1_flat @ self.proj_dirs.T).T.cpu()   # (L, B) on CPU

        a = torch.full((B,), 1.0 / B, dtype=torch.float64, device="cpu")
        b = torch.full((B,), 1.0 / B, dtype=torch.float64, device="cpu")

        f_grad, _, _ = emd1D_dual(
            proj_x0, proj_x1,
            u_weights=a, v_weights=b,
            p=2, require_sort=True,
        )  

        Phi = f_grad.T                                          
        Phi = Phi - Phi.mean(dim=0, keepdim=True)              
        return Phi   

    def _compute_sliced_potentials_f32(self, x0_flat: torch.Tensor,
                                       x1_flat: torch.Tensor) -> torch.Tensor:
        B   = x0_flat.shape[0]
        dev = x0_flat.device
        # Cast proj_dirs to float32 for fast GPU matmul (x0_flat is float32).
        # Then upcast the projected coords to float64 only for emd1D_dual.
        proj_f32 = self.proj_dirs.to(dtype=torch.float32, device=dev)       # (L, D)
        proj_x0 = (x0_flat @ proj_f32.T).T.cpu().double()   # (L, B) CPU f64
        proj_x1 = (x1_flat @ proj_f32.T).T.cpu().double()   # (L, B) CPU f64

        uni = torch.full((B,), 1.0 / B, dtype=torch.float64)
        f_grad, _, _ = emd1D_dual(
            proj_x0, proj_x1,
            u_weights=uni, v_weights=uni,
            p=2, require_sort=True,
        )  # f_grad: (L, B) CPU f64

        Phi = f_grad.T.float()                           # (B, L) float32 CPU
        Phi = Phi - Phi.mean(dim=0, keepdim=True)
        return Phi   # (B, L) float32, CPU

    def _sinkhorn_potential(self, x0_flat: torch.Tensor,
                            x1_flat: torch.Tensor):
        B   = x0_flat.shape[0]
        eps = self.eps                                   # fixed eps
        C   = torch.cdist(x0_flat, x1_flat).pow(2).cpu().numpy()

        a, b = ot.unif(B), ot.unif(B)
        _, log_d = ot.sinkhorn(
            a, b, C, reg=eps,
            numItermax=1000, stopThr=1e-9, log=True,
        )
        if "log_u" in log_d:
            f = eps * log_d["log_u"]
        else:
            u = log_d.get("u", np.ones(B))
            f = eps * np.log(np.clip(u, 1e-50, None))
        f = f - f.mean()
        return f, eps

    def pretrain(self, source_sampler, target_sampler, M: int = 50, B: int = 128):
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
              f"alpha norm={np.linalg.norm(self.alpha):.6f}")

        self.pretrain_time = time.time() - t0
        print(f"[RA-OT CIFAR] Pre-training total: {self.pretrain_time:.2f}s")
        return self.alpha

    def predict_plan(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        assert self.alpha is not None, "Call pretrain() first."
        B   = x0.shape[0]
        dev = x0.device

        # ── 1. Flatten to float32 on device ────────────────────────────────
        x0_flat = x0.reshape(B, -1).to(dtype=torch.float32, device=dev)  # (B, D)
        x1_flat = x1.reshape(B, -1).to(dtype=torch.float32, device=dev)  # (B, D)

        # ── 2. Sliced potentials (only emd1D_dual runs on CPU / float64) ───
        # proj_dirs kept in original dtype (float64); cast to float32 for matmul
        proj_f32 = self.proj_dirs.to(dtype=torch.float32, device=dev)     # (L, D)
        Phi = self._compute_sliced_potentials_f32(x0_flat, x1_flat)       # (B, L) CPU f32

        # ── 3. Amortised f prediction (fast linear map) ─────────────────────
        alpha_t = torch.tensor(self.alpha, dtype=torch.float32)           # (L,) CPU
        f = (Phi @ alpha_t).to(device=dev)                                # (B,) GPU f32
        f = f - f.mean()

        # ── 4. Cost matrix on GPU float32 ────────────────────────────────
        C   = torch.cdist(x0_flat, x1_flat).pow(2)  # (B, B) GPU float32
        eps = self.eps                               # fixed eps (no median needed)

        # ── 5. Sinkhorn refinement — all torch ops on GPU ───────────────────
        log_K   = -C / eps                           # (B, B)
        log_uni = float(np.log(1.0 / B))

        # Forward pass: f → g
        M_f   = f.unsqueeze(1) / eps + log_K        # (B, B)
        m     = M_f.max(dim=0, keepdim=True).values
        log_g = log_uni - ((M_f - m).exp().sum(dim=0).log() + m.squeeze(0))

        # Backward pass: g → f
        M_g   = log_g.unsqueeze(0) + log_K          # (B, B)
        m     = M_g.max(dim=1, keepdim=True).values
        log_f = log_uni - ((M_g - m).exp().sum(dim=1).log() + m.squeeze(1))

        # ── 6. Build plan (stay on GPU) ────────────────────────────────────
        log_P = log_f.unsqueeze(1) + log_g.unsqueeze(0) + log_K  # (B, B)
        log_P = log_P - log_P.max()
        P = log_P.exp().clamp(min=0.0)
        P = P / (P.sum(dim=1, keepdim=True) + 1e-30)
        P = P / (P.sum(dim=0, keepdim=True) + 1e-30)

        return P   # (B, B) float32 on GPU — no .numpy() copy!

    def sample_pairs(self, x0: torch.Tensor, x1: torch.Tensor,
                     cpu_ot: bool = False):
        B   = x0.shape[0]
        dev = x0.device

        if cpu_ot:
            P      = self.predict_plan_cpu(x0, x1)   # (B, B) CPU float32
            P_flat = P.reshape(-1)
            P_flat = (P_flat / (P_flat.sum() + 1e-30)).float()
            idx    = torch.multinomial(P_flat, num_samples=B, replacement=True)
            x0c, x1c = x0.cpu(), x1.cpu()
            return x0c[idx // B].to(dev), x1c[idx % B].to(dev)

        # ── default: GPU path (original behaviour) ────────────────────────
        P      = self.predict_plan(x0, x1)            # (B, B) GPU float32
        P_flat = P.reshape(-1)                         # (B*B,)
        P_flat = (P_flat / (P_flat.sum() + 1e-30)).float()
        idx = torch.multinomial(P_flat, num_samples=B, replacement=True)
        return x0[idx // B], x1[idx % B]

    def predict_plan_cpu(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]

        # ── 1. Flatten on CPU float32 ──────────────────────────────────────
        x0_flat = x0.reshape(B, -1).float().cpu()    # (B, D) CPU f32
        x1_flat = x1.reshape(B, -1).float().cpu()    # (B, D) CPU f32

        # ── 2. Sliced potentials — emd1D_dual is CPU-native, no transfer ──
        proj_cpu = self.proj_dirs.float().cpu()       # (L, D) CPU f32
        proj_x0  = (x0_flat @ proj_cpu.T).T.double() # (L, B) CPU f64
        proj_x1  = (x1_flat @ proj_cpu.T).T.double() # (L, B) CPU f64

        uni = torch.full((B,), 1.0 / B, dtype=torch.float64)
        f_grad, _, _ = emd1D_dual(
            proj_x0, proj_x1,
            u_weights=uni, v_weights=uni,
            p=2, require_sort=True,
        )  # (L, B) CPU f64
        Phi = f_grad.T.float()                        # (B, L) CPU f32
        Phi = Phi - Phi.mean(dim=0, keepdim=True)

        # ── 3. Amortised f prediction ──────────────────────────────────────
        alpha_t = torch.tensor(self.alpha, dtype=torch.float32)   # (L,) CPU
        f = Phi @ alpha_t                             # (B,) CPU f32
        f = f - f.mean()

        # ── 4. Cost matrix on CPU ──────────────────────────────────────────
        C   = torch.cdist(x0_flat, x1_flat).pow(2)   # (B, B) CPU f32
        eps = self.eps

        # ── 5. Sinkhorn (1 forward + 1 backward, CPU torch ops) ───────────
        log_K   = -C / eps
        log_uni = float(np.log(1.0 / B))

        M_f   = f.unsqueeze(1) / eps + log_K
        m     = M_f.max(dim=0, keepdim=True).values
        log_g = log_uni - ((M_f - m).exp().sum(dim=0).log() + m.squeeze(0))

        M_g   = log_g.unsqueeze(0) + log_K
        m     = M_g.max(dim=1, keepdim=True).values
        log_f = log_uni - ((M_g - m).exp().sum(dim=1).log() + m.squeeze(1))

        # ── 6. Build plan (CPU) ────────────────────────────────────────────
        log_P = log_f.unsqueeze(1) + log_g.unsqueeze(0) + log_K
        log_P = log_P - log_P.max()
        P = log_P.exp().clamp(min=0.)
        P = P / (P.sum(dim=1, keepdim=True) + 1e-30)
        P = P / (P.sum(dim=0, keepdim=True) + 1e-30)

        return P   # (B, B) CPU float32 — no GPU memory used!
