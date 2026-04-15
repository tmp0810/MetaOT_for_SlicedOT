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
    def __init__(self, L: int = 100, eps: float = 800.0,
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

        self.alpha = None           # np.ndarray (L,) float32 after pretrain
        self.pretrain_time = 0.0


    def _flatten_f32(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1).to(dtype=torch.float32, device=self.device)

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1).to(dtype=torch.float64, device=self.device)

    def _compute_sliced_potentials(self, x0_flat: torch.Tensor,
                                   x1_flat: torch.Tensor) -> torch.Tensor:
        B = x0_flat.shape[0]
        proj_x0 = (x0_flat @ self.proj_dirs.T).T.cpu()   # (L, B) CPU f64
        proj_x1 = (x1_flat @ self.proj_dirs.T).T.cpu()   # (L, B) CPU f64

        uni = torch.full((B,), 1.0 / B, dtype=torch.float64)
        f_grad, _, _ = emd1D_dual(
            proj_x0, proj_x1,
            u_weights=uni, v_weights=uni,
            p=2, require_sort=True,
        ) 

        Phi = f_grad.T                                    # (B, L) CPU f64
        Phi = Phi - Phi.mean(dim=0, keepdim=True)
        return Phi

    def _compute_sliced_potentials_f32(self, x0_flat: torch.Tensor,
                                       x1_flat: torch.Tensor) -> torch.Tensor:
        B   = x0_flat.shape[0]
        dev = x0_flat.device
        proj_f32 = self.proj_dirs.to(dtype=torch.float32, device=dev)  # (L, D)
        proj_x0  = (x0_flat @ proj_f32.T).T.cpu().double()  # (L, B) CPU f64
        proj_x1  = (x1_flat @ proj_f32.T).T.cpu().double()  # (L, B) CPU f64

        uni = torch.full((B,), 1.0 / B, dtype=torch.float64)
        f_grad, _, _ = emd1D_dual(
            proj_x0, proj_x1,
            u_weights=uni, v_weights=uni,
            p=2, require_sort=True,
        )  

        Phi = f_grad.T.float()                            
        Phi = Phi - Phi.mean(dim=0, keepdim=True)
        return Phi 
                                           
    @staticmethod
    def _g_from_f(f: torch.Tensor, b: torch.Tensor,
                  log_K: torch.Tensor, eps: float) -> torch.Tensor:
        log_b = torch.log(b.clamp(1e-38))
        M = log_K + f.unsqueeze(1) / eps           # (B, B)
        m = M.max(dim=0, keepdim=True).values
        lse = (M - m).exp().sum(dim=0).log() + m.squeeze(0)
        return eps * (log_b - lse)

    @staticmethod
    def _f_from_g(g: torch.Tensor, a: torch.Tensor,
                  log_K: torch.Tensor, eps: float) -> torch.Tensor:
        log_a = torch.log(a.clamp(1e-38))
        M = log_K + g.unsqueeze(0) / eps           # (B, B)
        m = M.max(dim=1, keepdim=True).values
        lse = (M - m).exp().sum(dim=1).log() + m.squeeze(1)
        return eps * (log_a - lse)

    def _dual_objective(self, a: torch.Tensor, b: torch.Tensor,
                        f: torch.Tensor, log_K: torch.Tensor,
                        eps: float) -> torch.Tensor:
        g = self._g_from_f(f, b, log_K, eps)

        M_fa = log_K + g.unsqueeze(0) / eps
        m    = M_fa.max(dim=1, keepdim=True).values
        fa   = eps * ((M_fa - m).exp().sum(1).log() + m.squeeze(1))

        M_gb = log_K + f.unsqueeze(1) / eps
        m    = M_gb.max(dim=0, keepdim=True).values
        gb   = eps * ((M_gb - m).exp().sum(0).log() + m.squeeze(0))

        div_a = (a * (f - fa)).sum()
        div_b = (b * (g - gb)).sum()

        log_P    = f.unsqueeze(1) / eps + g.unsqueeze(0) / eps + log_K
        lp_max   = log_P.detach().max()
        total_sum = (log_P - lp_max).exp().sum() * lp_max.exp()

        return div_a + div_b + eps * (1.0 - total_sum)

    def pretrain(self, source_sampler, target_sampler,
                 M: int = 50, B: int = 128, T: int = 5000):
        print(f"[OA-OT CIFAR] Pre-training  M={M}  B={B}  T={T}  "
              f"L={self.L}  adaptive_eps=median(C)/log(B)  dim={self.dim}")
        dev = self.device

        pool_Phi, pool_a, pool_b, pool_logK, pool_eps = [], [], [], [], []

        uni_f32 = torch.full((B,), 1.0 / B, dtype=torch.float32, device=dev)

        for _ in tqdm(range(M), desc="OA-OT collect"):
            x1 = target_sampler(B)
            x0 = source_sampler(x1)

            # float32 flatten on device
            x0_flat = x0.reshape(B, -1).to(dtype=torch.float32, device=dev)
            x1_flat = x1.reshape(B, -1).to(dtype=torch.float32, device=dev)

            # sliced potentials — GPU f32 matmul, CPU f64 emd1D_dual
            Phi = self._compute_sliced_potentials_f32(x0_flat, x1_flat)  # (B,L) CPU f32

            # cost matrix on GPU float32
            C     = torch.cdist(x0_flat, x1_flat).pow(2)  # (B, B) GPU f32
            eps_i = self.eps                               # fixed eps

            pool_Phi.append(Phi.to(device=dev))                        # (B,L) GPU f32
            pool_logK.append((-C / eps_i))                             # (B,B) GPU f32
            pool_eps.append(eps_i)
            pool_a.append(uni_f32.clone())
            pool_b.append(uni_f32.clone())

        alpha = nn.Parameter(torch.zeros(self.L, dtype=torch.float32, device=dev))
        opt   = torch.optim.Adam([alpha], lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T, eta_min=self.lr * 0.01)

        rng      = np.random.default_rng(42)
        t0       = time.time()
        pbar     = tqdm(total=T, desc="OA-OT optimise")
        loss_ema = None

        for step in range(T):
            idx    = int(rng.integers(0, M))
            f_pred = pool_Phi[idx] @ alpha                  # (B,) GPU f32
            loss   = -self._dual_objective(
                pool_a[idx], pool_b[idx],
                f_pred, pool_logK[idx], pool_eps[idx],
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([alpha], 1.0)
            opt.step()
            sched.step()

            lv       = loss.item()
            loss_ema = lv if loss_ema is None else 0.95 * loss_ema + 0.05 * lv
            pbar.update(1)
            if (step + 1) % 1000 == 0:
                pbar.set_description(
                    f"OA-OT  dual={-lv:.4e}  ema={loss_ema:.4e}")
        pbar.close()

        self.alpha = alpha.detach().cpu().numpy().astype(np.float32)
        print(f"[OA-OT CIFAR]  alpha norm={np.linalg.norm(self.alpha):.6f}")
        self.pretrain_time = time.time() - t0
        print(f"[OA-OT CIFAR] Pre-training total: {self.pretrain_time:.2f}s")
        return self.alpha

    def predict_plan(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        assert self.alpha is not None, "Call pretrain() first."
        B   = x0.shape[0]
        dev = x0.device

        x0_flat = x0.reshape(B, -1).to(dtype=torch.float32, device=dev)  # (B, D)
        x1_flat = x1.reshape(B, -1).to(dtype=torch.float32, device=dev)  # (B, D)
        Phi = self._compute_sliced_potentials_f32(x0_flat, x1_flat)       
        alpha_t = torch.tensor(self.alpha, dtype=torch.float32)           
        f = (Phi @ alpha_t).to(device=dev)                                
        f = f - f.mean()
        C     = torch.cdist(x0_flat, x1_flat).pow(2)  
        eps = self.eps

        log_K   = -C / eps                             # (B, B)
        log_uni = float(np.log(1.0 / B))

        M_f   = f.unsqueeze(1) / eps + log_K          # (B, B)
        m     = M_f.max(dim=0, keepdim=True).values
        log_g = log_uni - ((M_f - m).exp().sum(dim=0).log() + m.squeeze(0))

        M_g   = log_g.unsqueeze(0) + log_K            # (B, B)
        m     = M_g.max(dim=1, keepdim=True).values
        log_f = log_uni - ((M_g - m).exp().sum(dim=1).log() + m.squeeze(1))

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

        P      = self.predict_plan(x0, x1)            # (B, B) GPU float32
        P_flat = P.reshape(-1)                         # (B*B,)
        P_flat = (P_flat / (P_flat.sum() + 1e-30)).float()
        idx = torch.multinomial(P_flat, num_samples=B, replacement=True)
        return x0[idx // B], x1[idx % B]

    def predict_plan_cpu(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]

        x0_flat = x0.reshape(B, -1).float().cpu()    
        x1_flat = x1.reshape(B, -1).float().cpu()    

        proj_cpu = self.proj_dirs.float().cpu()     
        proj_x0  = (x0_flat @ proj_cpu.T).T.double() 
        proj_x1  = (x1_flat @ proj_cpu.T).T.double() 

        uni = torch.full((B,), 1.0 / B, dtype=torch.float64)
        f_grad, _, _ = emd1D_dual(
            proj_x0, proj_x1,
            u_weights=uni, v_weights=uni,
            p=2, require_sort=True,
        )  
        Phi = f_grad.T.float()                      
        Phi = Phi - Phi.mean(dim=0, keepdim=True)

        alpha_t = torch.tensor(self.alpha, dtype=torch.float32)  
        f = (Phi @ alpha_t)                        
        f = f - f.mean()

        C   = torch.cdist(x0_flat, x1_flat).pow(2) 
        eps = self.eps

        log_K   = -C / eps
        log_uni = float(np.log(1.0 / B))

        M_f   = f.unsqueeze(1) / eps + log_K
        m     = M_f.max(dim=0, keepdim=True).values
        log_g = log_uni - ((M_f - m).exp().sum(dim=0).log() + m.squeeze(0))

        M_g   = log_g.unsqueeze(0) + log_K
        m     = M_g.max(dim=1, keepdim=True).values
        log_f = log_uni - ((M_g - m).exp().sum(dim=1).log() + m.squeeze(1))

        log_P = log_f.unsqueeze(1) + log_g.unsqueeze(0) + log_K
        log_P = log_P - log_P.max()
        P = log_P.exp().clamp(min=0.)
        P = P / (P.sum(dim=1, keepdim=True) + 1e-30)
        P = P / (P.sum(dim=0, keepdim=True) + 1e-30)

        return P   
