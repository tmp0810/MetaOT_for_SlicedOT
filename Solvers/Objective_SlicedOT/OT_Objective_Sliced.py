import os
import time
import numpy as np
import torch
import torch.nn as nn
import ot
from tqdm import tqdm

from Solvers.Regression_SlicedOT.OT_Regression_Sliced import OT_Regression_Sliced
from regression_OT_utils import (
    generate_uniform_unit_sphere_projections,
    emd1D_dual,
)
from Utils import utils


class OT_Objective_Sliced(OT_Regression_Sliced):
    def __init__(self, cfg_proj, cfg_m):
        super().__init__(cfg_proj, cfg_m)
        self.name = "OT_Objective_Sliced"

    def _precompute_log_K(self) -> torch.Tensor:
        eps   = float(self.cfg_m.epsilon)
        C_t   = torch.tensor(self.C, dtype=torch.float64, device=self.device)
        return -C_t / eps   # (n, n)

    def _g_from_f(
        self,
        f:     torch.Tensor,   # (n,)
        b:     torch.Tensor,   # (n,)
        log_K: torch.Tensor,   # (n, n)
        eps:   float,
    ) -> torch.Tensor:
        log_b = torch.log(b.clamp(1e-300))            
        M     = log_K + f.unsqueeze(1) / eps        
        m     = M.max(dim=0, keepdim=True).values  
        lse   = (M - m).exp().sum(dim=0).log() + m.squeeze(0) 
        return eps * (log_b - lse)

    def _dual_obj_from_f(
        self,
        a:     torch.Tensor, 
        b:     torch.Tensor,
        f:     torch.Tensor,   
        log_K: torch.Tensor,   
        eps:   float,
    ) -> torch.Tensor:
        g = self._g_from_f(f, b, log_K, eps)             

        M_fa = log_K + g.unsqueeze(0) / eps               
        m    = M_fa.max(dim=1, keepdim=True).values
        fa   = eps * ((M_fa - m).exp().sum(1).log() + m.squeeze(1)) 

        M_gb = log_K + f.unsqueeze(1) / eps               # (n, n)
        m    = M_gb.max(dim=0, keepdim=True).values
        gb   = eps * ((M_gb - m).exp().sum(0).log() + m.squeeze(0))  # (n,)

        div_a = (a * (f  - fa)).sum()
        div_b = (b * (g  - gb)).sum()

        log_P = f.unsqueeze(1) / eps + g.unsqueeze(0) / eps + log_K
        lp_max    = log_P.detach().max()
        total_sum = (log_P - lp_max).exp().sum() * lp_max.exp()

        return div_a + div_b + eps * (1.0 - total_sum)


    def _fit(self, dataloader_train) -> np.ndarray:
        cfg    = self.cfg_m
        T      = int(cfg.num_train_iter)  
        M      = int(cfg.num_bootstrap)      
        L      = self.projection_matrix.shape[0]
        eps    = float(cfg.epsilon)
        lr     = float(getattr(cfg, "learning_rate", 1e-3))
        max_gn = float(getattr(cfg, "max_grad_norm", 1.0))
        log_iv = int(getattr(cfg, "log_interval", 100))

        self.logger.info(
            f"[OT_Objective_Sliced] Training  T={T} steps  "
            f"M={M} pair pool  L={L}  eps={eps}  lr={lr}"
        )

        log_K = self._precompute_log_K()   

        self.logger.info(f"[OT_Objective_Sliced] Collecting M={M} pair pool ...")
        pool_Phi, pool_a, pool_b = [], [], []
        collected = 0
        pbar_pool = tqdm(total=M, desc="Collecting pairs")

        for _, _, x_a, x_b in dataloader_train:
            for a_np, b_np in zip(x_a.numpy(), x_b.numpy()):
                if collected >= M:
                    break
                Phi_f, _, _ = self._compute_features_raw(a_np, b_np)
                pool_Phi.append(
                    torch.tensor(Phi_f, dtype=torch.float64, device=self.device))
                pool_a.append(
                    torch.tensor(a_np,  dtype=torch.float64, device=self.device))
                pool_b.append(
                    torch.tensor(b_np,  dtype=torch.float64, device=self.device))
                collected += 1
                pbar_pool.update(1)
            if collected >= M:
                break
        pbar_pool.close()
        self.logger.info(f"[OT_Objective_Sliced] Pool ready: {collected} pairs")

        alpha = nn.Parameter(
            torch.zeros(L, dtype=torch.float64, device=self.device))
        opt   = torch.optim.Adam([alpha], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T, eta_min=lr * 0.01)

        loss_ema = None
        t0       = time.time()
        pbar     = tqdm(total=T, desc="OT_Objective_Sliced")
        rng      = np.random.default_rng(42)

        for step in range(T):
            idx     = int(rng.integers(0, collected))
            Phi_f_t = pool_Phi[idx]  
            a_t     = pool_a[idx]  
            b_t     = pool_b[idx]

            f_pred = Phi_f_t @ alpha   # (n,)

            dual = self._dual_obj_from_f(a_t, b_t, f_pred, log_K, eps)
            loss = -dual

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([alpha], max_gn)
            opt.step()
            sched.step()

            lv       = loss.item()
            loss_ema = lv if loss_ema is None else 0.95*loss_ema + 0.05*lv
            pbar.update(1)

            if (step + 1) % log_iv == 0:
                msg = (f"[{step+1}/{T}]  "
                       f"dual={-lv:.4e}  loss_ema={loss_ema:.4e}  "
                       f"t={time.time()-t0:.1f}s")
                pbar.set_description(msg)
                self.logger.info(msg)

        pbar.close()

        alpha_np = alpha.detach().cpu().numpy()
        self.logger.info(
            f"[OT_Objective_Sliced] Done.  "
            f"alpha: min={alpha_np.min():.4f}  max={alpha_np.max():.4f}  "
            f"‖α‖={np.linalg.norm(alpha_np):.4f}"
        )
        np.save(os.path.join(self.log_sub_folder, "alpha.npy"), alpha_np)
        return alpha_np

    def _compute_features_raw(self, a: np.ndarray, b: np.ndarray):
        device = self.device
        L       = self.projection_matrix.shape[0]
        n       = len(a)

        proj_values = torch.tensor(
            (self.x_grid @ self.projection_matrix.T).T,  # (L, n)
            dtype=torch.float64, device=device)

        a_t = torch.tensor(a, dtype=torch.float64, device=device)
        b_t = torch.tensor(b, dtype=torch.float64, device=device)

        f_grad, g_grad, _ = emd1D_dual(
            proj_values, proj_values,
            u_weights=a_t, v_weights=b_t,
            p=2, require_sort=True,
        )

        Xf = f_grad.cpu().numpy().T   # (n, L)
        Xg = g_grad.cpu().numpy().T   # (n, L)

        Xf = Xf - Xf.mean(axis=0, keepdims=True)
        Xg = Xg - Xg.mean(axis=0, keepdims=True)
        return Xf, Xg, None

    def _predict_potentials(
        self,
        a:     np.ndarray,
        b:     np.ndarray,
        alpha: np.ndarray,
        beta:  np.ndarray = None,   # unused in Option A
    ):
        Xf, _, _ = self._compute_features_raw(a, b)
        f_clean  = Xf @ alpha            # (n,)

        eps   = float(self.cfg_m.epsilon)
        log_K = self._precompute_log_K()
        f_t   = torch.tensor(f_clean, dtype=torch.float64, device=self.device)
        b_t   = torch.tensor(b,       dtype=torch.float64, device=self.device)

        with torch.no_grad():
            g_t = self._g_from_f(f_t, b_t, log_K, eps)

        g_clean = g_t.cpu().numpy()
        return f_clean, g_clean


    def train(self, dataloader_train, dataloader_test):
        self.alpha = self._fit(dataloader_train)
        self.beta  = np.zeros(self.projection_matrix.shape[0])
        self._evaluate(dataloader_test, self.alpha, self.beta)
