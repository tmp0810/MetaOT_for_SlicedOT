import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from Solvers.Regression_SlicedOT.OT_Regression_Sliced_World import OT_Regression_Sliced_World


class OT_Objective_Sliced_World(OT_Regression_Sliced_World):
    def __init__(self, cfg_proj, cfg_m,
                 supply_euc, demand_euc,
                 supply_sph=None, demand_sph=None):
        super().__init__(cfg_proj, cfg_m,
                         supply_euc=supply_euc,
                         demand_euc=demand_euc,
                         supply_sph=supply_sph,
                         demand_sph=demand_sph)
        self.name = "OT_Objective_Sliced_World"
        self.logger.info(
            f"[OT_Objective_Sliced_World] Method 2  "
            f"n_supply={self.n_supply}  n_demand={self.n_demand}")

    def _precompute_log_K(self) -> torch.Tensor:
        """log_K = -C_arccos / eps. FIXED (supply/demand locations fixed)."""
        eps = float(self.cfg_m.epsilon)
        C_t = torch.tensor(self.C, dtype=torch.float64, device=self.device)
        return -C_t / eps   # (n_supply, n_demand)

    def _g_from_f(self,
                  f:     torch.Tensor,   # (n_supply,)
                  b:     torch.Tensor,   # (n_demand,)
                  log_K: torch.Tensor,   # (n_supply, n_demand)
                  eps:   float) -> torch.Tensor:
        """
        One Sinkhorn step: g update given f.
        g[j] = ε·log(b[j]) - ε·lse_i(log_K[i,j] + f[i]/ε)
        """
        log_b = torch.log(b.clamp(1e-300))                      # (n_demand,)
        M     = log_K + f.unsqueeze(1) / eps                    # (n_supply, n_demand)
        m     = M.max(dim=0, keepdim=True).values               # (1, n_demand)
        lse   = (M - m).exp().sum(dim=0).log() + m.squeeze(0)  # (n_demand,)
        return eps * (log_b - lse)                               # (n_demand,)

    def _dual_obj_from_f(self,
                         a:     torch.Tensor,   # (n_supply,)
                         b:     torch.Tensor,   # (n_demand,)
                         f:     torch.Tensor,   # (n_supply,) requires_grad
                         log_K: torch.Tensor,   # (n_supply, n_demand)
                         eps:   float) -> torch.Tensor:
        g = self._g_from_f(f, b, log_K, eps)                  

        # fa = ε·lse_j(log_K[i,j] + g[j]/ε)
        M_fa = log_K + g.unsqueeze(0) / eps                
        m    = M_fa.max(dim=1, keepdim=True).values
        fa   = eps * ((M_fa - m).exp().sum(1).log() + m.squeeze(1))  
        M_gb = log_K + f.unsqueeze(1) / eps                    
        m    = M_gb.max(dim=0, keepdim=True).values
        gb   = eps * ((M_gb - m).exp().sum(0).log() + m.squeeze(0))  

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
            f"[OT_Objective_Sliced_World] Training  T={T}  "
            f"M={M}  L={L}  eps={eps}  lr={lr}"
        )

        # log_K FIXED (supply/demand locations never change)
        log_K = self._precompute_log_K()   # (n_supply, n_demand)

        self.logger.info(f"[OT_Objective_Sliced_World] Collecting M={M} pair pool ...")
        pool_Phi, pool_a, pool_b = [], [], []
        collected = 0
        pbar_pool = tqdm(total=M, desc="Collecting pairs")

        for _, _, sw_b, dw_b in dataloader_train:
            for a_np, b_np in zip(sw_b.numpy(), dw_b.numpy()):
                if collected >= M:
                    break
                # Φ_f: (n_supply, L) — supply potentials via stereographic 1D OT
                Xf, _ = self._compute_features(a_np, b_np)
                Xf    = Xf - Xf.mean(axis=0, keepdims=True)   # center

                pool_Phi.append(
                    torch.tensor(Xf,    dtype=torch.float64, device=self.device))
                pool_a.append(
                    torch.tensor(a_np,  dtype=torch.float64, device=self.device))
                pool_b.append(
                    torch.tensor(b_np,  dtype=torch.float64, device=self.device))
                collected += 1
                pbar_pool.update(1)
            if collected >= M:
                break
        pbar_pool.close()
        self.logger.info(f"[OT_Objective_Sliced_World] Pool ready: {collected} pairs")

        alpha = nn.Parameter(
            torch.zeros(L, dtype=torch.float64, device=self.device))
        opt   = torch.optim.Adam([alpha], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T, eta_min=lr * 0.01)

        loss_ema = None
        t0       = time.time()
        pbar     = tqdm(total=T, desc="OT_Objective_Sliced_World")
        rng      = np.random.default_rng(42)

        for step in range(T):
            idx     = int(rng.integers(0, collected))
            Phi_f_t = pool_Phi[idx]   # (n_supply, L)
            a_t     = pool_a[idx]     # (n_supply,)
            b_t     = pool_b[idx]     # (n_demand,)

            f_pred = Phi_f_t @ alpha  # (n_supply,)

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
            f"[OT_Objective_Sliced_World] Done.  "
            f"alpha: min={alpha_np.min():.4f}  max={alpha_np.max():.4f}  "
            f"‖α‖={np.linalg.norm(alpha_np):.4f}"
        )
        np.save(os.path.join(self.log_sub_folder, "alpha.npy"), alpha_np)
        return alpha_np

    def _predict_potentials(self, a: np.ndarray, b: np.ndarray,
                            alpha: np.ndarray, beta: np.ndarray = None):
        Xf, _ = self._compute_features(a, b)
        Xf    = Xf - Xf.mean(axis=0, keepdims=True)
        f_pred = Xf @ alpha   # (n_supply,)

        eps   = float(self.cfg_m.epsilon)
        log_K = self._precompute_log_K()
        f_t   = torch.tensor(f_pred, dtype=torch.float64, device=self.device)
        b_t   = torch.tensor(b,      dtype=torch.float64, device=self.device)

        with torch.no_grad():
            g_t = self._g_from_f(f_t, b_t, log_K, eps)

        return f_pred, g_t.cpu().numpy()

    def train(self, dataloader_train, dataloader_test=None):
        self.alpha = self._fit(dataloader_train)
        self.beta  = np.zeros(self.projection_matrix.shape[0])  # unused, for interface
        self.logger.info("[OT_Objective_Sliced_World] Training complete.")
        return self.alpha, self.beta
