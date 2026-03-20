import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from Solvers.Regression_SlicedOT.OT_Regression_Sliced_Color import OT_Regression_Sliced_Color


class OT_Objective_Sliced_Color(OT_Regression_Sliced_Color):
    is_continuous = False   # eval_color_transfer.py dùng predict_plan

    def __init__(self, cfg_proj, cfg_m):
        super().__init__(cfg_proj, cfg_m)
        self.name = "OT_Objective_Sliced_Color"

    def _g_from_f(self,
                  f:     torch.Tensor,   # (n,)
                  b:     torch.Tensor,   # (n,)
                  log_K: torch.Tensor,   # (n, n)
                  eps:   float) -> torch.Tensor:
        log_b = torch.log(b.clamp(1e-300))
        M     = log_K + f.unsqueeze(1) / eps           # (n, n)
        m     = M.max(dim=0, keepdim=True).values
        lse   = (M - m).exp().sum(0).log() + m.squeeze(0)
        return eps * (log_b - lse)

    def _dual_obj_from_f(self,
                         a:     torch.Tensor,   # (n,)
                         b:     torch.Tensor,   # (n,)
                         f:     torch.Tensor,   # (n,) requires_grad
                         log_K: torch.Tensor,   # (n, n)
                         eps:   float) -> torch.Tensor:
        g    = self._g_from_f(f, b, log_K, eps)

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
            f"[OT_Objective_Sliced_Color] Training  T={T}  "
            f"M={M}  L={L}  eps={eps}  lr={lr}"
        )

        self.logger.info(f"[OT_Objective_Sliced_Color] Collecting M={M} pair pool ...")
        pool_Phi, pool_logK, pool_a, pool_b = [], [], [], []
        collected = 0
        pbar_pool = tqdm(total=M, desc="Collecting pairs")

        for src_w, src_c, tgt_w, tgt_c in dataloader_train:
            for i in range(src_w.shape[0]):
                if collected >= M:
                    break

                a_np    = src_w[i].numpy()
                x_src   = src_c[i].numpy()
                b_np    = tgt_w[i].numpy()
                x_tgt   = tgt_c[i].numpy()

                # Φ_f: (n, L)
                try:
                    Xf, _ = self._compute_features(a_np, b_np, x_src, x_tgt)
                except Exception:
                    continue
                Xf = Xf - Xf.mean(axis=0, keepdims=True)

                # log_K: (n, n) — per-pair cost
                C     = self._compute_cost(x_src, x_tgt)
                log_K = -C / eps

                pool_Phi.append(
                    torch.tensor(Xf,    dtype=torch.float32, device=self.device))
                pool_logK.append(
                    torch.tensor(log_K, dtype=torch.float32, device=self.device))
                pool_a.append(
                    torch.tensor(a_np,  dtype=torch.float32, device=self.device))
                pool_b.append(
                    torch.tensor(b_np,  dtype=torch.float32, device=self.device))
                collected += 1
                pbar_pool.update(1)

            if collected >= M:
                break
        pbar_pool.close()
        self.logger.info(f"[OT_Objective_Sliced_Color] Pool ready: {collected} pairs")

        alpha = nn.Parameter(
            torch.zeros(L, dtype=torch.float32, device=self.device))
        opt   = torch.optim.Adam([alpha], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T, eta_min=lr * 0.01)

        loss_ema = None
        t0       = time.time()
        pbar     = tqdm(total=T, desc="OT_Objective_Sliced_Color")
        rng      = np.random.default_rng(42)

        for step in range(T):
            idx     = int(rng.integers(0, collected))
            Phi_f_t = pool_Phi[idx]    # (n, L)
            log_K_t = pool_logK[idx]   # (n, n) — per-pair
            a_t     = pool_a[idx]      # (n,)
            b_t     = pool_b[idx]      # (n,)

            f_pred = Phi_f_t @ alpha   # (n,)

            dual = self._dual_obj_from_f(a_t, b_t, f_pred, log_K_t, eps)
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

        alpha_np = alpha.detach().cpu().numpy().astype(np.float64)
        self.logger.info(
            f"[OT_Objective_Sliced_Color] Done.  "
            f"alpha: min={alpha_np.min():.4f}  max={alpha_np.max():.4f}  "
            f"‖α‖={np.linalg.norm(alpha_np):.4f}"
        )
        np.save(os.path.join(self.log_sub_folder, "alpha.npy"), alpha_np)
        return alpha_np


    def predict_plan(self,
                     a:     np.ndarray,   # (n,)
                     b:     np.ndarray,   # (n,)
                     src_c: np.ndarray,   # (n, 3)
                     tgt_c: np.ndarray,   # (n, 3)
                     ) -> np.ndarray:
        eps = float(self.cfg_m.epsilon)

        # Features
        Xf, _ = self._compute_features(a, b, src_c, tgt_c)
        Xf    = Xf - Xf.mean(axis=0, keepdims=True)
        f_pred = Xf @ self.alpha   # (n,)

        # Per-pair log_K
        C     = self._compute_cost(src_c, tgt_c)
        log_K = -C / eps

        # g from f (Option A)
        f_t   = torch.tensor(f_pred, dtype=torch.float64, device=self.device)
        b_t   = torch.tensor(b,      dtype=torch.float64, device=self.device)
        log_Kt = torch.tensor(log_K, dtype=torch.float64, device=self.device)

        with torch.no_grad():
            g_t = self._g_from_f(f_t, b_t, log_Kt,
                                  float(self.cfg_m.epsilon))

        g_pred = g_t.cpu().numpy()
        return self._potentials_to_plan(a, b, f_pred, g_pred, C)


    def train(self, dataloader_train):
        self.alpha = self._fit(dataloader_train)
        self.logger.info("[OT_Objective_Sliced_Color] Training complete.")
        return self.alpha
