import os
import time
import numpy as np
import torch
import torch.nn as nn
import ot
from tqdm import tqdm

from Solvers.DefenseTrain import Defense_Train_Base
from regression_OT_utils import (
    generate_uniform_unit_sphere_projections,
    emd1D_dual,
)


class OT_Objective_Sliced_Color(Defense_Train_Base):
    is_continuous = False

    def __init__(self, cfg_proj, cfg_m):
        Defense_Train_Base.__init__(self, cfg_proj, cfg_m, name="OT_Objective_Sliced_Color")
        L = self.cfg_m.num_projections
        proj = generate_uniform_unit_sphere_projections(
            dim=3, num_projections=L, dtype=torch.float64, device="cpu")
        self.projection_matrix = proj.detach().numpy()

    def _compute_cost(self, x_src, x_tgt):
        diff = x_src[:, None, :] - x_tgt[None, :, :]
        return np.sum(diff ** 2, axis=-1)

    def _compute_features(self, a, b, x_src, x_tgt):
        device   = self.device
        proj_mat = torch.tensor(self.projection_matrix, dtype=torch.float64, device=device)
        src_t    = torch.tensor(x_src, dtype=torch.float64, device=device)
        tgt_t    = torch.tensor(x_tgt, dtype=torch.float64, device=device)
        proj_src = (src_t @ proj_mat.T).T
        proj_tgt = (tgt_t @ proj_mat.T).T
        a_t = torch.tensor(a, dtype=torch.float64, device=device)
        b_t = torch.tensor(b, dtype=torch.float64, device=device)
        f_grad, g_grad, _ = emd1D_dual(
            proj_src, proj_tgt,
            u_weights=a_t, v_weights=b_t,
            p=2, require_sort=True)
        Xf = f_grad.cpu().numpy().T
        Xg = g_grad.cpu().numpy().T
        return Xf, Xg

    def _potentials_to_plan(self, a, b, f, g, C):
        eps   = self.cfg_m.epsilon
        f_c   = f - f.mean()
        g_c   = g - g.mean()
        log_P = f_c[:, None] / eps - C / eps + g_c[None, :] / eps
        log_P -= log_P.max()
        P     = np.exp(log_P)
        r = P.sum(axis=1) + 1e-12
        P = P * (a / r)[:, None]
        c = P.sum(axis=0) + 1e-12
        P = P * (b / c)[None, :]
        return np.clip(P, 0.0, None)

    def _g_from_f(self, f, b, log_K, eps):
        log_b = torch.log(b.clamp(1e-300))
        M     = log_K + f.unsqueeze(1) / eps
        m     = M.max(dim=0, keepdim=True).values
        lse   = (M - m).exp().sum(0).log() + m.squeeze(0)
        return eps * (log_b - lse)

    def _dual_obj_from_f(self, a, b, f, log_K, eps):
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

    def _fit(self, dataloader_train):
        cfg    = self.cfg_m
        T      = int(cfg.num_train_iter)
        M      = int(cfg.num_bootstrap)
        L      = self.projection_matrix.shape[0]
        eps    = float(cfg.epsilon)
        lr     = float(getattr(cfg, "learning_rate", 1e-3))
        max_gn = float(getattr(cfg, "max_grad_norm", 1.0))
        log_iv = int(getattr(cfg, "log_interval", 100))

        pool_Phi, pool_logK, pool_a, pool_b = [], [], [], []
        collected = 0
        pbar_pool = tqdm(total=M, desc="Collecting pairs")
        for src_w, src_c, tgt_w, tgt_c in dataloader_train:
            for i in range(src_w.shape[0]):
                if collected >= M:
                    break
                a_np  = src_w[i].numpy()
                x_src = src_c[i].numpy()
                b_np  = tgt_w[i].numpy()
                x_tgt = tgt_c[i].numpy()
                try:
                    Xf, _ = self._compute_features(a_np, b_np, x_src, x_tgt)
                except Exception:
                    continue
                Xf    = Xf - Xf.mean(axis=0, keepdims=True)
                C     = self._compute_cost(x_src, x_tgt)
                log_K = -C / eps
                pool_Phi.append( torch.tensor(Xf,    dtype=torch.float32, device=self.device))
                pool_logK.append(torch.tensor(log_K, dtype=torch.float32, device=self.device))
                pool_a.append(   torch.tensor(a_np,  dtype=torch.float32, device=self.device))
                pool_b.append(   torch.tensor(b_np,  dtype=torch.float32, device=self.device))
                collected += 1
                pbar_pool.update(1)
            if collected >= M:
                break
        pbar_pool.close()

        alpha    = nn.Parameter(torch.zeros(L, dtype=torch.float32, device=self.device))
        opt      = torch.optim.Adam([alpha], lr=lr)
        sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T, eta_min=lr * 0.01)
        loss_ema = None
        t0       = time.time()
        pbar     = tqdm(total=T, desc="OT_Objective_Sliced_Color")
        rng      = np.random.default_rng(42)

        for step in range(T):
            idx    = int(rng.integers(0, collected))
            f_pred = pool_Phi[idx] @ alpha
            loss   = -self._dual_obj_from_f(
                pool_a[idx], pool_b[idx], f_pred, pool_logK[idx], eps)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([alpha], max_gn)
            opt.step()
            sched.step()
            lv       = loss.item()
            loss_ema = lv if loss_ema is None else 0.95 * loss_ema + 0.05 * lv
            pbar.update(1)
            if (step + 1) % log_iv == 0:
                msg = f"[{step+1}/{T}]  dual={-lv:.4e}  loss_ema={loss_ema:.4e}  t={time.time()-t0:.1f}s"
                pbar.set_description(msg)
                self.logger.info(msg)
        pbar.close()

        alpha_np = alpha.detach().cpu().numpy().astype(np.float64)
        np.save(os.path.join(self.log_sub_folder, "alpha.npy"), alpha_np)
        return alpha_np

    def predict_plan(self, a, b, src_c, tgt_c):
        eps    = float(self.cfg_m.epsilon)
        Xf, _ = self._compute_features(a, b, src_c, tgt_c)
        Xf     = Xf - Xf.mean(axis=0, keepdims=True)
        f_pred = Xf @ self.alpha

        C      = self._compute_cost(src_c, tgt_c)
        log_Kt = torch.tensor(-C / eps, dtype=torch.float64, device=self.device)
        f_t    = torch.tensor(f_pred, dtype=torch.float64, device=self.device)
        b_t    = torch.tensor(b,      dtype=torch.float64, device=self.device)
        with torch.no_grad():
            g_t = self._g_from_f(f_t, b_t, log_Kt, eps)

        return self._potentials_to_plan(a, b, f_pred, g_t.cpu().numpy(), C)

    def train(self, dataloader_train):
        self.alpha = self._fit(dataloader_train)
        self.beta  = np.zeros(self.projection_matrix.shape[0])
        return self.alpha
