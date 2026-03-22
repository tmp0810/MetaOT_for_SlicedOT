import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from Solvers.DefenseTrain import Defense_Train_Base
from regression_OT_utils import (
    generate_uniform_unit_sphere_projections,
    emd1D_dual,
)
from Utils import utils


class OT_Objective_Sliced(Defense_Train_Base):
    def __init__(self, cfg_proj, cfg_m):
        Defense_Train_Base.__init__(self, cfg_proj, cfg_m, name="OT_Objective_Sliced")
        self._build_grid()
        L = self.cfg_m.num_projections
        proj = generate_uniform_unit_sphere_projections(
            dim=2, num_projections=L, dtype=torch.float64, device="cpu")
        self.projection_matrix = proj.detach().numpy()

    def _build_grid(self):
        s = self.cfg_m.img_size
        grid = []
        for i in np.linspace(1, 0, num=s):
            for j in np.linspace(0, 1, num=s):
                grid.append([j, i])
        self.x_grid = np.array(grid, dtype=np.float64)
        diff = self.x_grid[:, None, :] - self.x_grid[None, :, :]
        self.C = np.sum(diff ** 2, axis=-1)

    def _compute_features(self, a, b):
        device = torch.device(
            f"cuda:{self.cfg_m.gpu}"
            if torch.cuda.is_available() and hasattr(self.cfg_m, "gpu")
            else "cpu")
        proj_values = torch.tensor(
            (self.x_grid @ self.projection_matrix.T).T,
            dtype=torch.float64, device=device)
        a_t = torch.tensor(a, dtype=torch.float64, device=device)
        b_t = torch.tensor(b, dtype=torch.float64, device=device)
        f_grad, g_grad, _ = emd1D_dual(
            proj_values, proj_values,
            u_weights=a_t, v_weights=b_t,
            p=2, require_sort=True)
        Xf = f_grad.cpu().numpy().T
        Xg = g_grad.cpu().numpy().T
        return Xf, Xg

    def _potentials_to_plan(self, a, b, f, g):
        eps   = self.cfg_m.epsilon
        f_c   = f - f.mean()
        g_c   = g - g.mean()
        log_P = f_c[:, None] / eps - self.C / eps + g_c[None, :] / eps
        log_P -= log_P.max()
        P = np.exp(log_P)
        r = P.sum(axis=1) + 1e-12
        P = P * (a / r)[:, None]
        c = P.sum(axis=0) + 1e-12
        P = P * (b / c)[None, :]
        return np.clip(P, 0.0, None)

    @staticmethod
    def interp(P, num_inter, batch_size, img_size):
        P_flatten = P.flatten()
        grid = []
        for i in np.linspace(1, 0, num=img_size):
            for j in np.linspace(0, 1, num=img_size):
                grid.append([j, i])
        x_grid = np.array(grid)
        y_grid = np.array(grid)
        n_pixels = img_size * img_size

        def get_hist(t, P_flat):
            map_samples = np.random.choice(range(len(P_flat)), size=batch_size, p=P_flat)
            a_samples = x_grid[map_samples // n_pixels]
            b_samples = y_grid[map_samples % n_pixels]
            proj_samples = (1.0 - t) * a_samples + t * b_samples
            hist, _, _ = np.histogram2d(
                proj_samples[:, 1], proj_samples[:, 0],
                bins=np.linspace(0.0, 1.0, num=img_size + 1))
            hist = np.flipud(hist)
            nonzero = hist[hist > 0]
            if len(nonzero) > 0:
                thresh = np.quantile(nonzero, 0.9)
                if thresh > 0:
                    hist = np.clip(hist, 0, thresh)
            if hist.max() > 0:
                hist = hist / hist.max()
            return hist

        return [get_hist(t, P_flatten) for t in np.linspace(0, 1, num=num_inter)]

    def _precompute_log_K(self):
        eps = float(self.cfg_m.epsilon)
        C_t = torch.tensor(self.C, dtype=torch.float64, device=self.device)
        return -C_t / eps

    def _g_from_f(self, f, b, log_K, eps):
        log_b = torch.log(b.clamp(1e-300))
        M     = log_K + f.unsqueeze(1) / eps
        m     = M.max(dim=0, keepdim=True).values
        lse   = (M - m).exp().sum(dim=0).log() + m.squeeze(0)
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

    def _compute_features_raw(self, a, b):
        device = self.device
        proj_values = torch.tensor(
            (self.x_grid @ self.projection_matrix.T).T,
            dtype=torch.float64, device=device)
        a_t = torch.tensor(a, dtype=torch.float64, device=device)
        b_t = torch.tensor(b, dtype=torch.float64, device=device)
        f_grad, g_grad, _ = emd1D_dual(
            proj_values, proj_values,
            u_weights=a_t, v_weights=b_t,
            p=2, require_sort=True)
        Xf = f_grad.cpu().numpy().T - 0
        Xg = g_grad.cpu().numpy().T - 0
        Xf = Xf - Xf.mean(axis=0, keepdims=True)
        Xg = Xg - Xg.mean(axis=0, keepdims=True)
        return Xf, Xg, None

    def _fit(self, dataloader_train):
        cfg    = self.cfg_m
        T      = int(cfg.num_train_iter)
        M      = int(cfg.num_bootstrap)
        L      = self.projection_matrix.shape[0]
        eps    = float(cfg.epsilon)
        lr     = float(getattr(cfg, "learning_rate", 1e-3))
        max_gn = float(getattr(cfg, "max_grad_norm", 1.0))
        log_iv = int(getattr(cfg, "log_interval", 100))

        log_K = self._precompute_log_K()

        pool_Phi, pool_a, pool_b = [], [], []
        collected = 0
        pbar_pool = tqdm(total=M, desc="Collecting pairs")
        for _, _, x_a, x_b in dataloader_train:
            for a_np, b_np in zip(x_a.numpy(), x_b.numpy()):
                if collected >= M:
                    break
                Xf, _, _ = self._compute_features_raw(a_np, b_np)
                pool_Phi.append(torch.tensor(Xf,   dtype=torch.float64, device=self.device))
                pool_a.append(  torch.tensor(a_np, dtype=torch.float64, device=self.device))
                pool_b.append(  torch.tensor(b_np, dtype=torch.float64, device=self.device))
                collected += 1
                pbar_pool.update(1)
            if collected >= M:
                break
        pbar_pool.close()

        alpha    = nn.Parameter(torch.zeros(L, dtype=torch.float64, device=self.device))
        opt      = torch.optim.Adam([alpha], lr=lr)
        sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T, eta_min=lr * 0.01)
        loss_ema = None
        t0       = time.time()
        pbar     = tqdm(total=T, desc="OT_Objective_Sliced")
        rng      = np.random.default_rng(42)

        for step in range(T):
            idx    = int(rng.integers(0, collected))
            f_pred = pool_Phi[idx] @ alpha
            loss   = -self._dual_obj_from_f(pool_a[idx], pool_b[idx], f_pred, log_K, eps)
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

        alpha_np = alpha.detach().cpu().numpy()
        np.save(os.path.join(self.log_sub_folder, "alpha.npy"), alpha_np)
        return alpha_np

    def _predict_potentials(self, a, b, alpha, beta=None):
        Xf, _, _ = self._compute_features_raw(a, b)
        f_clean  = Xf @ alpha
        eps      = float(self.cfg_m.epsilon)
        log_K    = self._precompute_log_K()
        f_t      = torch.tensor(f_clean, dtype=torch.float64, device=self.device)
        b_t      = torch.tensor(b,       dtype=torch.float64, device=self.device)
        with torch.no_grad():
            g_t = self._g_from_f(f_t, b_t, log_K, eps)
        return f_clean, g_t.cpu().numpy()

    def train(self, dataloader_train, dataloader_test):
        self.alpha = self._fit(dataloader_train)
        self.beta  = np.zeros(self.projection_matrix.shape[0])
