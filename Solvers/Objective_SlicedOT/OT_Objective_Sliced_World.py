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


def _epsilon_projection(x, epsilon=1e-6):
    north = torch.where(x[..., -1] == 1.0)
    if north[0].numel() > 0:
        x.data[north] = x[north] + (epsilon * torch.rand_like(x[north]) - epsilon / 2)
    x.data[..., -1] = torch.min(
        x[..., -1],
        torch.tensor(1.0 - epsilon, dtype=x.dtype, device=x.device))
    alpha = torch.sqrt(
        (1.0 - x[..., -1] ** 2) / (x[..., :-1] ** 2).sum(-1).clamp(min=1e-12))
    alpha[alpha.isnan()] = 1.0
    x.data[..., :-1] *= alpha.unsqueeze(-1)
    return x


def _get_stereo_proj_torch(x, epsilon=1e-6):
    d         = x.shape[-1] - 1
    numerator = 2.0 * x[..., :d]
    denom     = 1.0 - x[..., d]
    near_pole = torch.isclose(denom, torch.zeros_like(denom), atol=epsilon)
    proj      = torch.full_like(x[..., :d], float('inf'))
    proj[~near_pole] = numerator[~near_pole] / denom[~near_pole].unsqueeze(-1)
    return proj


def _sphere_cost(supply_euc, demand_euc):
    dots = supply_euc @ demand_euc.T
    dots = np.clip(dots, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.arccos(dots)


class OT_Objective_Sliced_World(Defense_Train_Base):
    def __init__(self, cfg_proj, cfg_m,
                 supply_euc, demand_euc,
                 supply_sph=None, demand_sph=None):
        self.supply_euc = supply_euc.astype(np.float64)
        self.demand_euc = demand_euc.astype(np.float64)
        self.supply_sph = supply_sph
        self.demand_sph = demand_sph
        self.n_supply   = len(supply_euc)
        self.n_demand   = len(demand_euc)

        Defense_Train_Base.__init__(self, cfg_proj, cfg_m, name="OT_Objective_Sliced_World")

        self.C = _sphere_cost(self.supply_euc, self.demand_euc)

        supply_t = torch.tensor(self.supply_euc, dtype=torch.float64)
        demand_t = torch.tensor(self.demand_euc, dtype=torch.float64)
        supply_t = _epsilon_projection(supply_t)
        demand_t = _epsilon_projection(demand_t)
        stereo_s = torch.nan_to_num(_get_stereo_proj_torch(supply_t), nan=0., posinf=0., neginf=0.)
        stereo_d = torch.nan_to_num(_get_stereo_proj_torch(demand_t), nan=0., posinf=0., neginf=0.)
        self.stereo_supply = stereo_s.numpy()
        self.stereo_demand = stereo_d.numpy()

        L = self.cfg_m.num_projections
        proj = generate_uniform_unit_sphere_projections(
            dim=2, num_projections=L, dtype=torch.float64, device="cpu")
        self.projection_matrix = proj.detach().numpy()

    def _compute_features(self, a, b):
        device   = self.device
        supply_t = torch.tensor(self.supply_euc, dtype=torch.float64, device=device)
        demand_t = torch.tensor(self.demand_euc, dtype=torch.float64, device=device)
        supply_t = _epsilon_projection(supply_t)
        demand_t = _epsilon_projection(demand_t)
        stereo_s = torch.nan_to_num(_get_stereo_proj_torch(supply_t), nan=0., posinf=0., neginf=0.)
        stereo_d = torch.nan_to_num(_get_stereo_proj_torch(demand_t), nan=0., posinf=0., neginf=0.)
        proj_mat    = torch.tensor(self.projection_matrix, dtype=torch.float64, device=device)
        proj_supply = (stereo_s @ proj_mat.T).T
        proj_demand = (stereo_d @ proj_mat.T).T
        a_t = torch.tensor(a, dtype=torch.float64, device=device)
        b_t = torch.tensor(b, dtype=torch.float64, device=device)
        f_grad, g_grad, _ = emd1D_dual(
            proj_supply, proj_demand,
            u_weights=a_t, v_weights=b_t,
            p=2, require_sort=True)
        Xf = f_grad.cpu().numpy().T
        Xg = g_grad.cpu().numpy().T
        return Xf, Xg

    def _potentials_to_plan(self, a, b, f, g):
        eps   = self.cfg_m.epsilon
        log_P = f[:, None] / eps - self.C / eps + g[None, :] / eps

        def lse(X, axis):
            m = X.max(axis=axis, keepdims=True)
            return np.log(np.exp(X - m).sum(axis=axis)) + m.squeeze(axis=axis)

        log_a = np.log(np.clip(a, 1e-300, None))
        log_u = log_a[:, None] - lse(log_P, axis=1)[:, None]
        log_P = log_P + log_u
        log_b = np.log(np.clip(b, 1e-300, None))
        log_v = log_b[None, :] - lse(log_P, axis=0)[None, :]
        log_P = log_P + log_v
        P = np.clip(np.exp(log_P), 0.0, None)
        s = P.sum()
        if s > 0:
            P /= s
        return P

    def predict_plan(self, a, b):
        f_pred, g_pred = self._predict_potentials(a, b, self.alpha)
        return self._potentials_to_plan(a, b, f_pred, g_pred)

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
        for _, _, sw_b, dw_b in dataloader_train:
            for a_np, b_np in zip(sw_b.numpy(), dw_b.numpy()):
                if collected >= M:
                    break
                Xf, _ = self._compute_features(a_np, b_np)
                Xf    = Xf - Xf.mean(axis=0, keepdims=True)
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
        pbar     = tqdm(total=T, desc="OT_Objective_Sliced_World")
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
        Xf, _ = self._compute_features(a, b)
        Xf    = Xf - Xf.mean(axis=0, keepdims=True)
        f_pred = Xf @ alpha
        eps    = float(self.cfg_m.epsilon)
        log_K  = self._precompute_log_K()
        f_t    = torch.tensor(f_pred, dtype=torch.float64, device=self.device)
        b_t    = torch.tensor(b,      dtype=torch.float64, device=self.device)
        with torch.no_grad():
            g_t = self._g_from_f(f_t, b_t, log_K, eps)
        return f_pred, g_t.cpu().numpy()

    def train(self, dataloader_train, dataloader_test=None):
        self.alpha = self._fit(dataloader_train)
        self.beta  = np.zeros(self.projection_matrix.shape[0])
        return self.alpha, self.beta
