import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from Solvers.DefenseTrain import Defense_Train_Base
from Solvers.Objective_SlicedOT.OT_Objective_Sliced import OT_Objective_Sliced
from regression_OT_utils import generate_uniform_unit_sphere_projections, emd1D_dual
from Data.multires_utils import MultiResGridMixin, RESOLUTIONS


class OT_Objective_Sliced_MultiRes(MultiResGridMixin, OT_Objective_Sliced):

    def __init__(self, cfg_proj, cfg_m):
        Defense_Train_Base.__init__(self, cfg_proj, cfg_m, name="OT_Objective_Sliced_MultiRes")
        self._build_grid()
        L = self.cfg_m.num_projections
        proj = generate_uniform_unit_sphere_projections(
            dim=2, num_projections=L, dtype=torch.float64, device="cpu")
        self.projection_matrix = proj.detach().numpy()

    # ---- override: build per-resolution grids instead of one fixed grid ----
    def _build_grid(self):
        self._init_multires(RESOLUTIONS)

    # ---- override: sliced potentials using each side's own grid ----
    def _compute_features_raw(self, a, b):
        device = self.device
        ra, rb = self._infer_res(a), self._infer_res(b)
        grid_a, grid_b = self._grid(ra), self._grid(rb)

        proj_a = torch.tensor((grid_a @ self.projection_matrix.T).T,
                               dtype=torch.float64, device=device)
        proj_b = torch.tensor((grid_b @ self.projection_matrix.T).T,
                               dtype=torch.float64, device=device)
        a_t = torch.tensor(a, dtype=torch.float64, device=device)
        b_t = torch.tensor(b, dtype=torch.float64, device=device)

        f_grad, g_grad, _ = emd1D_dual(
            proj_a, proj_b, u_weights=a_t, v_weights=b_t, p=2, require_sort=True)

        Xf = f_grad.cpu().numpy().T
        Xg = g_grad.cpu().numpy().T
        Xf = Xf - Xf.mean(axis=0, keepdims=True)
        Xg = Xg - Xg.mean(axis=0, keepdims=True)
        return Xf, Xg, None

    # ---- override: predict potentials with the pair's native log_K ----
    def _predict_potentials(self, a, b, alpha, beta=None):
        Xf, _, _ = self._compute_features_raw(a, b)
        f_clean = Xf @ alpha
        eps = float(self.cfg_m.epsilon)
        ra, rb = self._infer_res(a), self._infer_res(b)
        log_K = self._logK(ra, rb, eps)

        f_t = torch.tensor(f_clean, dtype=torch.float64, device=self.device)
        b_t = torch.tensor(b, dtype=torch.float64, device=self.device)
        a_t = torch.tensor(a, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            g_t = self._g_from_f(f_t, b_t, log_K, eps)
            f_t = self._f_from_g(g_t, a_t, log_K, eps)
        return f_t.cpu().numpy(), g_t.cpu().numpy()

    # ---- override: recover the plan with the pair's native cost matrix ----
    def _potentials_to_plan(self, a, b, f, g):
        eps = self.cfg_m.epsilon
        ra, rb = self._infer_res(a), self._infer_res(b)
        C = self._cost(ra, rb)
        f_c = f - f.mean()
        g_c = g - g.mean()
        log_P = f_c[:, None] / eps - C / eps + g_c[None, :] / eps
        log_P -= log_P.max()
        P = np.exp(log_P)
        r = P.sum(axis=1) + 1e-12
        P = P * (a / r)[:, None]
        c = P.sum(axis=0) + 1e-12
        P = P * (b / c)[None, :]
        return np.clip(P, 0.0, None)

    # ---- override: full retrain loop, each pooled pair keeps its own log_K ----
    def _fit(self, dataloader_train):
        cfg = self.cfg_m
        T = int(cfg.num_train_iter)
        M = int(cfg.num_bootstrap)
        L = self.projection_matrix.shape[0]
        eps = float(cfg.epsilon)
        lr = float(getattr(cfg, "learning_rate", 1e-3))
        max_gn = float(getattr(cfg, "max_grad_norm", 1.0))
        log_iv = int(getattr(cfg, "log_interval", 100))

        pool_Phi, pool_a, pool_b, pool_logK = [], [], [], []
        collected = 0
        pbar_pool = tqdm(total=M, desc="Collecting pairs (multi-res)")
        for _, _, x_a, x_b in dataloader_train:
            for a_np, b_np in zip(x_a.numpy(), x_b.numpy()):
                if collected >= M:
                    break
                ra, rb = self._infer_res(a_np), self._infer_res(b_np)
                Xf, _, _ = self._compute_features_raw(a_np, b_np)
                pool_Phi.append(torch.tensor(Xf, dtype=torch.float64, device=self.device))
                pool_a.append(torch.tensor(a_np, dtype=torch.float64, device=self.device))
                pool_b.append(torch.tensor(b_np, dtype=torch.float64, device=self.device))
                pool_logK.append(self._logK(ra, rb, eps))  # cached: <= len(RESOLUTIONS)**2 matrices total
                collected += 1
                pbar_pool.update(1)
            if collected >= M:
                break
        pbar_pool.close()

        alpha = nn.Parameter(torch.zeros(L, dtype=torch.float64, device=self.device))
        opt = torch.optim.Adam([alpha], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T, eta_min=lr * 0.01)
        loss_ema = None
        t0 = time.time()
        pbar = tqdm(total=T, desc="OT_Objective_Sliced_MultiRes")
        rng = np.random.default_rng(42)

        for step in range(T):
            idx = int(rng.integers(0, collected))
            f_pred = pool_Phi[idx] @ alpha
            loss = -self._dual_obj_from_f(pool_a[idx], pool_b[idx], f_pred, pool_logK[idx], eps)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([alpha], max_gn)
            opt.step()
            sched.step()
            lv = loss.item()
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
