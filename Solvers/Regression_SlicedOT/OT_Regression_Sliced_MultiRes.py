import numpy as np
import ot
import torch
from Solvers.DefenseTrain import Defense_Train_Base
from Solvers.Regression_SlicedOT.OT_Regression_Sliced import OT_Regression_Sliced
from regression_OT_utils import generate_uniform_unit_sphere_projections, emd1D_dual
from Data.multires_utils import MultiResGridMixin, RESOLUTIONS

class OT_Regression_Sliced_MultiRes(MultiResGridMixin, OT_Regression_Sliced):

    def __init__(self, cfg_proj, cfg_m):
        Defense_Train_Base.__init__(self, cfg_proj, cfg_m, name="OT_Regression_Sliced_MultiRes")
        self._build_grid()
        L = self.cfg_m.num_projections
        proj = generate_uniform_unit_sphere_projections(
            dim=2, num_projections=L, dtype=torch.float64, device="cpu")
        self.projection_matrix = proj.detach().numpy()
        self.logger.info(
            f"[OT_Regression_Sliced_MultiRes] resolutions={self._mr_resolutions}, "
            f"num_projections={L}, num_bootstrap={self.cfg_m.num_bootstrap}, ridge={self.cfg_m.ridge}"
        )

    # ---- override: build per-resolution grids instead of one fixed grid ----
    def _build_grid(self):
        self._init_multires(RESOLUTIONS)

    # ---- override: ground-truth entropic OT at the pair's native (n_a, n_b) ----
    def _solve_entropic_ot(self, a: np.ndarray, b: np.ndarray):
        eps = self.cfg_m.epsilon
        ra, rb = self._infer_res(a), self._infer_res(b)
        C = self._cost(ra, rb)
        a_safe = np.clip(a, 1e-10, None); a_safe /= a_safe.sum()
        b_safe = np.clip(b, 1e-10, None); b_safe /= b_safe.sum()
        _, log_dict = ot.sinkhorn(
            a_safe, b_safe, C,
            reg=eps, numItermax=self.cfg_m.sinkhorn_iters,
            stopThr=1e-5, log=True)
        if 'alpha' in log_dict:
            return log_dict['alpha'], log_dict['beta']
        elif 'log_u' in log_dict:
            return eps * log_dict['log_u'], eps * log_dict['log_v']
        else:
            u = log_dict.get('u', np.ones_like(a_safe))
            v = log_dict.get('v', np.ones_like(b_safe))
            return eps * np.log(np.clip(u, 1e-50, None)), eps * np.log(np.clip(v, 1e-50, None))

    # ---- override: sliced potentials using each side's own grid ----
    def _compute_features(self, a: np.ndarray, b: np.ndarray):
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
        return Xf, Xg

    # ---- override: predict potentials with the pair's native log_K ----
    def _predict_potentials(self, a: np.ndarray, b: np.ndarray,
                             alpha: np.ndarray, beta: np.ndarray = None):
        Xf, _ = self._compute_features(a, b)
        Xf = Xf - Xf.mean(axis=0, keepdims=True)
        f_pred = Xf @ alpha
        eps = float(self.cfg_m.epsilon)
        ra, rb = self._infer_res(a), self._infer_res(b)
        log_K = self._logK(ra, rb, eps)

        a_t = torch.tensor(a, dtype=torch.float64, device=self.device)
        f_t = torch.tensor(f_pred, dtype=torch.float64, device=self.device)
        b_t = torch.tensor(b, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            g_t = self._g_from_f(f_t, b_t, log_K, eps)
            f_t = self._f_from_g(g_t, a_t, log_K, eps)
        return f_t.cpu().numpy(), g_t.cpu().numpy()

    # ---- override: recover the plan with the pair's native cost matrix ----
    def _potentials_to_plan(self, a: np.ndarray, b: np.ndarray,
                             f: np.ndarray, g: np.ndarray) -> np.ndarray:
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
