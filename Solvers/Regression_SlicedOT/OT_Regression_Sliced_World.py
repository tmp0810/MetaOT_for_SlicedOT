import os
import numpy as np
import torch
from tqdm import tqdm

from Solvers.Regression_SlicedOT.OT_Regression_Sliced import OT_Regression_Sliced, _ridge_regression
from regression_OT_utils import (
    generate_uniform_unit_sphere_projections,
    emd1D_dual,
)

def _epsilon_projection(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    north = torch.where(x[..., -1] == 1.0)
    if north[0].numel() > 0:
        x.data[north] = x[north] + (
            epsilon * torch.rand_like(x[north]) - epsilon / 2
        )
    x.data[..., -1] = torch.min(
        x[..., -1],
        torch.tensor(1.0 - epsilon, dtype=x.dtype, device=x.device)
    )
    alpha = torch.sqrt(
        (1.0 - x[..., -1] ** 2) / (x[..., :-1] ** 2).sum(-1).clamp(min=1e-12)
    )
    alpha[alpha.isnan()] = 1.0
    x.data[..., :-1] *= alpha.unsqueeze(-1)
    return x

def _get_stereo_proj_torch(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    d         = x.shape[-1] - 1
    numerator = 2.0 * x[..., :d]
    denom     = 1.0 - x[..., d]
    near_pole = torch.isclose(
        denom, torch.zeros_like(denom), atol=epsilon
    )
    proj = torch.full_like(x[..., :d], float('inf'))
    proj[~near_pole] = numerator[~near_pole] / denom[~near_pole].unsqueeze(-1)
    return proj

def _sphere_cost(supply_euc: np.ndarray, demand_euc: np.ndarray) -> np.ndarray:
    dots = supply_euc @ demand_euc.T
    dots = np.clip(dots, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.arccos(dots)

class OT_Regression_Sliced_World(OT_Regression_Sliced):

    def __init__(
        self,
        cfg_proj,
        cfg_m,
        supply_euc: np.ndarray,
        demand_euc: np.ndarray,
        supply_sph: np.ndarray = None,
        demand_sph: np.ndarray = None,
    ):
        self.supply_euc = supply_euc.astype(np.float64)
        self.demand_euc = demand_euc.astype(np.float64)
        self.supply_sph = supply_sph
        self.demand_sph = demand_sph
        self.n_supply   = len(supply_euc)
        self.n_demand   = len(demand_euc)

        super().__init__(cfg_proj, cfg_m)

    def _build_grid(self):
        self.x_grid = None

        self.C = _sphere_cost(self.supply_euc, self.demand_euc)
        self.logger.info(
            f"[World] Cost matrix: {self.C.shape}  "
            f"min={self.C.min():.4f}  max={self.C.max():.4f}"
        )

        supply_t = torch.tensor(self.supply_euc, dtype=torch.float64)
        demand_t = torch.tensor(self.demand_euc, dtype=torch.float64)

        supply_t = _epsilon_projection(supply_t)
        demand_t = _epsilon_projection(demand_t)

        stereo_supply_t = _get_stereo_proj_torch(supply_t)
        stereo_demand_t = _get_stereo_proj_torch(demand_t)

        stereo_supply_t = torch.nan_to_num(stereo_supply_t, nan=0.0, posinf=0.0, neginf=0.0)
        stereo_demand_t = torch.nan_to_num(stereo_demand_t, nan=0.0, posinf=0.0, neginf=0.0)

        self.stereo_supply = stereo_supply_t.numpy()
        self.stereo_demand = stereo_demand_t.numpy()

        self.logger.info(
            f"[World] Stereo supply range: "
            f"u=[{self.stereo_supply[:,0].min():.2f}, {self.stereo_supply[:,0].max():.2f}]  "
            f"v=[{self.stereo_supply[:,1].min():.2f}, {self.stereo_supply[:,1].max():.2f}]"
        )

    def _compute_features(self, a: np.ndarray, b: np.ndarray):
        device = torch.device(
            f"cuda:{self.cfg_m.gpu}"
            if torch.cuda.is_available() and hasattr(self.cfg_m, "gpu")
            else "cpu"
        )
        L = self.projection_matrix.shape[0]

        supply_t = torch.tensor(self.supply_euc, dtype=torch.float64, device=device)
        demand_t = torch.tensor(self.demand_euc, dtype=torch.float64, device=device)

        supply_t = _epsilon_projection(supply_t)
        demand_t = _epsilon_projection(demand_t)

        stereo_supply = _get_stereo_proj_torch(supply_t)
        stereo_demand = _get_stereo_proj_torch(demand_t)

        stereo_supply = torch.nan_to_num(stereo_supply, nan=0.0, posinf=0.0, neginf=0.0)
        stereo_demand = torch.nan_to_num(stereo_demand, nan=0.0, posinf=0.0, neginf=0.0)

        proj_mat = torch.tensor(self.projection_matrix, dtype=torch.float64, device=device)
        proj_supply = (stereo_supply @ proj_mat.T).T
        proj_demand = (stereo_demand @ proj_mat.T).T

        a_t = torch.tensor(a, dtype=torch.float64, device=device)
        b_t = torch.tensor(b, dtype=torch.float64, device=device)

        f_grad, g_grad, _ = emd1D_dual(
            proj_supply, proj_demand,
            u_weights=a_t,
            v_weights=b_t,
            p=2,
            require_sort=True,
        )

        Xf = f_grad.cpu().numpy().T
        Xg = g_grad.cpu().numpy().T
        return Xf, Xg

    def _precompute_log_K(self) -> torch.Tensor:
        eps = float(self.cfg_m.epsilon)
        C_t = torch.tensor(self.C, dtype=torch.float64, device=self.device)
        return -C_t / eps

    def _g_from_f(self, f: torch.Tensor, b: torch.Tensor,
                  log_K: torch.Tensor, eps: float) -> torch.Tensor:
        log_b = torch.log(b.clamp(1e-300))
        M     = log_K + f.unsqueeze(1) / eps
        m     = M.max(dim=0, keepdim=True).values
        lse   = (M - m).exp().sum(dim=0).log() + m.squeeze(0)
        return eps * (log_b - lse)

    def _f_from_g(self, g: torch.Tensor, a: torch.Tensor,
                  log_K: torch.Tensor, eps: float) -> torch.Tensor:
        log_a = torch.log(a.clamp(1e-300))
        M     = log_K + g.unsqueeze(0) / eps
        m     = M.max(dim=1, keepdim=True).values
        lse   = (M - m).exp().sum(dim=1).log() + m.squeeze(1)
        return eps * (log_a - lse)

    def predict_plan(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        Xf, _ = self._compute_features(a, b)
        Xf    = Xf - Xf.mean(axis=0, keepdims=True)
        f_pred = Xf @ self.alpha

        eps   = float(self.cfg_m.epsilon)
        log_K = self._precompute_log_K()
        a_t   = torch.tensor(a,      dtype=torch.float64, device=self.device)
        f_t   = torch.tensor(f_pred, dtype=torch.float64, device=self.device)
        b_t   = torch.tensor(b,      dtype=torch.float64, device=self.device)
        with torch.no_grad():
            g_t = self._g_from_f(f_t, b_t, log_K, eps)
            f_t = self._f_from_g(g_t, a_t, log_K, eps)

        return self._potentials_to_plan(f_t.cpu().numpy(), g_t.cpu().numpy())

    def train(self, dataloader_train, dataloader_test=None):
        self.alpha, self.beta = self._fit(dataloader_train)
        self.logger.info("[World] Training complete. Call predict_plan(a, b) to use.")
        return self.alpha, self.beta
