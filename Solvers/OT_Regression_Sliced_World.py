import os
import numpy as np
import torch
from tqdm import tqdm

from Solvers.OT_Regression_Sliced import OT_Regression_Sliced, _ridge_regression
from regression_OT_utils import (
    generate_uniform_unit_sphere_projections,
    emd1D_dual,
)


def _epsilon_projection(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    # If any point is exactly at north pole, jitter it
    north = torch.where(x[..., -1] == 1.0)
    if north[0].numel() > 0:
        x.data[north] = x[north] + (
            epsilon * torch.rand_like(x[north]) - epsilon / 2
        )
    # Clamp last coordinate strictly below 1
    x.data[..., -1] = torch.min(
        x[..., -1],
        torch.tensor(1.0 - epsilon, dtype=x.dtype, device=x.device)
    )
    # Re-project onto unit sphere (rescale first d coords)
    alpha = torch.sqrt(
        (1.0 - x[..., -1] ** 2) / (x[..., :-1] ** 2).sum(-1).clamp(min=1e-12)
    )
    alpha[alpha.isnan()] = 1.0   # south-pole correction (from s3w.py)
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
    dots = supply_euc @ demand_euc.T            # (n_supply, n_demand)
    dots = np.clip(dots, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.arccos(dots)                       # (n_supply, n_demand)


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

        # Call parent (which will call our overridden _build_grid)
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

        stereo_supply_t = _get_stereo_proj_torch(supply_t)   # (n_supply, 2), factor-2
        stereo_demand_t = _get_stereo_proj_torch(demand_t)   # (n_demand, 2), factor-2

        stereo_supply_t = torch.nan_to_num(stereo_supply_t, nan=0.0, posinf=0.0, neginf=0.0)
        stereo_demand_t = torch.nan_to_num(stereo_demand_t, nan=0.0, posinf=0.0, neginf=0.0)

        self.stereo_supply = stereo_supply_t.numpy()  # (n_supply, 2)
        self.stereo_demand = stereo_demand_t.numpy()  # (n_demand, 2)

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

        supply_t = torch.tensor(self.supply_euc, dtype=torch.float64, device=device)  # (n_supply, 3)
        demand_t = torch.tensor(self.demand_euc, dtype=torch.float64, device=device)  # (n_demand, 3)
        supply_t = _epsilon_projection(supply_t)
        demand_t = _epsilon_projection(demand_t)

        stereo_supply = _get_stereo_proj_torch(supply_t)  # (n_supply, 2)
        stereo_demand = _get_stereo_proj_torch(demand_t)  # (n_demand, 2)

        stereo_supply = torch.nan_to_num(stereo_supply, nan=0.0, posinf=0.0, neginf=0.0)
        stereo_demand = torch.nan_to_num(stereo_demand, nan=0.0, posinf=0.0, neginf=0.0)

        proj_mat = torch.tensor(self.projection_matrix, dtype=torch.float64, device=device)  # (L, 2)
        proj_supply = (stereo_supply @ proj_mat.T).T   # (L, n_supply)
        proj_demand = (stereo_demand @ proj_mat.T).T   # (L, n_demand)

        a_t = torch.tensor(a, dtype=torch.float64, device=device)
        b_t = torch.tensor(b, dtype=torch.float64, device=device)

        f_grad, g_grad, _ = emd1D_dual(
            proj_supply, proj_demand,
            u_weights=a_t,
            v_weights=b_t,
            p=2,
            require_sort=True,
        )

        Xf = f_grad.cpu().numpy().T   # (n_supply, L)
        Xg = g_grad.cpu().numpy().T   # (n_demand, L)
        return Xf, Xg

    def _potentials_to_plan(
        self, 
        a: np.ndarray, 
        b: np.ndarray, 
        f: np.ndarray, 
        g: np.ndarray
    ) -> np.ndarray:
        eps = self.cfg_m.epsilon
        
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

        P = np.exp(log_P)
        P = np.clip(P, 0.0, None)

        P_sum = P.sum()
        if P_sum > 0:
            P /= P_sum
            
        return P

    def predict_plan(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        f_pred, g_pred = self._predict_potentials(a, b, self.alpha, self.beta)
        return _potentials_to_plan(a, b, f_pred, g_pred)


    def train(self, dataloader_train, dataloader_test=None):
        self.alpha, self.beta = self._fit(dataloader_train)
        self.logger.info("[World] Training complete. Call predict_plan(a, b) to use.")
        return self.alpha, self.beta
