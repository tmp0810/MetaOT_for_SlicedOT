import os
import time
import numpy as np
import torch
import ot

from Solvers.DefenseTrain import Defense_Train_Base
from SWGG import quantile_SWGG_CP


class min_SWGG_World(Defense_Train_Base):
    def __init__(
        self,
        cfg_proj,
        cfg_m,
        supply_euc:  np.ndarray,   # (n_supply, 3)
        demand_euc:  np.ndarray,   # (n_demand, 3)
        supply_sph:  np.ndarray = None,
        demand_sph:  np.ndarray = None,
    ):
        self.supply_euc = supply_euc.astype(np.float64)  # (100, 3)
        self.demand_euc = demand_euc.astype(np.float64)  # (10000, 3)
        self.supply_sph = supply_sph
        self.demand_sph = demand_sph
        self.n_supply   = supply_euc.shape[0]
        self.n_demand   = demand_euc.shape[0]

        Defense_Train_Base.__init__(self, cfg_proj, cfg_m, name="min_SWGG_World")
        self.C_np = self._sphere_cost(self.supply_euc, self.demand_euc)
        self.logger.info(
            f"[min_SWGG_World] n_supply={self.n_supply}  n_demand={self.n_demand}  "
            f"n_projections={cfg_m.n_projections}  "
            f"C=[{self.C_np.min():.3f}, {self.C_np.max():.3f}]"
        )
        self.X_t = torch.tensor(
            self.supply_euc, dtype=torch.float64, device=self.device)  # (100, 3)
        self.Y_t = torch.tensor(
            self.demand_euc, dtype=torch.float64, device=self.device)  # (10000, 3)

    @staticmethod
    def _sphere_cost(xs: np.ndarray, xt: np.ndarray) -> np.ndarray:
        dots = xs @ xt.T
        return np.arccos(np.clip(dots, -1 + 1e-7, 1 - 1e-7))

    def predict_plan(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        L      = int(self.cfg_m.n_projections)
        device = self.device

        a_t = torch.tensor(a, dtype=torch.float64, device=device)   # (100,)
        b_t = torch.tensor(b, dtype=torch.float64, device=device)   # (10000,)
        thetas = torch.randn(3, L, dtype=torch.float64, device=device)
        thetas = thetas / thetas.norm(dim=0, keepdim=True).clamp(1e-12)
        with torch.no_grad():
            W, delta_r, w_a, w_b, u, v = quantile_SWGG_CP(
                self.X_t, self.Y_t, a_t, b_t, thetas)

        W_safe = torch.where(torch.isfinite(W), W,
                             torch.full_like(W, float("inf")))
        best_l = int(W_safe.argmin().item())
        u_w = torch.take_along_dim(u, w_a, dim=0)[:, best_l].long()  # (n_r,)
        v_w = torch.take_along_dim(v, w_b, dim=0)[:, best_l].long()  # (n_r,)
        dr  = delta_r[:, best_l]                                       # (n_r,)

        mask  = dr > 1e-15
        u_w   = u_w[mask]
        v_w   = v_w[mask]
        dr    = dr[mask]

        P = torch.zeros(self.n_supply, self.n_demand,
                        dtype=torch.float64, device=device)
        P.index_put_((u_w, v_w), dr, accumulate=True)

        P_np = P.cpu().numpy()
        P_np = np.clip(P_np, 0.0, None)
        s    = P_np.sum()
        if s > 0:
            P_np /= s
        return P_np


    def train(self, dataloader_train, dataloader_test=None):
        self.logger.info(
            f"[min_SWGG_World] No training needed.  "
            f"n_projections={self.cfg_m.n_projections}  "
            f"Ready to call predict_plan(a, b).")
