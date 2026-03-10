import os
import numpy as np
import torch
from tqdm import tqdm

from OT_Regression_Sliced import OT_Regression_Sliced, _ridge_regression
from regression_OT_utils import (
    generate_uniform_unit_sphere_projections,
    emd1D_dual,
)

def _stereo_proj(xyz: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    denom    = (1.0 - z).clip(epsilon)           # avoid division by zero
    u        = 2 * x / denom
    v        = 2 * y / denom
    return np.stack([u, v], axis=1)              # (n, 2)


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
        # Store locations BEFORE calling parent __init__
        # (parent calls _build_grid which we override)
        self.supply_euc = supply_euc.astype(np.float64)   # (n_supply, 3)
        self.demand_euc = demand_euc.astype(np.float64)   # (n_demand, 3)
        self.supply_sph = supply_sph                       # (n_supply, 2), for plotting
        self.demand_sph = demand_sph                       # (n_demand, 2), for plotting
        self.n_supply   = len(supply_euc)
        self.n_demand   = len(demand_euc)

        # Call parent (which will call our overridden _build_grid)
        super().__init__(cfg_proj, cfg_m)

    def _build_grid(self):
        """
        Replace pixel grid with sphere geometry.
        Precomputes:
          self.C              : (n_supply, n_demand)  sphere cost matrix
          self.stereo_supply  : (n_supply, 2)  stereographic projections
          self.stereo_demand  : (n_demand, 2)  stereographic projections
          self.x_grid         : None (not used in world pair)
        """
        self.x_grid = None   # not used in world pair

        self.C = _sphere_cost(self.supply_euc, self.demand_euc)
        self.logger.info(
            f"[World] Cost matrix: {self.C.shape}  "
            f"min={self.C.min():.4f}  max={self.C.max():.4f}"
        )

        # Stereographic projection for feature extraction
        self.stereo_supply = _stereo_proj(self.supply_euc)  # (n_supply, 2)
        self.stereo_demand = _stereo_proj(self.demand_euc)  # (n_demand, 2)
        self.logger.info(
            f"[World] Stereo supply range: "
            f"u=[{self.stereo_supply[:,0].min():.2f}, {self.stereo_supply[:,0].max():.2f}]  "
            f"v=[{self.stereo_supply[:,1].min():.2f}, {self.stereo_supply[:,1].max():.2f}]"
        )

    # ------------------------------------------------------------------
    # Override: feature extraction using stereographic projection
    # ------------------------------------------------------------------

    def _compute_features(self, a: np.ndarray, b: np.ndarray):
        """
        Compute sliced-OT feature matrices for one world-pair (a, b).

        Steps:
          1. Map supply/demand locations to R² via stereographic projection.
          2. Project each 2-D point onto L random directions → 1-D positions.
          3. Run emd1D_dual to get dual potentials for all L directions at once.

        Parameters
        ----------
        a : (n_supply,)  supply weights
        b : (n_demand,)  demand weights

        Returns
        -------
        Xf : (n_supply, L)  source potentials per projection direction
        Xg : (n_demand, L)  target potentials per projection direction
        """
        device = torch.device(
            f"cuda:{self.cfg_m.gpu}"
            if torch.cuda.is_available() and hasattr(self.cfg_m, "gpu")
            else "cpu"
        )
        L = self.projection_matrix.shape[0]   # (L, 2)

        # Project stereographic coords onto L random directions
        # stereo_supply: (n_supply, 2) → proj_supply: (n_supply, L) → (L, n_supply)
        proj_supply = torch.tensor(
            (self.stereo_supply @ self.projection_matrix.T).T,  # (L, n_supply)
            dtype=torch.float64, device=device,
        )
        proj_demand = torch.tensor(
            (self.stereo_demand @ self.projection_matrix.T).T,  # (L, n_demand)
            dtype=torch.float64, device=device,
        )

        a_t = torch.tensor(a, dtype=torch.float64, device=device)  # (n_supply,)
        b_t = torch.tensor(b, dtype=torch.float64, device=device)  # (n_demand,)

        # emd1D_dual handles different source (N) and target (M) support sizes.
        # Returns: f_grad (L, n_supply), g_grad (L, n_demand)
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

    # ------------------------------------------------------------------
    # Override: predict transport plan
    # ------------------------------------------------------------------

    def predict_plan(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Predict the transport plan P for a new supply/demand pair.

        Parameters
        ----------
        a : (n_supply,)  supply weights
        b : (n_demand,)  demand weights

        Returns
        -------
        P : (n_supply, n_demand)  transport plan (sums to 1)
        """
        f_pred, g_pred = self._predict_potentials(a, b, self.alpha, self.beta)
        return self._potentials_to_plan(f_pred, g_pred)

    # ------------------------------------------------------------------
    # Override: train — skip MNIST _evaluate, just fit
    # ------------------------------------------------------------------

    def train(self, dataloader_train, dataloader_test=None):
        """Fit regression weights. No MNIST-style geodesic evaluation."""
        self.alpha, self.beta = self._fit(dataloader_train)
        self.logger.info("[World] Training complete. Call predict_plan(a, b) to use.")
        return self.alpha, self.beta
