import os
import time
import numpy as np
import torch

from Solvers.DefenseTrain import Defense_Train_Base
from SWGG import quantile_SWGG_CP


class min_SWGG_Color(Defense_Train_Base):
    is_continuous = False   

    def __init__(self, cfg_proj, cfg_m):
        Defense_Train_Base.__init__(self, cfg_proj, cfg_m,
                                    name="min_SWGG_Color")
        self.logger.info(
            f"[min_SWGG_Color] n_projections={cfg_m.n_projections}  "
            f"n_clusters={cfg_m.n_clusters}")

    @staticmethod
    def _compute_cost(x_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
        """Squared-Euclidean cost in [0,1]^3."""
        diff = x_src[:, None, :] - x_tgt[None, :, :]
        return np.sum(diff ** 2, axis=-1)

    def predict_plan(
        self,
        a:      np.ndarray,   # (n_clusters,)  source histogram weights
        b:      np.ndarray,   # (n_clusters,)  target histogram weights
        src_c:  np.ndarray,   # (n_clusters, 3) source KMeans centroids
        tgt_c:  np.ndarray,   # (n_clusters, 3) target KMeans centroids
    ) -> np.ndarray:
        L      = int(self.cfg_m.n_projections)
        device = self.device

        X_t = torch.tensor(src_c, dtype=torch.float32, device=device)  # (500, 3)
        Y_t = torch.tensor(tgt_c, dtype=torch.float32, device=device)  # (500, 3)
        a_t = torch.tensor(a,     dtype=torch.float32, device=device)  # (500,)
        b_t = torch.tensor(b,     dtype=torch.float32, device=device)  # (500,)
        thetas = torch.randn(3, L, dtype=torch.float32, device=device)
        thetas = thetas / thetas.norm(dim=0, keepdim=True).clamp(1e-12)
        with torch.no_grad():
            W, delta_r, w_a, w_b, u, v = quantile_SWGG_CP(
                X_t, Y_t, a_t, b_t, thetas)

        W_safe = torch.where(torch.isfinite(W), W,
                             torch.full_like(W, float("inf")))
        best_l = int(W_safe.argmin().item())
        u_w = torch.take_along_dim(u, w_a, dim=0)[:, best_l].long()   # (n_r-1,)
        v_w = torch.take_along_dim(v, w_b, dim=0)[:, best_l].long()   # (n_r-1,)
        dr  = delta_r[:, best_l]                                         # (n_r-1,)

        mask = dr > 1e-9
        u_w  = u_w[mask]
        v_w  = v_w[mask]
        dr   = dr[mask]

        n = a.shape[0]
        P = torch.zeros(n, n, dtype=torch.float32, device=device)
        P.index_put_((u_w, v_w), dr, accumulate=True)

        P_np = P.cpu().numpy().astype(np.float64)
        P_np = np.clip(P_np, 0.0, None)
        s    = P_np.sum()
        if s > 0:
            P_np /= s
        return P_np

    def train(self, dataloader_train):
        self.logger.info(
            f"[min_SWGG_Color] No training needed.  "
            f"n_projections={self.cfg_m.n_projections}  "
            f"Ready to call predict_plan(a, b, src_c, tgt_c).")
