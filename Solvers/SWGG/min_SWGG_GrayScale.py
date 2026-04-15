import os
import time
import numpy as np
import torch
import ot
from tqdm import tqdm

from Solvers.DefenseTrain import Defense_Train_Base
from SWGG import quantile_SWGG_CP
from Utils import utils


class min_SWGG_GrayScale(Defense_Train_Base):
    def __init__(self, cfg_proj, cfg_m):
        Defense_Train_Base.__init__(self, cfg_proj, cfg_m,
                                    name="min_SWGG_GrayScale")
        self._build_grid()

    def _build_grid(self):
        """Build pixel grid x_grid (784, 2) and cost matrix C."""
        s = self.cfg_m.img_size   # 28
        grid = []
        for i in np.linspace(1, 0, num=s):
            for j in np.linspace(0, 1, num=s):
                grid.append([j, i])

        self.x_grid_np = np.array(grid, dtype=np.float64)          # (784, 2)
        self.x_grid    = torch.tensor(self.x_grid_np,
                                      dtype=torch.float64).to(self.device)

        diff   = self.x_grid_np[:, None, :] - self.x_grid_np[None, :, :]
        self.C = np.sum(diff ** 2, axis=-1)                         # (784, 784)

        self.logger.info(
            f"[min_SWGG_GrayScale] img_size={s}  n_pixels={s**2}  "
            f"n_projections={self.cfg_m.n_projections}")

    def predict_plan(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        L      = int(self.cfg_m.n_projections)
        device = self.device

        a_t = torch.tensor(a, dtype=torch.float64, device=device)   # (784,)
        b_t = torch.tensor(b, dtype=torch.float64, device=device)   # (784,)
        X   = self.x_grid                                            # (784, 2)
        thetas = torch.randn(2, L, dtype=torch.float64, device=device)
        thetas = thetas / thetas.norm(dim=0, keepdim=True).clamp(1e-12)

        with torch.no_grad():
            W, delta_r, w_a, w_b, u, v = quantile_SWGG_CP(
                X, X, a_t, b_t, thetas)

        W_safe  = torch.where(torch.isfinite(W), W,
                              torch.full_like(W, float("inf")))
        best_l  = int(W_safe.argmin().item())
        u_w    = torch.take_along_dim(u, w_a, dim=0)   # (n_r, L)
        v_w    = torch.take_along_dim(v, w_b, dim=0)   # (n_r, L)

        u_best = u_w[:, best_l].long()                  # (n_r,)
        v_best = v_w[:, best_l].long()                  # (n_r,)
        dr     = delta_r[:, best_l]                     # (n_r,)

        # Filter negative mass (numerical noise)
        mask   = dr > 0
        u_best = u_best[mask]
        v_best = v_best[mask]
        dr     = dr[mask]

        n = a.shape[0]
        P = torch.zeros(n, n, dtype=torch.float64, device=device)
        P.index_put_((u_best, v_best), dr, accumulate=True)

        P_np = P.cpu().numpy()
        P_np = np.clip(P_np, 0.0, None)
        s    = P_np.sum()
        if s > 0:
            P_np /= s
        return P_np

    def _evaluate(self, dataloader_test):
        eps = float(self.cfg_m.epsilon)

        # Grab a batch
        for _, _, xs_a, xs_b in dataloader_test:
            xs_a_np = xs_a[:2].numpy()
            xs_b_np = xs_b[:2].numpy()
            break

        img_size = self.cfg_m.img_size

        for idx in range(len(xs_a_np)):
            a = xs_a_np[idx]
            b = xs_b_np[idx]

            # Sinkhorn ground truth
            a_safe = np.clip(a, 1e-10, None); a_safe /= a_safe.sum()
            b_safe = np.clip(b, 1e-10, None); b_safe /= b_safe.sum()

            t0   = time.time()
            P_gt = ot.sinkhorn(a_safe, b_safe, self.C, reg=eps,
                               numItermax=1000, stopThr=1e-9)
            t_sink = time.time() - t0

            # min-SWGG prediction
            t0     = time.time()
            P_pred = self.predict_plan(a, b)
            t_ours = time.time() - t0

            rmse_P = float(np.sqrt(np.mean((P_pred - P_gt) ** 2)))
            msg    = (f"[Eval {idx}]  RMSE_Plan={rmse_P:.8f}  "
                      f"sum_gt={P_gt.sum():.4f}  sum_pred={P_pred.sum():.4f}  "
                      f"t_sinkhorn={t_sink:.3f}s  t_swgg={t_ours:.3f}s  "
                      f"speedup={t_sink/max(t_ours,1e-9):.1f}x")
            print(msg)
            self.logger.info(msg)

            # Save interpolation images
            from Solvers.Regression_SlicedOT.OT_Regression_Sliced import OT_Regression_Sliced
            imgs_gt   = OT_Regression_Sliced.interp(
                P_gt,   num_inter=11, batch_size=50_000, img_size=img_size)
            imgs_pred = OT_Regression_Sliced.interp(
                P_pred, num_inter=11, batch_size=50_000, img_size=img_size)

            utils.save_r(imgs_gt, torch.tensor(a), torch.tensor(b),
                         path=self.log_sub_folder,
                         title=f"GroundTruth_{idx}")
            utils.save_r(imgs_pred, torch.tensor(a), torch.tensor(b),
                         path=self.log_sub_folder,
                         title=f"min_SWGG_{idx}")


    def train(self, dataloader_train, dataloader_test):
        self.logger.info(
            f"[min_SWGG_GrayScale] No training needed.  "
            f"n_projections={self.cfg_m.n_projections}  "
            f"Running test evaluation ...")
        self._evaluate(dataloader_test)
        self.logger.info("[min_SWGG_GrayScale] Done.")
