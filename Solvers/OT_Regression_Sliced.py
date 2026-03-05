"""
OT_Regression_Sliced.py
========================
Regression-based amortised Optimal Transport via Sliced-OT features.

Pipeline (from the NeurIPS 2026 draft):
  1.  Bootstrap M training pairs (µ_i, ν_i) from the dataloader.
  2.  Solve entropic OT (Sinkhorn) for each pair → Kantorovich potentials
      f_i, g_i  ∈ R^n.
  3.  Draw L fixed projection directions θ_1, …, θ_L  (unit sphere).
  4.  For each pair i and direction l: project masses onto θ_l and solve
      the resulting 1-D OT problem → 1-D potentials f_{i,θ_l}, g_{i,θ_l} ∈ R^n.
  5.  Build feature matrices  Φ_f^i, Φ_g^i  ∈ R^{n × L}
           Φ_f^i[:, l] = f_{i,θ_l}   (evaluated at projected positions of each grid point)
  6.  Solve joint simplex-constrained least squares across all M pairs:
           min_{α ≥ 0, Σα=1}  Σ_i  ||Φ_f^i α – f_i||²
      and the analogous problem for β, g.
  7.  Predict on a new pair: f̂ = Φ_f α,  ĝ = Φ_g β.
  8.  Recover transport plan:  P = exp(f̂/ε) ⊗ K ⊗ exp(ĝ/ε).
"""

import os
import numpy as np
import ot
import torch
from tqdm import tqdm

from Solvers.DefenseTrain import Defense_Train_Base
from regression_OT_utils import (
    generate_uniform_unit_sphere_projections,
    optimal_alpha_simplex,
    solve_1D_ot,                  # fixed version (np.int64)
    solve_1D_ot_unsorted,         # convenience wrapper that handles sorting
)


class OT_Regression_Sliced(Defense_Train_Base):
    """Regression-based amortised OT (Sliced-OT features)."""

    def __init__(self, cfg_proj, cfg_m):
        Defense_Train_Base.__init__(self, cfg_proj, cfg_m, name="OT_Regression_Sliced")
        self._build_grid()

        # --- fixed projection directions θ_1, …, θ_L  (L × 2) ---
        L = self.cfg_m.num_projections
        proj = generate_uniform_unit_sphere_projections(
            dim=2,
            num_projections=L,
            dtype=torch.float64,
            device="cpu",
        )
        self.projection_matrix = proj.detach().numpy()   # (L, 2)
        self.logger.info(
            f"[OT_Regression_Sliced] projection_matrix: {self.projection_matrix.shape}, "
            f"num_bootstrap={self.cfg_m.num_bootstrap}, ridge={self.cfg_m.ridge}"
        )

    # ------------------------------------------------------------------
    # Grid & squared-Euclidean cost matrix
    # ------------------------------------------------------------------

    def _build_grid(self):
        s = self.cfg_m.img_size
        grid = []
        for i in np.linspace(1, 0, num=s):
            for j in np.linspace(0, 1, num=s):
                grid.append([j, i])
        self.x_grid = np.array(grid, dtype=np.float64)          # (n, 2), n = s²
        diff = self.x_grid[:, None, :] - self.x_grid[None, :, :]
        self.C = np.sum(diff ** 2, axis=-1)                      # (n, n)

    # ------------------------------------------------------------------
    # Step 2: Entropic OT → ground-truth Kantorovich potentials
    # ------------------------------------------------------------------

    def _solve_entropic_ot(self, a: np.ndarray, b: np.ndarray):
        """
        Run Sinkhorn on the full 2-D problem.

        Returns
        -------
        f : (n,) float64  — source potential   (ε·log u)
        g : (n,) float64  — target potential   (ε·log v)
        """
        eps = self.cfg_m.epsilon
        _, log = ot.sinkhorn(
            a, b, self.C,
            reg=eps,
            log=True,
            numItermax=2000,
            stopThr=1e-9,
            warn=False,
        )
        # Sinkhorn scaling vectors: T = diag(u) K diag(v), K = exp(-C/ε)
        # Kantorovich potentials: f = ε log u, g = ε log v
        f = eps * np.log(np.maximum(log["u"], 1e-300))
        g = eps * np.log(np.maximum(log["v"], 1e-300))
        return f, g

    # ------------------------------------------------------------------
    # Steps 3-4: 1-D OT potentials along one projection direction
    # ------------------------------------------------------------------

    @staticmethod
    def _sliced_1d_potentials(
        a: np.ndarray,
        b: np.ndarray,
        proj_positions: np.ndarray,
        p: int = 2,
    ):
        """
        Project masses onto a 1-D line and solve 1-D OT.

        Both µ and ν share the same grid, so their projected positions
        ``proj_positions = x_grid @ θ`` are identical; only the weights
        a and b differ.

        Parameters
        ----------
        a, b          : (n,)  weight vectors (must sum to the same value)
        proj_positions: (n,)  projected scalar positions for every grid point
        p             : OT power

        Returns
        -------
        f1d : (n,) float — 1-D source potential in **original** grid ordering
        g1d : (n,) float — 1-D target potential in **original** grid ordering
        """
        # solve_1D_ot_unsorted handles sorting internally and maps back
        f1d, g1d, _ = solve_1D_ot_unsorted(a, b, proj_positions, proj_positions, p=p)
        return f1d, g1d

    # ------------------------------------------------------------------
    # Step 5: Feature matrix for a single pair
    # ------------------------------------------------------------------

    def _compute_features(self, a: np.ndarray, b: np.ndarray):
        """
        Build sliced-OT feature matrices for one distribution pair.

        Returns
        -------
        Xf : (n, L)  — Xf[:, l] = f_{θ_l}(P_{θ_l}(x_k))  for each grid point k
        Xg : (n, L)  — Xg[:, l] = g_{θ_l}(P_{θ_l}(x_k))
        """
        n = len(a)
        L = self.projection_matrix.shape[0]
        Xf = np.empty((n, L), dtype=np.float64)
        Xg = np.empty((n, L), dtype=np.float64)

        for l, theta in enumerate(self.projection_matrix):
            # Scalar projection of every grid point onto θ_l
            proj = self.x_grid @ theta   # (n,)
            f1d, g1d = self._sliced_1d_potentials(a, b, proj, p=2)
            Xf[:, l] = f1d
            Xg[:, l] = g1d

        return Xf, Xg

    # ------------------------------------------------------------------
    # Step 6: Fit regression coefficients α, β (joint across all pairs)
    # ------------------------------------------------------------------

    def _fit(self, dataloader_train):
        """
        Collect M bootstrap pairs, build feature matrices and targets,
        then solve the simplex-constrained least-squares problems.

        Returns
        -------
        alpha : (L,) — regression weights for f
        beta  : (L,) — regression weights for g
        """
        M = self.cfg_m.num_bootstrap
        self.logger.info(f"[Fit] Collecting M={M} bootstrap pairs …")

        Phi_f_list, Phi_g_list = [], []
        y_f_list,   y_g_list   = [], []
        count = 0

        pbar = tqdm(total=M, desc="Bootstrap pairs")
        for _, _, x_a, x_b in dataloader_train:
            x_a_np = x_a.numpy()   # (batch, n)
            x_b_np = x_b.numpy()

            for i in range(x_a_np.shape[0]):
                if count >= M:
                    break

                a = x_a_np[i]   # source histogram
                b = x_b_np[i]   # target histogram

                # --- ground-truth potentials via Sinkhorn ---
                f_gt, g_gt = self._solve_entropic_ot(a, b)

                # --- sliced-OT feature matrices ---
                Xf, Xg = self._compute_features(a, b)

                Phi_f_list.append(Xf)    # (n, L)
                Phi_g_list.append(Xg)
                y_f_list.append(f_gt)    # (n,)
                y_g_list.append(g_gt)

                count += 1
                pbar.update(1)
                self.logger.info(
                    f"Pair {count}/{M} | "
                    f"||f_gt||={np.linalg.norm(f_gt):.4f}, "
                    f"||g_gt||={np.linalg.norm(g_gt):.4f}"
                )

            if count >= M:
                break

        pbar.close()

        if count < M:
            self.logger.warning(
                f"Only {count} pairs collected (requested {M}). "
                "Consider increasing the dataset or reducing num_bootstrap."
            )

        # --- stack all pairs → joint regression ---
        Phi_f = np.vstack(Phi_f_list)        # (count * n, L)
        Phi_g = np.vstack(Phi_g_list)
        y_f   = np.concatenate(y_f_list)     # (count * n,)
        y_g   = np.concatenate(y_g_list)

        self.logger.info(
            f"[Fit] Phi_f shape: {Phi_f.shape} → solving simplex LS for α …"
        )
        alpha = optimal_alpha_simplex(Phi_f, y_f, ridge=self.cfg_m.ridge)

        self.logger.info("[Fit] Solving simplex LS for β …")
        beta  = optimal_alpha_simplex(Phi_g, y_g, ridge=self.cfg_m.ridge)

        self.logger.info(
            f"[Fit] α: min={alpha.min():.4f}, max={alpha.max():.4f}, "
            f"nnz={np.sum(alpha > 1e-6)}/{len(alpha)}"
        )
        self.logger.info(
            f"[Fit] β: min={beta.min():.4f},  max={beta.max():.4f}, "
            f"nnz={np.sum(beta  > 1e-6)}/{len(beta)}"
        )

        # Persist coefficients
        np.save(os.path.join(self.log_sub_folder, "alpha.npy"), alpha)
        np.save(os.path.join(self.log_sub_folder, "beta.npy"),  beta)
        self.logger.info(f"[Fit] Saved alpha/beta to {self.log_sub_folder}")

        return alpha, beta

    # ------------------------------------------------------------------
    # Steps 7-8: Predict potentials + transport plan for a new pair
    # ------------------------------------------------------------------

    def _predict_potentials(
        self,
        a: np.ndarray,
        b: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ):
        """
        Predict Kantorovich potentials for a new pair using learned α, β.

        f̂(x_k) = Σ_l α_l · f_{θ_l}(P_{θ_l}(x_k))
        ĝ(x_k) = Σ_l β_l · g_{θ_l}(P_{θ_l}(x_k))
        """
        Xf, Xg = self._compute_features(a, b)
        f_pred = Xf @ alpha    # (n,)
        g_pred = Xg @ beta     # (n,)
        return f_pred, g_pred

    def _potentials_to_plan(self, f: np.ndarray, g: np.ndarray) -> np.ndarray:
        """
        Recover the (regularised) transport plan from Kantorovich potentials.

        P_{ij} = exp(f_i / ε) · K_{ij} · exp(g_j / ε)
        where K = exp(-C / ε)  is the Gibbs kernel.
        """
        eps = self.cfg_m.epsilon
        K = np.exp(-self.C / eps)          # (n, n)
        P = np.exp(f[:, None] / eps) * K * np.exp(g[None, :] / eps)
        return P

    # ------------------------------------------------------------------
    # Evaluation & visualisation (mirrors OT_Discrete.OT_D_test)
    # ------------------------------------------------------------------

    def _evaluate(self, dataloader_test, alpha: np.ndarray, beta: np.ndarray):
        """Compute transport plans, report RMSE in potentials, and save geodesics."""
        
        from Utils import utils

        def interp(P, num_inter, batch_size, img_size):
            P_flatten = P.flatten()
            grid = []
            for i in np.linspace(1, 0, num = img_size):
                for j in np.linspace(0, 1, num = img_size):
                    grid.append([j, i])
            x_grid = np.array(grid)
            y_grid = np.array(grid)
        
            def get_hist(t, P_flat):
                map_samples = np.random.choice(range(len(P_flat)), size = batch_size, p = P_flat)
                a_samples = x_grid[map_samples // img_size**2]
                b_samples = y_grid[map_samples % img_size**2]
                proj_samples = (1.-t)*a_samples + t*b_samples
                hist, _, _ = np.histogram2d(proj_samples[:,1], proj_samples[:,0], bins = np.linspace(0., 1., num = img_size + 1))
        
                hist = np.flipud(hist)
                thresh = np.quantile(hist, 0.9)
                hist[hist > thresh] = thresh
                hist = hist / hist.max()
                return hist
        
            hists = []
            ts = np.linspace(0, 1, num = num_inter)
        
            for i, t in enumerate(ts):
                hist = get_hist(t, P_flatten)
                hists.append(hist)
            return hists

        # Grab a small test batch
        for _, _, xs_a, xs_b in dataloader_test:
            xs_a_np = xs_a[:2].numpy()
            xs_b_np = xs_b[:2].numpy()
            break

        img_size = self.cfg_m.img_size

        for idx in range(len(xs_a_np)):
            a, b = xs_a_np[idx], xs_b_np[idx]

            # Ground-truth potentials & plan
            f_gt, g_gt = self._solve_entropic_ot(a, b)
            P_gt        = self._potentials_to_plan(f_gt, g_gt)

            # Predicted potentials & plan
            f_pred, g_pred = self._predict_potentials(a, b, alpha, beta)
            P_pred          = self._potentials_to_plan(f_pred, g_pred)

            # Potential RMSE
            rmse_f = float(np.sqrt(np.mean((f_pred - f_gt) ** 2)))
            rmse_g = float(np.sqrt(np.mean((g_pred - g_gt) ** 2)))
            msg = (
                f"[Eval {idx}]  RMSE_f={rmse_f:.6f}  RMSE_g={rmse_g:.6f} | "
                f"plan_sum_gt={P_gt.sum():.4f}  plan_sum_pred={P_pred.sum():.4f}"
            )
            print(msg)
            self.logger.info(msg)

            # Geodesic interpolation images
            imgs_gt   = interp(P_gt,   num_inter=11, batch_size=50_000, img_size=img_size)
            imgs_pred = interp(P_pred, num_inter=11, batch_size=50_000, img_size=img_size)

            utils.save_r(
                imgs_gt,
                torch.tensor(a), torch.tensor(b),
                path=self.log_sub_folder,
                title=f"GroundTruth_{idx}",
            )
            utils.save_r(
                imgs_pred,
                torch.tensor(a), torch.tensor(b),
                path=self.log_sub_folder,
                title=f"Pred_{idx}",
            )

    # ------------------------------------------------------------------
    # Entry point (called by Pre_Model.train)
    # ------------------------------------------------------------------

    def train(self, dataloader_train, dataloader_test):
        alpha, beta = self._fit(dataloader_train)
        self._evaluate(dataloader_test, alpha, beta)
