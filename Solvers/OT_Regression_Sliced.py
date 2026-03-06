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
        Sinkhorn algorithm in log-space — Algorithm 1 from the paper.

        Iterates exactly as written:
            g_i = eps*log(b) - eps*log( K^T exp{f_{i-1}/eps} )
            f_i = eps*log(a) - eps*log( K   exp{g_i    /eps} )

        Returns f_N, g_N directly — no POT dependency, no log-key issues.

        Returns
        -------
        f : (n,) source Kantorovich potential
        g : (n,) target Kantorovich potential
        """
        eps = self.cfg_m.epsilon

        # Clamp weights to avoid log(0)
        a_safe = np.clip(a, 1e-10, None); a_safe /= a_safe.sum()
        b_safe = np.clip(b, 1e-10, None); b_safe /= b_safe.sum()

        log_a = np.log(a_safe)   # (n,)
        log_b = np.log(b_safe)   # (n,)
        log_K = -self.C / eps    # (n, n) — log Gibbs kernel

        def lse(X, axis):
            """Numerically stable log-sum-exp."""
            m = X.max(axis=axis, keepdims=True)
            return np.log(np.exp(X - m).sum(axis=axis)) + m.squeeze(axis=axis)

        # Initialise f = 0 (Algorithm 1 sets f_0 = 0)
        f = np.zeros_like(a_safe)

        for _ in range(self.cfg_m.sinkhorn_iters):
            # g = eps*log(b) - eps*lse_j( log_K[i,j] + f[j]/eps )
            #   K^T means we sum over rows (axis=0) for each column j
            g = eps * (log_b - lse(log_K + f[:, None] / eps, axis=0))
            # f = eps*log(a) - eps*lse_j( log_K[i,j] + g[j]/eps )
            f = eps * (log_a - lse(log_K + g[None, :] / eps, axis=1))

        if f.std() < 1e-8:
            raise RuntimeError(
                f"f_gt is constant (std={f.std():.2e}). "
                f"Check epsilon={eps} and that a/b are valid histograms."
            )
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
        f1d : (n,) float — 1-D source potential in original grid ordering
        g1d : (n,) float — 1-D target potential in original grid ordering
        """
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
        Sample M0 individual images, form all M0*(M0-1)/2 ordered pairs,
        then solve the regression for α and β.

        This follows the paper's protocol:
            "M0 samples drawn from the training set, yielding
             M = M0*(M0-1)/2 pairs used to estimate the RG coefficient."

        Returns
        -------
        alpha : (L,) — regression weights for f
        beta  : (L,) — regression weights for g
        """
        M0 = self.cfg_m.num_samples
        M  = M0 * (M0 - 1) // 2
        self.logger.info(
            f"[Fit] Collecting M0={M0} images → M={M} pairs …"
        )

        # ── Step 1: collect M0 individual histograms ──────────────────────
        # Pull from both x_a and x_b to maximise diversity of distributions.
        images = []
        for _, _, x_a, x_b in dataloader_train:
            for img in x_a.numpy():
                images.append(img)
                if len(images) >= M0:
                    break
            if len(images) < M0:
                for img in x_b.numpy():
                    images.append(img)
                    if len(images) >= M0:
                        break
            if len(images) >= M0:
                break

        if len(images) < M0:
            self.logger.warning(
                f"Only {len(images)} images available (requested M0={M0}). "
                "Reduce num_samples or increase dataset size."
            )
            M0 = len(images)
            M  = M0 * (M0 - 1) // 2

        self.logger.info(f"[Fit] Collected {M0} images → {M} pairs")

        # ── Step 2: enumerate all (i, j) pairs with i < j ─────────────────
        pairs = [(i, j) for i in range(M0) for j in range(i + 1, M0)]

        Phi_f_list, Phi_g_list = [], []
        y_f_list,   y_g_list   = [], []

        pbar = tqdm(total=M, desc="All pairs")
        for count, (i, j) in enumerate(pairs):
            a = images[i]
            b = images[j]

            # ground-truth potentials via Sinkhorn (Algorithm 1)
            f_gt, g_gt = self._solve_entropic_ot(a, b)

            # sliced-OT feature matrices
            Xf, Xg = self._compute_features(a, b)

            # centre to remove additive gauge constant of the dual
            f_gt = f_gt - f_gt.mean()
            g_gt = g_gt - g_gt.mean()
            Xf   = Xf - Xf.mean(axis=0, keepdims=True)
            Xg   = Xg - Xg.mean(axis=0, keepdims=True)

            Phi_f_list.append(Xf)
            Phi_g_list.append(Xg)
            y_f_list.append(f_gt)
            y_g_list.append(g_gt)

            pbar.update(1)
            if (count + 1) % 20 == 0:
                self.logger.info(
                    f"Pair {count+1}/{M} | "
                    f"||f_gt||={np.linalg.norm(f_gt):.4f}, "
                    f"||g_gt||={np.linalg.norm(g_gt):.4f}"
                )

        pbar.close()

        # --- stack all pairs → joint regression ---
        Phi_f = np.vstack(Phi_f_list)        # (count * n, L)
        Phi_g = np.vstack(Phi_g_list)
        y_f   = np.concatenate(y_f_list)     # (count * n,)
        y_g   = np.concatenate(y_g_list)

        # Normalise each feature column to unit std so that regression
        # coefficients are comparable across projection directions.
        # Save scale factors to apply the same normalisation at test time.
        self.Xf_col_scale = np.std(Phi_f, axis=0).clip(1e-12)
        self.Xg_col_scale = np.std(Phi_g, axis=0).clip(1e-12)
        Phi_f = Phi_f / self.Xf_col_scale[None, :]
        Phi_g = Phi_g / self.Xg_col_scale[None, :]

        self.logger.info(
            f"[Fit] Phi_f shape: {Phi_f.shape} → solving simplex LS for α …"
        )
        self.logger.info(
            f"[Fit] y_f range: [{y_f.min():.4f}, {y_f.max():.4f}] | "
            f"Phi_f range: [{Phi_f.min():.4f}, {Phi_f.max():.4f}]"
        )
        alpha = optimal_alpha_simplex(Phi_f, y_f, ridge=self.cfg_m.ridge)

        self.logger.info("[Fit] Solving simplex LS for β …")
        self.logger.info(
            f"[Fit] y_g range: [{y_g.min():.4f}, {y_g.max():.4f}] | "
            f"Phi_g range: [{Phi_g.min():.4f}, {Phi_g.max():.4f}]"
        )
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

        # ==========================================
        # ĐỒNG BỘ CENTERING (Nếu bạn đã dùng ở _fit)
        # ==========================================
        Xf = Xf - np.mean(Xf, axis=0, keepdims=True)
        Xg = Xg - np.mean(Xg, axis=0, keepdims=True)

        # Apply the same column normalisation used during fitting
        if hasattr(self, "Xf_col_scale"):
            Xf = Xf / self.Xf_col_scale[None, :]
            Xg = Xg / self.Xg_col_scale[None, :]

        f_pred = Xf @ alpha    # (n,)
        g_pred = Xg @ beta     # (n,)
        return f_pred, g_pred

    def _potentials_to_plan(self, f: np.ndarray, g: np.ndarray) -> np.ndarray:
        """
        Recover the (regularised) transport plan from Kantorovich potentials.

        P_{ij} = exp((f_i + g_j - C_ij) / eps)  (up to normalisation).

        Notes
        -----
        - Potentials are centred before exponentiation to prevent overflow.
          The dual (f, g) has a free additive constant: (f+c, g-c) leaves
          the plan invariant up to a global scalar, so subtracting means is safe.
        - The plan is clipped and normalised to sum=1 so it can be used
          directly as a probability distribution in np.random.choice.
        """
        eps = self.cfg_m.epsilon

        # Centre potentials to prevent exp overflow / underflow
        f_c = f - f.mean()
        g_c = g - g.mean()

        # Compute log-plan then shift by max for numerical stability
        log_P = f_c[:, None] / eps - self.C / eps + g_c[None, :] / eps
        log_P -= log_P.max()
        P = np.exp(log_P)

        # Clip tiny negatives from floating-point noise, normalise to sum=1
        P = np.clip(P, 0.0, None)
        P_sum = P.sum()
        if P_sum > 0:
            P /= P_sum
        return P

    # ------------------------------------------------------------------
    # Geodesic interpolation (moved from OT_Discrete)
    # ------------------------------------------------------------------

    @staticmethod
    def interp(P, num_inter, batch_size, img_size):
        P_flatten = P.flatten()
        grid = []
        for i in np.linspace(1, 0, num=img_size):
            for j in np.linspace(0, 1, num=img_size):
                grid.append([j, i])
        x_grid = np.array(grid)
        y_grid = np.array(grid)

        n_pixels = img_size * img_size   # n = 784 for img_size=28
        def get_hist(t, P_flat):
            map_samples = np.random.choice(range(len(P_flat)), size=batch_size, p=P_flat)
            # P has shape (n_pixels, n_pixels), flat index = i * n_pixels + j
            a_samples = x_grid[map_samples // n_pixels]   # source pixel index i
            b_samples = y_grid[map_samples % n_pixels]    # target pixel index j
            proj_samples = (1.0 - t) * a_samples + t * b_samples
            hist, _, _ = np.histogram2d(
                proj_samples[:, 1], proj_samples[:, 0],
                bins=np.linspace(0.0, 1.0, num=img_size + 1),
            )
            hist = np.flipud(hist)
            # Only clip if there are actually non-zero entries above the threshold.
            # Using quantile on a sparse hist (many zeros) gives thresh=0
            # which wipes everything → white image.
            nonzero = hist[hist > 0]
            if len(nonzero) > 0:
                thresh = np.quantile(nonzero, 0.9)  # 90th pctile of NON-ZERO bins only
                if thresh > 0:
                    hist = np.clip(hist, 0, thresh)
            if hist.max() > 0:
                hist = hist / hist.max()
            return hist

        return [get_hist(t, P_flatten) for t in np.linspace(0, 1, num=num_inter)]

    # ------------------------------------------------------------------
    # Evaluation & visualisation (mirrors OT_Discrete.OT_D_test)
    # ------------------------------------------------------------------

    def _evaluate(self, dataloader_test, alpha: np.ndarray, beta: np.ndarray):
        """Compute transport plans, report RMSE in potentials, and save geodesics."""
        from Utils import utils

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

            # Normm
            f_pred_c = f_pred - f_pred.mean()
            f_gt_c   = f_gt - f_gt.mean()
            g_pred_c = g_pred - g_pred.mean()
            g_gt_c   = g_gt - g_gt.mean()
            # Potential RMSE
            # rmse_f = float(np.sqrt(np.mean((f_pred - f_gt) ** 2)))
            # rmse_g = float(np.sqrt(np.mean((g_pred - g_gt) ** 2)))

            # Potential RMSE (on centered potentials — apples-to-apples)
            rmse_f = float(np.sqrt(np.mean((f_pred_c - f_gt_c) ** 2)))
            rmse_g = float(np.sqrt(np.mean((g_pred_c - g_gt_c) ** 2)))

            msg = (
                f"[Eval {idx}]  RMSE_f={rmse_f:.6f}  RMSE_g={rmse_g:.6f} | "
                f"plan_sum_gt={P_gt.sum():.4f}  plan_sum_pred={P_pred.sum():.4f}"
            )
            print(msg)
            self.logger.info(msg)

            # Geodesic interpolation images — call via class, not self, to avoid
            # Python treating P as a positional arg for the first parameter
            imgs_gt   = OT_Regression_Sliced.interp(P_gt,   num_inter=11, batch_size=50_000, img_size=img_size)
            imgs_pred = OT_Regression_Sliced.interp(P_pred, num_inter=11, batch_size=50_000, img_size=img_size)

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
