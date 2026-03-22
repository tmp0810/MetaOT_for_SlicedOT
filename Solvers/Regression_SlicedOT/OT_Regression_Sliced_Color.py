import os
import numpy as np
import torch
from tqdm import tqdm

from Solvers.Regression_SlicedOT.OT_Regression_Sliced import OT_Regression_Sliced, _ridge_regression
from regression_OT_utils import (
    generate_uniform_unit_sphere_projections,
    emd1D_dual,
)


class OT_Regression_Sliced_Color(OT_Regression_Sliced):

    def _build_grid(self):
        """No fixed pixel grid — support varies per pair."""
        self.x_grid = None
        self.C      = None

    def __init__(self, cfg_proj, cfg_m):
        # super().__init__ calls _build_grid (our override → no-op) and
        # then sets self.projection_matrix with dim=2 (from pixel space).
        super().__init__(cfg_proj, cfg_m)

        # Override projection directions: R^3 for RGB
        L    = self.cfg_m.num_projections
        proj = generate_uniform_unit_sphere_projections(
            dim=3, num_projections=L,
            dtype=torch.float64, device="cpu",
        )
        self.projection_matrix = proj.detach().numpy()   # (L, 3)
        self.logger.info(
            f"[Color] projection_matrix: {self.projection_matrix.shape}  "
            f"dim=3 (RGB), L={L}"
        )

    def _compute_cost(
        self,
        x_src: np.ndarray,
        x_tgt: np.ndarray,
    ) -> np.ndarray:
        diff = x_src[:, None, :] - x_tgt[None, :, :]   # (n_src, n_tgt, 3)
        return np.sum(diff ** 2, axis=-1)


    def _solve_entropic_ot(
        self,
        a: np.ndarray,
        b: np.ndarray,
        C: np.ndarray = None,
    ):
        """
        Log-space Sinkhorn with explicit cost matrix.
        Overrides parent (which uses self.C).
        """
        if C is None:
            raise ValueError("[Color] _solve_entropic_ot requires explicit C.")

        eps    = self.cfg_m.epsilon
        a_safe = np.clip(a, 1e-10, None); a_safe /= a_safe.sum()
        b_safe = np.clip(b, 1e-10, None); b_safe /= b_safe.sum()
        log_a  = np.log(a_safe)
        log_b  = np.log(b_safe)
        log_K  = -C / eps

        def lse(X, axis):
            m = X.max(axis=axis, keepdims=True)
            return np.log(np.exp(X - m).sum(axis=axis)) + m.squeeze(axis=axis)

        f = np.zeros_like(a_safe)
        g = np.zeros_like(b_safe)
        for _ in range(self.cfg_m.sinkhorn_iters):
            g_new = eps * (log_b - lse(log_K + f[:, None] / eps, axis=0))
            f_new = eps * (log_a - lse(log_K + g_new[None, :] / eps, axis=1))
            if np.max(np.abs(f_new - f)) < 1e-6:
                f, g = f_new, g_new
                break
            f, g = f_new, g_new

        if f.std() < 1e-8:
            raise RuntimeError(f"f_gt is constant (std={f.std():.2e}).")
        return f, g


    def _compute_features(
        self,
        a: np.ndarray,
        b: np.ndarray,
        x_src: np.ndarray = None,
        x_tgt: np.ndarray = None,
    ):
        if x_src is None or x_tgt is None:
            raise ValueError("[Color] _compute_features requires x_src and x_tgt.")

        device = torch.device(
            f"cuda:{self.cfg_m.gpu}"
            if torch.cuda.is_available() and hasattr(self.cfg_m, "gpu")
            else "cpu"
        )
        L        = self.projection_matrix.shape[0]
        proj_mat = torch.tensor(self.projection_matrix, dtype=torch.float64, device=device)   # (L, 3)
        src_t    = torch.tensor(x_src, dtype=torch.float64, device=device)   # (n_src, 3)
        tgt_t    = torch.tensor(x_tgt, dtype=torch.float64, device=device)   # (n_tgt, 3)

        # (L, n_src), (L, n_tgt)
        proj_src = (src_t @ proj_mat.T).T
        proj_tgt = (tgt_t @ proj_mat.T).T

        a_t = torch.tensor(a, dtype=torch.float64, device=device)
        b_t = torch.tensor(b, dtype=torch.float64, device=device)

        f_grad, g_grad, _ = emd1D_dual(
            proj_src, proj_tgt,
            u_weights=a_t,
            v_weights=b_t,
            p=2,
            require_sort=True,
        )

        Xf = f_grad.cpu().numpy().T    # (n_src, L)
        Xg = g_grad.cpu().numpy().T    # (n_tgt, L)
        return Xf, Xg

    def _potentials_to_plan(
        self,
        f: np.ndarray,
        g: np.ndarray,
        C: np.ndarray = None,
    ) -> np.ndarray:
        if C is None:
            raise ValueError("[Color] _potentials_to_plan requires explicit C.")

        eps   = self.cfg_m.epsilon
        f_c   = f - f.mean()
        g_c   = g - g.mean()

        log_P = f_c[:, None] / eps - C / eps + g_c[None, :] / eps
        log_P -= log_P.max()
        P     = np.exp(log_P)
        P     = np.clip(P, 0.0, None)
        P_sum = P.sum()
        if P_sum > 0:
            P /= P_sum
        return P


    def _fit(self, dataloader_train):
        M   = self.cfg_m.num_bootstrap
        eps = self.cfg_m.epsilon
        self.logger.info(f"[Color] Fitting on M={M} pairs ...")

        Phi_f_list, Phi_g_list = [], []
        y_f_list,   y_g_list   = [], []
        count = 0

        pbar = tqdm(total=M, desc="Color pairs")
        for src_w, src_c, tgt_w, tgt_c in dataloader_train:
            for i in range(src_w.shape[0]):
                if count >= M:
                    break

                a     = src_w[i].numpy()      # (n_src,)
                x_src = src_c[i].numpy()      # (n_src, 3)
                b     = tgt_w[i].numpy()      # (n_tgt,)
                x_tgt = tgt_c[i].numpy()      # (n_tgt, 3)

                # Per-pair cost matrix (n_src, n_tgt)
                C = self._compute_cost(x_src, x_tgt)

                # Ground-truth Sinkhorn potentials
                try:
                    f_gt, g_gt = self._solve_entropic_ot(a, b, C)
                except RuntimeError as e:
                    self.logger.warning(f"Skipping pair {count}: {e}")
                    continue

                f_clean = f_gt - eps * np.log(np.clip(a, 1e-10, None))
                g_clean = g_gt - eps * np.log(np.clip(b, 1e-10, None))
                f_clean -= f_clean.mean()
                g_clean -= g_clean.mean()

                # 1-D sliced features in R^3
                Xf, Xg = self._compute_features(a, b, x_src, x_tgt)
                Xf -= Xf.mean(axis=0, keepdims=True)
                Xg -= Xg.mean(axis=0, keepdims=True)

                Phi_f_list.append(Xf)
                Phi_g_list.append(Xg)
                y_f_list.append(f_clean)
                y_g_list.append(g_clean)

                count += 1
                pbar.update(1)

                if count % 20 == 0:
                    self.logger.info(
                        f"Pair {count}/{M}  "
                        f"||f_clean||={np.linalg.norm(f_clean):.4f}  "
                        f"||g_clean||={np.linalg.norm(g_clean):.4f}"
                    )
            if count >= M:
                break
        pbar.close()

        if count == 0:
            raise RuntimeError("[Color] No valid training pairs collected.")

        Phi_f = np.vstack(Phi_f_list)       # (count * n_src, L)
        Phi_g = np.vstack(Phi_g_list)       # (count * n_tgt, L)
        y_f   = np.concatenate(y_f_list)    # (count * n_src,)
        y_g   = np.concatenate(y_g_list)

        self.logger.info(
            f"[Color] Phi_f: {Phi_f.shape}  "
            f"y_f in [{y_f.min():.4f}, {y_f.max():.4f}]"
        )

        ridge = self.cfg_m.ridge
        self.logger.info("[Color] Solving ridge regression for alpha ...")
        alpha = _ridge_regression(Phi_f, y_f, ridge)
        self.logger.info("[Color] Solving ridge regression for beta ...")
        beta  = _ridge_regression(Phi_g, y_g, ridge)

        self.logger.info(
            f"[Color] alpha: [{alpha.min():.4f}, {alpha.max():.4f}]  "
            f"beta: [{beta.min():.4f}, {beta.max():.4f}]"
        )

        np.save(os.path.join(self.log_sub_folder, "alpha.npy"), alpha)
        np.save(os.path.join(self.log_sub_folder, "beta.npy"),  beta)
        self.logger.info(f"[Color] Saved alpha/beta -> {self.log_sub_folder}")
        return alpha, beta

    def _g_from_f(self, f: torch.Tensor, b: torch.Tensor,
                  log_K: torch.Tensor, eps: float) -> torch.Tensor:
        """One Sinkhorn step: g[j] = ε·log(b[j]) - ε·lse_i(log_K[i,j] + f[i]/ε)"""
        log_b = torch.log(b.clamp(1e-300))
        M     = log_K + f.unsqueeze(1) / eps
        m     = M.max(dim=0, keepdim=True).values
        lse   = (M - m).exp().sum(dim=0).log() + m.squeeze(0)
        return eps * (log_b - lse)

    def predict_plan(
        self,
        a: np.ndarray,
        b: np.ndarray,
        x_src: np.ndarray,
        x_tgt: np.ndarray,
    ) -> np.ndarray:
        """
        Inference: f = Φ_f @ α, g = g_from_f(f) via 1 Sinkhorn step.
        Same pipeline as Method 2 for fair inference time comparison.
        """
        eps = float(self.cfg_m.epsilon)
        C   = self._compute_cost(x_src, x_tgt)

        Xf, _ = self._compute_features(a, b, x_src, x_tgt)
        Xf    = Xf - Xf.mean(axis=0, keepdims=True)
        f_transport = Xf @ self.alpha   # transport-only
        f_pred = f_transport + eps * np.log(np.clip(a, 1e-10, None))

        # Derive g via 1 Sinkhorn step (not from β)
        log_K  = torch.tensor(-C / eps, dtype=torch.float64, device=self.device)
        f_t    = torch.tensor(f_pred, dtype=torch.float64, device=self.device)
        b_t    = torch.tensor(b,      dtype=torch.float64, device=self.device)
        with torch.no_grad():
            g_t = self._g_from_f(f_t, b_t, log_K, eps)

        return self._potentials_to_plan(f_pred, g_t.cpu().numpy(), C)
        
    def train(self, dataloader_train):
        """Fit and save regression weights."""
        self.alpha, self.beta = self._fit(dataloader_train)
        self.logger.info("[Color] Training complete.")
        return self.alpha, self.beta
