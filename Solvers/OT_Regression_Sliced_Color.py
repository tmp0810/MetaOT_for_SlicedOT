import os
import numpy as np
import torch
from tqdm import tqdm
import ot

from Solvers.OT_Regression_Sliced import OT_Regression_Sliced, _ridge_regression
from regression_OT_utils import (
    generate_uniform_unit_sphere_projections,
    emd1D_dual,
)


class OT_Regression_Sliced_Color(OT_Regression_Sliced):

    def _build_grid(self):
        self.x_grid = None
        self.C      = None

    def __init__(self, cfg_proj, cfg_m):
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
        if C is None:
            raise ValueError("[Color] _solve_entropic_ot requires explicit C.")
        
        eps = self.cfg_m.epsilon

        a_safe = np.clip(a, 1e-10, None)
        a_safe /= a_safe.sum()
        b_safe = np.clip(b, 1e-10, None)
        b_safe /= b_safe.sum()
 
        _, log_dict = ot.sinkhorn(
            a_safe, b_safe, C, 
            reg=eps, 
            numItermax=self.cfg_m.sinkhorn_iters, 
            stopThr=1e-5, 
            log=True
        )
        
        if 'alpha' in log_dict:
            f = log_dict['alpha']
            g = log_dict['beta']
        elif 'log_u' in log_dict:
            f = eps * log_dict['log_u']
            g = eps * log_dict['log_v']
        else:
            u_opt = log_dict.get('u', np.ones_like(a))
            v_opt = log_dict.get('v', np.ones_like(b))
            f = eps * np.log(np.clip(u_opt, 1e-50, None))
            g = eps * np.log(np.clip(v_opt, 1e-50, None))

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
        self, a: np.ndarray, b: np.ndarray,
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

        # Ép 1 bên về source a
        r = P.sum(axis=1) + 1e-12
        P = P * (a / r)[:, None]
        
        # Ép 1 bên về source b
        c = P.sum(axis=0) + 1e-12
        P = P * (b / c)[None, :]


        P     = np.clip(P, 0.0, None)
        P_sum = P.sum()
      
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

                a     = src_w[i].numpy()     
                x_src = src_c[i].numpy()    
                b     = tgt_w[i].numpy()     
                x_tgt = tgt_c[i].numpy()     

                # Per-pair cost matrix (n_src, n_tgt)
                C = self._compute_cost(x_src, x_tgt)

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


    def _predict_potentials_color(
        self,
        a: np.ndarray,
        b: np.ndarray,
        x_src: np.ndarray,
        x_tgt: np.ndarray,
    ):
        Xf, Xg = self._compute_features(a, b, x_src, x_tgt)
        Xf -= Xf.mean(axis=0, keepdims=True)
        Xg -= Xg.mean(axis=0, keepdims=True)

        f_transport = Xf @ self.alpha   # (n_src,) — transport-only
        g_transport = Xg @ self.beta    # (n_tgt,)

        # Add back log-density term
        eps    = self.cfg_m.epsilon
        f_pred = f_transport + eps * np.log(np.clip(a, 1e-10, None))
        g_pred = g_transport + eps * np.log(np.clip(b, 1e-10, None))
        return f_pred, g_pred

    def predict_plan(
        self,
        a: np.ndarray,
        b: np.ndarray,
        x_src: np.ndarray,
        x_tgt: np.ndarray,
    ) -> np.ndarray:
        C              = self._compute_cost(x_src, x_tgt)
        f_pred, g_pred = self._predict_potentials_color(a, b, x_src, x_tgt)
        return self._potentials_to_plan(a, b, f_pred, g_pred, C)

    def train(self, dataloader_train):
        """Fit and save regression weights."""
        self.alpha, self.beta = self._fit(dataloader_train)
        self.logger.info("[Color] Training complete.")
        return self.alpha, self.beta
