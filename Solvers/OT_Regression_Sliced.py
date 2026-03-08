import os
import numpy as np
import ot
import torch
from tqdm import tqdm

from Solvers.DefenseTrain import Defense_Train_Base
from regression_OT_utils import (
    generate_uniform_unit_sphere_projections,
    optimal_alpha_simplex,
    solve_1D_ot,                  
    solve_1D_ot_unsorted,         
)

class OT_Regression_Sliced(Defense_Train_Base):
    def __init__(self, cfg_proj, cfg_m):
        Defense_Train_Base.__init__(self, cfg_proj, cfg_m, name="OT_Regression_Sliced")
        self._build_grid()
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
    def _build_grid(self):
        s = self.cfg_m.img_size
        grid = []
        for i in np.linspace(1, 0, num=s):
            for j in np.linspace(0, 1, num=s):
                grid.append([j, i])
        self.x_grid = np.array(grid, dtype=np.float64)          
        diff = self.x_grid[:, None, :] - self.x_grid[None, :, :]
        self.C = np.sum(diff ** 2, axis=-1)                      

    def _solve_entropic_ot(self, a: np.ndarray, b: np.ndarray):
        eps = self.cfg_m.epsilon

        a_safe = np.clip(a, 1e-10, None)
        a_safe /= a_safe.sum()
        b_safe = np.clip(b, 1e-10, None)
        b_safe /= b_safe.sum()

        log_a = np.log(a_safe)   
        log_b = np.log(b_safe)   
        log_K = -self.C / eps    

        def lse(X, axis):
            m = X.max(axis=axis, keepdims=True)
            return np.log(np.exp(X - m).sum(axis=axis)) + m.squeeze(axis=axis)
        
        f = np.zeros_like(a_safe)

        for _ in range(self.cfg_m.sinkhorn_iters):
            g = eps * (log_b - lse(log_K + f[:, None] / eps, axis=0))
            f = eps * (log_a - lse(log_K + g[None, :] / eps, axis=1))

        return f, g
 

    @staticmethod
    def _sliced_1d_potentials(
        a: np.ndarray,
        b: np.ndarray,
        proj_positions: np.ndarray,
        p: int = 2,
    ):
        f1d, g1d, _ = solve_1D_ot_unsorted(a, b, proj_positions, proj_positions, p=p)
        return f1d, g1d

    def _compute_features(self, a: np.ndarray, b: np.ndarray):
 
        n = len(a)
        L = self.projection_matrix.shape[0]
        Xf = np.empty((n, L), dtype=np.float64)
        Xg = np.empty((n, L), dtype=np.float64)

        for l, theta in enumerate(self.projection_matrix):
            proj = self.x_grid @ theta   # (n,)
            f1d, g1d = self._sliced_1d_potentials(a, b, proj, p=2)
            Xf[:, l] = f1d
            Xg[:, l] = g1d

        return Xf, Xg
    
    def _fit(self, dataloader_train):
   
        M0 = self.cfg_m.num_samples
        M  = M0 * (M0 - 1) // 2
        self.logger.info(
            f"[Fit] Collecting M0={M0} images → M={M} pairs …"
        )

        # Collect M0
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


        pairs = [(i, j) for i in range(M0) for j in range(i + 1, M0)]

        Phi_f_list, Phi_g_list = [], []
        y_f_list,   y_g_list   = [], []

        pbar = tqdm(total=M, desc="All pairs")
        for count, (i, j) in enumerate(pairs):
            a = images[i]
            b = images[j]
            f_gt, g_gt = self._solve_entropic_ot(a, b)
            Xf, Xg = self._compute_features(a, b)

            # Centering tránh tràn số (overflow)

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

        Phi_f = np.vstack(Phi_f_list)        
        Phi_g = np.vstack(Phi_g_list)
        y_f   = np.concatenate(y_f_list)   
        y_g   = np.concatenate(y_g_list)

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

        alpha = alpha / self.Xf_col_scale
        beta  = beta  / self.Xg_col_scale

        self.logger.info(
            f"[Fit] α: min={alpha.min():.4f}, max={alpha.max():.4f}, "
            f"nnz={np.sum(alpha > 1e-6)}/{len(alpha)}"
        )
        self.logger.info(
            f"[Fit] β: min={beta.min():.4f},  max={beta.max():.4f}, "
            f"nnz={np.sum(beta  > 1e-6)}/{len(beta)}"
        )

        np.save(os.path.join(self.log_sub_folder, "alpha.npy"), alpha)
        np.save(os.path.join(self.log_sub_folder, "beta.npy"),  beta)
        self.logger.info(f"[Fit] Saved alpha/beta to {self.log_sub_folder}")

        return alpha, beta


    def _predict_potentials(
        self,
        a: np.ndarray,
        b: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ):  
        # Dùng Linear Regression
        Xf, Xg = self._compute_features(a, b)
        Xf = Xf - np.mean(Xf, axis=0, keepdims=True)
        Xg = Xg - np.mean(Xg, axis=0, keepdims=True)
        f_pred = Xf @ alpha    # (n,)
        g_pred = Xg @ beta     # (n,)
        return f_pred, g_pred

    def _potentials_to_plan(self, a: np.ndarray, b: np.ndarray, f: np.ndarray, g: np.ndarray) -> np.ndarray:
        eps = self.cfg_m.epsilon

        f_c = f - f.mean()
        g_c = g - g.mean()

        log_P = f_c[:, None] / eps - self.C / eps + g_c[None, :] / eps
        log_P -= log_P.max()
        P = np.exp(log_P)

        # Ép 1 bên về source a
        r = P.sum(axis=1) + 1e-12
        P = P * (a / r)[:, None]
        
        # Ép 1 bên về source b
        c = P.sum(axis=0) + 1e-12
        P = P * (b / c)[None, :]

        P = np.clip(P, 0.0, None)
        P_sum = P.sum()
        return P
    
    @staticmethod
    def interp(P, num_inter, batch_size, img_size):
        P_flatten = P.flatten()
        grid = []
        for i in np.linspace(1, 0, num=img_size):
            for j in np.linspace(0, 1, num=img_size):
                grid.append([j, i])
        x_grid = np.array(grid)
        y_grid = np.array(grid)

        n_pixels = img_size * img_size   
        def get_hist(t, P_flat):
            map_samples = np.random.choice(range(len(P_flat)), size=batch_size, p=P_flat)
            a_samples = x_grid[map_samples // n_pixels]  
            b_samples = y_grid[map_samples % n_pixels]    
            proj_samples = (1.0 - t) * a_samples + t * b_samples
            hist, _, _ = np.histogram2d(
                proj_samples[:, 1], proj_samples[:, 0],
                bins=np.linspace(0.0, 1.0, num=img_size + 1),
            )
            hist = np.flipud(hist)
            nonzero = hist[hist > 0]
            if len(nonzero) > 0:
                thresh = np.quantile(nonzero, 0.9) 
                if thresh > 0:
                    hist = np.clip(hist, 0, thresh)
            if hist.max() > 0:
                hist = hist / hist.max()
            return hist

        return [get_hist(t, P_flatten) for t in np.linspace(0, 1, num=num_inter)]


    def _evaluate(self, dataloader_test, alpha: np.ndarray, beta: np.ndarray):
        from Utils import utils

        for _, _, xs_a, xs_b in dataloader_test:
            xs_a_np = xs_a[:2].numpy()
            xs_b_np = xs_b[:2].numpy()
            break

        img_size = self.cfg_m.img_size

        for idx in range(len(xs_a_np)):
            a, b = xs_a_np[idx], xs_b_np[idx]
            f_gt, g_gt = self._solve_entropic_ot(a, b)
            P_gt = self._potentials_to_plan(a, b, f_gt, g_gt)
            f_pred, g_pred = self._predict_potentials(a, b, alpha, beta)
            P_pred = self._potentials_to_plan(a, b, f_pred, g_pred)

            # Normm
            f_pred_c = f_pred - f_pred.mean()
            f_gt_c   = f_gt - f_gt.mean()
            g_pred_c = g_pred - g_pred.mean()
            g_gt_c   = g_gt - g_gt.mean()

            rmse_f = float(np.sqrt(np.mean((f_pred_c - f_gt_c) ** 2)))
            rmse_g = float(np.sqrt(np.mean((g_pred_c - g_gt_c) ** 2)))

            msg = (
                f"[Eval {idx}]  RMSE_f={rmse_f:.6f}  RMSE_g={rmse_g:.6f} | "
                f"plan_sum_gt={P_gt.sum():.4f}  plan_sum_pred={P_pred.sum():.4f}"
            )
            print(msg)
            self.logger.info(msg)

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

    def train(self, dataloader_train, dataloader_test):
        alpha, beta = self._fit(dataloader_train)
        self._evaluate(dataloader_test, alpha, beta)
