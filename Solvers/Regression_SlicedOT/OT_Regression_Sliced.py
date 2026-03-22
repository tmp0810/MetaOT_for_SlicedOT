import os
import numpy as np
import ot
import torch
from tqdm import tqdm

from Solvers.DefenseTrain import Defense_Train_Base
from regression_OT_utils import (
    generate_uniform_unit_sphere_projections,
    emd1D,
    emd1D_dual,
)
def _ridge_regression(X, y, ridge=0.0):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    H   = X.T @ X
    Xty = X.T @ y
    if ridge > 0:
        H = H + ridge * np.eye(H.shape[0])
    return np.linalg.solve(H, Xty)

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
        self.projection_matrix = proj.detach().numpy()
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
        a_safe = np.clip(a, 1e-10, None); a_safe /= a_safe.sum()
        b_safe = np.clip(b, 1e-10, None); b_safe /= b_safe.sum()
        log_a = np.log(a_safe)
        log_b = np.log(b_safe)
        log_K = -self.C / eps

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

    def _compute_features(self, a: np.ndarray, b: np.ndarray):
        device = torch.device(
            f"cuda:{self.cfg_m.gpu}"
            if torch.cuda.is_available() and hasattr(self.cfg_m, "gpu")
            else "cpu"
        )
        n = len(a)
        L = self.projection_matrix.shape[0]

        proj_values = torch.tensor(
            (self.x_grid @ self.projection_matrix.T).T,
            dtype=torch.float64, device=device
        )

        a_t = torch.tensor(a, dtype=torch.float64, device=device)
        b_t = torch.tensor(b, dtype=torch.float64, device=device)

        f_grad, g_grad, _ = emd1D_dual(
            proj_values, proj_values,
            u_weights=a_t,
            v_weights=b_t,
            p=2,
            require_sort=True,
        )

        Xf = f_grad.cpu().numpy().T
        Xg = g_grad.cpu().numpy().T

        return Xf, Xg

    def _fit(self, dataloader_train):
        M = self.cfg_m.num_bootstrap
        self.logger.info(f"[Fit] Collecting M={M} pairs …")

        Phi_f_list = []
        y_f_list   = []
        count = 0

        pbar = tqdm(total=M, desc="Pairs")
        for _, _, x_a, x_b in dataloader_train:
            for a, b in zip(x_a.numpy(), x_b.numpy()):
                if count >= M:
                    break

                f_gt, g_gt = self._solve_entropic_ot(a, b)

                eps = self.cfg_m.epsilon
                f_gt = f_gt - eps * np.log(np.clip(a, 1e-10, None))
                f_gt = f_gt - f_gt.mean()

                Xf, _ = self._compute_features(a, b)
                Xf = Xf - Xf.mean(axis=0, keepdims=True)

                Phi_f_list.append(Xf)
                y_f_list.append(f_gt)

                count += 1
                pbar.update(1)
                if count % 20 == 0:
                    self.logger.info(f"Pair {count}/{M} | ||f_gt||={np.linalg.norm(f_gt):.4f}")
            if count >= M:
                break
        pbar.close()

        Phi_f = np.vstack(Phi_f_list)
        y_f   = np.concatenate(y_f_list)

        self.logger.info(
            f"[Fit] Phi_f: {Phi_f.shape} | "
            f"y_f range: [{y_f.min():.4f}, {y_f.max():.4f}]"
        )

        import time
        ridge = self.cfg_m.ridge
        self.logger.info(f"[Fit] Solving ridge regression for α …")
        with tqdm(total=1, desc="Ridge α", bar_format="{desc}: {elapsed}") as pbar:
            t0 = time.time()
            alpha = _ridge_regression(Phi_f, y_f, ridge)
            pbar.update(1)
            pbar.set_description(f"Ridge α done in {time.time()-t0:.2f}s")

        self.logger.info(
            f"[Fit] α: min={alpha.min():.4f}, max={alpha.max():.4f}, "
            f"nnz={np.sum(np.abs(alpha) > 1e-6)}/{len(alpha)}"
        )
        np.save(os.path.join(self.log_sub_folder, "alpha.npy"), alpha)
        self.logger.info(f"[Fit] Saved alpha to {self.log_sub_folder}")
        return alpha

    def _precompute_log_K(self) -> torch.Tensor:
        eps = float(self.cfg_m.epsilon)
        C_t = torch.tensor(self.C, dtype=torch.float64, device=self.device)
        return -C_t / eps

    def _g_from_f(self, f: torch.Tensor, b: torch.Tensor,
                  log_K: torch.Tensor, eps: float) -> torch.Tensor:
        log_b = torch.log(b.clamp(1e-300))
        M     = log_K + f.unsqueeze(1) / eps
        m     = M.max(dim=0, keepdim=True).values
        lse   = (M - m).exp().sum(dim=0).log() + m.squeeze(0)
        return eps * (log_b - lse)

    def _f_from_g(self, g: torch.Tensor, a: torch.Tensor,
                  log_K: torch.Tensor, eps: float) -> torch.Tensor:
        log_a = torch.log(a.clamp(1e-300))
        M     = log_K + g.unsqueeze(0) / eps
        m     = M.max(dim=1, keepdim=True).values
        lse   = (M - m).exp().sum(dim=1).log() + m.squeeze(1)
        return eps * (log_a - lse)

    def _predict_potentials(
        self,
        a: np.ndarray,
        b: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray = None,
    ):
        Xf, _ = self._compute_features(a, b)
        Xf    = Xf - Xf.mean(axis=0, keepdims=True)
        f_pred = Xf @ alpha
        eps    = float(self.cfg_m.epsilon)
        f_pred = f_pred + eps * np.log(np.clip(a, 1e-10, None))

        log_K = self._precompute_log_K()
        a_t   = torch.tensor(a,      dtype=torch.float64, device=self.device)
        f_t   = torch.tensor(f_pred, dtype=torch.float64, device=self.device)
        b_t   = torch.tensor(b,      dtype=torch.float64, device=self.device)
        with torch.no_grad():
            g_t = self._g_from_f(f_t, b_t, log_K, eps)
            f_t = self._f_from_g(g_t, a_t, log_K, eps)
        return f_t.cpu().numpy(), g_t.cpu().numpy()

    def _potentials_to_plan(self, f: np.ndarray, g: np.ndarray) -> np.ndarray:
        eps = self.cfg_m.epsilon

        f_c = f - f.mean()
        g_c = g - g.mean()

        log_P = f_c[:, None] / eps - self.C / eps + g_c[None, :] / eps
        log_P -= log_P.max()
        P = np.exp(log_P)

        P = np.clip(P, 0.0, None)
        P_sum = P.sum()
        if P_sum > 0:
            P /= P_sum
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
            P_gt        = self._potentials_to_plan(f_gt, g_gt)

            f_pred, g_pred = self._predict_potentials(a, b, alpha, beta)
            P_pred          = self._potentials_to_plan(f_pred, g_pred)

            rmse_f = float(np.sqrt(np.mean((f_pred - f_gt) ** 2)))
            rmse_g = float(np.sqrt(np.mean((g_pred - g_gt) ** 2)))
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
        self.alpha = self._fit(dataloader_train)
        self.beta  = np.zeros(self.projection_matrix.shape[0])
        self._evaluate(dataloader_test, self.alpha, self.beta)
