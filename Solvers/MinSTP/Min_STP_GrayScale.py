import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from Solvers.DefenseTrain import Defense_Train_Base

class _SoftTopK(torch.autograd.Function):
    @staticmethod
    def _solve(s, t, a, b, e, eps=1e-12):
        s, t, a, b, e = [x.to(torch.float64) for x in (s, t, a, b, e)]
        diff      = torch.clamp(s - t, max=40.0)
        exp_term  = torch.exp(diff)
        prod      = torch.clamp(a * b * exp_term, min=0.0, max=1e30)
        inside    = torch.clamp(e ** 2 + prod, min=0.0, max=1e30)
        sqrt_term = torch.sqrt(inside)
        z         = torch.clamp(torch.abs(e) + sqrt_term, min=eps)
        ab        = torch.clamp(torch.where(e > 0, a, b), min=eps)
        out_pos   = t + torch.log(z) - torch.log(ab)
        out_neg   = s - torch.log(z) + torch.log(ab)
        return torch.clamp(torch.where(e > 0, out_pos, out_neg), -1e6, 1e6).to(s.dtype)

    @staticmethod
    def forward(ctx, r, k, alpha, descending=False):
        batch_size, num_dim = r.shape
        x = torch.empty_like(r, requires_grad=False)

        def _finding_b():
            scaled = torch.sort(r, dim=1)[0].div_(alpha)
            eB = torch.logcumsumexp(scaled, dim=1).sub_(scaled).exp_()
            torch.neg(scaled, out=x)
            eA = torch.flip(x, dims=(1,))
            torch.logcumsumexp(eA, dim=1, out=x)
            idx = torch.arange(num_dim - 1, -1, -1, device=x.device)
            torch.index_select(x, 1, idx, out=eA)
            tmp = torch.clamp(eA + scaled, max=40.0)
            eA  = torch.exp(tmp)
            row = torch.arange(1, 2 * num_dim + 1, 2, device=r.device)
            torch.add(torch.add(eA, eB, alpha=-1, out=x), row.view(1, -1), out=x)
            w = (k if descending else num_dim - k).unsqueeze(1)
            i = torch.searchsorted(x, 2 * w)
            m = torch.clamp(i - 1, 0, num_dim - 1)
            n = torch.clamp(i,     0, num_dim - 1)
            zero_col = torch.zeros(batch_size, 1, dtype=eA.dtype, device=eA.device)
            a_val = torch.where(i < num_dim, eA.gather(1, n), zero_col.expand_as(n))
            b_val = torch.where(i > 0,       eB.gather(1, m), zero_col.expand_as(m))
            return _SoftTopK._solve(
                scaled.gather(1, m), scaled.gather(1, n),
                a_val, b_val, w - i)

        b_val = _finding_b()
        sign  = -1 if descending else 1
        torch.div(r, alpha * sign, out=x)
        x.sub_(sign * b_val)
        sign_x = x > 0
        p = torch.abs(x)
        p.neg_().exp_().mul_(0.5)
        inv_alpha = -sign / alpha
        S = torch.sum(p, dim=1, keepdim=True).mul_(inv_alpha)
        torch.where(sign_x, 1 - p, p, out=p)
        ctx.save_for_backward(r, x, S)
        ctx.alpha = alpha
        return p

    @staticmethod
    def backward(ctx, grad_output):
        r, x, S = ctx.saved_tensors
        alpha    = ctx.alpha
        x.abs_().neg_()
        q      = torch.softmax(x, dim=1)
        torch.mul(q, grad_output, out=x)
        grad_k = x.sum(dim=1, keepdim=True)
        grad_r = (grad_k - grad_output).mul_(q).mul_(S)
        q.mul_(r)
        x.mul_(S / alpha)
        r.sub_(q.sum(dim=1, keepdim=True))
        x.mul_(r)
        grad_alpha = x.sum()
        return grad_r, grad_k.squeeze(1), grad_alpha, None


def _soft_permutation(r: torch.Tensor, alpha: torch.Tensor,
                      descending: bool = False) -> torch.Tensor:
    n    = r.shape[0]
    r_   = r.unsqueeze(0)                          # (1, n)
    k    = torch.arange(1, n, device=r.device,
                         dtype=r.dtype)             # (n-1,)
    br   = r_.repeat(len(k), 1)                    # (n-1, n)
    sk   = _SoftTopK.apply(br, k, alpha, descending)  # (n-1, n)
    zeros = torch.zeros(1, n, dtype=r.dtype, device=r.device)
    ones  = torch.ones(1,  n, dtype=r.dtype, device=r.device)
    result = torch.cat([zeros, sk, ones], dim=0)   # (n+1, n)
    Pl = result[1:] - result[:-1]                  # (n, n)
    return Pl


def _hard_sort_matrix(scores: torch.Tensor) -> torch.Tensor:
    n   = len(scores)
    idx = torch.argsort(scores)
    P   = torch.zeros(n, n, dtype=scores.dtype, device=scores.device)
    P[torch.arange(n, device=scores.device), idx] = 1.0
    return P


class SlicerNet(nn.Module):
    def __init__(self, n_pixels: int = 784,
                 coord_dim: int = 2,
                 context_dim: int = 32,
                 hidden_dim: int = 128):
        super().__init__()
        self.context_enc = nn.Sequential(
            nn.Linear(2 * n_pixels, hidden_dim, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden_dim, context_dim, dtype=torch.float64),
            nn.Tanh(),
        )
        self.point_mlp = nn.Sequential(
            nn.Linear(coord_dim + context_dim, hidden_dim, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, dtype=torch.float64),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                x_grid: torch.Tensor) -> torch.Tensor:
        c     = self.context_enc(torch.cat([a, b]))          # (context_dim,)
        c_exp = c.unsqueeze(0).expand(x_grid.size(0), -1)    # (n, context_dim)
        inp   = torch.cat([x_grid, c_exp], dim=-1)           # (n, coord_dim+context_dim)
        return self.point_mlp(inp).squeeze(-1)                # (n,)


class Min_STP_GrayScale(Defense_Train_Base):
    def __init__(self, cfg_proj, cfg_m):
        Defense_Train_Base.__init__(self, cfg_proj, cfg_m, name="Min_STP_GrayScale")
        self._build_grid()
        self._build_slicer()

    def _build_grid(self):
        s = int(self.cfg_m.img_size)                          # 28
        grid = [[j, i]
                for i in np.linspace(1, 0, num=s)
                for j in np.linspace(0, 1, num=s)]
        self.x_grid_np = np.array(grid, dtype=np.float64)    # (784, 2)
        self.x_grid    = torch.tensor(
            self.x_grid_np, dtype=torch.float64).to(self.device)

        diff    = self.x_grid_np[:, None, :] - self.x_grid_np[None, :, :]
        self.C_np = np.sum(diff ** 2, axis=-1)                # (784, 784)
        self.C_t  = torch.tensor(
            self.C_np, dtype=torch.float64).to(self.device)

        self.logger.info(
            f"[Min_STP_GrayScale] grid={s}x{s}  "
            f"n_pixels={s**2}  C=[{self.C_np.min():.4f},{self.C_np.max():.4f}]")

    def _build_slicer(self):
        cfg         = self.cfg_m
        n_pixels    = int(self.cfg_m.img_size) ** 2         # 784
        context_dim = int(cfg.get("context_dim") or 32)
        hidden_dim  = int(cfg.get("hidden_dim")  or 128)

        self.slicer = SlicerNet(
            n_pixels    = n_pixels,
            coord_dim   = 2,
            context_dim = context_dim,
            hidden_dim  = hidden_dim,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.slicer.parameters())
        self.logger.info(
            f"[Min_STP_GrayScale] SlicerNet params={n_params:,}  "
            f"context_dim={context_dim}  hidden_dim={hidden_dim}")

    def train(self, dataloader_train):
        cfg    = self.cfg_m
        T      = int(cfg.get("num_train_iter") or 5000)
        lr     = float(cfg.get("learning_rate") or 1e-3)
        alpha  = float(cfg.get("alpha")         or 0.05)   # soft_perm temperature
        log_iv = int(cfg.get("log_interval")    or 100)
        max_gn = float(cfg.get("max_grad_norm") or 1.0)

        # Collect all training pairs into memory
        pool_a, pool_b = [], []
        for _, _, a_batch, b_batch in dataloader_train:
            for a, b in zip(a_batch, b_batch):
                pool_a.append(a.to(self.device, dtype=torch.float64))
                pool_b.append(b.to(self.device, dtype=torch.float64))
        M = len(pool_a)
        self.logger.info(
            f"[Min_STP_GrayScale] Training  T={T}  M={M}  lr={lr}  alpha={alpha}")

        alpha_t = torch.tensor(alpha, dtype=torch.float64, device=self.device)
        opt     = torch.optim.Adam(self.slicer.parameters(), lr=lr)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T, eta_min=lr * 0.01)

        loss_ema = None
        rng      = np.random.default_rng(42)
        t0       = time.time()
        pbar     = tqdm(total=T, desc="Min_STP_GrayScale")

        for step in range(T):
            idx   = int(rng.integers(0, M))
            a     = pool_a[idx]           # (784,) float64
            b     = pool_b[idx]           # (784,) float64

            self.slicer.train()
            scores = self.slicer(a, b, self.x_grid)   # (784,) float64

            P_soft = _soft_permutation(scores, alpha_t)    # (784, 784) [rank, elem]
            P_hard = _hard_sort_matrix(scores)             # (784, 784) [rank, elem]

            P1 = P_soft.T @ P_hard                         # (784, 784) [elemX, elemY]
            P2 = P_hard.T @ P_soft                         # symmetric counterpart
            pi = (P1 + P2) * 0.5

            loss = (pi * self.C_t).sum()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.slicer.parameters(), max_gn)
            opt.step()
            sched.step()

            lv       = loss.item()
            loss_ema = lv if loss_ema is None else 0.95 * loss_ema + 0.05 * lv
            pbar.update(1)

            if (step + 1) % log_iv == 0:
                msg = (f"[{step+1}/{T}]  cost_ema={loss_ema:.4e}  "
                       f"t={time.time()-t0:.1f}s")
                pbar.set_description(msg)
                self.logger.info(msg)

        pbar.close()

        ckpt = os.path.join(self.log_sub_folder, "slicer.pt")
        torch.save(self.slicer.state_dict(), ckpt)
        self.logger.info(f"[Min_STP_GrayScale] Saved → {ckpt}")

    def predict_plan(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        device = self.device
        a_t = torch.tensor(a, dtype=torch.float64, device=device)
        b_t = torch.tensor(b, dtype=torch.float64, device=device)

        self.slicer.eval()
        with torch.no_grad():
            scores = self.slicer(a_t, b_t, self.x_grid)   # (n,)

        n = len(a)
        u = torch.argsort(scores)   
        v = torch.argsort(scores)  
        rA     = torch.cumsum(a_t[u], dim=0)   # (n,)
        rB     = torch.cumsum(b_t[v], dim=0)   # (n,)
        rA_pad = torch.cat([torch.zeros(1, device=device), rA])  # (n+1,)
        rB_pad = torch.cat([torch.zeros(1, device=device), rB])  # (n+1,)

        r_all, _ = torch.sort(torch.cat([rA, rB]))
        r_all    = r_all[:-1]                  # remove trailing 1.0

        wA = (torch.searchsorted(
            rA_pad.contiguous(), r_all.contiguous(), side='right') - 1
              ).clamp(0, n - 1)
        wB = (torch.searchsorted(
            rB_pad.contiguous(), r_all.contiguous(), side='right') - 1
              ).clamp(0, n - 1)
        r_full  = torch.cat([torch.zeros(1, device=device),
                             r_all,
                             torch.ones(1, device=device)])
        delta_r = (r_full[1:] - r_full[:-1])[:-1]  # (len(r_all),)

        mask    = delta_r > 1e-15
        wA      = wA[mask]
        wB      = wB[mask]
        delta_r = delta_r[mask]

        orig_A = u[wA]   # source pixel indices
        orig_B = v[wB]   # target pixel indices

        # Accumulate into plan matrix
        P = torch.zeros(n, n, dtype=torch.float64, device=device)
        P.index_put_((orig_A, orig_B), delta_r, accumulate=True)

        P_np = P.cpu().numpy()
        P_np = np.clip(P_np, 0.0, None)
        s    = P_np.sum()
        if s > 0:
            P_np /= s
        return P_np