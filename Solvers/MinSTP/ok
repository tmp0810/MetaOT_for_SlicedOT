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


def _soft_permutation(r: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    n     = r.shape[0]
    r_    = r.unsqueeze(0)
    k     = torch.arange(1, n, device=r.device, dtype=r.dtype)
    br    = r_.repeat(len(k), 1)
    sk    = _SoftTopK.apply(br, k, alpha, False)
    zeros = torch.zeros(1, n, dtype=r.dtype, device=r.device)
    ones  = torch.ones(1,  n, dtype=r.dtype, device=r.device)
    result = torch.cat([zeros, sk, ones], dim=0)
    return result[1:] - result[:-1]


class SlicerNetWorld(nn.Module):
    def __init__(self, n_supply: int, n_demand: int,
                 context_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.enc_a = nn.Sequential(
            nn.Linear(n_supply, hidden_dim, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden_dim, context_dim, dtype=torch.float64),
            nn.Tanh(),
        )
        self.enc_b = nn.Sequential(
            nn.Linear(n_demand, hidden_dim, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden_dim, context_dim, dtype=torch.float64),
            nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Linear(context_dim * 2, hidden_dim, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3, dtype=torch.float64),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        za    = self.enc_a(a)
        zb    = self.enc_b(b)
        theta = self.head(torch.cat([za, zb]))
        return theta / theta.norm().clamp(min=1e-8)


def _compute_1d_coupling(a_sorted, b_sorted, device):
    n_s = len(a_sorted)
    n_d = len(b_sorted)
    rA  = torch.cumsum(a_sorted, dim=0)
    rB  = torch.cumsum(b_sorted, dim=0)
    rA_pad = torch.cat([torch.zeros(1, device=device, dtype=torch.float64), rA])
    rB_pad = torch.cat([torch.zeros(1, device=device, dtype=torch.float64), rB])
    r_all, _ = torch.sort(torch.cat([rA, rB]))
    r_all    = r_all[:-1]
    r_full   = torch.cat([torch.zeros(1, device=device, dtype=torch.float64),
                           r_all,
                           torch.ones(1, device=device, dtype=torch.float64)])
    delta_r  = (r_full[1:] - r_full[:-1])[:-1]
    wA = (torch.searchsorted(rA_pad.contiguous(), r_all.contiguous(), right=True) - 1).clamp(0, n_s - 1)
    wB = (torch.searchsorted(rB_pad.contiguous(), r_all.contiguous(), right=True) - 1).clamp(0, n_d - 1)
    mask    = delta_r > 1e-15
    return wA[mask], wB[mask], delta_r[mask]


class Min_STP_World(Defense_Train_Base):
    def __init__(self, cfg_proj, cfg_m,
                 supply_euc: np.ndarray, demand_euc: np.ndarray,
                 supply_sph: np.ndarray = None, demand_sph: np.ndarray = None):
        self.supply_euc = supply_euc.astype(np.float64)
        self.demand_euc = demand_euc.astype(np.float64)
        self.supply_sph = supply_sph
        self.demand_sph = demand_sph
        self.n_supply   = len(supply_euc)
        self.n_demand   = len(demand_euc)

        Defense_Train_Base.__init__(self, cfg_proj, cfg_m, name="Min_STP_World")
        self._build_geometry()
        self._build_slicer()

    def _build_geometry(self):
        from Solvers.Regression_SlicedOT.OT_Regression_Sliced_World import _sphere_cost
        self.C_np = _sphere_cost(self.supply_euc, self.demand_euc)
        self.C_t  = torch.tensor(self.C_np, dtype=torch.float64).to(self.device)
        self.supply_t = torch.tensor(self.supply_euc, dtype=torch.float64).to(self.device)
        self.demand_t = torch.tensor(self.demand_euc, dtype=torch.float64).to(self.device)
        self.logger.info(
            f"[Min_STP_World] n_supply={self.n_supply}  n_demand={self.n_demand}  "
            f"C=[{self.C_np.min():.4f},{self.C_np.max():.4f}]")

    def _build_slicer(self):
        cfg         = self.cfg_m
        context_dim = int(cfg.get("context_dim") or 32)
        hidden_dim  = int(cfg.get("hidden_dim")  or 128)
        self.slicer = SlicerNetWorld(
            n_supply    = self.n_supply,
            n_demand    = self.n_demand,
            context_dim = context_dim,
            hidden_dim  = hidden_dim,
        ).to(self.device)
        n_params = sum(p.numel() for p in self.slicer.parameters())
        self.logger.info(
            f"[Min_STP_World] SlicerNet params={n_params:,}  "
            f"context_dim={context_dim}  hidden_dim={hidden_dim}")

    def train(self, dataloader_train):
        cfg    = self.cfg_m
        T      = int(cfg.get("num_train_iter") or 5000)
        lr     = float(cfg.get("learning_rate") or 1e-3)
        alpha  = float(cfg.get("alpha")         or 0.05)
        log_iv = int(cfg.get("log_interval")    or 100)
        max_gn = float(cfg.get("max_grad_norm") or 1.0)
        device = self.device

        pool_a, pool_b = [], []
        for _, _, a_batch, b_batch in dataloader_train:
            for a, b in zip(a_batch, b_batch):
                pool_a.append(a.to(device, dtype=torch.float64))
                pool_b.append(b.to(device, dtype=torch.float64))
        M = len(pool_a)
        self.logger.info(
            f"[Min_STP_World] Training  T={T}  M={M}  lr={lr}  alpha={alpha}")

        alpha_t = torch.tensor(alpha, dtype=torch.float64, device=device)
        opt     = torch.optim.Adam(self.slicer.parameters(), lr=lr)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T, eta_min=lr * 0.01)

        loss_ema = None
        rng      = np.random.default_rng(42)
        t0       = time.time()
        pbar     = tqdm(total=T, desc="Min_STP_World")

        for step in range(T):
            idx = int(rng.integers(0, M))
            a   = pool_a[idx]   # (n_supply,)
            b   = pool_b[idx]   # (n_demand,)

            self.slicer.train()
            theta          = self.slicer(a, b)              # (3,) normalized, differentiable
            scores_supply  = self.supply_t @ theta          # (n_supply,) differentiable
            scores_demand  = (self.demand_t @ theta).detach()  # (n_demand,) hard sort only

            with torch.no_grad():
                u_s     = torch.argsort(scores_supply.detach())
                u_d     = torch.argsort(scores_demand)
                a_u     = a[u_s]
                b_u     = b[u_d]
                wA, wB, delta_r = _compute_1d_coupling(a_u, b_u, device)

            # LapSum on supply (n_supply=100 only — n_demand=10000 is infeasible):
            #   π_approx = P_soft_supply.T @ π_1D_hard @ P_hard_demand
            #   loss = <π_approx, C>
            #        = sum_k delta_r[k] * (P_soft_supply @ C[:, u_d])[wA[k], wB[k]]
            P_soft  = _soft_permutation(scores_supply, alpha_t)  # (n_s, n_s)
            C_proj  = P_soft @ self.C_t        # (n_s, n_d)
            C_rank  = C_proj[:, u_d]           # (n_s, n_d) demand cols sorted by score
            loss    = (delta_r * C_rank[wA, wB]).sum()

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
        self.logger.info(f"[Min_STP_World] Saved → {ckpt}")

    def predict_plan(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        device = self.device
        a_t    = torch.tensor(a, dtype=torch.float64, device=device)
        b_t    = torch.tensor(b, dtype=torch.float64, device=device)

        self.slicer.eval()
        with torch.no_grad():
            theta          = self.slicer(a_t, b_t)
            scores_supply  = self.supply_t @ theta
            scores_demand  = self.demand_t @ theta

            u_s = torch.argsort(scores_supply)
            u_d = torch.argsort(scores_demand)
            a_u = a_t[u_s]
            b_u = b_t[u_d]
            wA, wB, delta_r = _compute_1d_coupling(a_u, b_u, device)

            orig_A = u_s[wA]
            orig_B = u_d[wB]
            P = torch.zeros(self.n_supply, self.n_demand,
                            dtype=torch.float64, device=device)
            P.index_put_((orig_A, orig_B), delta_r, accumulate=True)

        P_np = P.cpu().numpy()
        P_np = np.clip(P_np, 0.0, None)
        s    = P_np.sum()
        if s > 0:
            P_np /= s
        return P_np
