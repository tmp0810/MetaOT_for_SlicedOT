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
    mask = delta_r > 1e-15
    return wA[mask], wB[mask], delta_r[mask]


class PointCloudEncoder(nn.Module):
    def __init__(self, coord_dim: int = 3, phi_hidden: int = 64,
                 enc_dim: int = 32, dtype=torch.float64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(coord_dim + 1, phi_hidden, dtype=dtype), nn.Tanh(),
            nn.Linear(phi_hidden,    enc_dim,     dtype=dtype),
        )

    def forward(self, weights: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x   = torch.cat([weights.unsqueeze(-1), coords], dim=-1)
        phi = self.phi(x)
        return (weights.unsqueeze(-1) * phi).sum(dim=0)


class SlicerNetColor(nn.Module):
    def __init__(self, enc_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.enc_src = PointCloudEncoder(coord_dim=3, phi_hidden=hidden_dim,
                                         enc_dim=enc_dim)
        self.enc_tgt = PointCloudEncoder(coord_dim=3, phi_hidden=hidden_dim,
                                         enc_dim=enc_dim)
        self.head = nn.Sequential(
            nn.Linear(enc_dim * 2, hidden_dim, dtype=torch.float64), nn.Tanh(),
            nn.Linear(hidden_dim,  3,          dtype=torch.float64),
        )

    def forward(self, a: torch.Tensor, src_c: torch.Tensor,
                b: torch.Tensor, tgt_c: torch.Tensor) -> torch.Tensor:
        z_src = self.enc_src(a, src_c)
        z_tgt = self.enc_tgt(b, tgt_c)
        theta = self.head(torch.cat([z_src, z_tgt]))
        return theta / theta.norm().clamp(min=1e-8)


class Min_STP_Color(Defense_Train_Base):
    is_continuous = False

    def __init__(self, cfg_proj, cfg_m):
        Defense_Train_Base.__init__(self, cfg_proj, cfg_m, name="Min_STP_Color")
        self._build_slicer()

    def _build_slicer(self):
        cfg        = self.cfg_m
        enc_dim    = int(cfg.get("enc_dim")    or 32)
        hidden_dim = int(cfg.get("hidden_dim") or 64)
        self.slicer = SlicerNetColor(
            enc_dim    = enc_dim,
            hidden_dim = hidden_dim,
        ).to(self.device)
        n_params = sum(p.numel() for p in self.slicer.parameters())
        self.logger.info(
            f"[Min_STP_Color] SlicerNet params={n_params:,}  "
            f"enc_dim={enc_dim}  hidden_dim={hidden_dim}")

    def train(self, dataloader_train):
        cfg    = self.cfg_m
        T      = int(cfg.get("num_train_iter") or 5000)
        lr     = float(cfg.get("learning_rate") or 1e-3)
        alpha  = float(cfg.get("alpha")         or 0.05)
        log_iv = int(cfg.get("log_interval")    or 100)
        max_gn = float(cfg.get("max_grad_norm") or 1.0)
        device = self.device

        pool_a, pool_sc, pool_b, pool_tc = [], [], [], []
        for src_w, src_c, tgt_w, tgt_c in dataloader_train:
            for i in range(src_w.shape[0]):
                pool_a.append( src_w[i].to(device, dtype=torch.float64))
                pool_sc.append(src_c[i].to(device, dtype=torch.float64))
                pool_b.append( tgt_w[i].to(device, dtype=torch.float64))
                pool_tc.append(tgt_c[i].to(device, dtype=torch.float64))
        M = len(pool_a)
        self.logger.info(
            f"[Min_STP_Color] Training  T={T}  M={M}  lr={lr}  alpha={alpha}")

        alpha_t = torch.tensor(alpha, dtype=torch.float64, device=device)
        opt     = torch.optim.Adam(self.slicer.parameters(), lr=lr)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T, eta_min=lr * 0.01)

        loss_ema = None
        rng      = np.random.default_rng(42)
        t0       = time.time()
        pbar     = tqdm(total=T, desc="Min_STP_Color")

        for step in range(T):
            idx  = int(rng.integers(0, M))
            a    = pool_a[idx]    # (n_clusters,)
            sc   = pool_sc[idx]   # (n_clusters, 3)
            b    = pool_b[idx]    # (n_clusters,)
            tc   = pool_tc[idx]   # (n_clusters, 3)

            self.slicer.train()
            theta        = self.slicer(a, sc, b, tc)        # (3,) normalized
            scores_src   = sc @ theta                        # (n,) differentiable
            scores_tgt   = (tc @ theta).detach()             # (n,) hard sort only

            with torch.no_grad():
                u_s     = torch.argsort(scores_src.detach())
                u_t     = torch.argsort(scores_tgt)
                a_u     = a[u_s]
                b_u     = b[u_t]
                wA, wB, delta_r = _compute_1d_coupling(a_u, b_u, device)

                diff  = sc.unsqueeze(1) - tc.unsqueeze(0)   # (n, n, 3)
                C_t   = (diff ** 2).sum(-1).to(torch.float64)  # (n, n)

            # LapSum on src side:
            #   loss = sum_k delta_r[k] * (P_soft_src @ C[:, u_t])[wA[k], wB[k]]
            P_soft = _soft_permutation(scores_src, alpha_t)  # (n, n) differentiable
            C_proj = P_soft @ C_t        # (n, n)
            C_rank = C_proj[:, u_t]      # (n, n) tgt cols sorted by score
            loss   = (delta_r * C_rank[wA, wB]).sum()

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
        self.logger.info(f"[Min_STP_Color] Saved → {ckpt}")

    def predict_plan(self, a: np.ndarray, b: np.ndarray,
                     src_c: np.ndarray, tgt_c: np.ndarray) -> np.ndarray:
        device = self.device
        a_t  = torch.tensor(a,     dtype=torch.float64, device=device)
        b_t  = torch.tensor(b,     dtype=torch.float64, device=device)
        sc_t = torch.tensor(src_c, dtype=torch.float64, device=device)
        tc_t = torch.tensor(tgt_c, dtype=torch.float64, device=device)

        self.slicer.eval()
        with torch.no_grad():
            theta      = self.slicer(a_t, sc_t, b_t, tc_t)
            scores_src = sc_t @ theta
            scores_tgt = tc_t @ theta
            u_s = torch.argsort(scores_src)
            u_t = torch.argsort(scores_tgt)
            a_u = a_t[u_s]
            b_u = b_t[u_t]
            wA, wB, delta_r = _compute_1d_coupling(a_u, b_u, device)
            orig_A = u_s[wA]
            orig_B = u_t[wB]
            n = len(a)
            P = torch.zeros(n, n, dtype=torch.float64, device=device)
            P.index_put_((orig_A, orig_B), delta_r, accumulate=True)

        P_np = P.cpu().numpy()
        P_np = np.clip(P_np, 0.0, None)
        s    = P_np.sum()
        if s > 0:
            P_np /= s
        return P_np
