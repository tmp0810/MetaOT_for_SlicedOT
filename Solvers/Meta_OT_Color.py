import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from Solvers.DefenseTrain import Defense_Train_Base


class PointCloudEncoder(nn.Module):
    def __init__(self, coord_dim: int = 3, phi_hidden: int = 256, enc_dim: int = 256):
        super().__init__()
        inp = coord_dim + 1   # weight scalar + coords
        self.phi = nn.Sequential(
            nn.Linear(inp,        phi_hidden), nn.Tanh(),
            nn.Linear(phi_hidden, phi_hidden), nn.Tanh(),
            nn.Linear(phi_hidden, enc_dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(enc_dim, enc_dim), nn.Tanh(),
            nn.Linear(enc_dim, enc_dim),
        )

    def forward(self, weights: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, n, d = coords.shape
        x   = torch.cat([weights.unsqueeze(-1), coords], dim=-1)   # (B, n, d+1)
        phi = self.phi(x.view(B * n, d + 1)).view(B, n, -1)        # (B, n, enc_dim)
        agg = (weights.unsqueeze(-1) * phi).sum(dim=1)              # (B, enc_dim)
        return self.rho(agg)


class PotentialNet(nn.Module):
    def __init__(self, n_clusters: int, coord_dim: int = 3,
                 enc_dim: int = 256, head_hidden: int = 512):
        super().__init__()
        self.encoder = PointCloudEncoder(coord_dim=coord_dim,
                                         phi_hidden=enc_dim, enc_dim=enc_dim)
        self.f_head  = nn.Sequential(
            nn.Linear(enc_dim * 2, head_hidden), nn.Tanh(),
            nn.Linear(head_hidden, head_hidden), nn.Tanh(),
            nn.Linear(head_hidden, n_clusters),
        )

    def forward(
        self,
        src_w: torch.Tensor,   # (B, n)
        src_c: torch.Tensor,   # (B, n, 3)
        tgt_w: torch.Tensor,   # (B, n)
        tgt_c: torch.Tensor,   # (B, n, 3)
    ) -> torch.Tensor:         # (B, n)
        z_src = self.encoder(src_w, src_c)
        z_tgt = self.encoder(tgt_w, tgt_c)
        return self.f_head(torch.cat([z_src, z_tgt], dim=-1))



class Meta_OT_Color_Discrete(Defense_Train_Base):
    # is_continuous = False  (default) → eval_color_transfer.py dùng predict_plan
    is_continuous = False

    def __init__(self, cfg_proj, cfg_m):
        Defense_Train_Base.__init__(self, cfg_proj, cfg_m,
                                    name="Meta_OT_Color_Discrete")
        self._build_network()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _device(self):
        if torch.cuda.is_available() and hasattr(self.cfg_m, "gpu"):
            return torch.device(f"cuda:{self.cfg_m.gpu}")
        return torch.device("cpu")

    def _build_network(self):
        cfg        = self.cfg_m
        n_clusters = int(cfg.get("n_clusters")   or 500)
        enc_dim    = int(cfg.get("enc_dim")       or 256)
        head_hidden= int(cfg.get("head_hidden")   or 512)

        self.net = PotentialNet(
            n_clusters  = n_clusters,
            coord_dim   = 3,
            enc_dim     = enc_dim,
            head_hidden = head_hidden,
        ).to(self._device())

        n_p = sum(p.numel() for p in self.net.parameters())
        self.logger.info(
            f"[Meta_OT_Color_Discrete] PotentialNet  params={n_p:,}  "
            f"n_clusters={n_clusters}  enc_dim={enc_dim}  head_hidden={head_hidden}"
        )

    # ── core OT operations (faithful to JAX train_discrete.py) ───────────────

    @staticmethod
    def _compute_log_K(src_c: torch.Tensor, tgt_c: torch.Tensor,
                       eps: float) -> torch.Tensor:
        """
        log_K[i,j] = -||src_c[i] - tgt_c[j]||² / eps
        src_c : (n,3)   tgt_c : (m,3)
        Returns : (n, m)
        """
        diff  = src_c.unsqueeze(1) - tgt_c.unsqueeze(0)   # (n, m, 3)
        C     = (diff ** 2).sum(-1)                         # (n, m)
        return -C / eps

    def _g_from_f(self, f: torch.Tensor, b: torch.Tensor,
                  log_K: torch.Tensor, eps: float) -> torch.Tensor:
        """
        One Sinkhorn step: g update given f.
        JAX: g = geom.update_potential(f, zeros, log_b, axis=0)
           = eps * (log_b - lse(log_K + f/eps, axis=0))

        f     : (B, n_src)
        b     : (B, n_tgt)
        log_K : (B, n_src, n_tgt)  or  (n_src, n_tgt) broadcastable
        """
        log_b = torch.log(b.clamp(1e-300))                         # (B, n_tgt)
        M     = log_K + (f / eps).unsqueeze(-1)                    # (B, n_src, n_tgt)
        m     = M.max(dim=-2, keepdim=True).values
        lse   = (M - m).exp().sum(dim=-2).log() + m.squeeze(-2)    # (B, n_tgt)
        return eps * (log_b - lse)

    def _dual_obj_from_f(self, a: torch.Tensor, b: torch.Tensor,
                         f: torch.Tensor, log_K: torch.Tensor,
                         eps: float) -> torch.Tensor:
        """
        Exact port of JAX dual_obj_from_f.

        dual = div_a + div_b + eps*(1 - total_sum)

        a, b  : (B, n)
        f     : (B, n_src)
        log_K : (B, n_src, n_tgt)
        Returns: scalar
        """
        g = self._g_from_f(f, b, log_K, eps)                       # (B, n_tgt)

        # fa_i = eps*log(sum_j exp(log_K_ij + g_j/eps))
        M_fa = log_K + (g / eps).unsqueeze(-2)                      # (B, n_src, n_tgt)
        m    = M_fa.max(dim=-1, keepdim=True).values
        fa   = eps * ((M_fa - m).exp().sum(-1).log() + m.squeeze(-1))  # (B, n_src)

        # gb_j = eps*log(sum_i exp(log_K_ij + f_i/eps))
        M_gb = log_K + (f / eps).unsqueeze(-1)                      # (B, n_src, n_tgt)
        m    = M_gb.max(dim=-2, keepdim=True).values
        gb   = eps * ((M_gb - m).exp().sum(-2).log() + m.squeeze(-2))  # (B, n_tgt)

        div_a = (a * (f - fa)).sum(-1)                              # (B,)
        div_b = (b * (g - gb)).sum(-1)                              # (B,)

        log_P  = (f.unsqueeze(-1) + g.unsqueeze(-2)) / eps + log_K # (B, n_src, n_tgt)
        lp_max = log_P.detach().max()
        total  = (log_P - lp_max).exp().sum(dim=(-2, -1)) * lp_max.exp()

        dual = div_a + div_b + eps * (1.0 - total)
        return dual.mean()

    # ── training ─────────────────────────────────────────────────────────────

    def train(self, dataloader_train):
        """
        Dataloader yields: (src_w, src_c, tgt_w, tgt_c)
          src_w, tgt_w : (B, n_clusters)
          src_c, tgt_c : (B, n_clusters, 3)
        """
        device = self._device()
        cfg    = self.cfg_m
        eps    = float(cfg.epsilon)

        n_iters      = int(cfg.get("num_train_iter") or 5000)
        lr           = float(cfg.get("learning_rate") or 1e-3)
        log_interval = int(cfg.get("log_interval")    or 100)
        max_gn       = float(cfg.get("max_grad_norm") or 1.0)

        opt   = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_iters, eta_min=lr * 0.01)

        self.logger.info(
            f"[Meta_OT_Color_Discrete] Training  n_iters={n_iters}  "
            f"lr={lr}  eps={eps}"
        )

        loss_ema = None
        step     = 0
        t0       = time.time()
        pbar     = tqdm(total=n_iters, desc="Meta_OT_Color_Discrete")

        while step < n_iters:
            for src_w, src_c, tgt_w, tgt_c in dataloader_train:
                if step >= n_iters:
                    break

                # color_transfer_data.py yields float64 → cast to float32
                src_w = src_w.to(device, dtype=torch.float32)   # (B, n)
                src_c = src_c.to(device, dtype=torch.float32)   # (B, n, 3)
                tgt_w = tgt_w.to(device, dtype=torch.float32)   # (B, n)
                tgt_c = tgt_c.to(device, dtype=torch.float32)   # (B, n, 3)
                B     = src_w.shape[0]

                # Per-batch log_K (cost changes per image pair)
                # (B, n_src, n_tgt) — computed on GPU
                diff  = src_c.unsqueeze(2) - tgt_c.unsqueeze(1)   # (B, n, n, 3)
                C_bat = (diff ** 2).sum(-1)                         # (B, n, n)
                log_K = -C_bat / eps

                # Forward: PotentialNet → f
                f = self.net(src_w, src_c, tgt_w, tgt_c)           # (B, n)

                # Loss = -dual_obj (maximize)
                loss = -self._dual_obj_from_f(src_w, tgt_w, f, log_K, eps)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_gn)
                opt.step()
                sched.step()

                lv       = loss.item()
                loss_ema = lv if loss_ema is None else 0.95*loss_ema + 0.05*lv
                step    += 1
                pbar.update(1)

                if step % log_interval == 0:
                    msg = (f"[{step}/{n_iters}] "
                           f"loss_ema={loss_ema:.4e}  t={time.time()-t0:.1f}s")
                    pbar.set_description(msg)
                    self.logger.info(msg)

        pbar.close()
        ckpt = os.path.join(self.log_sub_folder, "net.pt")
        torch.save(self.net.state_dict(), ckpt)
        self.logger.info(f"[Meta_OT_Color_Discrete] Saved → {ckpt}")

    # ── inference ─────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_cost(x_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
        """Squared-Euclidean cost. Used by eval_color_transfer.py."""
        diff = x_src[:, None, :] - x_tgt[None, :, :]
        return np.sum(diff ** 2, axis=-1)

    def predict_plan(
        self,
        a: np.ndarray,      # (n_src,)  source weights
        b: np.ndarray,      # (n_tgt,)  target weights
        src_c: np.ndarray,  # (n_src, 3) source cluster centers
        tgt_c: np.ndarray,  # (n_tgt, 3) target cluster centers
    ) -> np.ndarray:
        """
        Predict transport plan P (n_src, n_tgt).

        Faithful to JAX pred_transport:
          f = net(src_w, src_c, tgt_w, tgt_c)
          g = g_from_f(f, b, log_K)  [1 Sinkhorn step]
          f_new = update_f(f, g, log_K)  [1 more Sinkhorn step]
          P = exp((f_new + g - C) / eps)
        """
        device = self._device()
        eps    = float(self.cfg_m.epsilon)

        a_t   = torch.tensor(a,     dtype=torch.float32, device=device).unsqueeze(0)
        b_t   = torch.tensor(b,     dtype=torch.float32, device=device).unsqueeze(0)
        sc_t  = torch.tensor(src_c, dtype=torch.float32, device=device).unsqueeze(0)
        tc_t  = torch.tensor(tgt_c, dtype=torch.float32, device=device).unsqueeze(0)

        diff  = sc_t.unsqueeze(2) - tc_t.unsqueeze(1)     # (1, n_src, n_tgt, 3)
        log_K = -(diff**2).sum(-1) / eps                   # (1, n_src, n_tgt)

        with torch.no_grad():
            f_t = self.net(a_t, sc_t, b_t, tc_t)          # (1, n_src)

        # g from f (1 Sinkhorn step, same as training)
        g_t = self._g_from_f(f_t, b_t, log_K, eps)        # (1, n_tgt)

        # update f once more (JAX pred_error does this for eval)
        M_f   = log_K + (g_t / eps).unsqueeze(-2)          # (1, n_src, n_tgt)
        m_f   = M_f.max(dim=-1, keepdim=True).values
        lse_f = (M_f - m_f).exp().sum(-1).log() + m_f.squeeze(-1)
        f_new = eps * (torch.log(a_t.clamp(1e-300)) - lse_f)   # (1, n_src)

        f_np  = f_new[0].cpu().numpy()
        g_np  = g_t[0].cpu().numpy()
        C_np  = self._compute_cost(src_c, tgt_c)

        # Plan recovery (same as Meta_OT_World)
        log_P  = f_np[:, None] / eps - C_np / eps + g_np[None, :] / eps
        log_P -= log_P.max()
        P      = np.clip(np.exp(log_P), 0.0, None)

        # Sinkhorn marginal projection (5 rounds)
        for _ in range(1):
            P = P * (a / P.sum(1).clip(1e-300))[:, None]
            P = P * (b / P.sum(0).clip(1e-300))[None, :]
        P = np.clip(P, 0.0, None)
        s = P.sum()
        if s > 0:
            P /= s
        return P
