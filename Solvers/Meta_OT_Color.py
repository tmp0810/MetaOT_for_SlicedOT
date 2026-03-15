import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from Solvers.DefenseTrain import Defense_Train_Base
from Models.ot_models import DenseICNN
from Models.ot_models import (
    MetaICNN_Cloud,
    build_icnn_param_info,
    unravel_icnn_params,
    _dense_icnn_forward_gs,
    _dense_icnn_push_gs,
)

# Monkey-patch gradient-safe methods onto DenseICNN
DenseICNN._forward_gs = _dense_icnn_forward_gs
DenseICNN._push_gs    = _dense_icnn_push_gs


class Meta_OT_Color(Defense_Train_Base):
    INPUT_DIM  = 3   # RGB ∈ [0,1]^3
    COORD_DIM  = 3

    def __init__(self, cfg_proj, cfg_m):
        Defense_Train_Base.__init__(self, cfg_proj, cfg_m, name="Meta_OT_Color")
        self._build_networks()

    def _build_networks(self):
        cfg = self.cfg_m
        device = self._get_device()

        # Two small ICNNs in R^3 (D and its approximate conjugate D_conj)
        self.D      = DenseICNN(
            input_dim  = self.INPUT_DIM,
            hidden_dim = cfg.icnn_hidden_dim,
            hidden_num = cfg.icnn_hidden_num,
        ).to(device)
        self.D_conj = DenseICNN(
            input_dim  = self.INPUT_DIM,
            hidden_dim = cfg.icnn_hidden_dim,
            hidden_num = cfg.icnn_hidden_num,
        ).to(device)

        self.param_info, total = build_icnn_param_info(self.D)
        self.icnn_param_dim    = total
        self.logger.info(
            f"[Meta_OT_Color] ICNN param_dim={total} "
            f"(hidden_dim={cfg.icnn_hidden_dim}, hidden_num={cfg.icnn_hidden_num})"
        )

        # Meta-network: point-cloud pair → ICNN params
        self.meta_net = MetaICNN_Cloud(
            icnn_param_dim  = total,
            coord_dim       = self.COORD_DIM,
            enc_dim         = cfg.enc_dim,
            head_hidden_dim = cfg.meta_hidden_dim,
        ).to(device)
        n_meta = sum(p.numel() for p in self.meta_net.parameters())
        self.logger.info(f"[Meta_OT_Color] MetaICNN_Cloud params: {n_meta:,}")

    def _get_device(self):
        if torch.cuda.is_available() and hasattr(self.cfg_m, "gpu"):
            return torch.device(f"cuda:{self.cfg_m.gpu}")
        return torch.device("cpu")

    def _compute_cost(self, x_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
        """Squared Euclidean cost in [0,1]^3. (n_src, n_tgt)."""
        diff = x_src[:, None, :] - x_tgt[None, :, :]
        return (diff ** 2).sum(axis=-1)

    def _sample_from_measure(
        self,
        weights: torch.Tensor,   # (n,)
        coords:  torch.Tensor,   # (n, d)
        n_samples: int,
    ) -> torch.Tensor:
        """Sample n_samples points ~ weights (with replacement). → (n_samples, d)"""
        idx = torch.multinomial(weights.float(), n_samples, replacement=True)
        return coords[idx]                                    # (n_samples, d)

    def _dual_loss_single(
        self,
        D_params:      dict,
        D_conj_params: dict,
        X:  torch.Tensor,    # (n_X, d)  samples from µ
        Y:  torch.Tensor,    # (n_Y, d)  samples from ν
        cycle_weight: float,
    ) -> torch.Tensor:
        """
        W2GN dual loss for one (µ, ν) pair.

        L = E_X[D(X)] + E_Y[<T_c(Y), Y> - D(T_c(Y))]
          + λ * cycle(D, D_c, X, Y)
        """
        # ── Dual term ─────────────────────────────────────────────────
        T_conj_Y = self.D_conj._push_gs(Y, D_conj_params, create_graph=True)  # (n_Y, d)
        T_conj_Y_det = T_conj_Y.detach()

        D_X          = self.D._forward_gs(X,           D_params)        # (n_X, 1)
        D_TY         = self.D._forward_gs(T_conj_Y_det, D_params)       # (n_Y, 1)
        inner_TY     = (T_conj_Y_det * Y).sum(dim=-1, keepdim=True)     # (n_Y, 1)
        dual_loss    = D_X.mean() + (inner_TY - D_TY).mean()

        # ── Cycle regularization ───────────────────────────────────────
        T_X    = self.D._push_gs(X, D_params, create_graph=True)         # (n_X, d)
        cyc_XY = ((self.D_conj._push_gs(T_X.detach(),      D_conj_params) - X) ** 2).mean()
        cyc_YX = ((self.D._push_gs(T_conj_Y.detach(),      D_params)     - Y) ** 2).mean()

        return dual_loss + cycle_weight * (cyc_XY + cyc_YX)

    def _pretrain_identity(self, n_iter: int = 500):
        """
        Warm-start the meta-network so D(x) ≈ 0.5||x||^2 (gradient = identity map).
        Trains on random RGB points X ~ U[0,1]^3.
        """
        device  = self._get_device()
        cfg     = self.cfg_m
        opt     = torch.optim.Adam(self.meta_net.parameters(), lr=cfg.learning_rate)
        n_pts   = cfg.n_inner_samples

        # Synthetic "same-distribution" pair: both src=tgt=uniform
        dummy_w = torch.ones(1, cfg.n_clusters, dtype=torch.float64, device=device)
        dummy_w = dummy_w / dummy_w.sum(dim=-1, keepdim=True)
        dummy_c = torch.rand(1, cfg.n_clusters, 3, dtype=torch.float64, device=device)

        self.logger.info(f"[Meta_OT_Color] Pretraining identity ({n_iter} iters) ...")
        for step in range(n_iter):
            X = torch.rand(n_pts, self.INPUT_DIM, dtype=torch.float64, device=device)

            D_flat, Dc_flat = self.meta_net(dummy_w, dummy_c, dummy_w, dummy_c)
            D_p  = unravel_icnn_params(D_flat[0],  self.param_info)
            Dc_p = unravel_icnn_params(Dc_flat[0], self.param_info)

            # Loss: push(D, X) ≈ X  and  push(D_conj, X) ≈ X
            # create_graph=True required — see Meta_OT_World._pretrain_identity comment
            T_X  = self.D._push_gs(X,  D_p,  create_graph=True)
            Tc_X = self.D_conj._push_gs(X, Dc_p, create_graph=True)
            loss = ((T_X - X) ** 2).mean() + ((Tc_X - X) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 100 == 0:
                self.logger.info(f"  pretrain step {step}/{n_iter}  loss={loss.item():.4e}")

        self.logger.info("[Meta_OT_Color] Pretrain done.")


    def train(self, dataloader_train):
        """
        Main meta-training loop.

        Dataloader yields: (src_w, src_c, tgt_w, tgt_c)
            src_w : (B, n_clusters)
            src_c : (B, n_clusters, 3)
        """
        device = self._get_device()
        cfg    = self.cfg_m

        # ── Optional pretrain ──────────────────────────────────────────
        if cfg.pretrain_iter > 0:
            self._pretrain_identity(cfg.pretrain_iter)

        opt = torch.optim.Adam(self.meta_net.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.num_train_iter, eta_min=cfg.learning_rate * 0.01)

        loss_ema = None
        step     = 0
        t0       = time.time()

        self.logger.info(
            f"[Meta_OT_Color] Training for {cfg.num_train_iter} iters, "
            f"batch_size={cfg.batch_size}"
        )

        pbar = tqdm(total=cfg.num_train_iter, desc="Meta_OT_Color")
        while step < cfg.num_train_iter:
            for src_w, src_c, tgt_w, tgt_c in dataloader_train:
                if step >= cfg.num_train_iter:
                    break

                src_w = src_w.to(device)   # (B, n)
                src_c = src_c.to(device)   # (B, n, 3)
                tgt_w = tgt_w.to(device)
                tgt_c = tgt_c.to(device)
                B     = src_w.shape[0]

                # ── Forward meta-network ───────────────────────────────
                D_flat, Dc_flat = self.meta_net(src_w, src_c, tgt_w, tgt_c)
                # D_flat: (B, icnn_param_dim)

                # ── Compute loss over batch ────────────────────────────
                total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
                for b in range(B):
                    D_p  = unravel_icnn_params(D_flat[b],  self.param_info)
                    Dc_p = unravel_icnn_params(Dc_flat[b], self.param_info)

                    X = self._sample_from_measure(
                        src_w[b], src_c[b], cfg.n_inner_samples)  # (n_smp, 3)
                    Y = self._sample_from_measure(
                        tgt_w[b], tgt_c[b], cfg.n_inner_samples)

                    loss_b = self._dual_loss_single(
                        D_p, Dc_p, X, Y, cfg.cycle_loss_weight)
                    total_loss = total_loss + loss_b

                loss = total_loss / B

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.meta_net.parameters(), 1.0)
                opt.step()
                scheduler.step()

                # ── Logging ───────────────────────────────────────────
                lv      = loss.item()
                loss_ema = lv if loss_ema is None else 0.95 * loss_ema + 0.05 * lv
                step    += 1
                pbar.update(1)

                if step % cfg.log_interval == 0:
                    elapsed = time.time() - t0
                    msg = (f"[{step}/{cfg.num_train_iter}] "
                           f"loss_ema={loss_ema:.4e}  elapsed={elapsed:.1f}s")
                    pbar.set_description(msg)
                    self.logger.info(msg)

        pbar.close()

        # Save the trained meta-network
        ckpt_path = os.path.join(self.log_sub_folder, "meta_net.pt")
        torch.save(self.meta_net.state_dict(), ckpt_path)
        self.logger.info(f"[Meta_OT_Color] Saved meta_net -> {ckpt_path}")


    def predict_plan(
        self,
        a:     np.ndarray,    # (n_src,)
        b:     np.ndarray,    # (n_tgt,)
        x_src: np.ndarray,   # (n_src, 3)
        x_tgt: np.ndarray,   # (n_tgt, 3)
    ) -> np.ndarray:
        """
        Predict transport plan P (n_src, n_tgt) for a new color pair.

        Pipeline:
            1. Forward meta-net → ICNN params
            2. f_i = D(x_src_i)          — source potentials
            3. g_j = D_conj(x_tgt_j)     — target potentials
            4. P_ij ∝ exp((f_i + g_j - C_ij) / ε)
        """
        device = self._get_device()
        eps    = float(getattr(self.cfg_m, 'epsilon', 0.5))

        # Tensors (add batch dim = 1)
        sw = torch.tensor(a,     dtype=torch.float64, device=device).unsqueeze(0)   # (1, n_src)
        sc = torch.tensor(x_src, dtype=torch.float64, device=device).unsqueeze(0)   # (1, n_src, 3)
        tw = torch.tensor(b,     dtype=torch.float64, device=device).unsqueeze(0)
        tc = torch.tensor(x_tgt, dtype=torch.float64, device=device).unsqueeze(0)

        with torch.no_grad():
            D_flat, Dc_flat = self.meta_net(sw, sc, tw, tc)
            D_p  = unravel_icnn_params(D_flat[0],  self.param_info)
            Dc_p = unravel_icnn_params(Dc_flat[0], self.param_info)

            X_t = sc[0]    # (n_src, 3)
            Y_t = tc[0]    # (n_tgt, 3)

            f = self.D._forward_gs(X_t,  D_p).squeeze(-1).cpu().numpy()   # (n_src,)
            g = self.D_conj._forward_gs(Y_t, Dc_p).squeeze(-1).cpu().numpy()  # (n_tgt,)

        C = self._compute_cost(x_src, x_tgt)    # (n_src, n_tgt)

        # Log-domain plan
        f_c   = f - f.mean()
        g_c   = g - g.mean()
        log_P = f_c[:, None] / eps - C / eps + g_c[None, :] / eps
        log_P -= log_P.max()
        P     = np.exp(log_P)
        P     = np.clip(P, 0.0, None)

        # Sinkhorn marginal normalization (3 rounds) to enforce P@1≈a, P.T@1≈b
        def _norm(M, target, axis):
            s = M.sum(axis=axis, keepdims=True).clip(1e-300)
            return M * (target.reshape(s.shape) / s)
        for _ in range(3):
            P = _norm(P, a, axis=1)
            P = _norm(P, b, axis=0)

        P = np.clip(P, 0.0, None)
        P_sum = P.sum()
        if P_sum > 0:
            P /= P_sum
        return P
