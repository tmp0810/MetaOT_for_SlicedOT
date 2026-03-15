import numpy as np
import torch

from Solvers.Meta_OT_Color import Meta_OT_Color
from Models.ot_models_additions import unravel_icnn_params


class Meta_OT_World(Meta_OT_Color):
    INPUT_DIM = 3   # euclidean sphere coords ∈ R³
    COORD_DIM = 3

    def __init__(
        self,
        cfg_proj,
        cfg_m,
        supply_euc: np.ndarray,
        demand_euc: np.ndarray,
        supply_sph: np.ndarray = None,
        demand_sph: np.ndarray = None,
    ):
        self.supply_euc = supply_euc.astype(np.float64)   # (n_supply, 3)
        self.demand_euc = demand_euc.astype(np.float64)   # (n_demand, 3)
        self.supply_sph = supply_sph
        self.demand_sph = demand_sph

        # Call Defense_Train_Base directly (skip Meta_OT_Color.__init__'s build_grid etc.)
        # Meta_OT_Color.__init__ calls _build_networks → fine, no overrides needed there
        super().__init__(cfg_proj, cfg_m)

        self.logger.info(
            f"[Meta_OT_World] supply={supply_euc.shape}  demand={demand_euc.shape}"
        )

    def _compute_cost(
        self,
        x_src: np.ndarray,    # (n_src, 3) euclidean unit sphere
        x_tgt: np.ndarray,    # (n_tgt, 3)
    ) -> np.ndarray:
        """Great-circle distance: C_ij = arccos(x_src_i · x_tgt_j)."""
        dots = x_src @ x_tgt.T
        dots = np.clip(dots, -1.0 + 1e-7, 1.0 - 1e-7)
        return np.arccos(dots)    # (n_src, n_tgt)

    def _sphere_project(self, x: torch.Tensor) -> torch.Tensor:
        """Project R³ points onto the unit sphere S²."""
        nrm = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return x / nrm

    def _dual_loss_single(
        self,
        D_params,
        D_conj_params,
        X: torch.Tensor,     # (n_X, 3)  on unit sphere
        Y: torch.Tensor,     # (n_Y, 3)
        cycle_weight: float,
    ) -> torch.Tensor:
        # ── Dual term ─────────────────────────────────────────────────
        T_conj_Y_raw = self.D_conj._push_gs(Y, D_conj_params, create_graph=True)
        T_conj_Y     = self._sphere_project(T_conj_Y_raw)
        T_conj_Y_det = T_conj_Y.detach()

        D_X      = self.D._forward_gs(X,           D_params)
        D_TY     = self.D._forward_gs(T_conj_Y_det, D_params)
        inner_TY = (T_conj_Y_det * Y).sum(dim=-1, keepdim=True)
        dual_loss = D_X.mean() + (inner_TY - D_TY).mean()

        # ── Cycle regularization on sphere ─────────────────────────────
        T_X_raw = self.D._push_gs(X, D_params, create_graph=True)
        T_X     = self._sphere_project(T_X_raw)

        cyc_XY = ((self._sphere_project(
            self.D_conj._push_gs(T_X.detach(), D_conj_params)) - X) ** 2).mean()
        cyc_YX = ((self._sphere_project(
            self.D._push_gs(T_conj_Y.detach(), D_params))      - Y) ** 2).mean()

        return dual_loss + cycle_weight * (cyc_XY + cyc_YX)

    def _pretrain_identity(self, n_iter: int = 500):
        device = self._get_device()
        cfg    = self.cfg_m
        opt    = torch.optim.Adam(self.meta_net.parameters(), lr=cfg.learning_rate)
        n_pts  = cfg.n_inner_samples

        # Dummy pair: spherical uniform supply and demand
        n_s = self.supply_euc.shape[0]
        n_d = self.demand_euc.shape[0]
        dummy_sw = torch.full((1, n_s), 1.0 / n_s, dtype=torch.float64, device=device)
        dummy_sc = torch.tensor(self.supply_euc, dtype=torch.float64, device=device).unsqueeze(0)
        dummy_dw = torch.full((1, n_d), 1.0 / n_d, dtype=torch.float64, device=device)
        dummy_dc = torch.tensor(self.demand_euc, dtype=torch.float64, device=device).unsqueeze(0)

        self.logger.info(f"[Meta_OT_World] Pretraining identity ({n_iter} iters) ...")
        for step in range(n_iter):
            # Sample sphere-uniform points
            raw = torch.randn(n_pts, 3, dtype=torch.float64, device=device)
            X   = raw / raw.norm(dim=-1, keepdim=True)

            D_flat, Dc_flat = self.meta_net(dummy_sw, dummy_sc, dummy_dw, dummy_dc)
            D_p  = unravel_icnn_params(D_flat[0],  self.param_info)
            Dc_p = unravel_icnn_params(Dc_flat[0], self.param_info)

            T_X  = self._sphere_project(self.D._push_gs(X, D_p))
            Tc_X = self._sphere_project(self.D_conj._push_gs(X, Dc_p))

            # For sphere: T should approximate identity → T(x) ≈ x
            loss = ((T_X - X) ** 2).mean() + ((Tc_X - X) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 100 == 0:
                self.logger.info(f"  pretrain step {step}/{n_iter}  loss={loss.item():.4e}")
        self.logger.info("[Meta_OT_World] Pretrain done.")


    def train(self, dataloader_train):
        device  = self._get_device()
        cfg     = self.cfg_m

        supply_euc_t = torch.tensor(
            self.supply_euc, dtype=torch.float64, device=device)   # (n_supply, 3)
        demand_euc_t = torch.tensor(
            self.demand_euc, dtype=torch.float64, device=device)   # (n_demand, 3)
        n_supply = supply_euc_t.shape[0]
        n_demand = demand_euc_t.shape[0]

        # ── Optional pretrain ──────────────────────────────────────────
        if cfg.pretrain_iter > 0:
            self._pretrain_identity(cfg.pretrain_iter)

        opt = torch.optim.Adam(self.meta_net.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.num_train_iter, eta_min=cfg.learning_rate * 0.01)

        import time
        loss_ema = None
        step     = 0
        t0       = time.time()

        self.logger.info(
            f"[Meta_OT_World] Training for {cfg.num_train_iter} iters")

        import torch.nn as nn
        from tqdm import tqdm
        pbar = tqdm(total=cfg.num_train_iter, desc="Meta_OT_World")

        while step < cfg.num_train_iter:
            for _, _, supply_w, demand_w in dataloader_train:
                if step >= cfg.num_train_iter:
                    break

                supply_w = supply_w.to(device)   # (B, n_supply)
                demand_w = demand_w.to(device)   # (B, n_demand)
                B        = supply_w.shape[0]

                # Expand fixed locations to batch
                sc = supply_euc_t.unsqueeze(0).expand(B, -1, -1)  # (B, n_supply, 3)
                dc = demand_euc_t.unsqueeze(0).expand(B, -1, -1)  # (B, n_demand, 3)

                # ── Forward meta-network ───────────────────────────────
                D_flat, Dc_flat = self.meta_net(supply_w, sc, demand_w, dc)

                # ── Compute loss ───────────────────────────────────────
                total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
                for b in range(B):
                    D_p  = unravel_icnn_params(D_flat[b],  self.param_info)
                    Dc_p = unravel_icnn_params(Dc_flat[b], self.param_info)

                    # Sample supply (all n_supply=100) and n_inner from demand
                    n_smp = min(cfg.n_inner_samples, n_demand)
                    X = self._sample_from_measure(
                        supply_w[b], sc[b], min(cfg.n_inner_samples, n_supply))
                    Y = self._sample_from_measure(
                        demand_w[b], dc[b], n_smp)

                    loss_b = self._dual_loss_single(
                        D_p, Dc_p, X, Y, cfg.cycle_loss_weight)
                    total_loss = total_loss + loss_b

                loss = total_loss / B

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.meta_net.parameters(), 1.0)
                opt.step()
                scheduler.step()

                lv       = loss.item()
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

        import os
        ckpt_path = os.path.join(self.log_sub_folder, "meta_net.pt")
        torch.save(self.meta_net.state_dict(), ckpt_path)
        self.logger.info(f"[Meta_OT_World] Saved meta_net -> {ckpt_path}")

    def predict_plan(
        self,
        a: np.ndarray,   # (n_supply,)
        b: np.ndarray,   # (n_demand,)
    ) -> np.ndarray:
        """
        Predict transport plan P (n_supply, n_demand) for a WorldPair.

        Note: x_src and x_tgt come from self.supply_euc / self.demand_euc.
        """
        device = self._get_device()
        eps    = float(getattr(self.cfg_m, 'epsilon', 0.5))

        sw = torch.tensor(a, dtype=torch.float64, device=device).unsqueeze(0)
        dw = torch.tensor(b, dtype=torch.float64, device=device).unsqueeze(0)
        sc = torch.tensor(self.supply_euc, dtype=torch.float64, device=device).unsqueeze(0)
        dc = torch.tensor(self.demand_euc, dtype=torch.float64, device=device).unsqueeze(0)

        with torch.no_grad():
            D_flat, Dc_flat = self.meta_net(sw, sc, dw, dc)
            D_p  = unravel_icnn_params(D_flat[0],  self.param_info)
            Dc_p = unravel_icnn_params(Dc_flat[0], self.param_info)

            f = self.D._forward_gs(sc[0], D_p).squeeze(-1).cpu().numpy()       # (n_supply,)
            g = self.D_conj._forward_gs(dc[0], Dc_p).squeeze(-1).cpu().numpy() # (n_demand,)

        C = self._compute_cost(self.supply_euc, self.demand_euc)   # (n_supply, n_demand)

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
