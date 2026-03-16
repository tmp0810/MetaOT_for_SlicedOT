import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from Solvers.DefenseTrain import Defense_Train_Base

class PotentialMLP(nn.Module):
    def __init__(self, n_input: int, n_output: int,
                 n_hidden: int = 512, n_hidden_layer: int = 3):
        super().__init__()
        layers = []
        in_dim = n_input
        for _ in range(n_hidden_layer):
            layers += [nn.Linear(in_dim, n_hidden, dtype=torch.float64), nn.ReLU()]
            in_dim = n_hidden
        layers += [nn.Linear(n_hidden, n_output, dtype=torch.float64)]
        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a : (..., n_supply)
        b : (..., n_demand)
        → f : (..., n_supply)
        """
        z = torch.cat([a, b], dim=-1)
        return self.net(z)

class Meta_OT_World(Defense_Train_Base):
    def __init__(
        self,
        cfg_proj,
        cfg_m,
        supply_euc:  np.ndarray,   # (n_supply, 3)
        demand_euc:  np.ndarray,   # (n_demand, 3)
        supply_sph:  np.ndarray = None,
        demand_sph:  np.ndarray = None,
    ):
        self.supply_euc = supply_euc.astype(np.float64)
        self.demand_euc = demand_euc.astype(np.float64)
        self.supply_sph = supply_sph
        self.demand_sph = demand_sph
        self.n_supply   = supply_euc.shape[0]
        self.n_demand   = demand_euc.shape[0]

        Defense_Train_Base.__init__(self, cfg_proj, cfg_m, name="Meta_OT_World")

        # Precompute FIXED cost matrix once
        self.C_np = self._sphere_cost(self.supply_euc, self.demand_euc)
        self.logger.info(
            f"[Meta_OT_World] n_supply={self.n_supply}  n_demand={self.n_demand}  "
            f"C=[{self.C_np.min():.3f}, {self.C_np.max():.3f}]  eps={cfg_m.epsilon}"
        )

        self._build_network()

    def _device(self):
        if torch.cuda.is_available() and hasattr(self.cfg_m, "gpu"):
            return torch.device(f"cuda:{self.cfg_m.gpu}")
        return torch.device("cpu")

    @staticmethod
    def _sphere_cost(xs, xt):
        d = xs @ xt.T
        return np.arccos(np.clip(d, -1 + 1e-7, 1 - 1e-7))

    def _build_network(self):
        cfg = self.cfg_m
        n_hidden       = cfg.get("n_hidden")       or 512
        n_hidden_layer = cfg.get("n_hidden_layer") or 3

        self.mlp = PotentialMLP(
            n_input        = self.n_supply + self.n_demand,
            n_output       = self.n_supply,
            n_hidden       = n_hidden,
            n_hidden_layer = n_hidden_layer,
        ).to(self._device())

        n_p = sum(p.numel() for p in self.mlp.parameters())
        self.logger.info(
            f"[Meta_OT_World] PotentialMLP  params={n_p:,}  "
            f"n_hidden={n_hidden}  n_hidden_layer={n_hidden_layer}"
        )

 
    def _g_and_f_from_f_pred(
        self,
        f_pred: torch.Tensor,  # (B, n_supply)
        a:      torch.Tensor,  # (B, n_supply)
        b:      torch.Tensor,  # (B, n_demand)
        log_K:  torch.Tensor,  # (n_supply, n_demand)
        eps:    float,
    ):
        log_b = torch.log(b.clamp(min=1e-300))
        M_g = log_K.unsqueeze(0) + (f_pred / eps).unsqueeze(-1)
        m_g = M_g.max(dim=-2, keepdim=True).values
        lse_g = (M_g - m_g).exp().sum(dim=-2).log() + m_g.squeeze(-2)
        g_sink = eps * (log_b - lse_g)
        log_a = torch.log(a.clamp(min=1e-300))
        M_f = log_K.unsqueeze(0) + (g_sink / eps).unsqueeze(-2)
        m_f = M_f.max(dim=-1, keepdim=True).values
        lse_f = (M_f - m_f).exp().sum(dim=-1).log() + m_f.squeeze(-1)
        f_sink = eps * (log_a - lse_f)

        return g_sink, f_sink

    def _dual_obj_from_f(
        self,
        a:     torch.Tensor,
        b:     torch.Tensor,
        f:     torch.Tensor,   # Đây là f_pred từ MLP
        log_K: torch.Tensor,
        eps:   float,
    ) -> torch.Tensor:
        
        g_sink, f_sink = self._g_and_f_from_f_pred(f, a, b, log_K, eps)
        dual_obj_left = (f_sink * a).sum(dim=-1) + (g_sink * b).sum(dim=-1)
        log_P = (f_sink.unsqueeze(-1) + g_sink.unsqueeze(-2)) / eps + log_K.unsqueeze(0)
        lp_max = log_P.detach().max() 
        P_sum = (log_P - lp_max).exp().sum(dim=(-2, -1)) * lp_max.exp()
        dual = dual_obj_left - eps * P_sum + eps

        return dual.mean() if dual.dim() > 0 else dual



    def train(self, dataloader_train):
        device = self._device()
        cfg    = self.cfg_m
        eps    = float(cfg.epsilon)

        n_iters      = int(cfg.get("num_train_iter") or 2000)
        lr           = float(cfg.get("learning_rate") or 1e-3)
        log_interval = int(cfg.get("log_interval")    or 100)
        max_grad_norm = float(cfg.get("max_grad_norm") or 1.0)

        # Precompute log_K on device
        C_t     = torch.tensor(self.C_np, dtype=torch.float64, device=device)
        log_K   = -C_t / eps   # (n_supply, n_demand) — fixed

        opt   = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_iters, eta_min=lr * 0.01)

        self.logger.info(
            f"[Meta_OT_World] Training  n_iters={n_iters}  lr={lr}  eps={eps}"
        )

        loss_ema = None
        step = 0
        t0   = time.time()
        pbar = tqdm(total=n_iters, desc="Meta_OT_World")

        while step < n_iters:
            for _, _, sw_b, dw_b in dataloader_train:
                if step >= n_iters:
                    break

                a = sw_b.to(device)   # (B, n_supply)
                b = dw_b.to(device)   # (B, n_demand)

                # Forward: MLP predicts f
                f = self.mlp(a, b)    # (B, n_supply)

                # Loss = -dual_obj (maximize dual = stable)
                loss = -self._dual_obj_from_f(a, b, f, log_K, eps)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.mlp.parameters(), max_grad_norm)
                opt.step()
                sched.step()

                lv       = loss.item()
                loss_ema = lv if loss_ema is None else 0.95 * loss_ema + 0.05 * lv
                step    += 1
                pbar.update(1)

                if step % log_interval == 0:
                    msg = (f"[{step}/{n_iters}] "
                           f"loss_ema={loss_ema:.4e}  t={time.time()-t0:.1f}s")
                    pbar.set_description(msg)
                    self.logger.info(msg)

        pbar.close()
        ckpt = os.path.join(self.log_sub_folder, "meta_net.pt")
        torch.save(self.mlp.state_dict(), ckpt)
        self.logger.info(f"[Meta_OT_World] Saved -> {ckpt}")


    def predict_plan(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        device = self._device()
        eps    = float(self.cfg_m.epsilon)

        a_t = torch.tensor(a, dtype=torch.float64, device=device).unsqueeze(0)
        b_t = torch.tensor(b, dtype=torch.float64, device=device).unsqueeze(0)
        C_t = torch.tensor(self.C_np, dtype=torch.float64, device=device)
        log_K = -C_t / eps

        with torch.no_grad():
            f_pred = self.mlp(a_t, b_t)                      # (1, n_supply)
            g_t, f_t = self._g_and_f_from_f_pred(f_pred, a_t, b_t, log_K, eps)  
            
            f = f_t[0].cpu().numpy()
            g = g_t[0].cpu().numpy()

        log_P = f[:, None] / eps - self.C_np / eps + g[None, :] / eps

        def lse(X, axis):
            m = X.max(axis=axis, keepdims=True)
            return np.log(np.exp(X - m).sum(axis=axis)) + m.squeeze(axis=axis)

        log_a = np.log(np.clip(a, 1e-300, None))
        log_b = np.log(np.clip(b, 1e-300, None))
        for _ in range(1):
            log_u = log_a[:, None] - lse(log_P, axis=1)[:, None]
            log_P = log_P + log_u
            
            log_v = log_b[None, :] - lse(log_P, axis=0)[None, :]
            log_P = log_P + log_v

        log_P -= log_P.max() 
        P = np.exp(log_P)
        P = np.clip(P, 0.0, None)
        s = P.sum()
        if s > 0:
            P /= s
            
        return P
