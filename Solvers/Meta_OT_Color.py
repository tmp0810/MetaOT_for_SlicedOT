import os
import csv
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from tqdm import tqdm
from PIL import Image

from Solvers.DefenseTrain import Defense_Train_Base
from Data.color_meta_data import (
    ImagePairSampler, get_image_paths, load_val_pairs,
    IMAGENET_MEAN, IMAGENET_STD,
)

def build_icnn_layout(dim_hidden=(128,), input_dim=3, quad_rank=3):
    layout     = []
    num_hidden = len(dim_hidden)

    # w_xs: num_hidden+1 linear layers (all take x as skip input)
    for i in range(num_hidden):
        layout.append((f"wx{i}_w", (dim_hidden[i], input_dim)))
        layout.append((f"wx{i}_b", (dim_hidden[i],)))
    layout.append((f"wx{num_hidden}_w", (1, input_dim)))
    layout.append((f"wx{num_hidden}_b", (1,)))

    # w_zs: PositiveDense layers, no bias
    for i in range(num_hidden - 1):
        layout.append((f"wz{i}_w_raw", (dim_hidden[i + 1], dim_hidden[i])))
    layout.append((f"wz{num_hidden - 1}_w_raw", (1, dim_hidden[-1])))

    # Quadratic term
    layout.append(("L", (quad_rank, input_dim)))

    num_params = sum(int(np.prod(s)) for _, s in layout)
    return layout, num_params


def unpack_icnn_params(params_flat, layout):
    """Unpack (P,) flat tensor → dict of named tensors."""
    d, offset = {}, 0
    for name, shape in layout:
        n = int(np.prod(shape))
        d[name] = params_flat[offset: offset + n].reshape(shape)
        offset  += n
    return d

def icnn_forward(X, params_flat, layout, dim_hidden, act=F.leaky_relu):
    p          = unpack_icnn_params(params_flat, layout)
    num_hidden = len(dim_hidden)

    # First wx layer
    z = act(X @ p["wx0_w"].T + p["wx0_b"])           # (N, dim_hidden[0])

    # Intermediate layers (empty for dim_hidden=[128])
    for i in range(num_hidden - 1):
        wz = F.softplus(p[f"wz{i}_w_raw"])            # (dim_hidden[i+1], dim_hidden[i])
        wx = p[f"wx{i + 1}_w"]
        bx = p[f"wx{i + 1}_b"]
        z  = act(z @ wz.T + X @ wx.T + bx)

    # Final layer: wz_last(z) + wx_last(x) + ||Lx||^2
    wz_last = F.softplus(p[f"wz{num_hidden - 1}_w_raw"])   # (1, dim_hidden[-1])
    wx_last = p[f"wx{num_hidden}_w"]                         # (1, input_dim)
    bx_last = p[f"wx{num_hidden}_b"]                         # (1,)
    L       = p["L"]                                          # (quad_rank, input_dim)
    quad    = ((X @ L.T) ** 2).sum(dim=-1)                   # (N,)

    y = (z @ wz_last.T + X @ wx_last.T + bx_last).squeeze(-1) + quad
    return y  # (N,)


def push_grad(X, params_flat, layout, dim_hidden, create_graph=True):
    if not X.requires_grad:
        X = X.detach().requires_grad_(True)
    y    = icnn_forward(X, params_flat, layout, dim_hidden)   # (N,)
    grad = torch.autograd.grad(
        y.sum(), X,
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    return grad   # (N, d)


class MetaICNN_Color(nn.Module):
    def __init__(self, num_icnn_params: int, bottleneck_size: int = 512,
                 fc_num_hidden_units: int = 512, fc_num_hidden_layers: int = 2):
        super().__init__()
        assert bottleneck_size % 2 == 0

        # Shared ResNet18 (same for X and Y, matching JAX self.resnet)
        resnet    = tv_models.resnet18(weights=None)
        resnet.fc = nn.Linear(512, bottleneck_size // 2)
        self.resnet = resnet   # shared weights, called twice per forward

        # FC head: concat(zx, zy) → 2 * num_icnn_params
        layers, in_d = [], bottleneck_size
        for _ in range(fc_num_hidden_layers):
            layers += [nn.Linear(in_d, fc_num_hidden_units), nn.ReLU()]
            in_d    = fc_num_hidden_units
        layers += [nn.Linear(fc_num_hidden_units, 2 * num_icnn_params)]
        self.fc = nn.Sequential(*layers)

        self.num_icnn_params = num_icnn_params

    def forward(self, X_sq, Y_sq):
        """
        X_sq, Y_sq : (B, 3, 224, 224) float32
        Returns    : D_flat (B, P), Dc_flat (B, P)
        """
        zx  = self.resnet(X_sq)
        zy  = self.resnet(Y_sq)
        z   = torch.cat([zx, zy], dim=-1)
        out = self.fc(z)
        return out.split(self.num_icnn_params, dim=-1)

class Meta_OT_Color(Defense_Train_Base):

    is_continuous = True   # flag for eval_color_transfer.py dispatch

    def __init__(self, cfg_proj, cfg_m):
        Defense_Train_Base.__init__(self, cfg_proj, cfg_m, name="Meta_OT_Color")

        dim_hidden   = tuple(cfg_m.get("dim_hidden") or (128,))
        quad_rank    = int(cfg_m.get("quad_rank")    or 3)
        input_dim    = 3

        self.dim_hidden = dim_hidden
        self.quad_rank  = quad_rank
        self.input_dim  = input_dim

        self.icnn_layout, self.num_icnn_params = build_icnn_layout(
            dim_hidden, input_dim, quad_rank)

        self.logger.info(
            f"[Meta_OT_Color] ICNN dim_hidden={dim_hidden}  "
            f"num_icnn_params={self.num_icnn_params}  "
            f"MetaICNN outputs 2x{self.num_icnn_params}={2*self.num_icnn_params}"
        )
        self._build_network()


    def _device(self):
        if torch.cuda.is_available() and hasattr(self.cfg_m, "gpu"):
            return torch.device(f"cuda:{self.cfg_m.gpu}")
        return torch.device("cpu")

    def _build_network(self):
        cfg = self.cfg_m
        self.meta_icnn = MetaICNN_Color(
            num_icnn_params      = self.num_icnn_params,
            bottleneck_size      = int(cfg.get("bottleneck_size")      or 512),
            fc_num_hidden_units  = int(cfg.get("fc_num_hidden_units")  or 512),
            fc_num_hidden_layers = int(cfg.get("fc_num_hidden_layers") or 2),
        ).to(self._device())
        n_p = sum(p.numel() for p in self.meta_icnn.parameters())
        self.logger.info(f"[Meta_OT_Color] MetaICNN_Color  params={n_p:,}")


    def _loss_single(self, D_p, Dc_p, X, Y):
        cfg = self.cfg_m
        li  = self.icnn_layout
        dh  = self.dim_hidden

        # ── X_hat = nabla Dc(Y) ───────────────────────────────────────────
        X_hat   = push_grad(Y, Dc_p, li, dh, create_graph=True)
        X_hat_d = X_hat.detach()   # stop_gradient for dual term only

        # ── Dual loss ─────────────────────────────────────────────────────
        D_X     = icnn_forward(X, D_p, li, dh)        # (N,)
        D_Xhatd = icnn_forward(X_hat_d, D_p, li, dh)  # (N,)
        dual    = (D_X + (X_hat_d * Y).sum(-1) - D_Xhatd).mean()

        # ── Cycle loss ────────────────────────────────────────────────────
        # Y_hat = nabla D(X)
        Y_hat = push_grad(X, D_p, li, dh, create_graph=True)

        # cyc1: nabla D(X_hat) ≈ Y   — X_hat ALIVE (not detached)
        cyc1  = ((push_grad(X_hat, D_p, li, dh, create_graph=True) - Y) ** 2).mean()

        # cyc2: nabla Dc(Y_hat) ≈ X  — Y_hat ALIVE (not detached)
        cyc2  = ((push_grad(Y_hat, Dc_p, li, dh, create_graph=True) - X) ** 2).mean()

        cycle = cyc1 + cyc2

        # ── L2 regularisation ─────────────────────────────────────────────
        l2  = float(cfg.get("l2_penalty") or 1e-5)
        reg = l2 * (D_p ** 2).mean() + l2 * (Dc_p ** 2).mean()

        cyc_w = float(cfg.get("cycle_loss_weight") or 0.1)
        loss  = dual + cyc_w * cycle + reg
        return loss, dual.detach().item(), cycle.detach().item()

    # ── pretrain identity ─────────────────────────────────────────────────────

    def _pretrain_identity(self, pair_sampler, val_pairs=None):
        """
        Pretrain MetaICNN so push_grad(D, x) ≈ push_grad(Dc, x) ≈ x.

        JAX: X = 2.*(uniform([N,3])-.5)+.5  → range [-0.5, 1.5]^3
        """
        cfg    = self.cfg_m
        device = self.meta_icnn.resnet.fc.weight.device
        n_iter = int(cfg.get("num_pretrain_iter") or 5000)
        lr_pre = float(cfg.get("pretrain_lr")     or 1e-3)
        thresh = float(cfg.get("pretrain_loss_threshold") or 1e-3)
        B      = int(cfg.get("meta_batch_size")   or 4)
        N      = int(cfg.get("inner_batch_size")  or 256)
        l2     = float(cfg.get("l2_penalty")      or 1e-5)
        max_gn = float(cfg.get("max_grad_norm")   or 1.0)
        li, dh = self.icnn_layout, self.dim_hidden

        opt = torch.optim.Adam(self.meta_icnn.parameters(), lr=lr_pre)
        self.meta_icnn.train()
        loss_ema = None
        pbar = tqdm(range(n_iter), desc="Pretrain identity")

        for step in pbar:
            _, _, X_sq, Y_sq, _, _ = pair_sampler.sample_image_pair_batch(B, val_pairs)
            X_sq_t = torch.tensor(X_sq, dtype=torch.float32, device=device)
            Y_sq_t = torch.tensor(Y_sq, dtype=torch.float32, device=device)

            D_flat, Dc_flat = self.meta_icnn(X_sq_t, Y_sq_t)   # (B, P)

            # JAX: X = 2.*(uniform([N,3])-.5)+.5  → [-0.5, 1.5]^3
            X_rand = 2.0 * (torch.rand(N, 3, device=device) - 0.5) + 0.5

            total = torch.zeros(1, device=device)
            for b in range(B):
                push_D  = push_grad(X_rand, D_flat[b],  li, dh, create_graph=True)
                push_Dc = push_grad(X_rand, Dc_flat[b], li, dh, create_graph=True)
                loss_b  = ((push_D  - X_rand) ** 2).sum(-1).mean()
                loss_b += ((push_Dc - X_rand) ** 2).sum(-1).mean()
                loss_b += l2 * (D_flat[b]  ** 2).mean()
                loss_b += l2 * (Dc_flat[b] ** 2).mean()
                total   = total + loss_b
            loss = total / B

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.meta_icnn.parameters(), max_gn)
            opt.step()

            lv       = loss.item()
            loss_ema = lv if loss_ema is None else 0.95 * loss_ema + 0.05 * lv
            if step % 500 == 0:
                msg = f"pretrain [{step}/{n_iter}] loss={loss_ema:.4e}"
                pbar.set_description(msg)
                self.logger.info(msg)
            if loss_ema is not None and loss_ema < thresh:
                self.logger.info(f"Pretrain converged at step {step}")
                break

        pbar.close()

    # ── main training ─────────────────────────────────────────────────────────

    def train(self, image_dir: str):
        """
        Port of JAX Workspace.run() in train_color_meta.py.
        """
        cfg    = self.cfg_m
        device = self.meta_icnn.resnet.fc.weight.device

        # Load images
        image_paths = get_image_paths(image_dir)
        assert len(image_paths) >= 2, f"Need >= 2 images in {image_dir}"
        self.logger.info(f"[Meta_OT_Color] {len(image_paths)} images")

        pairs_file = os.path.join(image_dir, "pairs.txt")
        val_pairs  = load_val_pairs(pairs_file, image_dir)
        val_set    = set(val_pairs)
        self.logger.info(f"[Meta_OT_Color] val_pairs={len(val_pairs)}")

        num_rgb = cfg.get("num_rgb_sample")   # None = use all pixels
        pair_sampler = ImagePairSampler(image_paths, num_rgb_sample=num_rgb)

        # Pretrain identity
        self.logger.info("[Meta_OT_Color] Pretrain identity ...")
        self._pretrain_identity(pair_sampler, val_set)

        # Main training
        n_iters      = int(cfg.get("num_train_iter")   or 50000)
        lr           = float(cfg.get("lr")             or 1e-4)
        B            = int(cfg.get("meta_batch_size")  or 4)
        N            = int(cfg.get("inner_batch_size") or 256)
        log_interval = int(cfg.get("log_interval")     or 1000)
        max_gn       = float(cfg.get("max_grad_norm")  or 1.0)
        li, dh       = self.icnn_layout, self.dim_hidden

        # LR schedule: warmup 5000 steps + cosine decay (mirrors JAX optax)
        opt = torch.optim.Adam(self.meta_icnn.parameters(), lr=lr)
        warmup = min(5000, n_iters // 10)
        def lr_lambda(s):
            if s < warmup:
                return s / max(1, warmup)
            p = (s - warmup) / max(1, n_iters - warmup)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * p)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        # CSV logging (matches JAX log.csv fields)
        log_csv  = os.path.join(self.log_sub_folder, "log.csv")
        log_file = open(log_csv, "w", newline="")
        writer   = csv.DictWriter(
            log_file,
            fieldnames=["iter", "time", "loss", "dual_obj", "cycle_loss"])
        writer.writeheader(); log_file.flush()

        self.meta_icnn.train()
        rng      = np.random.default_rng(0)
        loss_ema = dual_ema = cycle_ema = None
        t0       = time.time()
        self.logger.info(
            f"[Meta_OT_Color] Training  n_iters={n_iters}  lr={lr}  B={B}  N={N}")

        pbar = tqdm(range(n_iters), desc="Meta_OT_Color")
        for step in pbar:
            _, _, X_sq, Y_sq, X_full, Y_full =                 pair_sampler.sample_image_pair_batch(B, val_set)

            X_sq_t = torch.tensor(X_sq, dtype=torch.float32, device=device)
            Y_sq_t = torch.tensor(Y_sq, dtype=torch.float32, device=device)

            D_flat, Dc_flat = self.meta_icnn(X_sq_t, Y_sq_t)   # (B, P) float32

            total_loss = torch.zeros(1, device=device)
            t_dual = t_cycle = 0.0
            for b in range(B):
                # Sample inner pixels
                Xi = torch.tensor(
                    rng.choice(X_full[b], N), dtype=torch.float32, device=device)
                Yi = torch.tensor(
                    rng.choice(Y_full[b], N), dtype=torch.float32, device=device)

                loss_b, d_b, c_b = self._loss_single(D_flat[b], Dc_flat[b], Xi, Yi)
                total_loss = total_loss + loss_b
                t_dual    += d_b
                t_cycle   += c_b

            loss = total_loss / B
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.meta_icnn.parameters(), max_gn)
            opt.step()
            scheduler.step()

            lv        = loss.item()
            dv        = t_dual  / B
            cv        = t_cycle / B
            loss_ema  = lv if loss_ema  is None else 0.99 * loss_ema  + 0.01 * lv
            dual_ema  = dv if dual_ema  is None else 0.99 * dual_ema  + 0.01 * dv
            cycle_ema = cv if cycle_ema is None else 0.99 * cycle_ema + 0.01 * cv

            if step % log_interval == 0:
                elapsed = time.time() - t0
                msg = (f"[{step}/{n_iters}] "
                       f"loss={loss_ema:.4e}  dual={dual_ema:.4e}  "
                       f"cycle={cycle_ema:.4e}  t={elapsed:.0f}s")
                pbar.set_description(msg)
                self.logger.info(msg)
                writer.writerow({
                    "iter": step, "time": elapsed,
                    "loss": loss_ema, "dual_obj": dual_ema, "cycle_loss": cycle_ema,
                })
                log_file.flush()
                torch.save(self.meta_icnn.state_dict(),
                           os.path.join(self.log_sub_folder, "meta_icnn_latest.pt"))

        pbar.close()
        log_file.close()
        ckpt = os.path.join(self.log_sub_folder, "meta_icnn_final.pt")
        torch.save(self.meta_icnn.state_dict(), ckpt)
        self.logger.info(f"[Meta_OT_Color] Saved → {ckpt}")

    # ── inference ─────────────────────────────────────────────────────────────

    def _img_to_square_tensor(self, img_np):
        """(H,W,3) uint8 → (1,3,224,224) float32 tensor on device."""
        device = self.meta_icnn.resnet.fc.weight.device
        pil = Image.fromarray(img_np.astype(np.uint8)).resize((224, 224), Image.LANCZOS)
        sq  = np.array(pil, dtype=np.float32) / 255.0
        sq  = (sq - IMAGENET_MEAN) / IMAGENET_STD
        sq  = sq.transpose(2, 0, 1)[None]   # (1,3,224,224)
        return torch.tensor(sq, dtype=torch.float32, device=device)

    def apply_map(self, src_img_np: np.ndarray, tgt_img_np: np.ndarray,
                  pixel_batch: int = 5000) -> np.ndarray:
        """
        Apply T(x) = nabla D(x; D_params) pixel-wise.

        Port of JAX push_image() from train_color_single.py.

        src_img_np, tgt_img_np : (H, W, 3) uint8
        Returns                : (H, W, 3) uint8
        """
        device = self.meta_icnn.resnet.fc.weight.device
        li, dh = self.icnn_layout, self.dim_hidden

        X_sq = self._img_to_square_tensor(src_img_np)
        Y_sq = self._img_to_square_tensor(tgt_img_np)

        self.meta_icnn.eval()
        with torch.no_grad():
            D_flat, _ = self.meta_icnn(X_sq, Y_sq)   # (1, P)
        D_p = D_flat[0]   # (P,)

        H, W   = src_img_np.shape[:2]
        pixels = src_img_np.reshape(-1, 3).astype(np.float32) / 255.0  # (N, 3) in [0,1]
        result = np.zeros_like(pixels)

        for start in range(0, len(pixels), pixel_batch):
            end     = min(start + pixel_batch, len(pixels))
            X_batch = torch.tensor(
                pixels[start:end], dtype=torch.float32, device=device)
            with torch.no_grad():
                pushed = push_grad(X_batch, D_p, li, dh, create_graph=False).detach()
            result[start:end] = pushed.cpu().numpy()

        result = np.clip(result, 0.0, 1.0)
        return (result * 255).astype(np.uint8).reshape(H, W, 3)

    @staticmethod
    def _compute_cost(x_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
        """Squared-Euclidean cost — used by eval_color_transfer.py for Sinkhorn baseline."""
        diff = x_src[:, None, :] - x_tgt[None, :, :]
        return np.sum(diff ** 2, axis=-1)
