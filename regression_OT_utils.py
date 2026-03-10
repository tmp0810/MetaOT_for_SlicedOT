import torch
import torch.nn as nn
import ot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

def _ridge_regression(X, y, ridge=0.0):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    H   = X.T @ X
    Xty = X.T @ y
    if ridge > 0:
        H = H + ridge * np.eye(H.shape[0])
    return np.linalg.solve(H, Xty)

def generate_uniform_unit_sphere_projections(
    dim,
    requires_grad=False,
    num_projections=1000,
    dtype=torch.float32,
    device="cpu",
):
    projection_matrix = torch.randn((num_projections, dim), dtype=dtype, device=device)
    projection_matrix = projection_matrix / projection_matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
    if requires_grad:
        projection_matrix.requires_grad_(True)
    return projection_matrix


def quantile_function(qs, cws, xs):
    cws, _ = torch.sort(cws, dim=0)
    qs, _ = torch.sort(qs, dim=0)
    num_dist = xs.shape[0]
    num_projections = xs.shape[-1]
    cws = cws.t().contiguous()
    qs = qs.t().contiguous()
    idx = torch.searchsorted(cws, qs).t()
    return torch.take_along_dim(
        input=xs,
        indices=idx.expand(num_projections, idx.shape[-1]).t().expand(num_dist, idx.shape[-1], num_projections),
        dim=-2,
    )

def cost_function(p):
    """Returns elementwise cost C(x) = |x|^p as a torch function."""
    if p == 1:
        return torch.abs
    if p == 2:
        return torch.square
    else:
        return lambda x: torch.pow(torch.abs(x), p)


def emd1D(u_values, v_values, u_weights=None, v_weights=None, p=2, require_sort=True):
    """
    Vectorized 1-D OT loss via inverse-CDF formula.

    Parameters
    ----------
    u_values  : (Proj, N)  support of source measures
    v_values  : (Proj, M)  support of target measures
    u_weights : (Proj, N) or (N,)  weights of source (uniform if None)
    v_weights : (Proj, M) or (M,)  weights of target (uniform if None)
    p         : cost exponent C(x)=|x|^p
    require_sort : sort supports if not already sorted

    Returns
    -------
    loss : (Proj,)  1-D OT cost for each projection
    """
    proj = u_values.shape[0]
    n    = u_values.shape[-1]
    m    = v_values.shape[-1]
    device, dtype = u_values.device, u_values.dtype

    if u_weights is None:
        u_weights = torch.full((proj, n), 1/n, dtype=dtype, device=device)
    elif u_weights.dim() == 1:
        u_weights = u_weights.unsqueeze(0).expand(proj, -1)

    if v_weights is None:
        v_weights = torch.full((proj, m), 1/m, dtype=dtype, device=device)
    elif v_weights.dim() == 1:
        v_weights = v_weights.unsqueeze(0).expand(proj, -1)

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)
        u_weights = torch.gather(u_weights, -1, u_sorter)
        v_weights = torch.gather(v_weights, -1, v_sorter)

    u_cdf = torch.clamp(torch.cumsum(u_weights, -1), max=1.)
    v_cdf = torch.clamp(torch.cumsum(v_weights, -1), max=1.)

    cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)

    u_index = torch.searchsorted(u_cdf, cdf_axis)
    v_index = torch.searchsorted(v_cdf, cdf_axis)

    u_icdf = torch.gather(u_values, -1, u_index.clamp(0, n-1))
    v_icdf = torch.gather(v_values, -1, v_index.clamp(0, m-1))

    cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
    delta = cdf_axis[..., 1:] - cdf_axis[..., :-1]

    return torch.sum(delta * cost_function(p)(u_icdf - v_icdf), dim=-1)


def emd1D_dual(u_values, v_values, u_weights=None, v_weights=None, p=2, require_sort=True):
    """
    Kantorovich dual potentials for a batch of 1-D OT problems via autograd.

    By the envelope theorem, the gradient of W_p(a, b) w.r.t. a_i equals
    the optimal source potential f_i*, and w.r.t. b_j equals g_j*.
    This avoids the sequential NW-corner loop entirely.

    Parameters
    ----------
    u_values  : (Proj, N)  support of source measures  (shared grid ok)
    v_values  : (Proj, M)  support of target measures
    u_weights : (Proj, N) or (N,)  source weights
    v_weights : (Proj, M) or (M,)  target weights
    p         : cost exponent
    require_sort : passed through to emd1D

    Returns
    -------
    f_grad : (Proj, N)  source potentials  ∂W/∂a
    g_grad : (Proj, M)  target potentials  ∂W/∂b
    loss   : scalar     total OT cost (sum over projections)

    Notes
    -----
    - Potentials are unique only up to an additive constant per pair.
      Centering (subtract mean) before regression is recommended.
    - Gradients are computed jointly for all Proj directions in one
      backward pass — fully vectorized on GPU.
    """
    proj = u_values.shape[0]
    n    = u_values.shape[-1]
    m    = v_values.shape[-1]
    device, dtype = u_values.device, u_values.dtype

    # Detach and enable grad on weights only
    if u_weights is None:
        mu = torch.full((proj, n), 1/n, dtype=dtype, device=device)
    elif u_weights.dim() == 1:
        mu = u_weights.unsqueeze(0).expand(proj, -1).clone().detach()
    else:
        mu = u_weights.clone().detach()

    if v_weights is None:
        nu = torch.full((proj, m), 1/m, dtype=dtype, device=device)
    elif v_weights.dim() == 1:
        nu = v_weights.unsqueeze(0).expand(proj, -1).clone().detach()
    else:
        nu = v_weights.clone().detach()

    mu.requires_grad_(True)
    nu.requires_grad_(True)

    loss = emd1D(u_values, v_values,
                 u_weights=mu, v_weights=nu,
                 p=p, require_sort=require_sort).sum()
    loss.backward()

    return mu.grad, nu.grad, loss
