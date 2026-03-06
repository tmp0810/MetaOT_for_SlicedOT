import torch
import torch.nn as nn
import ot
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp


# ---------------------------------------------------------------------------
# Simplex-constrained least-squares
# ---------------------------------------------------------------------------

def optimal_alpha_simplex(X, y, ridge=0.0, solver="OSQP"):
    """
    Solve:  min ||X a - y||^2 + ridge * ||a||^2
            s.t. a >= 0, sum(a) = 1
    # Ở đây change lại không cần sum(a) = 1 nữa
    Parameters
    ----------
    X     : (n, d) ndarray
    y     : (n,)   ndarray
    ridge : float >= 0.  0 = no regularisation; > 0 helps when columns of X
            are nearly collinear.

    Returns
    -------
    a : (d,) ndarray  (on the probability simplex)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    n, d = X.shape
    a = cp.Variable(d)

    obj = cp.sum_squares(X @ a - y)
    if ridge and ridge > 0:
        obj = obj + ridge * cp.sum_squares(a)

    #constraints = [a >= 0, cp.sum(a) == 1]
    constraints = [a >= 0]
    prob = cp.Problem(cp.Minimize(obj), constraints)

    try:
        prob.solve(solver=solver, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if a.value is None:
        raise RuntimeError("optimal_alpha_simplex: solver failed to find a solution.")
    return a.value  # np.ndarray shape (d,)


# ---------------------------------------------------------------------------
# Projection directions
# ---------------------------------------------------------------------------

def generate_uniform_unit_sphere_projections(
    dim,
    requires_grad=False,
    num_projections=1000,
    dtype=torch.float32,
    device="cpu",
):
    """Generate uniformly random unit-sphere directions of shape (num_projections, dim)."""
    projection_matrix = torch.randn((num_projections, dim), dtype=dtype, device=device)
    projection_matrix = projection_matrix / projection_matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
    if requires_grad:
        projection_matrix.requires_grad_(True)
    return projection_matrix


# ---------------------------------------------------------------------------
# Quantile helper (used internally by Sliced-Wasserstein)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 1-D Optimal Transport  (BUG FIX: was using numba.int64, now uses np.int64)
# ---------------------------------------------------------------------------

def solve_1D_ot(a, b, x, y, p):
    """Compute 1-D Optimal Transport between two histograms.

    **Important**: ``np.sum(a)`` must equal ``np.sum(b)``.
    **Important**: ``x`` and ``y`` must be **pre-sorted** in ascending order.

    Parameters
    ----------
    a : (n,) ndarray  — source weights (positive, sum to 1)
    b : (m,) ndarray  — target weights (positive, sum to 1)
    x : (n,) ndarray  — source support positions (sorted)
    y : (m,) ndarray  — target support positions (sorted)
    p : float         — power (>= 1)

    Returns
    -------
    I    : (q,) int64  — source indices of the sparse OT plan
    J    : (q,) int64  — target indices of the sparse OT plan
    P    : (q,) float  — masses of the sparse OT plan
    f    : (n,) float  — source dual potential
    g    : (m,) float  — target dual potential
    cost : float       — dual OT cost  (= sum_i f_i a_i + sum_j g_j b_j)

    Notes
    -----
    This is the North-West-corner / staircase algorithm for 1-D OT.
    """
    n = len(a)
    m = len(b)
    q = m + n - 1

    a1 = a.copy()
    b1 = b.copy()
    I = np.zeros(q, dtype=np.int64)   # fixed: was numba.int64
    J = np.zeros(q, dtype=np.int64)   # fixed: was numba.int64
    P = np.zeros(q)
    f = np.zeros(n)
    g = np.zeros(m)

    g[0] = np.abs(x[0] - y[0]) ** p

    for k in range(q - 1):
        i = I[k]
        j = J[k]
        if (a1[i] <= b1[j]) and (i < n - 1):
            I[k + 1] = i + 1
            J[k + 1] = j
            f[i + 1] = np.abs(x[i + 1] - y[j]) ** p - g[j]
        elif (a1[i] > b1[j]) and (j < m - 1):
            I[k + 1] = i
            J[k + 1] = j + 1
            g[j + 1] = np.abs(x[i] - y[j + 1]) ** p - f[i]
        elif i == n - 1:
            I[k + 1] = i
            J[k + 1] = j + 1
            g[j + 1] = np.abs(x[i] - y[j + 1]) ** p - f[i]
        elif j == m - 1:
            I[k + 1] = i + 1
            J[k + 1] = j
            f[i + 1] = np.abs(x[i + 1] - y[j]) ** p - g[j]

        t = min(a1[i], b1[j])
        P[k] = t
        a1[i] -= t
        b1[j] -= t

    P[k + 1] = max(a1[-1], b1[-1])
    cost = np.sum(f * a) + np.sum(g * b)
    return I, J, P, f, g, cost


def solve_1D_ot_unsorted(a, b, x, y, p=2):
    """
    Convenience wrapper: sorts ``x`` and ``y``, calls ``solve_1D_ot``, then
    maps the dual potentials ``f`` and ``g`` back to the **original** (unsorted)
    index ordering.

    Parameters
    ----------
    a, b : (n,) / (m,) weight vectors
    x, y : (n,) / (m,) support positions (need NOT be sorted)
    p    : power

    Returns
    -------
    f_orig : (n,) dual potential for the source, in original order
    g_orig : (m,) dual potential for the target, in original order
    cost   : dual OT cost
    """
    idx_a = np.argsort(x)
    idx_b = np.argsort(y)

    x_s = x[idx_a]; a_s = a[idx_a]
    y_s = y[idx_b]; b_s = b[idx_b]

    _, _, _, f_s, g_s, cost = solve_1D_ot(a_s, b_s, x_s, y_s, p)

    f_orig = np.empty_like(f_s)
    g_orig = np.empty_like(g_s)
    f_orig[idx_a] = f_s
    g_orig[idx_b] = g_s

    return f_orig, g_orig, cost


# ---------------------------------------------------------------------------
# Sliced-Wasserstein Distance
# ---------------------------------------------------------------------------

def Sliced_Wasserstein_Distance(
    X,
    Y,
    num_projections=1000,
    projection_matrix=None,
    p=2,
    device="cuda",
    chunk=1000,
    dtype=torch.float16,
    return_vectors=False,
):
    """
    Compute Sliced-Wasserstein Distance efficiently (supports backprop).

    Parameters
    ----------
    X : (n_src, d)  source support
    Y : (n_tgt, d)  target support
    num_projections : int
    projection_matrix : optional precomputed (num_projections, d) tensor
    p : Wasserstein-p order
    chunk : projections per chunk (memory management)
    return_vectors : if True, return per-projection distances instead of mean

    Returns
    -------
    Scalar SWD (or vector if return_vectors=True).
    """
    assert X.shape[-1] == Y.shape[-1], "Source and target must have the same dimension"
    dims = X.shape[-1]

    if projection_matrix is not None:
        num_projections = projection_matrix.shape[0]

    if num_projections < chunk:
        chunk = num_projections
        chunk_num_projections = 1
    else:
        chunk_num_projections = num_projections // chunk

    sum_w_p = [] if return_vectors else torch.tensor(0.0, device=device)

    for i in range(chunk_num_projections):
        if projection_matrix is None:
            projection_vectors = generate_uniform_unit_sphere_projections(
                dim=dims, num_projections=chunk, device=device
            )
        else:
            projection_vectors = projection_matrix[i * chunk : (i + 1) * chunk]

        projection_vectors = projection_vectors.to(dtype)
        X_proj = torch.matmul(X.to(dtype), projection_vectors.t())  # (n_src, chunk)
        Y_proj = torch.matmul(Y.to(dtype), projection_vectors.t())  # (n_tgt, chunk)

        X_sorted, _ = torch.sort(X_proj, dim=0)
        Y_sorted, _ = torch.sort(Y_proj, dim=0)

        diff = torch.abs(X_sorted - Y_sorted)
        w_1d = diff.mean(dim=0) if p == 1 else diff.pow(p).mean(dim=0)

        if return_vectors:
            sum_w_p.append(w_1d.pow(1.0 / p))
        else:
            sum_w_p = sum_w_p + torch.sum(w_1d, dim=0)

    if return_vectors:
        return torch.cat(sum_w_p, dim=0)
    else:
        mean_w_p = sum_w_p / num_projections
        return mean_w_p.pow(1.0 / p) if p != 1 else mean_w_p


# ---------------------------------------------------------------------------
# Cost / kernel matrix helpers
# ---------------------------------------------------------------------------

class cost_matrix_calculator:
    def __init__(self, img_size, epsilon):
        self.img_size = img_size
        x_grid = []
        for i in np.linspace(1, 0, num=img_size):
            for j in np.linspace(0, 1, num=img_size):
                x_grid.append([j, i])
        x_grid = np.array(x_grid, dtype=np.float64)
        self.x = x_grid
        self.y = x_grid
        self.epsilon = epsilon
        self.power = 2.0

    def compute_C_K(self):
        C = -2 * self.x @ self.y.T
        C += self.norm(self.x)[:, np.newaxis] + self.norm(self.y)[np.newaxis, :]
        K = np.exp(-C / self.epsilon)
        return C, K

    def norm(self, x):
        return np.sum(x ** 2, axis=-1)


class potentials_f_g:
    def __init__(self, img_size, epsilon, device):
        self.img_size = img_size
        self.device = device
        self.epsilon = epsilon
        c_cal = cost_matrix_calculator(img_size, epsilon)
        cost_matrix, kernel_matrix = c_cal.compute_C_K()
        self.K = torch.tensor(kernel_matrix, dtype=torch.float64).to(device)
        self.C = torch.tensor(cost_matrix, dtype=torch.float64).to(device)

    def pred_transport(self, f_pred, g_pred):
        """
        Recover transport plan from potentials.

        f_pred, g_pred : (..., n) tensors  (e.g. output of X_f @ alpha)
        """
        f_expand = f_pred.unsqueeze(-1)   # (..., n, 1)
        g_expand = g_pred.unsqueeze(-2)   # (..., 1, n)
        P = torch.exp(f_expand / self.epsilon) * self.K * torch.exp(g_expand / self.epsilon)
        return P.data.cpu().numpy()
