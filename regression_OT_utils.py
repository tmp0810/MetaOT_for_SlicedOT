import torch
import torch.nn as nn
import ot
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

def optimal_alpha_simplex(X, y, ridge=0.0, solver="OSQP"):
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
    return a.value  

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

def solve_1D_ot(a, b, x, y, p):  
    #The North-West-corner / staircase algorithm for 1-D OT.
    n = len(a)
    m = len(b)
    q = m + n - 1

    a1 = a.copy()
    b1 = b.copy()
    I = np.zeros(q, dtype=np.int64)   
    J = np.zeros(q, dtype=np.int64)   
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
