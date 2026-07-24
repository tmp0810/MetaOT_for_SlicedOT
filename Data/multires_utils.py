import numpy as np
import torch
import torch.nn.functional as F

# The 3 resolutions used in the W1 rebuttal experiment: n in {784, 400, 196}
RESOLUTIONS = [28, 20, 14]


def make_grid(img_size):
    #Coordinate grid in [0,1]^2
    grid = np.array(
        [[j, i] for i in np.linspace(1, 0, num=img_size)
                for j in np.linspace(0, 1, num=img_size)],
        dtype=np.float64,
    )
    return grid

def build_cost_cross(grid_a, grid_b):
    diff = grid_a[:, None, :] - grid_b[None, :, :]
    return np.sum(diff ** 2, axis=-1)


def resize_prob(vec, in_size, out_size):
    #Resize a flattened (in_size**2,) probability vector to a (out_size**2,) probability vector and renormalize

    if in_size == out_size:
        return vec.copy() if isinstance(vec, np.ndarray) else vec.clone()

    is_np = isinstance(vec, np.ndarray)
    x = torch.as_tensor(vec, dtype=torch.float64).reshape(1, 1, in_size, in_size)
    if out_size < in_size:
        x = F.adaptive_avg_pool2d(x, out_size)
    else:
        x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)
    x = x.clamp(min=0).reshape(-1)
    x = x / x.sum()
    return x.numpy() if is_np else x


def infer_res(vec_or_len):
    n = vec_or_len if isinstance(vec_or_len, (int, np.integer)) else vec_or_len.shape[-1]
    r = int(round(n ** 0.5))
    assert r * r == n, f"length {n} is not a perfect square"
    return r


class MultiResGridMixin:
    def _init_multires(self, resolutions=RESOLUTIONS):
        self._mr_resolutions = list(resolutions)
        self._mr_grids = {r: make_grid(r) for r in self._mr_resolutions}
        self._mr_cost_cache = {}
        self._mr_logK_cache = {}

    def _grid(self, r):
        return self._mr_grids[r]

    def _cost(self, ra, rb):
        key = (ra, rb)
        if key not in self._mr_cost_cache:
            self._mr_cost_cache[key] = build_cost_cross(self._mr_grids[ra], self._mr_grids[rb])
        return self._mr_cost_cache[key]

    def _logK(self, ra, rb, eps):
        key = (ra, rb, eps)
        if key not in self._mr_logK_cache:
            C = self._cost(ra, rb)
            C_t = torch.tensor(C, dtype=torch.float64, device=self.device)
            self._mr_logK_cache[key] = -C_t / eps
        return self._mr_logK_cache[key]

    @staticmethod
    def _infer_res(vec):
        return infer_res(vec)
