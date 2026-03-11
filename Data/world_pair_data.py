"""
world_pair_data.py
==================
PyTorch replacement for meta_ot/data.py WorldPairSampler.

Fixed geometry:
  - supply_locs (n_supply, 3): euclidean positions on unit sphere,
    sampled once from uniform distribution over landmass.
  - demand_locs (n_demand, 3): euclidean positions on unit sphere,
    sampled once from population density distribution.

Per-batch randomness (weights only):
  - supply weights: bernoulli mask × uniform → normalized
  - demand weights: uniform → normalized
"""

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


# ---------------------------------------------------------------------------
# Location loading (run once)
# ---------------------------------------------------------------------------

def load_world_locations(
    population_fname: str,
    n_supply: int = 100,
    n_demand: int = 10_000,
    seed: int = 0,
    downsample: int = 4,
):
    """
    Load population tiff, sample fixed supply and demand locations on the sphere.

    Parameters
    ----------
    population_fname : path to .tif population raster
    n_supply         : number of supply locations (sparse)
    n_demand         : number of demand locations (dense)
    seed             : random seed
    downsample       : spatial downsampling factor to reduce memory usage.
                       downsample=4 on a 21600×43200 raster → 5400×10800 (~470MB float64).
                       Increase if still OOM; set to 1 for full resolution.

    Returns
    -------
    supply_spherical : (n_supply, 2)  [phi, theta] spherical coordinates
    supply_euclidean : (n_supply, 3)  unit-sphere euclidean
    demand_spherical : (n_demand, 2)
    demand_euclidean : (n_demand, 3)
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(population_fname) as src:
        full_h, full_w = src.height, src.width
        out_h = max(1, full_h // downsample)
        out_w = max(1, full_w // downsample)

        print(f"  Raster: {full_h}×{full_w}  → downsampled to {out_h}×{out_w}  "
              f"({out_h * out_w * 8 / 1e9:.2f} GB float64)")

        # Read with spatial downsampling — avoids allocating full raster
        raw = src.read(
            1,
            out_shape=(1, out_h, out_w),
            resampling=Resampling.average,
        )[0].astype(np.float64)   # (out_h, out_w)

        nodata = src.nodata   # e.g. -9999, -3.4e+38, or None

    # Mask out nodata values before anything else
    P = raw.copy()
    if nodata is not None:
        P[np.isclose(P, nodata, rtol=1e-3)] = 0.0
    P[~np.isfinite(P)] = 0.0   # NaN / inf → 0
    P[P < 0] = 0.0

    print(f"  After cleaning: nonzero pixels = {(P > 0).sum():,} / {P.size:,}  "
          f"max={P.max():.4f}  nodata_value={nodata}")

    # Population distribution (for demand)
    Pflat = P.ravel().copy()
    p_max = Pflat.max()
    if p_max <= 0:
        raise ValueError(
            f"Population raster is all-zero after cleaning. "
            f"Check that {population_fname} is a valid population tiff."
        )
    Pflat /= p_max           # numerical stability
    Pflat /= Pflat.sum()

    # Uniform-over-landmass (for supply)
    Uflat = (Pflat > 0).astype(np.float64)
    u_sum = Uflat.sum()
    if u_sum <= 0:
        raise ValueError("No landmass pixels found after cleaning raster.")
    Uflat /= u_sum

    def _sample(p, num_samples, rng_seed):
        rng      = np.random.default_rng(rng_seed)
        idxs     = rng.choice(len(p), p=p, size=num_samples)
        row, col = np.divmod(idxs, P.shape[1])

        theta = (1.0 - row / P.shape[0]) * np.pi       # colatitude [0, π]
        phi   = (col / P.shape[1]) * 2 * np.pi - np.pi # longitude  [-π, π]

        spherical = np.stack([phi, theta], axis=1)       # (n, 2)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        euclidean = np.stack([x, y, z], axis=1)          # (n, 3)
        return spherical, euclidean

    demand_sph, demand_euc = _sample(Pflat, n_demand, seed)
    supply_sph, supply_euc = _sample(Uflat, n_supply, seed + 1)

    return supply_sph, supply_euc, demand_sph, demand_euc


# ---------------------------------------------------------------------------
# Streaming dataset
# ---------------------------------------------------------------------------

class WorldPairDataset(IterableDataset):
    """
    Infinite stream of (supply_weights, demand_weights) pairs.

    Supply weights: bernoulli(p) mask × U[0,1], then L1-normalised.
    Demand weights: U[0,1], then L1-normalised.

    Yields 4-tuples  (dummy, dummy, supply_w, demand_w)
    to match the interface expected by OT_Regression_Sliced._fit:
        for _, _, x_a, x_b in dataloader: ...
    """

    def __init__(
        self,
        n_supply: int = 100,
        n_demand: int = 10_000,
        supply_bernoulli_p: float = 0.5,
        num_pairs: int = None,   # None → infinite
        seed: int = 42,
    ):
        self.n_supply           = n_supply
        self.n_demand           = n_demand
        self.supply_bernoulli_p = supply_bernoulli_p
        self.num_pairs          = num_pairs
        self.seed               = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed if worker_info is None else self.seed + worker_info.id
        rng  = np.random.default_rng(seed)

        count = 0
        while self.num_pairs is None or count < self.num_pairs:
            # Supply: bernoulli mask + random weights
            mask     = rng.binomial(1, self.supply_bernoulli_p, self.n_supply).astype(np.float64)
            supply_w = mask * rng.uniform(0.0, 1.0, self.n_supply)
            if supply_w.sum() < 1e-12:          # edge case: all masked out
                supply_w = np.ones(self.n_supply, dtype=np.float64)
            supply_w /= supply_w.sum()

            # Demand: uniform weights
            demand_w  = rng.uniform(0.0, 1.0, self.n_demand).astype(np.float64)
            demand_w /= demand_w.sum()

            yield (
                torch.zeros(1),                                         # dummy label a
                torch.zeros(1),                                         # dummy label b
                torch.tensor(supply_w, dtype=torch.float64),           # (n_supply,)
                torch.tensor(demand_w, dtype=torch.float64),           # (n_demand,)
            )
            count += 1


def get_world_pair_dataloader(
    n_supply: int  = 100,
    n_demand: int  = 10_000,
    batch_size: int = 1,
    supply_bernoulli_p: float = 0.5,
    num_pairs: int  = None,
    seed: int       = 42,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader that streams WorldPair samples.

    Each batch: (dummy, dummy, supply_w, demand_w)
      supply_w : (batch, n_supply)
      demand_w : (batch, n_demand)
    """
    dataset = WorldPairDataset(
        n_supply=n_supply,
        n_demand=n_demand,
        supply_bernoulli_p=supply_bernoulli_p,
        num_pairs=num_pairs,
        seed=seed,
    )

    def _collate(batch):
        _, _, x_a, x_b = zip(*batch)
        return (
            torch.zeros(len(batch)),      # dummy
            torch.zeros(len(batch)),      # dummy
            torch.stack(x_a),            # (B, n_supply)
            torch.stack(x_b),            # (B, n_demand)
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_collate,
        num_workers=num_workers,
    )
