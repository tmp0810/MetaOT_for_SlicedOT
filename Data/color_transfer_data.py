"""
color_transfer_data.py
======================
Data loading and KMeans quantization for the color transfer experiment.

Each image is quantized once → (weights, centroids).
The dataset yields ordered (src, tgt) pairs for training OT regression.

Interface
---------
    weights  : (n_clusters,)   float64  — normalized histogram
    centroids: (n_clusters, 3) float64  — cluster centers in [0, 1]
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import MiniBatchKMeans
from PIL import Image


# ---------------------------------------------------------------------------
# Single-image quantization
# ---------------------------------------------------------------------------

def load_and_quantize(
    img_path: str,
    n_clusters: int = 500,
    seed: int = 0,
    max_size: int = 512,
):
    """
    Load an image, resize if needed, and quantize colors via MiniBatchKMeans.

    Parameters
    ----------
    img_path   : path to image (JPG / PNG / …)
    n_clusters : number of KMeans clusters  (= support size of the histogram)
    seed       : random seed for reproducibility
    max_size   : downsample so max(H, W) <= max_size before clustering

    Returns
    -------
    weights   : (n_clusters,)   float64  normalised histogram weights
    centroids : (n_clusters, 3) float64  cluster centers in [0, 1]
    labels    : (H*W,)          int64    per-pixel cluster index
    orig_shape: (H, W)          int      shape of the (possibly resized) image
    """
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    pixels     = np.array(img, dtype=np.float64) / 255.0   # (H, W, 3) in [0,1]
    orig_shape = pixels.shape[:2]                           # (H, W)
    X          = pixels.reshape(-1, 3)                      # (n_pixels, 3)

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init=4,
        batch_size=min(4096, len(X)),
        random_state=seed,
    )
    km.fit(X)

    centroids = km.cluster_centers_.astype(np.float64)      # (n_clusters, 3)
    labels    = km.labels_.astype(np.int64)                 # (n_pixels,)
    counts    = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    weights   = counts / counts.sum()

    return weights, centroids, labels, orig_shape


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ColorTransferDataset(Dataset):
    """
    Dataset of ordered image pairs for training OT regression.

    All images are pre-quantized at construction time and cached.
    Pairs are all ordered (i, j) with i != j, optionally subsampled.

    __getitem__ yields:
        src_weights   : (n_clusters,)   float64
        src_centroids : (n_clusters, 3) float64 in [0, 1]
        tgt_weights   : (n_clusters,)   float64
        tgt_centroids : (n_clusters, 3) float64 in [0, 1]
    """

    def __init__(
        self,
        image_dir: str,
        n_clusters: int = 500,
        seed: int = 0,
        max_pairs: int = None,
        max_img_size: int = 512,
    ):
        self.n_clusters = n_clusters
        self.seed       = seed

        # Discover images
        exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_dir, ext)))
        self.image_paths = sorted(paths)
        n = len(self.image_paths)
        assert n >= 2, f"Need >= 2 images in {image_dir}, found {n}"

        # Pre-quantize all images and cache (weights, centroids only — no labels)
        print(f"  Quantizing {n} images ({n_clusters} clusters each) ...")
        self._cache = {}
        for p in self.image_paths:
            w, c, _, _ = load_and_quantize(p, n_clusters, seed, max_img_size)
            self._cache[p] = (w, c)
        print(f"  Quantization done.")

        # Build ordered pair list
        self.pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        if max_pairs is not None and max_pairs < len(self.pairs):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(self.pairs), size=max_pairs, replace=False)
            self.pairs = [self.pairs[k] for k in idx]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j   = self.pairs[idx]
        sw, sc = self._cache[self.image_paths[i]]
        tw, tc = self._cache[self.image_paths[j]]
        return (
            torch.tensor(sw, dtype=torch.float64),    # (n_clusters,)
            torch.tensor(sc, dtype=torch.float64),    # (n_clusters, 3)
            torch.tensor(tw, dtype=torch.float64),    # (n_clusters,)
            torch.tensor(tc, dtype=torch.float64),    # (n_clusters, 3)
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_color_transfer_dataloader(
    image_dir: str,
    n_clusters: int = 500,
    batch_size: int = 1,
    seed: int = 0,
    max_pairs: int = None,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for color transfer training."""
    dataset = ColorTransferDataset(
        image_dir, n_clusters, seed, max_pairs,
    )

    def _collate(batch):
        sw, sc, tw, tc = zip(*batch)
        return (
            torch.stack(sw),   # (B, n_clusters)
            torch.stack(sc),   # (B, n_clusters, 3)
            torch.stack(tw),   # (B, n_clusters)
            torch.stack(tc),   # (B, n_clusters, 3)
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=num_workers,
    )
