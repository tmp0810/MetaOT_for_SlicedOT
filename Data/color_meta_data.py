import os
import glob
import random
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageSampler:
    def __init__(self, image_path: str, square_size: int = 224,
                 num_rgb_sample: int = None, seed: int = 0):
        self.path  = image_path
        self.image = Image.open(image_path).convert("RGB")
        flat = np.array(self.image, dtype=np.float32).reshape(-1, 3) / 255.0  # (N, 3)
        if num_rgb_sample is not None and len(flat) > num_rgb_sample:
            rng  = np.random.default_rng(seed)
            idx  = rng.choice(len(flat), size=num_rgb_sample, replace=False)
            flat = flat[idx]
        self.flat_pixels = flat  # (N, 3)
        sq  = self.image.resize((square_size, square_size), Image.LANCZOS)
        sq  = np.array(sq, dtype=np.float32) / 255.0          # (H, W, 3)
        sq  = (sq - IMAGENET_MEAN) / IMAGENET_STD              # ImageNet normalise
        self.image_square = sq.transpose(2, 0, 1)              # (3, H, W) float32

    def sample_pixels(self, n: int, rng=None) -> np.ndarray:
        if rng is not None:
            idx = rng.integers(0, len(self.flat_pixels), size=n)
        else:
            idx = np.random.randint(0, len(self.flat_pixels), size=n)
        return self.flat_pixels[idx]


class ImagePairSampler:

    def __init__(self, image_paths: list, num_rgb_sample: int = None, seed: int = 0):
        print(f"  Loading {len(image_paths)} images ...")
        self.samplers = []
        for i, p in enumerate(image_paths):
            try:
                s = ImageSampler(p, num_rgb_sample=num_rgb_sample, seed=seed + i)
                self.samplers.append(s)
            except Exception as e:
                print(f"  Warning: skip {os.path.basename(p)}: {e}")
        print(f"  Loaded {len(self.samplers)} images.")

    def sample_image_pair(self, val_pairs=None):
        """Return (X_sampler, Y_sampler) avoiding val_pairs."""
        while True:
            X_s, Y_s = random.sample(self.samplers, 2)
            if val_pairs is None or (X_s.path, Y_s.path) not in val_pairs:
                return X_s, Y_s

    def sample_image_pair_batch(self, batch_size: int = 1, val_pairs=None):
        X_samps, Y_samps = [], []
        for _ in range(batch_size):
            xs, ys = self.sample_image_pair(val_pairs)
            X_samps.append(xs)
            Y_samps.append(ys)

        X_sq   = np.stack([s.image_square for s in X_samps])   # (B, 3, 224, 224)
        Y_sq   = np.stack([s.image_square for s in Y_samps])
        X_full = [s.flat_pixels for s in X_samps]              # list of (Ni, 3)
        Y_full = [s.flat_pixels for s in Y_samps]
        return X_samps, Y_samps, X_sq, Y_sq, X_full, Y_full


def get_image_paths(image_dir: str) -> list:
    exts  = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(paths)


def load_val_pairs(pairs_file: str, data_dir: str) -> list:
    """Load validation pairs from pairs.txt (same format as JAX)."""
    if not os.path.exists(pairs_file):
        return []
    pairs = []
    with open(pairs_file, "rb") as f:
        for line in f.readlines():
            line = line.decode().strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            im1 = os.path.join(data_dir, parts[0].strip() + ".jpg")
            im2 = os.path.join(data_dir, parts[1].strip() + ".jpg")
            if os.path.exists(im1) and os.path.exists(im2):
                pairs.append((im1, im2))
    return pairs
