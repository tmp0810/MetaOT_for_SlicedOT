import numpy as np
from scipy.ndimage import rotate as ndi_rotate

ROTATION_ANGLES = [0, 15, 30, 45, 60, 75, 90]  # degrees; 0 = sanity check (no shift)


def rotate_prob(vec, img_size, angle_deg):
    if angle_deg == 0:
        return vec.copy() if isinstance(vec, np.ndarray) else np.array(vec, dtype=np.float64)
    img = np.asarray(vec, dtype=np.float64).reshape(img_size, img_size)
    img_r = ndi_rotate(img, angle_deg, reshape=False, order=1, mode="constant", cval=0.0)
    img_r = np.clip(img_r, 0.0, None)
    s = img_r.sum()
    if s > 0:
        img_r = img_r / s
    return img_r.reshape(-1)


def make_rotated_pairs(pairs, angle_deg, img_size=28):
    return [(rotate_prob(a, img_size, angle_deg), rotate_prob(b, img_size, angle_deg))
            for a, b in pairs]
