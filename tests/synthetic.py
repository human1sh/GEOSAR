"""Synthetic optical/SAR-like pair with a known displacement, for unit tests and demos."""
import cv2
import numpy as np


def make_pair(size=1024, shift=(23.0, -17.0), seed=0, looks=4):
    rng = np.random.default_rng(seed)
    base = np.full((size, size), 90, np.float32)
    for _ in range(150):
        x0, y0 = rng.integers(0, size - 90, 2)
        w, h = rng.integers(12, 90, 2)
        cv2.rectangle(base, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), float(rng.integers(30, 230)), -1)
    for _ in range(50):
        p = rng.integers(0, size, 4)
        cv2.line(base, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), float(rng.integers(30, 230)),
                 int(rng.integers(2, 5)))
    base = cv2.GaussianBlur(base, (0, 0), 1.2)
    optical = np.stack([base, base * 0.95, base * 0.9])            # (3,H,W)
    tx, ty = shift
    shifted = cv2.warpAffine(base, np.float32([[1, 0, tx], [0, 1, ty]]), (size, size),
                             borderMode=cv2.BORDER_REPLICATE)
    amp = np.sqrt(np.clip(255.0 - shifted, 1, 255))                 # contrast reversal, non-linear
    speckle = np.sqrt(rng.gamma(looks, 1.0 / looks, amp.shape)).astype(np.float32)
    sar = (amp * speckle).astype(np.float32)
    return optical.astype(np.float32), sar, (tx, ty)
