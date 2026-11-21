"""Cross-modal (optical <-> SAR) registration of image patches on a common grid.

Both patches must already be resampled onto the same north-up grid (see dataset.py), so the
identity transform means "trust the raw georeferencing". The estimated 2x3 matrix M maps SAR
patch pixel coordinates to optical patch pixel coordinates.

Methods:
    identity       no correction (baseline)
    ncc            translation by masked normalised cross-correlation of gradient-magnitude maps
    oriented_ncc   same, on oriented-gradient channels (invariant to SAR/optical contrast reversal)
    sift           SIFT + ratio test + RANSAC similarity/affine, with sanity limits

Each image has its own validity mask: zero-padded raster margins create strong artificial edges
that otherwise dominate gradient-based matching. Masked NCC (Padfield 2012) uses only pixels that
are valid in BOTH images at each trial shift, so the mask boundary itself cannot create a
common signal. Tie-points are never used for estimation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


@dataclass
class RegResult:
    M: np.ndarray
    success: bool
    method: str
    n_matches: int = 0
    n_inliers: int = 0
    note: str = ""
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- preprocessing
def to_u8(img, p_lo=2.0, p_hi=98.0, mask=None):
    """Percentile-stretch any numeric array (NaN allowed) to uint8; percentiles use only ``mask``."""
    img = np.asarray(img, dtype=np.float32)
    valid = np.isfinite(img) if mask is None else (np.isfinite(img) & mask)
    if valid.sum() < 16:
        return np.zeros(img.shape, np.uint8)
    lo, hi = np.percentile(img[valid], [p_lo, p_hi])
    if hi - lo < 1e-9:
        return np.zeros(img.shape, np.uint8)
    out = np.clip((np.nan_to_num(img, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def prep_optical(rgb):
    """(3,H,W) or (H,W) optical patch -> uint8 luminance."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim == 3:
        rgb = rgb[:3]
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] if rgb.shape[0] == 3 else rgb[0]
    else:
        lum = rgb
    return to_u8(lum)


def prep_sar(amplitude, log_scale=True, median=3):
    """SAR amplitude patch -> uint8. Use log_scale=False when the GeoTIFF is already uint8/display-scaled."""
    x = np.asarray(amplitude, dtype=np.float32)
    if log_scale:
        with np.errstate(divide="ignore", invalid="ignore"):
            x = 20.0 * np.log10(np.where(x > 0, x, np.nan))
    u8 = to_u8(x)
    if median and median >= 3:
        u8 = cv2.medianBlur(u8, int(median))
    return u8


def _clean_mask(valid, erode_px):
    m = valid.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))      # fill isolated dark pixels
    m = cv2.erode(m, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    return m.astype(bool)


def mask_optical(rgb, erode_px=12):
    """True where the optical patch has real data. Margins are zero-padded (nodata is not flagged in
    the SN9 GeoTIFFs). erode_px should exceed ~4*sigma so margin edges cannot leak into gradients."""
    rgb = np.asarray(rgb, dtype=np.float32)
    valid = np.isfinite(rgb).all(axis=0) & (np.nan_to_num(rgb).sum(axis=0) > 0)
    return _clean_mask(valid, erode_px)


def mask_sar(sar, erode_px=12):
    sar = np.asarray(sar, dtype=np.float32)
    return _clean_mask(np.isfinite(sar) & (np.nan_to_num(sar) > 0), erode_px)


def _grad(u8, sigma):
    g = cv2.GaussianBlur(u8.astype(np.float32), (0, 0), sigma)
    return cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)


def _norm_mag(mag, mask):
    vals = mag[mask] if mask is not None and mask.any() else mag
    return mag / (np.percentile(vals, 99.5) + 1e-9)


def gradient_float(u8, sigma=2.0, mask=None):
    gx, gy = _grad(u8, sigma)
    mag = cv2.magnitude(gx, gy)
    if mask is not None:
        mag = mag * mask
    return _norm_mag(mag, mask).astype(np.float32)


def gradient_u8(u8, sigma=2.0, mask=None):
    return (np.clip(gradient_float(u8, sigma, mask), 0, 1) * 255).astype(np.uint8)


def oriented_channels(u8, sigma=2.0, mask=None, bins=8):
    """Unsigned oriented-gradient channels (K,H,W): invariant to contrast reversal, which is
    the main radiometric difference between SAR and optical."""
    gx, gy = _grad(u8, sigma)
    mag = cv2.magnitude(gx, gy)
    theta = np.arctan2(gy, gx)
    if mask is not None:
        mag = mag * mask
    mag = _norm_mag(mag, mask)
    return np.stack([mag * np.cos(theta - k * np.pi / bins) ** 2 for k in range(bins)]).astype(np.float32)


def representation(u8, kind="gradient", sigma=2.0, mask=None):
    """Single-channel image used for feature detection / visual debugging."""
    if kind == "gradient":
        return gradient_u8(u8, sigma, mask)
    if kind == "intensity":
        return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(u8)
    raise ValueError(f"Unknown representation {kind!r}")


# ----------------------------------------------------------------------------- masked NCC
def masked_ncc_shift(A, B, mA, mB, max_shift_px, min_overlap_frac=0.3):
    """Shift s such that B(x) ~ A(x - s), by masked normalised cross-correlation over K channels.

    A, B: (K,H,W) float32; mA, mB: bool validity masks of the two images.
    The score at each trial shift uses only pixels valid in both images (Padfield 2012), so
    zero-padded margins cannot create a spurious zero-shift peak. Shifts with too little overlap
    are ignored, and the search is limited to +-max_shift_px.
    Returns (sx, sy, peak_ncc, psr).
    """
    K, H, W = A.shape
    PH = int(2 ** np.ceil(np.log2(2 * H - 1)))
    PW = int(2 ** np.ceil(np.log2(2 * W - 1)))
    fa = mA.astype(np.float32)
    fb = mB.astype(np.float32)
    A = A * fa
    B = B * fb
    R = int(min(max_shift_px, H // 2 - 2, W // 2 - 2))
    ys = np.arange(-R, R + 1) % PH
    xs = np.arange(-R, R + 1) % PW
    ix = np.ix_(ys, xs)
    rf = lambda x: np.fft.rfft2(x, s=(PH, PW))
    xc = lambda Fa, Fb: np.fft.irfft2(np.conj(Fa) * Fb, s=(PH, PW))[ix]   # sum_x a(x) b(x+s)

    FmA, FmB = rf(fa), rf(fb)
    overlap = np.maximum(xc(FmA, FmB), 1e-6)
    s_ab = s_a2 = s_b2 = 0.0
    corr_ab = 0.0
    sum_a2_fft = sum_b2_fft = sum_ab_fft = 0.0
    for k in range(K):
        Fa, Fb = rf(A[k]), rf(B[k])
        sum_ab_fft = sum_ab_fft + np.conj(Fa) * Fb
        sum_a2_fft = sum_a2_fft + np.conj(rf(A[k] ** 2))
        sum_b2_fft = sum_b2_fft + rf(B[k] ** 2)
        a_mb = xc(Fa, FmB)                    # sum A_k(x) mB(x+s)
        ma_b = xc(FmA, Fb)                    # sum mA(x) B_k(x+s)
        corr_ab = corr_ab + a_mb * ma_b / overlap
        s_a2 = s_a2 + a_mb ** 2 / overlap
        s_b2 = s_b2 + ma_b ** 2 / overlap
    num = np.fft.irfft2(sum_ab_fft, s=(PH, PW))[ix] - corr_ab
    den_a = np.fft.irfft2(sum_a2_fft * FmB, s=(PH, PW))[ix] - s_a2
    den_b = np.fft.irfft2(np.conj(FmA) * sum_b2_fft, s=(PH, PW))[ix] - s_b2
    ncc = num / np.sqrt(np.maximum(den_a, 0) * np.maximum(den_b, 0) + 1e-9)
    ncc = np.clip(ncc, -1.0, 1.0)
    ncc[overlap < min_overlap_frac * min(fa.sum(), fb.sum())] = -1.0
    py, px = np.unravel_index(np.argmax(ncc), ncc.shape)
    peak = float(ncc[py, px])

    def refine(m1, m0, p1):
        d = m1 - 2 * m0 + p1
        return 0.0 if abs(d) < 1e-12 else float(np.clip(0.5 * (m1 - p1) / d, -0.5, 0.5))

    dx = refine(ncc[py, px - 1], ncc[py, px], ncc[py, px + 1]) if 0 < px < ncc.shape[1] - 1 else 0.0
    dy = refine(ncc[py - 1, px], ncc[py, px], ncc[py + 1, px]) if 0 < py < ncc.shape[0] - 1 else 0.0
    side = np.where(overlap >= min_overlap_frac * min(fa.sum(), fb.sum()), ncc, np.nan)
    side[max(0, py - 5):py + 6, max(0, px - 5):px + 6] = np.nan
    psr = float((peak - np.nanmean(side)) / (np.nanstd(side) + 1e-12))
    return float(px - R + dx), float(py - R + dy), peak, psr


# ----------------------------------------------------------------------------- registrar
class CrossModalRegistrar:
    def __init__(self, method="sift", rep="gradient", model="similarity", sigma=2.0, bins=8, ratio=0.8,
                 ransac_px=3.0, min_inliers=10, max_shift_px=125.0, scale_range=(0.9, 1.1),
                 max_rot_deg=5.0, min_peak=0.0, min_overlap_frac=0.3):
        if method not in ("identity", "ncc", "oriented_ncc", "sift"):
            raise ValueError(f"Unknown method {method!r}")
        if model not in ("similarity", "affine"):
            raise ValueError(f"Unknown model {model!r}")
        self.method, self.rep, self.model, self.sigma, self.bins = method, rep, model, sigma, bins
        self.ratio, self.ransac_px, self.min_inliers = ratio, ransac_px, min_inliers
        self.max_shift_px, self.scale_range = max_shift_px, scale_range
        self.max_rot_deg, self.min_peak, self.min_overlap_frac = max_rot_deg, min_peak, min_overlap_frac
        cv2.setRNGSeed(0)

    def register(self, opt_u8, sar_u8, mask_opt=None, mask_sar=None):
        mask_opt = np.ones(opt_u8.shape, bool) if mask_opt is None else mask_opt
        mask_sar = np.ones(sar_u8.shape, bool) if mask_sar is None else mask_sar
        if self.method == "identity":
            return RegResult(IDENTITY.copy(), True, "identity")
        if self.method in ("ncc", "oriented_ncc"):
            return self._ncc(opt_u8, sar_u8, mask_opt, mask_sar)
        return self._sift(opt_u8, sar_u8, mask_opt, mask_sar)

    def _ncc(self, opt_u8, sar_u8, mask_opt, mask_sar):
        if self.method == "oriented_ncc":
            A = oriented_channels(opt_u8, self.sigma, mask_opt, self.bins)
            B = oriented_channels(sar_u8, self.sigma, mask_sar, self.bins)
        else:
            A = gradient_float(opt_u8, self.sigma, mask_opt)[None]
            B = gradient_float(sar_u8, self.sigma, mask_sar)[None]
        sx, sy, peak, psr = masked_ncc_shift(A, B, mask_opt, mask_sar, self.max_shift_px, self.min_overlap_frac)
        # (sx, sy) is the shift of the SAR content relative to the optical content
        # (SAR sits at optical position + s), so the SAR -> optical map is a translation by -s.
        M = np.array([[1.0, 0.0, -sx], [0.0, 1.0, -sy]])
        ok = peak >= self.min_peak
        note = "" if ok else f"rejected (ncc={peak:.2f})"
        return RegResult(M if ok else IDENTITY.copy(), bool(ok), self.method, note=note,
                         extra={"ncc": peak, "psr": psr, "shift_px": float(np.hypot(sx, sy))})

    def _sift(self, opt_u8, sar_u8, mask_opt, mask_sar):
        a = representation(opt_u8, self.rep, self.sigma, mask_opt)
        b = representation(sar_u8, self.rep, self.sigma, mask_sar)
        sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
        kp_o, des_o = sift.detectAndCompute(a, mask_opt.astype(np.uint8) * 255)
        kp_s, des_s = sift.detectAndCompute(b, mask_sar.astype(np.uint8) * 255)
        if des_o is None or des_s is None or len(kp_o) < 4 or len(kp_s) < 4:
            return RegResult(IDENTITY.copy(), False, "sift", note="too few keypoints")
        knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_s, des_o, k=2)
        good = [p[0] for p in knn if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]
        if len(good) < 4:
            return RegResult(IDENTITY.copy(), False, "sift", len(good), note="too few matches")
        src = np.float32([kp_s[m.queryIdx].pt for m in good])
        dst = np.float32([kp_o[m.trainIdx].pt for m in good])
        est = cv2.estimateAffinePartial2D if self.model == "similarity" else cv2.estimateAffine2D
        M, inl = est(src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_px,
                     maxIters=5000, confidence=0.995)
        if M is None:
            return RegResult(IDENTITY.copy(), False, "sift", len(good), note="RANSAC failed")
        n_in = int(inl.sum())
        problem = self._sanity(M, a.shape, n_in)
        if problem:
            return RegResult(IDENTITY.copy(), False, "sift", len(good), n_in, note=problem)
        return RegResult(M.astype(np.float64), True, "sift", len(good), n_in)

    def _sanity(self, M, shape, n_in):
        if n_in < self.min_inliers:
            return f"only {n_in} inliers"
        A = M[:, :2]
        scale = float(np.sqrt(abs(np.linalg.det(A))))
        rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
        cx, cy = shape[1] / 2.0, shape[0] / 2.0
        shift = float(np.hypot(*(M @ np.array([cx, cy, 1.0]) - np.array([cx, cy]))))
        if not (self.scale_range[0] <= scale <= self.scale_range[1]):
            return f"implausible scale {scale:.3f}"
        if abs(rot) > self.max_rot_deg:
            return f"implausible rotation {rot:.1f} deg"
        if shift > self.max_shift_px:
            return f"implausible shift {shift:.0f}px"
        return ""


def warp_sar(sar_u8, M, shape):
    """Warp a SAR patch into the optical frame using the SAR->optical matrix M."""
    return cv2.warpAffine(sar_u8, M.astype(np.float32), (shape[1], shape[0]))
