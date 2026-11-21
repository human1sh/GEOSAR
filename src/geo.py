"""Georeferencing helpers. Pure numpy/pandas, so they can be unit-tested without rasterio.

Transforms are 6-tuples (a, b, c, d, e, f), the same order as rasterio's Affine:
    x = a*col + b*row + c
    y = d*col + e*row + f
Obtain one from an open raster with ``tuple(src.transform)[:6]``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TP_COLS = ("optical_col", "optical_row", "sar_col", "sar_row")


def px_to_world(t, cols, rows):
    a, b, c, d, e, f = t
    cols = np.asarray(cols, dtype=float)
    rows = np.asarray(rows, dtype=float)
    return a * cols + b * rows + c, d * cols + e * rows + f


def world_to_px(t, xs, ys):
    a, b, c, d, e, f = t
    det = a * e - b * d
    if abs(det) < 1e-12:
        raise ValueError("Degenerate affine transform")
    xs = np.asarray(xs, dtype=float) - c
    ys = np.asarray(ys, dtype=float) - f
    return (e * xs - b * ys) / det, (-d * xs + a * ys) / det


def grid_transform(center_xy, size_px, gsd):
    """North-up transform of a size_px x size_px window centred on center_xy (map units, metres)."""
    cx, cy = center_xy
    half = size_px * gsd / 2.0
    return (gsd, 0.0, cx - half, 0.0, -gsd, cy + half)


def utm_epsg(lon, lat):
    """EPSG code of the UTM zone containing (lon, lat)."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def load_tiepoints(src):
    """Read a tie-point CSV (or copy a DataFrame) and check that the expected columns exist."""
    df = src.copy() if isinstance(src, pd.DataFrame) else pd.read_csv(src)
    missing = [c for c in TP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Tie-point data is missing columns {missing}; found {list(df.columns)}")
    df = df.dropna(subset=list(TP_COLS)).reset_index(drop=True)
    if df.empty:
        raise ValueError("Tie-point data has no valid rows")
    return df


def tiepoints_to_world(tp, opt_transform, sar_transform, opt_to_work=None, sar_to_work=None):
    """Convert each image's own pixel coordinates to common map coordinates (metres).

    ``opt_to_work`` / ``sar_to_work`` are optional callables (xs, ys) -> (xs, ys) that
    reproject from each raster's CRS to the working CRS.
    """
    ox, oy = px_to_world(opt_transform, tp["optical_col"], tp["optical_row"])
    sx, sy = px_to_world(sar_transform, tp["sar_col"], tp["sar_row"])
    if opt_to_work is not None:
        ox, oy = opt_to_work(ox, oy)
    if sar_to_work is not None:
        sx, sy = sar_to_work(sx, sy)
    return np.column_stack([ox, oy]), np.column_stack([sx, sy])


def raw_displacement(opt_xy, sar_xy):
    """Displacement (metres) that moves each SAR tie-point onto its optical partner, and its length."""
    d = np.asarray(opt_xy) - np.asarray(sar_xy)
    return d, np.hypot(d[:, 0], d[:, 1])
