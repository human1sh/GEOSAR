"""Tie-point based evaluation. Errors are in metres on the common grid."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_affine(M, pts):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    return pts @ np.asarray(M, dtype=float)[:, :2].T + np.asarray(M, dtype=float)[:, 2]


def residual_m(M, sar_px, opt_px, gsd):
    """Distance (metres) between each registered SAR tie-point and its optical partner."""
    pred = apply_affine(M, sar_px)
    return np.hypot(*(pred - np.asarray(opt_px, dtype=float).reshape(-1, 2)).T) * gsd


def summarize(df):
    """Per-method summary of a per-tie-point results table.

    Columns needed: method, before_m, after_m (identity fallback if the method failed), success.
    """
    rows = []
    for method, g in df.groupby("method"):
        ok = g[g["success"]]
        rows.append({
            "method": method,
            "n_tiepoints": len(g),
            "success_rate": float(g["success"].mean()),
            "median_before_m": float(g["before_m"].median()),
            "median_after_m": float(g["after_m"].median()),
            "mean_after_m": float(g["after_m"].mean()),
            "p90_after_m": float(g["after_m"].quantile(0.9)),
            "median_after_m_successes_only": float(ok["after_m"].median()) if len(ok) else float("nan"),
            # 5 cm tolerance: float rounding must not count as improvement
            "share_improved": float((g["after_m"] < g["before_m"] - 0.05).mean()),
            "share_worsened": float((g["after_m"] > g["before_m"] + 0.05).mean()),
        })
    return pd.DataFrame(rows)
