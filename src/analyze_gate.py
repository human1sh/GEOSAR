"""Does the confidence score (PSR or NCC peak) separate good registrations from bad ones?

    python -m src.analyze_gate outputs/dev/per_tiepoint.csv --method oriented_ncc
    python -m src.analyze_gate outputs/results/per_tiepoint.csv --method oriented_ncc --apply-psr 6.5

Rule fixed in advance: on the DEVELOPMENT scene choose the PSR threshold that minimises the overall
median error (rejected tie-points keep their unregistered error), keeping at least 40% of tie-points.
Then apply that threshold unchanged to the other scenes with --apply-psr. Never tune it there.
"""
import argparse

import numpy as np
import pandas as pd

TOL = 0.05   # metres; float noise must not count as improvement


def auc(score, label):
    """Probability that a randomly chosen improved case has a higher score than a worsened/unchanged one."""
    label = np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(score).rank().values
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gated(g, thr, col):
    keep = (g[col] >= thr).values
    return np.where(keep, g["after_m"].values, g["before_m"].values), keep


def report(g, thr, col, label):
    after, keep = gated(g, thr, col)
    print(f"{label}: threshold {col} >= {thr:.2f} keeps {keep.mean():.0%} | overall median {np.median(after):.2f} m, "
          f"mean {after.mean():.2f} m, p90 {np.percentile(after, 90):.2f} m "
          f"(unregistered median {g['before_m'].median():.2f} m)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--method", default="oriented_ncc")
    ap.add_argument("--col", default="psr", choices=["psr", "ncc"])
    ap.add_argument("--apply-psr", type=float, default=None, help="apply a threshold chosen on the dev scene")
    ap.add_argument("--min-keep", type=float, default=0.4)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    g = df[df.method == a.method].reset_index(drop=True)
    g["improved"] = g.after_m < g.before_m - TOL
    g["worsened"] = g.after_m > g.before_m + TOL
    print(f"{a.method}: {len(g)} tie-points; {g.improved.mean():.0%} improved, {g.worsened.mean():.0%} worsened")
    print(f"AUC of {a.col} for predicting improvement: {auc(g[a.col], g.improved):.2f} (0.5 = useless)\n")
    g["bin"] = pd.qcut(g[a.col], 4, duplicates="drop")
    print(g.groupby("bin", observed=True).agg(n=("after_m", "size"), median_before=("before_m", "median"),
                                              median_after=("after_m", "median"),
                                              share_improved=("improved", "mean"),
                                              share_worsened=("worsened", "mean")).round(2).to_string(), "\n")
    if a.apply_psr is not None:
        report(g, a.apply_psr, a.col, "applied unchanged")
        return
    best = None
    for q in np.arange(0, 0.61, 0.1):
        thr = float(g[a.col].quantile(q))
        after, keep = gated(g, thr, a.col)
        med = float(np.median(after))
        print(f"  quantile {q:.1f}: thr {thr:6.2f} keeps {keep.mean():4.0%} overall median {med:6.2f} m "
              f"| median among kept {g.after_m[keep].median():6.2f} m")
        if keep.mean() >= a.min_keep and (best is None or med < best[1] - 1e-9):
            best = (thr, med)
    print()
    report(g, best[0], a.col, "chosen on this (development) scene")
    print(f"\nNext: python -m src.analyze_gate <final per_tiepoint.csv> --method {a.method} --apply-psr {best[0]:.2f}")


if __name__ == "__main__":
    main()
