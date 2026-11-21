"""Print the numbers for the README straight from per_tiepoint.csv, so nothing is typed by hand.

    python -m src.report outputs/results/per_tiepoint.csv
"""
import sys

import pandas as pd

from .evaluation import summarize

df = pd.read_csv(sys.argv[1])
s = summarize(df)
names = {"identity": "no correction", "ncc_grad": "masked NCC (gradient)",
         "oriented_ncc": "masked NCC (oriented channels)", "sift_grad_sim": "SIFT + RANSAC"}
print(f"Tie-points evaluated: {df.groupby('method').size().iloc[0]} across scenes {sorted(df.scene.unique())}\n")
print("| Method | Median error (m) | Mean (m) | P90 (m) | Improved | Worsened | Method succeeded |")
print("|---|---|---|---|---|---|---|")
for _, r in s.iterrows():
    print(f"| {names.get(r.method, r.method)} | {r.median_after_m:.1f} | {r.mean_after_m:.1f} | {r.p90_after_m:.1f} "
          f"| {r.share_improved:.0%} | {r.share_worsened:.0%} | {r.success_rate:.0%} |")
print("\nPer scene (oriented NCC vs no correction), median error in metres:")
for sc, g in df.groupby("scene"):
    a = g[g.method == "oriented_ncc"]
    print(f"  {sc}: n={len(a)}, before {a.before_m.median():.1f} -> after {a.after_m.median():.1f}, "
          f"improved {(a.after_m < a.before_m - 0.05).mean():.0%}")
o = df[df.method == "oriented_ncc"].copy()
o["start_error_bin"] = pd.cut(o.before_m, [0, 10, 20, 40, 1000])
print("\nBy starting error (oriented NCC):")
print(o.groupby("start_error_bin", observed=True).agg(
    n=("after_m", "size"), median_before=("before_m", "median"), median_after=("after_m", "median"),
    improved=("after_m", lambda x: float((x < o.loc[x.index, "before_m"] - 0.05).mean()))).round(2).to_string())
