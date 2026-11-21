"""Create the two README figures from existing results (no recomputation of registration).

    python -m src.make_figures --results outputs/results --out results
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="outputs/results")
ap.add_argument("--out", default="results")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

df = pd.read_csv(os.path.join(a.results, "per_tiepoint.csv"))
labels = {"identity": "no correction", "ncc_grad": "masked NCC (gradient)", "oriented_ncc": "masked NCC (oriented)"}
fig, ax = plt.subplots(figsize=(6, 4))
for m, lab in labels.items():
    e = np.sort(df[df.method == m].after_m.values)
    ax.plot(e, np.arange(1, len(e) + 1) / len(e), label=f"{lab} (median {np.median(e):.1f} m)")
ax.set_xlabel("tie-point error (m)")
ax.set_ylabel("fraction of tie-points")
ax.set_xlim(0, 80)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8)
ax.set_title("Tie-point error, cumulative distribution")
fig.tight_layout()
fig.savefig(os.path.join(a.out, "error_cdf.png"), dpi=150)
plt.close(fig)

ex = os.path.join(a.results, "examples.npz")
if os.path.exists(ex):
    z = np.load(ex)
    k = sorted({n.split("__")[0] for n in z.files})[0]
    opt, sar, warped = z[f"{k}__opt"], z[f"{k}__sar"], z[f"{k}__sar_warped"]
    comp = lambda s: np.dstack([opt, s, s])          # optical red, SAR cyan: colour fringes = misalignment
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax, im, t in ((axs[0], comp(sar), "before"), (axs[1], comp(warped), "after oriented-NCC shift")):
        ax.imshow(im)
        ax.set_title(f"{k}: {t}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "overlay_before_after.png"), dpi=150)
    plt.close(fig)
print("wrote figures to", a.out)
