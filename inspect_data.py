"""Run this first and paste the output back if anything looks wrong:

    python inspect_data.py --data-dir ./data --split train

It prints how files were paired, each raster's georeferencing, and the raw tie-point
misregistration in metres under the pixel-coordinate convention the pipeline assumes.
A median between roughly 5 and 100 m is plausible for SpaceNet 9. Thousands of metres,
or wildly different numbers for (col,row) vs (row,col), mean the convention is wrong.
"""
import argparse

import numpy as np
import pandas as pd

from src.dataset import ScenePair, discover_scenes
from src.geo import load_tiepoints, raw_displacement, tiepoints_to_world

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="./data")
ap.add_argument("--split", default="train")
a = ap.parse_args()

for sc in discover_scenes(a.data_dir, a.split):
    print("=" * 78)
    print(f"scene {sc.scene_id}\n  optical : {sc.optical}\n  sar     : {sc.sar}\n  tiepts  : {sc.tiepoints}")
    with ScenePair(sc) as p:
        for name, src in (("optical", p.opt), ("sar", p.sar)):
            print(f"  [{name}] size={src.width}x{src.height} bands={src.count} dtype={src.dtypes[0]} "
                  f"nodata={src.nodata}\n           crs={src.crs} res={src.res} bounds={tuple(round(v, 1) for v in src.bounds)}")
        print(f"  working CRS: {p.work_crs}")
        tp = load_tiepoints(sc.tiepoints)
        print(f"  tie-points: {len(tp)} | columns: {list(tp.columns)}")
        print(tp.head(3).to_string(index=False))
        for label, df in (("(col,row) as named", tp),
                          ("swapped (row,col)", tp.rename(columns={"optical_col": "optical_row", "optical_row": "optical_col",
                                                                    "sar_col": "sar_row", "sar_row": "sar_col"}))):
            o, s = tiepoints_to_world(df, p.opt_transform, p.sar_transform, p.opt_to_work, p.sar_to_work)
            _, mag = raw_displacement(o, s)
            print(f"  raw misregistration, {label:20s}: median {np.median(mag):9.1f} m  p90 {np.percentile(mag, 90):9.1f} m")
        naive = np.hypot(tp.optical_col - tp.sar_col, tp.optical_row - tp.sar_row)
        print(f"  (for comparison: naive pixel-index difference, which the old code used: median {naive.median():.0f} px)")
