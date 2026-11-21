"""Compare several development runs (different patch/sigma settings) side by side.

    python -m src.compare_runs outputs/dev_p384_s2 outputs/dev_p512_s2 outputs/dev_p768_s2 ...
Choose ONE configuration on the development scene, then run the final scenes once with it.
"""
import json
import os
import sys

import pandas as pd

rows = []
for d in sys.argv[1:]:
    info = json.load(open(os.path.join(d, "run_info.json")))
    s = pd.read_csv(os.path.join(d, "summary.csv")).set_index("method")
    for m in ("ncc_grad", "oriented_ncc"):
        r = s.loc[m]
        rows.append(dict(run=os.path.basename(d.rstrip("/")), patch=info["patch_px"], sigma=info["sigma_px"],
                         gsd=info["gsd_m"], method=m, n=int(r.n_tiepoints), median_before=r.median_before_m,
                         median_after=r.median_after_m, mean_after=r.mean_after_m, p90_after=r.p90_after_m,
                         improved=r.share_improved, worsened=r.share_worsened))
print(pd.DataFrame(rows).round(2).to_string(index=False))
