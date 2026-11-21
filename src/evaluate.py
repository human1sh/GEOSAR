"""Run the registration methods over SN9 tie-points and write honest, reproducible results.

    python -m src.evaluate --data-dir ./data --split train --scenes 02_02 --out outputs/dev      # development scene
    python -m src.evaluate --data-dir ./data --split train --scenes 02_01,03_01 --out outputs/results   # report these

Protocol: any choice of preprocessing, resolution or thresholds is made on the development scene only;
the final numbers come from the other scenes. For each tie-point a patch centred on it is read onto a
common grid, each method estimates a SAR->optical transform from image content alone (tie-points are
never used for estimation), and the residual at that tie-point is measured in metres. If a method
rejects its own estimate, the error falls back to the unregistered error and the rejection is counted.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd

from .dataset import ScenePair, discover_scenes
from .evaluation import residual_m, summarize
from .geo import grid_transform, load_tiepoints, raw_displacement, tiepoints_to_world, world_to_px
from .gis_export import export_displacements
from .registration import (CrossModalRegistrar, mask_optical, mask_sar, prep_optical, prep_sar,
                           representation, warp_sar)

# NCC methods are run ungated (min_peak=0) so their true error distribution is visible; the NCC peak
# and PSR are saved so a confidence gate can be analysed afterwards (on the development scene only).
METHODS = {
    "identity": dict(method="identity"),
    "ncc_grad": dict(method="ncc"),
    "oriented_ncc": dict(method="oriented_ncc"),
    "sift_grad_sim": dict(method="sift", rep="gradient", model="similarity"),
}


def run(data_dir, split, gsd, patch, max_points, out_dir, sar_log="auto", max_shift_m=150.0,
        debug_patches=3, scenes=None, sigma=2.0):
    os.makedirs(os.path.join(out_dir, "debug"), exist_ok=True)
    erode_px = int(np.ceil(5 * sigma))               # keeps margin edges out of the gradient maps
    registrars = {k: CrossModalRegistrar(max_shift_px=max_shift_m / gsd, sigma=sigma, **v)
                  for k, v in METHODS.items()}
    rows, vec_rows, examples, work_crs, n_debug = [], [], {}, None, 0
    t0 = time.time()
    for scene in discover_scenes(data_dir, split):
        if scenes and scene.scene_id not in scenes:
            continue
        tp = load_tiepoints(scene.tiepoints)
        with ScenePair(scene) as pair:
            work_crs = pair.work_crs
            opt_xy, sar_xy = tiepoints_to_world(tp, pair.opt_transform, pair.sar_transform,
                                                pair.opt_to_work, pair.sar_to_work)
            _, before = raw_displacement(opt_xy, sar_xy)
            # uint8 SAR is already display-scaled, so a further dB transform would distort it.
            use_log = (pair.sar.dtypes[0] != "uint8") if sar_log == "auto" else (sar_log == "yes")
            print(f"[{scene.scene_id}] SAR dtype {pair.sar.dtypes[0]}: dB scaling {'on' if use_log else 'off'}")
            print(f"[{scene.scene_id}] {len(tp)} tie-points | raw misregistration: "
                  f"median {np.median(before):.1f} m, p90 {np.percentile(before, 90):.1f} m")
            if not (2.0 <= np.median(before) <= 200.0):
                print("   WARNING: median misregistration outside 2-200 m. The tie-point coordinate "
                      "convention or the file pairing may be wrong; run inspect_data.py.")
            idx = np.arange(len(tp))
            if max_points and max_points < len(idx):
                idx = np.random.default_rng(0).choice(idx, max_points, replace=False)
            for i in idx:
                t6 = grid_transform(opt_xy[i], patch, gsd)
                opt_px = np.column_stack(world_to_px(t6, *opt_xy[i:i + 1].T))
                sar_px = np.column_stack(world_to_px(t6, *sar_xy[i:i + 1].T))
                sar_raw = pair.read_sar(t6, patch)
                opt_rgb = pair.read_optical(t6, patch)
                m_opt, m_sar = mask_optical(opt_rgb, erode_px), mask_sar(sar_raw, erode_px)
                if m_opt.mean() < 0.3 or m_sar.mean() < 0.3 or (m_opt & m_sar).mean() < 0.25:
                    continue                                    # not enough overlapping valid data
                opt_u8 = prep_optical(opt_rgb)
                sar_u8 = prep_sar(sar_raw, log_scale=use_log)
                if n_debug < debug_patches:      # look at these images before trusting any number
                    tag = os.path.join(out_dir, "debug", f"{scene.scene_id}_{i}")
                    for nm, im in (("opt", opt_u8), ("sar", sar_u8),
                                   ("opt_grad", representation(opt_u8, "gradient", sigma, m_opt)),
                                   ("sar_grad", representation(sar_u8, "gradient", sigma, m_sar)),
                                   ("mask_opt", m_opt.astype(np.uint8) * 255),
                                   ("mask_sar", m_sar.astype(np.uint8) * 255)):
                        cv2.imwrite(f"{tag}_{nm}.png", im)
                    n_debug += 1
                for name, reg in registrars.items():
                    res = reg.register(opt_u8, sar_u8, m_opt, m_sar)
                    after = float(residual_m(res.M, sar_px, opt_px, gsd)[0])
                    rows.append(dict(scene=scene.scene_id, tp_index=int(i), method=name, success=res.success,
                                     n_matches=res.n_matches, n_inliers=res.n_inliers,
                                     ncc=res.extra.get("ncc", np.nan), psr=res.extra.get("psr", np.nan),
                                     before_m=float(before[i]), after_m=after, note=res.note))
                    if name == "oriented_ncc" and res.success and len(examples) < 3:
                        examples[f"{scene.scene_id}_{i}"] = (opt_u8, sar_u8, warp_sar(sar_u8, res.M, opt_u8.shape))
                vec_rows.append(dict(scene=scene.scene_id, tp_index=int(i), opt_x=opt_xy[i, 0], opt_y=opt_xy[i, 1],
                                     sar_x=sar_xy[i, 0], sar_y=sar_xy[i, 1], before_m=float(before[i])))
    if not rows:
        raise RuntimeError("No tie-points were evaluated (no matching scenes, or coverage too low)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "per_tiepoint.csv"), index=False)
    summary = summarize(df)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    vec = pd.DataFrame(vec_rows)
    best = df[df.method == "oriented_ncc"][["scene", "tp_index", "after_m", "success"]].rename(
        columns={"after_m": "after_m_oriented_ncc", "success": "oriented_ncc_success"})
    export_displacements(vec.merge(best, on=["scene", "tp_index"], how="left"), work_crs.to_string(),
                         os.path.join(out_dir, "displacements.gpkg"))
    np.savez_compressed(os.path.join(out_dir, "examples.npz"),
                        **{f"{k}__{n}": a for k, v in examples.items() for n, a in zip(("opt", "sar", "sar_warped"), v)})
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(dict(gsd_m=gsd, patch_px=patch, sigma_px=sigma, max_shift_m=max_shift_m, sar_log=str(sar_log),
                       scenes=sorted(df.scene.unique().tolist()), seconds=round(time.time() - t0, 1)), f, indent=2)
    print(summary.round(2).to_string(index=False))
    fails = df[(~df.success) & (df.method != "identity")]
    if len(fails):
        print("\nWhy methods failed (counts):")
        print(fails.assign(note=fails.note.str.replace(r"[-\d.]+", "#", regex=True))
                   .groupby(["method", "note"]).size().to_string())
    g = df[df.method == "sift_grad_sim"]
    print(f"sift_grad_sim: matches median {g.n_matches.median():.0f}, inliers median {g.n_inliers.median():.0f}")
    for m in ("ncc_grad", "oriented_ncc"):
        g = df[df.method == m]
        print(f"{m}: NCC peak median {g.ncc.median():.3f}, p90 {g.ncc.quantile(0.9):.3f}; PSR median {g.psr.median():.1f}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--scenes", default="", help="comma-separated scene ids, e.g. 02_02 (default: all)")
    ap.add_argument("--gsd", type=float, default=1.2, help="working resolution in metres")
    ap.add_argument("--patch", type=int, default=512, help="patch size in pixels at the working resolution")
    ap.add_argument("--sigma", type=float, default=2.0, help="gradient smoothing in working pixels")
    ap.add_argument("--max-points", type=int, default=0, help="subsample tie-points per scene (0 = all)")
    ap.add_argument("--max-shift-m", type=float, default=150.0)
    ap.add_argument("--sar-log", choices=["auto", "yes", "no"], default="auto",
                    help="dB-scale the SAR amplitude; auto = only when the GeoTIFF is not uint8")
    ap.add_argument("--debug-patches", type=int, default=3, help="save this many preprocessed patches as PNG")
    ap.add_argument("--out", default="outputs/results")
    a = ap.parse_args()
    run(a.data_dir, a.split, a.gsd, a.patch, a.max_points, a.out, a.sar_log, a.max_shift_m, a.debug_patches,
        [s for s in a.scenes.split(",") if s] or None, a.sigma)
