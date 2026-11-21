"""SpaceNet 9 scene discovery and georeferenced patch reading.

SN9 optical and SAR rasters differ in extent, pixel size and origin, and the tie-point CSVs give
pixel coordinates *in each image's own grid*. Patches are therefore cut by map coordinates and
both modalities are resampled onto one common north-up grid, never by reusing pixel indices.

NOTE: this module needs rasterio. It is written against the documented rasterio API but has not
been run against the real SN9 files by its author; run ``python inspect_data.py`` first.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np

from .geo import utm_epsg


@dataclass
class Scene:
    scene_id: str
    optical: str
    sar: str
    tiepoints: str


def discover_scenes(data_dir, split="train"):
    root = os.path.join(data_dir, split) if os.path.isdir(os.path.join(data_dir, split)) else data_dir
    find = lambda pat: sorted(glob.glob(os.path.join(root, "**", pat), recursive=True))
    opt, sar, tps = find("*optical*.tif"), find("*sar*.tif"), find("*tiepoints*.csv")
    if not opt or not sar or not tps:
        raise FileNotFoundError(
            f"Under {root!r} found {len(opt)} optical, {len(sar)} SAR and {len(tps)} tie-point files. "
            "Check data_dir and the file-name patterns in discover_scenes().")
    if not (len(opt) == len(sar) == len(tps)):
        raise ValueError(f"Unequal file counts: optical={len(opt)}, sar={len(sar)}, tiepoints={len(tps)}. "
                         "Pair the files by hand.")
    scenes = []
    for o, s, t in zip(opt, sar, tps):
        m = re.search(r"(\d+)_tiepoints\w*?_(\d+)", os.path.basename(t))
        sid = f"{m.group(1)}_{m.group(2)}" if m else os.path.splitext(os.path.basename(t))[0]
        scenes.append(Scene(sid, o, s, t))
    return scenes


class ScenePair:
    """Open optical + SAR rasters and read patches onto a common working grid."""

    def __init__(self, scene: Scene):
        self.scene = scene

    def __enter__(self):
        import rasterio
        from rasterio.crs import CRS
        self._rio = rasterio
        self.opt = rasterio.open(self.scene.optical)
        self.sar = rasterio.open(self.scene.sar)
        if self.opt.crs is None or self.sar.crs is None:
            raise ValueError("Both rasters need a CRS (georeferencing) for this pipeline")
        if self.opt.crs.is_projected:
            self.work_crs = self.opt.crs
        else:  # geographic CRS: switch to the UTM zone of the scene centre so units are metres
            from rasterio.warp import transform
            b = self.opt.bounds
            lon, lat = (b.left + b.right) / 2, (b.bottom + b.top) / 2
            self.work_crs = CRS.from_epsg(utm_epsg(lon, lat))
        self.opt_transform = tuple(self.opt.transform)[:6]
        self.sar_transform = tuple(self.sar.transform)[:6]
        return self

    def __exit__(self, *exc):
        self.opt.close()
        self.sar.close()

    def _to_work(self, crs):
        if crs == self.work_crs:
            return None
        from rasterio.warp import transform
        return lambda xs, ys: tuple(np.asarray(v) for v in
                                    transform(crs, self.work_crs, np.atleast_1d(xs).tolist(), np.atleast_1d(ys).tolist()))

    @property
    def opt_to_work(self):
        return self._to_work(self.opt.crs)

    @property
    def sar_to_work(self):
        return self._to_work(self.sar.crs)

    def _read(self, src, band, t6, size):
        from rasterio.transform import Affine
        from rasterio.warp import Resampling, reproject
        dst = np.full((size, size), np.nan, dtype=np.float32)
        reproject(source=self._rio.band(src, band), destination=dst,
                  dst_transform=Affine(*t6), dst_crs=self.work_crs,
                  resampling=Resampling.average, src_nodata=src.nodata, dst_nodata=np.nan)
        return dst

    def read_optical(self, t6, size):
        bands = min(3, self.opt.count)
        return np.stack([self._read(self.opt, b, t6, size) for b in range(1, bands + 1)])

    def read_sar(self, t6, size):
        return self._read(self.sar, 1, t6, size)
