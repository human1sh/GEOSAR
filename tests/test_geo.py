import numpy as np
import pandas as pd
import pytest

from src.geo import (grid_transform, load_tiepoints, px_to_world, raw_displacement,
                     tiepoints_to_world, utm_epsg, world_to_px)


def test_roundtrip():
    t = (0.3, 0.0, 500000.0, 0.0, -0.3, 4000000.0)
    x, y = px_to_world(t, [10, 2000.5], [20, 3000.25])
    c, r = world_to_px(t, x, y)
    assert np.allclose(c, [10, 2000.5]) and np.allclose(r, [20, 3000.25])


def test_different_grids_agree_in_world_space():
    opt_t = (0.3, 0.0, 500000.0, 0.0, -0.3, 4000000.0)
    sar_t = (0.4, 0.0, 499900.0, 0.0, -0.4, 4000100.0)
    wx, wy = 500123.0, 3999876.0                       # one ground point, seen by both rasters
    oc, orow = world_to_px(opt_t, wx, wy)
    sc, srow = world_to_px(sar_t, wx + 12.0, wy - 5.0)  # SAR sees it displaced by (+12, -5) m
    tp = pd.DataFrame({"optical_col": [oc], "optical_row": [orow], "sar_col": [sc], "sar_row": [srow]})
    opt_xy, sar_xy = tiepoints_to_world(tp, opt_t, sar_t)
    d, mag = raw_displacement(opt_xy, sar_xy)
    assert np.allclose(d[0], [-12.0, 5.0]) and np.isclose(mag[0], 13.0)


def test_pixel_index_difference_is_not_displacement():
    """Same pixel indices on two grids with different origin and GSD are NOT the same place."""
    opt_t = (0.3, 0.0, 500000.0, 0.0, -0.3, 4000000.0)
    sar_t = (0.4, 0.0, 499850.0, 0.0, -0.4, 4000100.0)
    tp = pd.DataFrame({"optical_col": [1000.0], "optical_row": [1000.0], "sar_col": [1000.0], "sar_row": [1000.0]})
    _, mag = raw_displacement(*tiepoints_to_world(tp, opt_t, sar_t))
    assert mag[0] > 40.0


def test_grid_transform_centres_point():
    t = grid_transform((1000.0, 2000.0), 512, 0.6)
    c, r = world_to_px(t, 1000.0, 2000.0)
    assert np.isclose(c, 256.0) and np.isclose(r, 256.0)


def test_missing_columns_raises():
    with pytest.raises(ValueError):
        load_tiepoints(pd.DataFrame({"a": [1]}))


def test_utm():
    assert utm_epsg(77.6, 12.9) == 32643
    assert utm_epsg(-70.0, -33.0) == 32719
