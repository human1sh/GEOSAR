"""GeoPackage export of tie-point displacement vectors in real map coordinates."""
from __future__ import annotations

import pandas as pd


def export_displacements(df: pd.DataFrame, crs, out_path):
    """Write one LineString per tie-point from its SAR position to its optical position.

    ``df`` needs: opt_x, opt_y, sar_x, sar_y (map metres in ``crs``) plus any attribute columns.
    The line therefore points along the shift that would move the SAR feature onto the optical one.
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    if df.empty:
        raise ValueError("No displacement rows to export")
    if crs is None:
        raise ValueError("A CRS is required so QGIS places the vectors correctly")
    geoms = [LineString([(r.sar_x, r.sar_y), (r.opt_x, r.opt_y)]) for r in df.itertuples()]
    attrs = df.drop(columns=["opt_x", "opt_y", "sar_x", "sar_y"]).copy()
    attrs["dx_m"] = (df["opt_x"] - df["sar_x"]).values
    attrs["dy_m"] = (df["opt_y"] - df["sar_y"]).values
    gdf = gpd.GeoDataFrame(attrs, geometry=geoms, crs=crs)
    gdf.to_file(out_path, layer="displacements", driver="GPKG")
    return out_path
