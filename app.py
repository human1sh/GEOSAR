"""Streamlit viewer for the precomputed results.  Run: streamlit run app.py
Generate the results first: python -m src.evaluate && python -m src.edge_benchmark"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

RES = os.environ.get("OPTOSAR_RESULTS", "outputs/results")
st.set_page_config(page_title="OptoSAR registration on SpaceNet 9", layout="wide")
st.title("SAR-optical registration on SpaceNet 9")
st.caption("Independent portfolio project on public data. Not affiliated with any company.")

summary_path = os.path.join(RES, "summary.csv")
if not os.path.exists(summary_path):
    st.error(f"No results in {RES}. Run `python -m src.evaluate` first.")
    st.stop()
summary = pd.read_csv(summary_path)
per = pd.read_csv(os.path.join(RES, "per_tiepoint.csv"))
tabs = st.tabs(["Registration accuracy", "Examples", "INT8 benchmark", "GIS layer"])

with tabs[0]:
    st.subheader("Tie-point error in metres (lower is better)")
    st.caption("Tie-points are used only for scoring. When a method fails, its error falls back to the "
               "unregistered error and the failure is counted in success_rate.")
    st.dataframe(summary.round(2), width="stretch")
    bins = np.array([0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 90, 1e9])
    labels = [f"{a:g}-{b:g} m" if b < 1e8 else f">{a:g} m" for a, b in zip(bins[:-1], bins[1:])]
    hist = pd.DataFrame({m: np.histogram(g["after_m"], bins=bins)[0] for m, g in per.groupby("method")}, index=labels)
    st.bar_chart(hist)

with tabs[1]:
    z = np.load(os.path.join(RES, "examples.npz")) if os.path.exists(os.path.join(RES, "examples.npz")) else None
    if z is None:
        st.info("No example patches saved.")
    else:
        for k in sorted({n.split("__")[0] for n in z.files}):
            opt, sar, warped = z[f"{k}__opt"], z[f"{k}__sar"], z[f"{k}__sar_warped"]
            comp = lambda s: np.dstack([opt, s, s])      # optical in red, SAR in cyan: fringes show misalignment
            c = st.columns(3)
            c[0].image(opt, caption=f"{k}: optical", width="stretch")
            c[1].image(comp(sar), caption="before: optical (red) + SAR (cyan)", width="stretch")
            c[2].image(comp(warped), caption="after registration (oriented NCC)", width="stretch")

with tabs[2]:
    p = os.path.join(RES, "benchmark.json")
    if os.path.exists(p):
        st.json(json.load(open(p)))
    else:
        st.info("Run `python -m src.edge_benchmark` to measure FP32 vs INT8 on this machine.")

with tabs[3]:
    gp = os.path.join(RES, "displacements.gpkg")
    if os.path.exists(gp):
        import geopandas as gpd
        gdf = gpd.read_file(gp)
        st.write(f"{len(gdf)} displacement vectors, CRS {gdf.crs}")
        st.dataframe(pd.DataFrame(gdf.drop(columns="geometry")).head(20), width="stretch")
        with open(gp, "rb") as f:
            st.download_button("Download GeoPackage", f, "displacements.gpkg")
