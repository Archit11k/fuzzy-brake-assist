"""
app.py - interactive demo for the viva.

    pip install streamlit
    streamlit run app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from braking_system import (SYSTEM, RULES, speed, distance, drowsiness, brake,
                            crisp_brake)
from fuzzy_engine import defuzz_centroid, defuzz_bisector, defuzz_mom
from simulation import simulate
from braking_system import fuzzy_brake

COLORS = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#9b2226"]
st.set_page_config(page_title="Fuzzy Braking Assist", layout="wide")
st.title("🚗 Fuzzy Driver-Drowsiness & Braking Assist")
st.caption("Mamdani fuzzy inference engine built from scratch · CSE3121 AI & Soft Computing")

with st.sidebar:
    st.header("Inputs")
    s = st.slider("Vehicle speed (km/h)", 0, 120, 70)
    d = st.slider("Gap to car ahead (m)", 0, 100, 18)
    z = st.slider("Driver drowsiness (0 alert – 10 asleep)", 0.0, 10.0, 6.0, 0.5)
    method = st.radio("Defuzzification", ["centroid", "bisector", "mom"])

crisp, info = SYSTEM.infer(method=method, speed=s, distance=d, drowsiness=z)

c1, c2, c3 = st.columns(3)
c1.metric("Fuzzy brake force", f"{crisp:.1f} %")
c2.metric("Crisp if-else brake", f"{crisp_brake(s, d, z):.0f} %")
level = max(brake.fuzzify(crisp), key=brake.fuzzify(crisp).get)
c3.metric("Dominant output term", level)
st.progress(min(int(crisp), 100))

tab1, tab2, tab3 = st.tabs(["Inference steps", "Rules fired", "Simulation"])

with tab1:
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    for ax, var, val in zip(axes[:3], [speed, distance, drowsiness], [s, d, z]):
        for (term, _), c in zip(var.terms.items(), COLORS):
            ax.plot(var.universe, var.curve(term), color=c, lw=2,
                    label=f"{term}: {var.fuzzify(val)[term]:.2f}")
        ax.axvline(val, color="k", ls="--"); ax.set_title(var.name); ax.legend(fontsize=7)
    x, agg = brake.universe, info["aggregated"]
    for (term, _), c in zip(brake.terms.items(), COLORS):
        axes[3].plot(x, brake.curve(term), color=c, ls=":", lw=1)
    axes[3].fill_between(x, agg, color="#264653", alpha=0.5)
    axes[3].axvline(crisp, color="red", lw=2)
    axes[3].set_title(f"aggregated output → {method} = {crisp:.1f}%")
    st.pyplot(fig)
    st.write({"Centroid": round(defuzz_centroid(x, agg), 2),
              "Bisector": round(defuzz_bisector(x, agg), 2),
              "Mean of Maximum": round(defuzz_mom(x, agg), 2)})

with tab2:
    fired = [(a, str(r)) for r, a in zip(RULES, info["firing"]) if a > 0]
    st.write(f"**{len(fired)} of {len(RULES)} rules fired** (AND = min):")
    for a, r in sorted(fired, reverse=True):
        st.write(f"`α = {a:.2f}`  {r}")

with tab3:
    st.write("Lead car brakes hard (80 → 10 km/h) at t = 4 s; starting gap 45 m.")
    fz, cr = simulate(fuzzy_brake, z), simulate(crisp_brake, z)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].plot(fz["t"], fz["gap"], label="fuzzy", color="#2a9d8f", lw=2)
    ax[0].plot(cr["t"], cr["gap"], label="crisp", color="#e76f51", ls="--", lw=2)
    ax[0].axhline(0, color="k"); ax[0].set_title("gap (m)"); ax[0].legend()
    ax[1].plot(fz["t"], fz["brake"], label="fuzzy", color="#2a9d8f", lw=2)
    ax[1].plot(cr["t"], cr["brake"], label="crisp", color="#e76f51", ls="--", lw=2)
    ax[1].set_title("brake command (%)"); ax[1].legend()
    st.pyplot(fig)
    st.write(f"Fuzzy: {'💥 COLLISION' if fz['collision'] else '✅ safe'} "
             f"(min gap {fz['min_gap']:.1f} m) · Crisp: "
             f"{'💥 COLLISION' if cr['collision'] else '✅ safe'} "
             f"(min gap {cr['min_gap']:.1f} m)")
