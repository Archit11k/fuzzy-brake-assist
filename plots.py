"""
plots.py - every figure used in the report / viva demo.
All figures are saved to ./outputs/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from braking_system import (speed, distance, drowsiness, brake, SYSTEM,
                            RULE_TABLE, DIST_TERMS, fuzzy_brake, crisp_brake)
from fuzzy_engine import defuzz_centroid, defuzz_bisector, defuzz_mom
from simulation import run_all, SCENARIOS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
COLORS = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#9b2226"]
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# 1 ---------------------------------------------------------------------------
def plot_membership_functions():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, var in zip(axes.flat, [speed, distance, drowsiness, brake]):
        for (term, _), c in zip(var.terms.items(), COLORS):
            ax.plot(var.universe, var.curve(term), lw=2.2, label=term, color=c)
            ax.fill_between(var.universe, var.curve(term), alpha=0.12, color=c)
        ax.set_title(f"{var.name.capitalize()} ({var.unit})", fontweight="bold")
        ax.set_ylim(-0.03, 1.08); ax.set_ylabel("membership μ")
        ax.legend(loc="center right", fontsize=8)
    fig.suptitle("Membership functions (fuzzification)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "1_membership_functions.png")


# 2 ---------------------------------------------------------------------------
def plot_worked_example(s=70, d=18, z=6):
    crisp, info = SYSTEM.infer(speed=s, distance=d, drowsiness=z)
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, var, val in zip(axes[:3], [speed, distance, drowsiness], [s, d, z]):
        for (term, _), c in zip(var.terms.items(), COLORS):
            mu = var.fuzzify(val)[term]
            ax.plot(var.universe, var.curve(term), color=c, lw=2,
                    label=f"{term}: μ={mu:.2f}")
        ax.axvline(val, color="k", ls="--", lw=1.5)
        ax.set_title(f"{var.name} = {val} {var.unit}", fontweight="bold")
        ax.legend(fontsize=8)
    ax = axes[3]
    x, agg = brake.universe, info["aggregated"]
    for (term, _), c in zip(brake.terms.items(), COLORS):
        ax.plot(x, brake.curve(term), color=c, lw=1, ls=":")
    ax.fill_between(x, agg, color="#264653", alpha=0.45, label="aggregated output")
    for fn, name, col in [(defuzz_centroid, "Centroid", "red"),
                          (defuzz_bisector, "Bisector", "blue"),
                          (defuzz_mom, "MOM", "green")]:
        v = fn(x, agg)
        ax.axvline(v, color=col, lw=2, label=f"{name} = {v:.1f}%")
    ax.set_title("Brake: aggregation + defuzzification", fontweight="bold")
    ax.legend(fontsize=8)
    fig.suptitle(f"Worked example: speed={s} km/h, gap={d} m, drowsiness={z}  →  "
                 f"brake = {crisp:.1f} %", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "2_worked_example.png")


# 3 ---------------------------------------------------------------------------
def plot_control_surfaces(levels=(1, 5, 9), n=41):
    S = np.linspace(0, 120, n); D = np.linspace(0, 100, n)
    SS, DD = np.meshgrid(S, D)
    fig = plt.figure(figsize=(18, 5.5))
    for i, z in enumerate(levels, 1):
        B = np.vectorize(lambda s, d: fuzzy_brake(s, d, z))(SS, DD)
        ax = fig.add_subplot(1, len(levels), i, projection="3d")
        ax.plot_surface(SS, DD, B, cmap="inferno_r", edgecolor="none", alpha=0.95)
        ax.set_xlabel("speed (km/h)"); ax.set_ylabel("gap (m)"); ax.set_zlabel("brake %")
        ax.set_zlim(0, 100); ax.view_init(30, 35)
        ax.set_title(f"drowsiness = {z}", fontweight="bold")
    fig.suptitle("Control surfaces: brake % as a continuous function of speed and gap",
                 fontsize=14, fontweight="bold")
    return _save(fig, "3_control_surfaces.png")


# 4 ---------------------------------------------------------------------------
def plot_fuzzy_vs_crisp_curve(s=80):
    D = np.linspace(0, 80, 400)
    fig, ax = plt.subplots(figsize=(10, 5))
    for z, ls in [(1, "-"), (9, "--")]:
        ax.plot(D, [fuzzy_brake(s, d, z) for d in D], lw=2.5, ls=ls,
                color="#2a9d8f", label=f"Fuzzy (drowsiness {z})")
        ax.plot(D, [crisp_brake(s, d, z) for d in D], lw=2.5, ls=ls,
                color="#e76f51", label=f"Crisp if-else (drowsiness {z})")
    ax.set_xlabel("gap to car ahead (m)"); ax.set_ylabel("brake force (%)")
    ax.set_title(f"Fuzzy vs crisp at {s} km/h - fuzzy is smooth, crisp jumps in steps",
                 fontweight="bold")
    ax.legend()
    return _save(fig, "4_fuzzy_vs_crisp_curve.png")


# 5 ---------------------------------------------------------------------------
def plot_rule_table():
    order = list(brake.terms)
    cmap = ListedColormap(COLORS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (dz, rows) in zip(axes, RULE_TABLE.items()):
        grid = np.array([[order.index(t) for t in rows[sp]] for sp in rows])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=len(order) - 1)
        for i, sp in enumerate(rows):
            for j, t in enumerate(rows[sp]):
                ax.text(j, i, t, ha="center", va="center", fontsize=10,
                        color="white" if order.index(t) >= 3 else "black",
                        fontweight="bold")
        ax.set_xticks(range(3), DIST_TERMS); ax.set_yticks(range(3), list(rows))
        ax.set_xlabel("distance"); ax.set_ylabel("speed"); ax.grid(False)
        ax.set_title(f"Driver: {dz}", fontweight="bold")
    fig.suptitle("Fuzzy rule base (27 rules = 3 × 3 × 3)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "5_rule_base.png")


# 6 ---------------------------------------------------------------------------
def plot_simulation(results=None):
    results = results or run_all()
    fig, axes = plt.subplots(3, len(SCENARIOS), figsize=(17, 10), sharex=True)
    for j, name in enumerate(SCENARIOS):
        fz, cr = results[f"Fuzzy  | {name}"], results[f"Crisp  | {name}"]
        t = fz["t"]
        ax = axes[0, j]
        ax.plot(t, fz["v_lead"], color="gray", lw=2, label="lead car")
        ax.plot(t, fz["v"], color="#2a9d8f", lw=2, label="ego - fuzzy")
        ax.plot(t, cr["v"], color="#e76f51", lw=2, ls="--", label="ego - crisp")
        ax.set_title(name, fontweight="bold"); ax.set_ylabel("speed (km/h)")
        ax.legend(fontsize=8)
        ax = axes[1, j]
        ax.plot(t, fz["gap"], color="#2a9d8f", lw=2, label="fuzzy")
        ax.plot(t, cr["gap"], color="#e76f51", lw=2, ls="--", label="crisp")
        ax.axhline(0, color="k", lw=1)
        if cr["collision"]:
            ax.plot(cr["collided_at"], 0, "X", ms=14, color="#9b2226", label="CRASH (crisp)")
        if fz["collision"]:
            ax.plot(fz["collided_at"], 0, "X", ms=14, color="k", label="CRASH (fuzzy)")
        ax.set_ylabel("gap (m)"); ax.legend(fontsize=8)
        ax = axes[2, j]
        ax.plot(t, fz["brake"], color="#2a9d8f", lw=2, label="fuzzy")
        ax.plot(t, cr["brake"], color="#e76f51", lw=2, ls="--", label="crisp")
        ax.set_ylabel("brake cmd (%)"); ax.set_xlabel("time (s)"); ax.legend(fontsize=8)
    fig.suptitle("Simulation: lead car brakes hard at t = 4 s (80 → 10 km/h)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "6_simulation.png")


# 7 ---------------------------------------------------------------------------
def plot_drowsiness_effect(s=70, d=25):
    Z = np.linspace(0, 10, 200)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for method, c in [("centroid", "red"), ("bisector", "blue"), ("mom", "green")]:
        ax.plot(Z, [fuzzy_brake(s, d, z, method) for z in Z], lw=2.2, color=c,
                label=method)
    ax.set_xlabel("driver drowsiness score"); ax.set_ylabel("brake force (%)")
    ax.set_title(f"Effect of drowsiness (speed {s} km/h, gap {d} m) "
                 "for 3 defuzzification methods", fontweight="bold")
    ax.legend()
    return _save(fig, "7_drowsiness_and_defuzz.png")


def make_all():
    results = run_all()
    return [plot_membership_functions(), plot_worked_example(),
            plot_control_surfaces(), plot_fuzzy_vs_crisp_curve(),
            plot_rule_table(), plot_simulation(results), plot_drowsiness_effect()]


if __name__ == "__main__":
    for p in make_all():
        print("saved", p)
