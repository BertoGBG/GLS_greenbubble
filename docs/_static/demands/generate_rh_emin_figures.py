# -*- coding: utf-8 -*-
"""Generate the e_min figures for the rolling-horizon / demand-bin explanation.

rh_emin_scaling.svg   e_min(t) before the last delivery for 1, 12, 52 and 365 bins
rh_pf_vs_rh.svg       one year solved with PyPSA, for 12 bins and 1 bin: perfect
                        foresight, and rolling horizon with each delivery_target
                        rule of scripts/delivery_targets.py (none, e_min, pro_rata)

All runs share the plant, the demand and the windows. The delivery store has no
size limit; it starts empty and holds at most the deliveries still to come, so
it is empty after the last one.

Run from the repository root:  python docs/_static/demands/generate_rh_emin_figures.py
"""
import numpy as np
import pandas as pd
import pypsa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import sys
sys.path.insert(0, ".")
from scripts.delivery_targets import set_delivery_targets, unmet_deliveries  # noqa: E402

OUT_SCALING = "docs/_static/demands/rh_emin_scaling.svg"
OUT_COMPARE = "docs/_static/demands/rh_pf_vs_rh.svg"
SOLVER = "gurobi"

T = 8760                    # h, Δt = 1 h
C = 1.0                     # MW
U = 0.5                     # utilisation
A = U * C * T               # MWh/y
H, O = 168, 72              # h
S = H - O

INK, INK2, GRID = "#14202c", "#7d8fa1", "#e3e8ee"
C_DEM, C_PF, C_RH, C_FLOOR, C_BAD, C_E = ("#1668c4", "#44586b", "#d0352a",
                                           "#2f7d4f", "#8b1a10", "#c9820f")

snaps = pd.date_range("2023-01-01", periods=T, freq="h")
h = np.arange(T)
# production cost: dearer in winter, daily cycle
cost = pd.Series(50 + 20 * np.cos(2 * np.pi * h / T) + 15 * np.sin(2 * np.pi * h / 24),
                 index=snaps)


def deliveries(n_bins):
    """Equal bins, endpoints as in scripts/preprocessing.py (bins_flat)."""
    d = np.zeros(T)
    for i in range(n_bins):
        d[min(int(round((i + 1) * T / n_bins)) - 1, T - 1)] += A / n_bins
    return d


def e_min(d, cap=C, dt=1.0):
    """Smallest stock after each snapshot that keeps every later delivery reachable.

    e_min(T-1) = 0;  e_min(t-1) = max(0, e_min(t) + d(t) - C*dt)
    """
    e = np.zeros(len(d))
    for t in range(len(d) - 1, 0, -1):
        e[t - 1] = max(0.0, e[t] + d[t] - cap * dt)
    return e


def e_prorata(d):
    """Stock after t when each bin is produced at a constant rate d_k / T_b."""
    e = np.zeros(len(d))
    ends = np.flatnonzero(d)
    start = 0
    for t_k in ends:
        n = t_k - start + 1
        e[start:t_k] = d[t_k] * np.arange(1, n) / n
        start = t_k + 1
    return e


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8.5)


# ───────────────────────── scaling figure ─────────────────────────
fig, axs = plt.subplots(2, 2, figsize=(11, 6.6), constrained_layout=True)
for ax, nb in zip(axs.flat, [1, 12, 52, 365]):
    d = deliveries(nb)
    em = e_min(d)
    Tb = T / nb
    tau = (A / nb) / C
    span = int(min(T, max(2.2 * Tb, 2.2 * H)))      # show the last two bins at least
    x = np.arange(T - span, T)
    style(ax)
    ax.fill_between(x, em[x], color=C_FLOOR, alpha=0.18, lw=0)
    ax.plot(x, em[x], color=C_FLOOR, lw=1.8, label="e_min(t) [MWh]")
    for t in np.flatnonzero(d[x[0]:]) + x[0]:
        ax.bar(t, d[t], width=max(1, span / 200), color=C_DEM, zorder=3)
    # lead-time range of the last delivery
    t_k = T - 1
    ax.axvspan(t_k - H + 1, t_k - O, color="#bcd3ea", alpha=0.6, lw=0,
               label="first window sees t_k between these hours")
    ax.axvline(t_k - H + 1, color=C_DEM, lw=0.8, ls=":")
    ax.axvline(t_k - O, color=C_DEM, lw=0.8, ls=":")
    regime = ("θ ≤ O + 1: floor never binds" if tau <= O + 1 else
              "O + 1 < θ ≤ H: binds for some alignments" if tau <= H else
              "θ > H: floor binds in every bin")
    ax.set_title(f"n_bins = {nb}:  T_b = {Tb:,.0f} h,  θ = {tau:,.0f} h,  "
                 f"peak e_min = {em.max():,.0f} MWh\n{regime}",
                 loc="left", fontsize=9.5, color=INK)
    ax.set_xlim(x[0], T + span * 0.01)
    ax.set_ylim(0, 1.15 * A / nb)
    ax.set_xlabel("hour of the year", fontsize=8.5, color=INK2)
    ax.set_ylabel("MWh", fontsize=8.5, color=INK2)
    hd, _ = ax.get_legend_handles_labels()
    ax.legend(handles=hd + [Patch(color=C_DEM, label="delivery d_k [MWh]")],
              loc="upper left", fontsize=7.5, frameon=False)
fig.suptitle(f"The floor fades out as bins shorten: C = {C:g} MW, u = {U}, "
             f"H = {H} h, O = {O} h (last bins of the year shown)", fontsize=10.5, color=INK)
fig.savefig(OUT_SCALING)
print("wrote", OUT_SCALING)


# ───────────────────────── PF vs RH ─────────────────────────
def network(d):
    """One producer feeding a delivery bus with a Load and an uncapped store."""
    n = pypsa.Network()
    n.set_snapshots(snaps)
    n.add("Bus", "delivery")
    n.add("Generator", "producer", bus="delivery", p_nom=C, marginal_cost=cost)
    n.add("Load", "delivery", bus="delivery", p_set=pd.Series(d, index=snaps))
    e_max_pu = pd.Series((d.sum() - np.cumsum(d)) / d.sum(), index=snaps)
    n.add("Store", "delivery store", bus="delivery", e_nom=d.sum(), e_cyclic=False,
          e_initial=0.0, e_max_pu=e_max_pu)
    return n


def run(d, rule):
    n = network(d)
    opts = dict(solver_name=SOLVER, solver_options={"OutputFlag": 0})
    if rule == "pf":
        n.optimize(**opts)
    else:
        set_delivery_targets(n, H, O, rule=rule)
        n.optimize.optimize_with_rolling_horizon(horizon=H, overlap=O, **opts)
    prod_cost = float((n.generators_t.p["producer"] * cost).sum())
    target = (n.stores_t.e_min_pu["delivery store"] * n.stores.at["delivery store", "e_nom"]
              if rule != "pf" else None)
    return dict(e=n.stores_t.e["delivery store"].values, unmet=float(unmet_deliveries(n).sum()),
                cost=prod_cost, target=target)


STYLE = {"pf": (C_PF, "--", 1.3, "perfect foresight"),
         "none": ("#e8a093", "-", 1.1, "rolling horizon, no target"),
         "e_min": (C_RH, "-", 1.5, "rolling horizon, e_min target"),
         "pro_rata": ("#7b4fb0", "-", 1.5, "rolling horizon, pro_rata target")}
fig, axs = plt.subplots(2, 1, figsize=(11, 8.2), constrained_layout=True)
for ax, nb in zip(axs, [12, 1]):
    d = deliveries(nb)
    em = e_min(d)
    res = {r: run(d, r) for r in STYLE}
    for r, v in res.items():
        print(f"n_bins={nb:3d} {r:9s} unmet {v['unmet']:9.1f} MWh   production cost "
              f"{v['cost']:12,.0f} EUR   vs PF {100 * (v['cost'] / res['pf']['cost'] - 1):+6.1f} %")
    style(ax)
    ax.fill_between(h, em, color=C_FLOOR, alpha=0.15, lw=0, label="e_min(t)")
    for r, (col, ls, lw, lab) in STYLE.items():
        ax.plot(h, res[r]["e"], color=col, ls=ls, lw=lw, label=f"store, {lab}")
    tg = res["pro_rata"]["target"]
    pts = tg[tg > 1e-6]
    ax.plot([snaps.get_loc(t) for t in pts.index], pts.values, ".", color="#7b4fb0", ms=4,
            label="pro_rata target at window ends")
    for t in np.flatnonzero(d):
        ax.bar(t, d[t], width=T / 250, color=C_DEM, zorder=3, alpha=0.9)
    pc = {r: 100 * (res[r]["cost"] / res["pf"]["cost"] - 1) for r in res}
    ax.set_title(f"n_bins = {nb}:  no target leaves {res['none']['unmet']:,.0f} MWh unmet.  "
                 f"Cost vs PF: e_min {pc['e_min']:+.1f} %, pro_rata {pc['pro_rata']:+.1f} %",
                 loc="left", fontsize=10, color=INK)
    ax.set_xlim(-50, T + 50)
    ax.set_ylim(0, 1.15 * max(d.max(), max(v["e"].max() for v in res.values())))
    ax.set_xticks(np.arange(0, T + 1, 730))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                        "Sep", "Oct", "Nov", "Dec", ""], fontsize=8.5)
    ax.set_ylabel("MWh", fontsize=8.5, color=INK2)
    hd, _ = ax.get_legend_handles_labels()
    ax.legend(handles=hd + [Patch(color=C_DEM, label="delivery d_k")],
              loc="upper left", fontsize=7.8, frameon=False, ncol=2)
fig.suptitle(f"Same plant (C = {C:g} MW), demand (u = {U}) and windows (H = {H} h, O = {O} h); "
             f"store uncapped, empty at start and end", fontsize=10.5, color=INK)
fig.savefig(OUT_COMPARE)
print("wrote", OUT_COMPARE)
