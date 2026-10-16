# -*- coding: utf-8 -*-
"""Tutorial 3b figures: hydrogen deliveries under perfect foresight and rolling horizon.

Run from the repository root after step 1 and the step-2 variants:

    python tutorials/3b_rh_bins/plot_rh_deliveries.py [run_name ...]

Each argument is ``run_name`` or ``run_name=label`` of a rolling-horizon run
(default: tut3b_rh). The first one is used for the late-year figure.
Writes docs/_static/tutorials/tut3b_*.png.
"""
import glob
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pypsa

OUT = "docs/_static/tutorials"
STORE, LOAD = "H2 delivery store", "H2 delivery"
COLORS = ["#d0352a", "#7b4fb0", "#2f7d4f", "#c9820f"]


def load(run, pattern="*_OPT.nc"):
    return pypsa.Network(glob.glob(f"outputs/single_analysis/{run}/networks/{pattern}")[0])


def unmet(n):
    g = [x for x in n.generators.index if n.generators.at[x, "carrier"] == "unmet delivery"]
    w = n.snapshot_weightings.generators
    return (n.generators_t.p[g].sum(axis=1) * w) if g else w * 0


args = sys.argv[1:] or ["tut3b_rh"]
labels = dict(a.split("=", 1) if "=" in a else (a, a) for a in args)
runs = list(labels)
pf = load("tut3b_pf")
rh = {r: load(r) for r in runs}
w = pf.snapshot_weightings.stores
d = pf.loads_t.p_set[LOAD] * w
t = pf.snapshots

# ── store content over the year ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
ax.bar(t[d > 0], d[d > 0], width=3, color="#1668c4", label="delivery d_k [MWh]", zorder=3)
ax.plot(t, pf.stores_t.e[STORE], "--", color="#44586b", lw=1.3, label="stock, perfect foresight")
ax.axhline(pf.stores.at[STORE, "e_nom_max"], color="#44586b", lw=0.8, ls=":")
ax.text(t[5], pf.stores.at[STORE, "e_nom_max"], " E, perfect-foresight store limit",
        va="bottom", fontsize=8, color="#44586b")
for (r, n), c in zip(rh.items(), COLORS):
    ax.plot(n.snapshots, n.stores_t.e[STORE], color=c, lw=1.3,
            label=f"{labels[r]}: unmet {unmet(n).sum():,.0f} MWh")
    if STORE in n.stores_t.e_min_pu:          # an all-zero target is not exported
        f = n.stores_t.e_min_pu[STORE] * n.stores.at[STORE, "e_nom"]
        ax.plot(f.index[f > 0], f[f > 0], ".", color=c, ms=5)
ax.plot([], [], ".", color="#7d8fa1", label="target f(τ_j) at window ends")
ax.set_ylabel("MWh"); ax.set_xlim(t[0], t[-1])
ax.grid(axis="y", color="#e3e8ee")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0, 0.97))
ax.set_title("Hydrogen delivery store: 26 two-weekly bins", loc="left", fontsize=10.5)
fig.savefig(f"{OUT}/tut3b_H2_store.png", dpi=150)

# ── late November: what the electrolysers could really make ──────────────────
n = rh[runs[0]]
w_n = n.snapshot_weightings.stores
plants = [l for l in n.links.index
          if n.links.at[l, "bus1"] == "H2 collection" and n.links.at[l, "p_nom"] > 1e-3]
h2 = sum(-n.links_t.p1[l] for l in plants)
rating = sum(n.links.at[l, "p_nom"] * n.links.at[l, "efficiency"] for l in plants)
el_in = sum(n.links.at[l, "p_nom"] for l in plants)
grid = n.links.at["DK1_to_ElDK1_buy", "p_nom"]          # shared grid connection
eff = rating / el_in
wind = sum(n.generators.at[g, "p_nom"] * n.generators_t.p_max_pu[g]
           for g in n.generators.index if n.generators.at[g, "bus"] == "El3"
           and g in n.generators_t.p_max_pu)
possible = np.minimum(rating, (grid + wind) * eff)
win = slice("2024-11-01", "2024-12-31")
fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
ax.fill_between(h2[win].index, h2[win], step="post", color="#1668c4", alpha=0.25, lw=0)
ax.step(h2[win].index, h2[win], where="post", color="#1668c4", lw=1.2,
        label="hydrogen produced, rolling horizon [MW]")
ax.axhline(rating, color="#d0352a", lw=1.2, label=f"C = {rating:.1f} MW, the rate the targets assume")
ax.step(possible[win].index, possible[win], where="post", color="#c9820f", lw=1.0, ls="--",
        label=f"at most: ({grid:.1f} MW grid + available wind) x efficiency, for {el_in:.1f} MW electrolysis")
for ts, v in d[win][d[win] > 0].items():
    ax.axvline(ts, color="#14202c", lw=0.8, ls=":")
    ax.text(ts, rating * 1.02, f" {v:,.0f} MWh due", fontsize=8, va="bottom")
ax.set_ylim(0, rating * 1.25); ax.set_ylabel("MW")
ax.grid(axis="y", color="#e3e8ee")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
u = unmet(n)[win]
for ts, v in u[u > 1e-3].items():
    ax.axvspan(ts, ts + np.timedelta64(8, "h"), color="#d0352a", alpha=0.15, lw=0)
ax.plot([], [], color="#d0352a", alpha=0.3, lw=6, label=f"unmet delivery generator used ({u.sum():,.0f} MWh)")
ax.legend(loc="lower left", fontsize=8, frameon=False)
ax.set_title("November and December: below C whenever the wind is low", loc="left", fontsize=10.5)
fig.savefig(f"{OUT}/tut3b_electrolysis_late_year.png", dpi=150)
print("wrote", f"{OUT}/tut3b_H2_store.png", f"{OUT}/tut3b_electrolysis_late_year.png")
