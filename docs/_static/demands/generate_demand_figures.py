# -*- coding: utf-8 -*-
"""Generate the demand-mode figure for docs/guide_demands.rst.

Each panel is a solved toy network, not a sketch: one producer with a fixed
capacity and a seasonal marginal cost feeds a delivery bus that carries the
Load and the delivery Store exactly as ``add_targets_per_product`` builds them.
Demand series and store sizes come from ``build_product_demand_ts``.

Run from the repository root:  python docs/_static/demands/generate_demand_figures.py
"""
import sys
import numpy as np
import pandas as pd
import pypsa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from scripts.preprocessing import build_product_demand_ts  # noqa: E402

OUT = "docs/_static/demands/demand_modes_overview.svg"
SOLVER = "gurobi"

ANNUAL = 100_000.0          # MWh/y
N_BINS = 4
FLEX = 0.05                 # flexibility for flat / profile
BUFFER = 0.5                # demand_store_buffer for bins_profile
CAP = 1.6                   # producer capacity, multiple of the mean rate

snaps = pd.date_range("2023-01-01", periods=8760, freq="h")
prof = pd.read_csv("data/common/NG_demand_DK_profile.csv", sep=";",
                   index_col=0, parse_dates=True).iloc[:, 0]
prof = prof.reindex(snaps, method="ffill")
# winter-expensive production: the store has a reason to carry summer output
doy = np.arange(8760) / 8760
cost = pd.Series(50 + 20 * np.cos(2 * np.pi * doy), index=snaps)

INK, INK2, GRID = "#14202c", "#7d8fa1", "#e3e8ee"
C_DEM, C_PROD, C_SOC, C_CARRY = "#1668c4", "#6b7a87", "#d0352a", "#2f7d4f"


def solve(mode):
    ts, e_nom_max = build_product_demand_ts(
        ANNUAL, mode, snaps, profile_ts=prof, n_bins=N_BINS,
        flexibility_fraction=FLEX, store_buffer=BUFFER)
    d = ts.iloc[:, 0]
    n = pypsa.Network()
    n.set_snapshots(snaps)
    n.add("Bus", "delivery")
    n.add("Generator", "producer", bus="delivery",
          p_nom=CAP * ANNUAL / 8760, marginal_cost=cost)
    n.add("Load", "delivery", bus="delivery", p_set=d)
    n.add("Store", "delivery store", bus="delivery", e_nom_extendable=True,
          e_cyclic=True, e_nom_max=e_nom_max, capital_cost=1.0)
    n.optimize(solver_name=SOLVER, solver_options={"OutputFlag": 0})
    e = n.stores_t.e["delivery store"]
    p = n.generators_t.p["producer"]
    # the store balance the figure relies on: SOC after a delivery is what is left
    e_prev = e.shift(1, fill_value=e.iloc[-1])
    assert np.allclose(e, e_prev + p - d, atol=1e-4)
    assert e.min() > -1e-4
    return d, p, e, e_nom_max


def axes_style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(snaps[0] - pd.Timedelta(days=4), snaps[-1] + pd.Timedelta(days=8))
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.tick_params(colors=INK2, labelsize=9)


def daily(s):
    return s.resample("D").mean()


fig, axs = plt.subplots(2, 2, figsize=(12, 7.6), constrained_layout=True)
titles = {"flat": "(A) flat, flexibility 0.05",
          "profile": "(B) profile, flexibility 0.05",
          "bins_flat": f"(C) bins_flat, {N_BINS} bins",
          "bins_profile": f"(D) bins_profile, {N_BINS} bins, buffer {BUFFER}"}

for ax, mode in zip(axs.flat, titles):
    d, p, e, e_max = solve(mode)
    axes_style(ax)
    ax.set_title(titles[mode], loc="left", fontsize=11, color=INK, fontweight="bold")
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(colors=C_SOC, labelsize=9)
    ax2.set_ylabel("store SOC [MWh]", color=C_SOC, fontsize=9)

    if mode in ("flat", "profile"):
        ax.plot(daily(d).index, daily(d), color=C_DEM, lw=1.6, label="demand [MW]")
        ax.plot(daily(p).index, daily(p), color=C_PROD, lw=1.0, label="production [MW]")
        ax.set_ylabel("MW (daily mean)", fontsize=9, color=INK2)
        ax.set_ylim(0, 2.2 * ANNUAL / 8760)
        ax2.fill_between(e.index, e, color=C_SOC, alpha=0.12, lw=0)
        ax2.plot(e.index, e, color=C_SOC, lw=0.9, label="store SOC [MWh]")
        ax2.set_ylim(0, 2.2 * e_max)
    else:
        eps = d[d > 0]
        ax.bar(eps.index, eps.values, width=4, color=C_DEM, label="delivery [MWh]")
        ax.set_ylabel("delivery [MWh]", fontsize=9, color=C_DEM)
        ax.set_ylim(0, 1.9 * eps.max())
        ax2.fill_between(e.index, e, color=C_SOC, alpha=0.12, lw=0)
        ax2.plot(e.index, e, color=C_SOC, lw=0.9, label="store SOC [MWh]")
        ax2.axhline(e_max, color=C_SOC, ls=":", lw=0.9)
        ax2.text(snaps[-1], e_max, "e_nom_max ", color=C_SOC, fontsize=8,
                 va="bottom", ha="right")
        # cyclic store: the year opens with the stock left after the last delivery
        if e.iloc[0] > 1:
            ax2.plot(snaps[0], e.iloc[0], "s", ms=5, color=C_CARRY, zorder=5)
            ax2.annotate(f"start = end: {e.iloc[0]:,.0f}", (snaps[0], e.iloc[0]),
                         xytext=(6, -12), textcoords="offset points",
                         fontsize=8, color=C_CARRY)
        ax2.set_ylim(0, 1.9 * eps.max())
        # SOC left after each delivery is the start of the next bin
        for t in eps.index:
            ax2.plot(t, e[t], "o", ms=5, color=C_CARRY, zorder=5)
            ax2.annotate(f"{e[t]:,.0f}", (t, e[t]), xytext=(-4, 6),
                         textcoords="offset points", ha="right",
                         fontsize=8, color=C_CARRY)
        ax2.plot([], [], "o", color=C_CARRY, ms=5,
                 label="SOC after delivery = start of next bin")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, frameon=False, ncol=2)

    print(f"{mode:13s} e_nom_max={e_max:>10,.0f}  e_opt={e.max():>10,.0f}  "
          + ("SOC after deliveries: "
             + ", ".join(f"{e[t]:,.0f}" for t in d[d > 0].index) if "bins" in mode else ""))

fig.suptitle(f"Demand modes, solved toy network: {ANNUAL/1e3:.0f} GWh/y, "
             f"producer capacity {CAP:.2f} x mean rate, production cheaper in summer",
             fontsize=11, color=INK)
fig.savefig(OUT)
print("wrote", OUT)
