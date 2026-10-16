# -*- coding: utf-8 -*-
"""Generate the rolling-horizon / demand-bin concept figure.

Three toy cases on four weeks, each solved twice with PyPSA: once with perfect
foresight and once with ``optimize_with_rolling_horizon``. The producer capacity
is fixed in both, as in a GreenBubble rolling-horizon run. An expensive
``unmet`` generator on the delivery bus shows what the rolling horizon cannot
deliver instead of letting the window fail.

Run from the repository root:  python docs/_static/demands/generate_rh_bins_figure.py
"""
import numpy as np
import pandas as pd
import pypsa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "docs/_static/demands/rh_bins_concept.svg"
SOLVER = "gurobi"

T = 4 * 168                 # h
C = 1.0                     # MW, fixed producer capacity
U = 0.4                     # utilisation: A = U * C * T
A = U * C * T               # MWh over the period
H, O = 168, 72              # h, rolling horizon
S = H - O

snaps = pd.date_range("2024-01-01", periods=T, freq="h")
hours = np.arange(T)
cost = pd.Series(40 + 15 * np.sin(2 * np.pi * hours / 24), index=snaps)  # EUR/MWh

INK, INK2, GRID = "#14202c", "#7d8fa1", "#e3e8ee"
C_DEM, C_PF, C_RH, C_BAD, C_WIN, C_WIN2 = ("#1668c4", "#7d8fa1", "#d0352a",
                                           "#8b1a10", "#2f6fb0", "#bcd3ea")


def demand(case):
    d = np.zeros(T)
    if case == "flat":
        d[:] = A / T
    elif case == "weekly":
        for k in range(4):
            d[(k + 1) * 168 - 1] = A / 4
    else:  # one bin at the end
        d[-1] = A
    return pd.Series(d, index=snaps)


def network(d, store):
    n = pypsa.Network()
    n.set_snapshots(snaps)
    n.add("Bus", "delivery")
    n.add("Generator", "producer", bus="delivery", p_nom=C, marginal_cost=cost)
    # unmet delivery is only possible at a delivery hour, so it cannot fill the store
    n.add("Generator", "unmet", bus="delivery", p_nom=1e6, marginal_cost=1e4,
          p_max_pu=(d > 0).astype(float))
    n.add("Load", "delivery", bus="delivery", p_set=d)
    if store:
        n.add("Store", "delivery store", bus="delivery", e_nom=A,
              e_cyclic=False, e_initial=0.0)
    return n


def solve(d, store, rh):
    n = network(d, store)
    opts = dict(solver_name=SOLVER, solver_options={"OutputFlag": 0})
    if rh:
        n.optimize.optimize_with_rolling_horizon(horizon=H, overlap=O, **opts)
    else:
        n.optimize(**opts)
    e = n.stores_t.e["delivery store"] if store else pd.Series(0.0, index=snaps)
    return e, n.generators_t.p["unmet"]


cases = [("flat", "(a) Continuous demand, no store: every window sees its own demand"),
         ("weekly", "(b) Weekly bins: θ = 67 h ≤ O + 1 = 73 h, so it fits wherever the delivery falls"),
         ("single", "(c) One bin at the end: θ = 269 h > H = 168 h, so no window sees it early enough")]

fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True,
                        gridspec_kw={"height_ratios": [0.6, 1, 1]})
for ax, (case, title) in zip(axs, cases):
    d = demand(case)
    store = case != "flat"
    e_pf, u_pf = solve(d, store, rh=False)
    e_rh, u_rh = solve(d, store, rh=True)
    print(f"{case:7s} unmet PF {u_pf.sum():8.1f} MWh   RH {u_rh.sum():8.1f} MWh")

    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)

    if case == "flat":
        ax.plot(hours, d.values, color=C_DEM, lw=1.6, label="demand [MW]")
        ax.plot(hours, np.minimum(C, d.values), color=C_RH, lw=1.0, ls="--",
                label="production, PF and RH [MW]")
        ax.axhline(C, color=INK2, lw=0.8, ls=":")
        ax.text(T - 2, C, "capacity C ", ha="right", va="bottom", fontsize=8, color=INK2)
        ax.set_ylim(0, 1.5 * C)
        ax.set_ylabel("MW", fontsize=9, color=INK2)
    else:
        top = 1.25 * A
        eps = d[d > 0]
        for t in np.flatnonzero(d.values):
            ax.bar(t, d.iloc[t], width=4, color=C_DEM, zorder=3)
        ax.plot(hours, e_pf.values, color=C_PF, lw=1.4, ls="--", label="store SOC, perfect foresight [MWh]")
        ax.plot(hours, e_rh.values, color=C_RH, lw=1.6, label="store SOC, rolling horizon [MWh]")
        handles = [Patch(color=C_DEM, label="delivery [MWh]")]
        miss = u_rh[u_rh > 1e-6]
        for t in miss.index:
            i = snaps.get_loc(t)
            ax.bar(i, miss[t], bottom=d.iloc[i] - miss[t], width=5, color=C_BAD, zorder=4)
            ax.annotate(f"unmet {miss[t]:,.0f} MWh", (i, d.iloc[i] - miss[t] / 2),
                        xytext=(-8, 0), textcoords="offset points", ha="right",
                        va="center", fontsize=8.5, color=C_BAD)
        if len(miss):
            handles.append(Patch(color=C_BAD, label="unmet by rolling horizon [MWh]"))
        ax.set_ylim(0, top)
        ax.set_ylabel("MWh", fontsize=9, color=INK2)

        # the window that first sees the last delivery, and its lead time
        t_k = int(np.flatnonzero(d.values)[-1])
        j = max(0, (t_k - H) // S + 1)
        s_k = j * S
        L = t_k - s_k + 1
        y = 0.93 * top
        ax.add_patch(plt.Rectangle((s_k, y), S, 0.05 * top, color=C_WIN, lw=0))
        ax.add_patch(plt.Rectangle((s_k + S, y), min(O, T - s_k - S), 0.05 * top, color=C_WIN2, lw=0))
        ax.annotate("", xy=(t_k, 0.86 * top), xytext=(s_k, 0.86 * top),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
        ax.text((s_k + t_k) / 2, 0.875 * top, f"lead time L = {L} h", ha="center",
                fontsize=8.5, color=INK)
        ax.text(s_k - 4, y + 0.025 * top, "first window that sees the last delivery",
                ha="right", va="center", fontsize=8.5, color=C_WIN)
        tau = d.iloc[t_k] / C
        ax.annotate("", xy=(t_k, 0.78 * top), xytext=(t_k - tau, 0.78 * top),
                    arrowprops=dict(arrowstyle="<->", color=C_RH, lw=1))
        ax.text(min(t_k - tau / 2, t_k - 60), 0.795 * top, f"needed θ = d / C = {tau:.0f} h",
                ha="center", fontsize=8.5, color=C_RH)
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h + (handles if case != "flat" else []), loc="upper left",
              fontsize=8, frameon=False, ncol=1)

axs[-1].set_xlim(-4, T + 4)
axs[-1].set_xticks(range(0, T + 1, 168))
axs[-1].set_xlabel("hour", fontsize=9, color=INK2)
fig.suptitle(f"Same plant (C = {C:g} MW), same annual demand (u = {U}), same rolling horizon "
             f"(H = {H} h, O = {O} h): only the demand structure changes",
             fontsize=10.5, color=INK)
fig.savefig(OUT)
print("wrote", OUT)
