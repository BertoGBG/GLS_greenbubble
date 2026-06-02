# Run from the repo root:
#   python tutorials/2_brownfield_heat/plot_heat_network.py
#
# REMINDER: the network topology plots do not filter out components with
# small optimal capacities due to numerical tolerance of the solver.

import pathlib
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import pypsatopo

NETWORK_NC = pathlib.Path(
    "outputs/single_analysis/"
    "B_H_RE_H2_METH_SN_ST_CO2_100_tD_H2_580_MeOH_0_CH4_262_2024_El_0.1_DET_3h_tut2_brownfield_heat/"
    "networks/"
    "B_H_RE_H2_METH_SN_ST_CO2_100_tD_H2_580_MeOH_0_CH4_262_2024_El_0.1_DET_3h_tut2_brownfield_heat_OPT.nc"
)

n = pypsa.Network(str(NETWORK_NC))

# ── Heat-subsystem topology SVG ──────────────────────────────────────────────
if isinstance(n.buses.index, pd.MultiIndex):
    bus_names = n.buses.index.get_level_values("name").unique()
else:
    bus_names = n.buses.index

heat_buses = list(dict.fromkeys(b for b in bus_names if "Heat" in b or "DH" in b))
print("Heat buses:", heat_buses)

svg_out = pathlib.Path("docs/_static/tutorials/tut2_heat_heat_subsystem")
svg_out.parent.mkdir(parents=True, exist_ok=True)

pypsatopo.generate(
    n,
    focus=heat_buses,
    neighbourhood=1,
    carrier_color=True,
    file_output=str(svg_out),
    file_format="svg",
)

# ── DH demand: time series + load-duration curve ─────────────────────────────
dh = n.loads_t.p_set["DH load"]          # MW
peak = dh.max()
annual_gwh = (dh * n.snapshot_weightings["objective"]).sum() / 1e3

fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)

# top: full-year time series
ax = axes[0]
ax.fill_between(dh.index, dh.values, alpha=0.6, color="#e91e63", linewidth=0)
ax.plot(dh.index, dh.values, color="#e91e63", linewidth=0.6)
ax.axhline(peak, color="black", linewidth=0.8, linestyle="--",
           label=f"Peak {peak:.1f} MW")
ax.set_ylabel("DH load (MW)")
ax.set_title("District-heating demand — full year")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.legend(fontsize=9)
ax.set_xlim(dh.index[0], dh.index[-1])
ax.grid(axis="y", alpha=0.3)

# bottom: load-duration curve
ax2 = axes[1]
ldc = dh.sort_values(ascending=False).values
step_h = dh.index.to_series().diff().dt.total_seconds().median() / 3600
x = [i * step_h for i in range(len(ldc))]
ax2.fill_between(x, ldc, alpha=0.6, color="#e91e63", linewidth=0)
ax2.set_xlabel("Hours per year (sorted)")
ax2.set_ylabel("DH load (MW)")
ax2.set_title(
    f"Load duration curve  |  Peak {peak:.1f} MW  |  Annual {annual_gwh:.1f} GWh/y"
)
ax2.grid(axis="y", alpha=0.3)

png_out = pathlib.Path("docs/_static/tutorials/tut2_heat_DH_load_profile.png")
png_out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(png_out, dpi=150, bbox_inches="tight")
print(f"Saved → {png_out}")
plt.show()
