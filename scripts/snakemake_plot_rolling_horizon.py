# SPDX-License-Identifier: MIT
"""Snakemake wrapper: generate operational plots for a rolling horizon result.

Reuses the same plotting functions as the CD optimisation but skips
capacity and cost steps that are not meaningful for a dispatch-only network.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pypsa
from scripts.helpers import create_folder_if_not_exists
from scripts.plots import run_plot_operational
from scripts import config as c

n = pypsa.Network(snakemake.input.network)

# Output folder sits next to the RH network file
rh_net_path   = Path(snakemake.input.network)
results_folder = rh_net_path.parent.parent   # …/networks/rolling_horizon/ → …/
plot_folder    = create_folder_if_not_exists(str(results_folder), "plots_rh")

# Resolve symbolic threshold keys to numeric values (same as snakemake_plot.py)
for it in c.items:
    if isinstance(it.get("th"), str):
        it["th"] = float(c.thresholds[it["th"]])

run_plot_operational(
    n            = n,
    c            = c,
    plot_folder  = plot_folder,
    thresholds   = c.thresholds,
    items        = c.items,
    bus_list_mp  = c.bus_list_mp,
)

Path(snakemake.output.done).touch()
