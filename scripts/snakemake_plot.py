"""Snakemake wrapper: generate analysis plots and CSV exports (run_plot_and_export).

Touches output .done marker on successful completion.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pypsa
from scripts.helpers import create_folder_if_not_exists
from scripts.plots import run_plot_and_export
from scripts import config as c

n = pypsa.Network(snakemake.input.network)

results_folder = str(Path(snakemake.input.network).parent.parent)
plot_folder    = create_folder_if_not_exists(results_folder, "plots")
csv_folder     = create_folder_if_not_exists(results_folder, "csv")

# resolve symbolic threshold keys to numeric values
for it in c.items:
    if isinstance(it.get("th"), str):
        it["th"] = float(c.thresholds[it["th"]])

run_plot_and_export(
    n                       = n,
    c                       = c,
    csv_folder              = csv_folder,
    plot_folder             = plot_folder,
    thresholds              = c.thresholds,
    items                   = c.items,
    bus_list_mp             = c.bus_list_mp,
    network_comp_allocation = getattr(n, "network_comp_allocation", None),
    scenarios               = None,
    networks_dict           = None,
)

Path(snakemake.output.done).touch()
