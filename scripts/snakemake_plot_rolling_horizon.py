# SPDX-License-Identifier: MIT
"""Snakemake wrapper: generate plots and CSVs for a rolling horizon result.

Runs the full ``run_plot_and_export`` suite on the RH network (carrier-level
costs, operational heatmaps, shadow prices, etc.) and then generates
side-by-side PF vs RH comparison plots via ``run_plot_rh_comparison``.

Steps that require ``network_comp_allocation`` (agent-level cost breakdown,
capacity allocation table) are skipped because no allocation pickle is
produced for the RH solve.  All other steps — including cost-by-carrier which
uses ``n.statistics`` — work correctly because the RH network has its
extendability flags restored and its CAPEX embedded as ``objective_constant``.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pypsa
from scripts.helpers import create_folder_if_not_exists
from scripts.plots import run_plot_and_export, run_plot_rh_comparison
from scripts import config as c

n_rh = pypsa.Network(snakemake.input.network)
n_pf = pypsa.Network(snakemake.input.network_pf)

results_folder = Path(snakemake.input.network).parent.parent
plot_folder    = create_folder_if_not_exists(str(results_folder), "plots_rh")
csv_folder     = create_folder_if_not_exists(str(results_folder), "csv_rh")

# ── Full plot suite for the RH network ────────────────────────────────────────
# network_comp_allocation=None → agent/allocation steps are skipped with a
# warning; all carrier-level cost, operational and shadow-price steps run.
run_plot_and_export(
    n                       = n_rh,
    c                       = c,
    csv_folder              = csv_folder,
    plot_folder             = plot_folder,
    items                   = c.items,
    bus_list_mp             = c.bus_list_mp,
    network_comp_allocation = None,
    comp_tech_map           = {},
    tech_costs_used         = None,
    scenarios               = None,
    networks_dict           = None,
)

# ── PF vs RH comparison plots ─────────────────────────────────────────────────
run_plot_rh_comparison(
    n_pf        = n_pf,
    n_rh        = n_rh,
    plot_folder = plot_folder,
    csv_folder  = csv_folder,
    c           = c,
)

Path(snakemake.output.done).touch()
