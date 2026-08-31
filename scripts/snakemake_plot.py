# SPDX-License-Identifier: MIT
"""Snakemake wrapper: generate analysis plots and CSV exports (run_plot_and_export).

Touches output .done marker on successful completion.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import pypsa
from scripts.helpers import (
    create_folder_if_not_exists, zero_small_capacities,
    load_run_config, apply_run_config_overrides,
)
from scripts.plots import run_plot_and_export
from scripts import config as c

n = pypsa.Network(snakemake.input.network)

results_folder = str(Path(snakemake.input.network).parent.parent)

# Describe this network using the config that actually produced it (recorded
# in networks/config_run.yaml at solve time), not the live config.yaml/
# n_config.yaml -- which may have changed since if this rule is rerun on its
# own (see docs/issue #14: individual-rule reruns picking up config drift).
# No-op fallback to live config if no config_run.yaml is found.
c = apply_run_config_overrides(c, load_run_config(results_folder))

# Clean solver-noise artifacts in-memory before any analysis.
# The .nc on disk is never touched — this mirrors the pypsa-eur brownfield
# capacity_threshold approach (add_brownfield.py) for single-period networks.
_zero_th = float(c.optimization.get("zero_threshold_MW", 0.0))
zero_small_capacities(n, _zero_th)

with open(snakemake.input.comp_alloc, "rb") as fh:
    _alloc_payload = pickle.load(fh)
if isinstance(_alloc_payload, dict) and "allocation" in _alloc_payload:
    network_comp_allocation = _alloc_payload["allocation"]
    comp_tech_map    = _alloc_payload.get("tech_mapping", {})
    tech_costs_used  = _alloc_payload.get("tech_costs_used", None)
else:
    network_comp_allocation = _alloc_payload  # backward compat
    comp_tech_map   = {}
    tech_costs_used = None

plot_folder    = create_folder_if_not_exists(results_folder, "plots")
csv_folder     = create_folder_if_not_exists(results_folder, "csv")

run_plot_and_export(
    n                       = n,
    c                       = c,
    csv_folder              = csv_folder,
    plot_folder             = plot_folder,
    items                   = c.items,
    bus_list_mp             = c.bus_list_mp,
    network_comp_allocation = network_comp_allocation,
    comp_tech_map           = comp_tech_map,
    tech_costs_used         = tech_costs_used,
    scenarios               = None,
    networks_dict           = None,
)

Path(snakemake.output.done).touch()
