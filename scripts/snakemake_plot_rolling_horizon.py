# SPDX-License-Identifier: MIT
"""Snakemake wrapper: generate plots and CSVs for a rolling horizon result.

Runs the full ``run_plot_and_export`` suite on the RH network (carrier-level
costs, operational heatmaps, shadow prices, etc.) and then generates
side-by-side PF vs RH comparison plots via ``run_plot_rh_comparison``.

The RH network keeps the components and capacities of the PF network it was
built from, so the PF run's config fingerprint and component allocation pickle
describe it too. Both are read from the PF run. If the allocation pickle is
missing, the agent-level steps are skipped with a warning.
"""
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pypsa
from scripts.helpers import (
    create_folder_if_not_exists, load_run_config, apply_run_config_overrides,
    tighten_negligible_capacities,
)
from scripts.plots import run_plot_and_export, run_plot_rh_comparison
from scripts import config as c

n_rh = pypsa.Network(snakemake.input.network)
n_pf = pypsa.Network(snakemake.input.network_pf)

# Report free and loop_tol-cost connectors at the flow they carry (in-memory
# only); see snakemake_plot.py.
tighten_negligible_capacities(n_rh)

results_folder = Path(snakemake.input.network).parent.parent
pf_folder      = Path(snakemake.input.network_pf).parent.parent

# Describe this network using the config that actually produced it, not
# whatever config.yaml/n_config.yaml currently say -- see snakemake_plot.py
# for the full rationale. The capacities come from the PF run, so its
# fingerprint is the one to use; the RH output folder may belong to another run.
c = apply_run_config_overrides(c, load_run_config(str(pf_folder)))

# Component allocation of the PF network, written by prepare_inputs as
# resources/{pf_run}/{pf_network}_comp_alloc.pkl.
_pf_name    = Path(snakemake.input.network_pf).stem.removesuffix("_OPT")
_alloc_path = Path("resources") / pf_folder.name / f"{_pf_name}_comp_alloc.pkl"
network_comp_allocation, comp_tech_map, tech_costs_used = None, {}, None
if _alloc_path.exists():
    with open(_alloc_path, "rb") as fh:
        _alloc_payload = pickle.load(fh)
    if isinstance(_alloc_payload, dict) and "allocation" in _alloc_payload:
        network_comp_allocation = _alloc_payload["allocation"]
        comp_tech_map   = _alloc_payload.get("tech_mapping", {})
        tech_costs_used = _alloc_payload.get("tech_costs_used", None)
    else:
        network_comp_allocation = _alloc_payload  # backward compat
else:
    print(f"[plot_rh] no allocation pickle at {_alloc_path}; agent-level steps skipped")

plot_folder    = create_folder_if_not_exists(str(results_folder), "plots_rh")
csv_folder     = create_folder_if_not_exists(str(results_folder), "csv_rh")

# ── Full plot suite for the RH network ────────────────────────────────────────
run_plot_and_export(
    n                       = n_rh,
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

# ── PF vs RH comparison plots ─────────────────────────────────────────────────
run_plot_rh_comparison(
    n_pf        = n_pf,
    n_rh        = n_rh,
    plot_folder = plot_folder,
    csv_folder  = csv_folder,
    c           = c,
)

Path(snakemake.output.done).touch()
