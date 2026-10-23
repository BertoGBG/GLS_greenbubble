# SPDX-License-Identifier: MIT
"""Snakemake wrapper: solve network (deterministic or stochastic).

Handles:
  - single deterministic solve
  - stochastic solve (single multi-scenario network)
  - EVPI wait-and-see deterministic networks per scenario
Exports OPT network, saves config, network_comp_allocation and EVPI CSV.
"""
from pathlib import Path
import sys
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

import pypsa
from scripts.helpers import (
    build_model_solve_network,
    export_network,
    save_config,
    save_network_comp_allocation,
    create_folder_if_not_exists,
    compare_objective,
)
from scripts.plots import print_network
from scripts.delivery_targets import unmet_deliveries, drop_unused_unmet
from scripts import config as c, parameters as p

# ---- Folder layout derived from the declared output path
networks_folder = str(Path(snakemake.output.network).parent)
results_folder  = str(Path(snakemake.output.network).parent.parent)
plot_folder     = create_folder_if_not_exists(results_folder, "plots")
csv_folder      = create_folder_if_not_exists(results_folder, "csv")
create_folder_if_not_exists(results_folder, "networks")

network_name = snakemake.wildcards.network

# ---- Load pre-built network and supporting data
n = pypsa.Network(snakemake.input.network)

with open(snakemake.input.comp_alloc, "rb") as fh:
    _alloc_payload = pickle.load(fh)
if isinstance(_alloc_payload, dict) and "allocation" in _alloc_payload:
    network_comp_allocation = _alloc_payload["allocation"]
else:
    network_comp_allocation = _alloc_payload  # backward compat

# ---- Print PRE network topology (before solving mutates n)
print_network(
    n             = n,
    n_flags       = c.n_flags,
    nc_path       = snakemake.input.network,
    network_name  = network_name,
    suffix        = "_PRE",
    plot_folder   = plot_folder,
    is_stochastic = c.stochastic["stochastic"],
)

# ---- Assemble networks dict (main + WS for EVPI)
networks_dict = {}
n_names_dict  = {}

if c.stochastic["stochastic"]:
    from scripts.create_stoch_scenarios import scenarios

    networks_dict["stoch"] = n
    n_names_dict["stoch"]  = network_name

    if c.stochastic["EVPI"]:
        # Wait-and-see networks are taken from the unsolved stochastic network,
        # so they share its scenario data (prices, capacity factors, CO2 costs)
        # exactly; rebuilding them per year priced the grid differently.
        for year in n.scenarios:
            networks_dict[str(year)] = n.get_scenario(year)
            n_names_dict[str(year)]  = f"{network_name}_WS_{year}"

    c.n_flags["print"]     = False
    c.n_flags_opt["print"] = False

else:
    networks_dict["network"] = n
    n_names_dict["network"]  = network_name

# ---- Solve all networks
for key, net in networks_dict.items():
    name = n_names_dict[key]
    print(f"Solving: {name}")
    status, condition, used_solver, used_opts, model = build_model_solve_network(
        net,
        results_folder    = results_folder,
        solver            = c.optimization["solver"],
        profile           = c.optimization["solver_profile"],
        n_config          = c.n_config,
        overrides         = c.optimization["overrides"],
        collect_all_duals = c.optimization["collect_all_duals"],
        return_model      = c.optimization["return_model"],
        n_name            = name,
    )
    networks_dict[key] = net

    # Load shedding: report undelivered demand, then drop generators never used.
    # The stochastic network keeps them: its components are indexed per scenario.
    unmet = unmet_deliveries(net)
    for gen, mwh in unmet[unmet > 0.01].items():
        print(f"WARNING: {mwh:,.1f} MWh of '{gen}' demand unmet in {name} "
              f"(load shedding at {c.targets_dict['load_shedding_cost']:g} EUR/MWh)")
    if key != "stoch":
        drop_unused_unmet(net)

def zero_small_capacities(n, threshold_mw):
    """Zero solver-noise artifacts before export.

    Mirrors pypsa-eur add_brownfield capacity_threshold logic but for
    single-period networks.  Extendable components whose p_nom_opt (or
    e_nom_opt for stores) is strictly below threshold_mw are treated as
    "not built": the optimal capacity is set to zero and all result
    timeseries columns for those components are zeroed so that downstream
    analysis (LCOP, plots, CSV exports) sees a clean network.
    """
    import pandas as pd

    _RESULT_TS = {
        "generators":    ["p"],
        "links":         ["p0", "p1", "p2", "p3", "p4"],
        "storage_units": ["p", "state_of_charge", "spill"],
        "stores":        ["p", "e"],
    }

    for c_name, attr in [
        ("generators",    "p"),
        ("links",         "p"),
        ("storage_units", "p"),
        ("stores",        "e"),
    ]:
        comp      = getattr(n, c_name)
        nom_opt   = f"{attr}_nom_opt"
        nom_ext   = f"{attr}_nom_extendable"
        if nom_opt not in comp.columns or nom_ext not in comp.columns:
            continue

        small = comp[nom_ext] & (comp[nom_opt].abs() < threshold_mw)
        small_idx = comp.index[small]
        if small_idx.empty:
            continue

        comp.loc[small_idx, nom_opt] = 0.0

        ts = getattr(n, f"{c_name}_t", None)
        if ts is not None:
            for ts_attr in _RESULT_TS.get(c_name, []):
                df = ts.get(ts_attr)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    cols = small_idx.intersection(df.columns)
                    if not cols.empty:
                        df.loc[:, cols] = 0.0

        print(f"[zero_small_capacities] {c_name}: zeroed {len(small_idx)} component(s) "
              f"with {nom_opt} < {threshold_mw} MW → {list(small_idx)}")


# ---- Export main (stoch or det) OPT network to the declared Snakemake output
main_key = "stoch" if c.stochastic["stochastic"] else "network"
n_solved = networks_dict[main_key]

_zero_th = float(c.optimization.get("zero_threshold_MW", 0.0))
if _zero_th > 0:
    zero_small_capacities(n_solved, _zero_th)

n_solved.export_to_netcdf(snakemake.output.network)

# Export WS networks if EVPI
for key, net in networks_dict.items():
    if key == main_key:
        continue
    if _zero_th > 0:
        zero_small_capacities(net, _zero_th)
    export_network(net, c.n_flags_opt, n_names_dict[key], networks_folder, "_OPT")

# Print OPT network topology diagram
print_network(
    n             = n_solved,
    n_flags       = c.n_flags_opt,
    nc_path       = snakemake.output.network,
    network_name  = network_name,
    suffix        = "_OPT",
    plot_folder   = plot_folder,
    is_stochastic = c.stochastic["stochastic"],
)

# ---- Save run metadata
save_config(results_folder, c)
save_network_comp_allocation(results_folder, network_comp_allocation)

# ---- EVPI
if c.stochastic["EVPI"]:
    df_evpi = compare_objective(
        networks_dict["stoch"],
        {k: v for k, v in networks_dict.items() if k != "stoch"},
        scenarios,
    )
    df_evpi.to_csv(Path(csv_folder) / "EVPI.csv")
    print(df_evpi)
