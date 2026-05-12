# SPDX-License-Identifier: MIT
"""Snakemake wrapper: build PyPSA network, add stochastic scenarios if configured.

Outputs:
  {network}_PRE.nc       — pre-optimisation network (NetCDF)
  {network}_comp_alloc.pkl — network_comp_allocation dict (pickle)
                             saved separately because PyPSA netcdf does not
                             preserve custom Python attributes.
"""
from pathlib import Path
import sys
import pickle
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prepare_network import network_dependencies, build_network
from scripts.helpers import assert_stochastic_schema_consistent, create_folder_if_not_exists, prepare_costs
from scripts.plots import print_network
from scripts import config as c, parameters as p
from scripts.technology_inputs import tech_inputs

tech_costs = prepare_costs(
    latitude        = c.latitude,
    longitude       = c.longitude,
    tech_inputs     = tech_inputs,
    USD_to_EUR      = c.USD_to_EUR,
    discount_rate   = c.discount_rate,
    cost_path_EU    = snakemake.input.costs_eu,
    cost_path_US    = p.cost_path_US,
    dict_tech_US_EU = p.dict_tech_US_EU,
)
with open(snakemake.input.inputs, "rb") as fh:
    inputs_dict = pickle.load(fh)

n_flags_OK = network_dependencies(c.n_flags)
n = build_network(tech_costs, inputs_dict, n_flags_OK, c.n_options, p)
n.consistency_check()

if c.stochastic["stochastic"]:
    from scripts.create_stoch_scenarios import (
        create_scenarios, scenarios, CO2_cost_s, CO2_cost_ref_year_s,
    )
    create_scenarios(n, scenarios, CO2_cost_s, CO2_cost_ref_year_s, n_flags_OK, tech_costs)
    assert_stochastic_schema_consistent(n, where="after create_scenarios")

n.export_to_netcdf(snakemake.output.network)

# Build enriched comp_alloc payload:
#   allocation      — plant → {generators, links, stores, storage_units}
#   tech_mapping    — component name → tech_costs index key
#   tech_costs_used — tech_costs rows for all referenced technologies
_used_techs = sorted(set(n.comp_tech_map.values()) & set(tech_costs.index))
_tc_used = tech_costs.loc[_used_techs].dropna(axis=1, how="all") if _used_techs else pd.DataFrame()
with open(snakemake.output.comp_alloc, "wb") as fh:
    pickle.dump(
        {
            "allocation":      n.network_comp_allocation,
            "tech_mapping":    n.comp_tech_map,
            "tech_costs_used": _tc_used,
        },
        fh,
    )

plot_folder = create_folder_if_not_exists(
    str(Path(snakemake.params.plot_folder).parent), "plots"
)
print_network(
    n             = n,
    n_flags       = c.n_flags,
    nc_path       = snakemake.output.network,
    network_name  = snakemake.wildcards.network,
    suffix        = "_PRE",
    plot_folder   = plot_folder,
    is_stochastic = c.stochastic["stochastic"],
)
