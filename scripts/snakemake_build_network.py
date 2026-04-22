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

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prepare_network import network_dependencies, build_network
from scripts.helpers import assert_stochastic_schema_consistent
from scripts import config as c, parameters as p

with open(snakemake.input.tech_costs, "rb") as fh:
    tech_costs = pickle.load(fh)
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

with open(snakemake.output.comp_alloc, "wb") as fh:
    pickle.dump(n.network_comp_allocation, fh)
