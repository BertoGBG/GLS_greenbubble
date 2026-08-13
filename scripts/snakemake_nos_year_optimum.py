# SPDX-License-Identifier: MIT
"""Snakemake wrapper: nos_year_optimum (wildcard {year}) of the staged NOS pipeline.

Builds + solves one mga.robustness.years cost-optimal network, mirroring the
reference implementation's compute_optimum.py -- one instance per robustness
year instead of per weather year. See rules/near_optimal_staged.smk.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import config as c, parameters as p
from scripts.helpers import prepare_costs, build_model_solve_network
from scripts.technology_inputs import tech_inputs
from scripts.create_stoch_scenarios import set_input_paths
from scripts.preprocessing import prepare_all_inputs
from scripts.prepare_network import network_dependencies, build_network
from scripts import near_optimal as nos

year = snakemake.wildcards.year
ydict = c.mga["robustness"]["years"][year]

tech_costs = prepare_costs(
    latitude=c.latitude, longitude=c.longitude, tech_inputs=tech_inputs,
    USD_to_EUR=c.USD_to_EUR, discount_rate=c.discount_rate,
    cost_path_EU=snakemake.input.costs_eu,
    cost_path_US=p.cost_path_US, dict_tech_US_EU=p.dict_tech_US_EU,
)
n_flags_OK = network_dependencies(c.n_flags)

set_input_paths(p, year)
inputs = prepare_all_inputs(
    targets_dict=c.targets_dict,
    CO2_cost=ydict.get("CO2_cost", c.CO2_cost),
    CO2_cost_ref_year=ydict.get("CO2_cost_ref_year", c.CO2_cost_ref_year),
    max_RE_to_grid=c.max_RE_to_grid,
)
ny = build_network(tech_costs, inputs, n_flags_OK, c.n_options, p)

solver = c.optimization["solver"]
profile = c.optimization["solver_profile"]
build_model_solve_network(
    ny, results_folder=str(Path(snakemake.output.network).parent),
    solver=solver, profile=profile,
    n_config=c.n_config, collect_all_duals=False, return_model=False,
    n_name=f"nos_year_{year}",
)
ny.export_to_netcdf(snakemake.output.network)

c_opt = nos.optimal_objective(ny)
with open(snakemake.output.obj, "w") as f:
    f.write(str(c_opt))

print(f"[nos_year_optimum] {year}: c_opt={c_opt:.1f} -> {snakemake.output.network}")
