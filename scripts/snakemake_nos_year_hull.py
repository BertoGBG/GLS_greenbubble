# SPDX-License-Identifier: MIT
"""Snakemake wrapper: nos_year_hull (wildcard {year}) of the staged NOS pipeline.

Adaptive Tier-2 hull for one robustness year's cost-optimal network, under the
shared c* bound from nos_cost_bound. Unlike nos_hull_adaptive (which consumes
a separately-cached nos_seed), this seeds itself internally -- one seed rule
per robustness year would double the rule count in rules/near_optimal_staged.smk
for a caching benefit that matters less here (a failed year-hull run is rare
enough not to warrant separating its own seed stage out).
"""
from pathlib import Path
import sys

import pypsa

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import config as c, parameters as p
from scripts.helpers import prepare_costs
from scripts.solver_profiles import get_solver_options
from scripts.technology_inputs import tech_inputs
from scripts import near_optimal as nos

mga = c.mga
ny = pypsa.Network(snakemake.input.network)
year = snakemake.wildcards.year

tech_costs = prepare_costs(
    latitude=c.latitude, longitude=c.longitude, tech_inputs=tech_inputs,
    USD_to_EUR=c.USD_to_EUR, discount_rate=c.discount_rate,
    cost_path_EU=snakemake.input.costs_eu,
    cost_path_US=p.cost_path_US, dict_tech_US_EU=p.dict_tech_US_EU,
)
comp_tech_map = nos.build_comp_tech_map(ny, tech_costs.index)
n_flags_OK = c.n_flags  # nos_year_optimum already solved under network_dependencies(c.n_flags)

dimensions = nos.resolve_dimensions(ny, mga["dimensions"], comp_tech_map, c.n_config.index,
                                     weight_by=mga.get("dimension_weight", "capacity"))
solver = c.optimization["solver"]
profile = c.optimization["solver_profile"]
solver_options = get_solver_options(solver, profile) if profile else None

with open(snakemake.input.bound) as f:
    c_star = float(f.read())

adaptive_cfg = mga.get("adaptive", {})
result = nos.explore_hull_adaptive(
    ny, dimensions, slack=mga["slack"],
    direction_method=adaptive_cfg.get("direction_method", "maximal-centre-then-facets"),
    direction_angle_sep=adaptive_cfg.get("direction_angle_sep", 15.0),
    angle_tolerance=adaptive_cfg.get("angle_tolerance", 1.0),
    conv_method=adaptive_cfg.get("conv_method", "volume"),
    conv_eps=adaptive_cfg.get("conv_eps", 2.0),
    conv_iter=adaptive_cfg.get("conv_iter", 2),
    max_iter=adaptive_cfg.get("max_iter", 20),
    n_flags=n_flags_OK, solver_name=solver, solver_options=solver_options,
    seed=mga["seed"], c_opt=c_star,
)
result["points"].to_csv(snakemake.output.points, index=False)

print(f"[nos_year_hull] {year}: {result['iterations']} iterations, "
      f"converged={result['converged']}, {len(result['points'])} points -> {snakemake.output.points}")
