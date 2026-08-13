# SPDX-License-Identifier: MIT
"""Snakemake wrapper: Tier 1 (nos_seed) of the staged NOS pipeline.

Cardinal +/- unit-axis solves on the current-config network -> per-technology
ranges (same numbers explore_near_optimal's Tier 1 would report) plus the
real seed point set nos_hull_adaptive builds on (mga_ranges(return_points=True)
derives both from the same solves, one per cardinal direction).
See rules/near_optimal_staged.smk.
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
n = pypsa.Network(snakemake.input.network)

tech_costs = prepare_costs(
    latitude=c.latitude, longitude=c.longitude, tech_inputs=tech_inputs,
    USD_to_EUR=c.USD_to_EUR, discount_rate=c.discount_rate,
    cost_path_EU=snakemake.input.costs_eu,
    cost_path_US=p.cost_path_US, dict_tech_US_EU=p.dict_tech_US_EU,
)
comp_tech_map = nos.build_comp_tech_map(n, tech_costs.index)

run_cfg = nos.load_run_config(snakemake.input.network) if mga["network_path"] else {}
n_flags = run_cfg.get("n_flags", c.n_flags)
re_alpha = run_cfg.get("max_RE_to_grid", c.max_RE_to_grid)

dimensions = nos.resolve_dimensions(n, mga["dimensions"], comp_tech_map, c.n_config.index,
                                     weight_by=mga.get("dimension_weight", "capacity"))
solver = c.optimization["solver"]
profile = c.optimization["solver_profile"]
solver_options = get_solver_options(solver, profile) if profile else None

ranges, seed_points = nos.mga_ranges(
    n, dimensions, slack=mga["slack"], n_flags=n_flags, re_alpha=re_alpha,
    solver_name=solver, solver_options=solver_options, return_points=True,
)
ranges.to_csv(snakemake.output.ranges)
seed_points.to_csv(snakemake.output.points, index=False)

print(f"[nos_seed] ranges + {len(seed_points)} real seed points -> {snakemake.output.ranges}, {snakemake.output.points}")
