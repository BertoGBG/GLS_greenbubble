# SPDX-License-Identifier: MIT
"""Snakemake wrapper: Tier 2 (nos_hull_adaptive) of the staged NOS pipeline.

Adaptive facet/Chebyshev-ball hull refinement on the current-config network,
seeded from nos_seed's real cardinal-direction points (not re-solved here).
See rules/near_optimal_staged.smk and scripts.near_optimal.explore_hull_adaptive.
"""
import json
from pathlib import Path
import sys

import pandas as pd
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

seed_points = pd.read_csv(snakemake.input.seed)

adaptive_cfg = mga.get("adaptive", {})
result = nos.explore_hull_adaptive(
    n, dimensions, slack=mga["slack"],
    direction_method=adaptive_cfg.get("direction_method", "maximal-centre-then-facets"),
    direction_angle_sep=adaptive_cfg.get("direction_angle_sep", 15.0),
    angle_tolerance=adaptive_cfg.get("angle_tolerance", 1.0),
    conv_method=adaptive_cfg.get("conv_method", "volume"),
    conv_eps=adaptive_cfg.get("conv_eps", 2.0),
    conv_iter=adaptive_cfg.get("conv_iter", 2),
    max_iter=adaptive_cfg.get("max_iter", 20),
    n_flags=n_flags, re_alpha=re_alpha,
    solver_name=solver, solver_options=solver_options,
    seed=mga["seed"], seed_points=seed_points,
)
result["points"].to_csv(snakemake.output.points, index=False)

summary = {
    "dimensions": result["dimensions"],
    "slack": mga["slack"],
    "c_opt": result["c_opt"],
    "iterations": result["iterations"],
    "converged": result["converged"],
    "volume": result["volume"],
    "n_points": len(result["points"]),
}
with open(snakemake.output.summary, "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"[nos_hull_adaptive] {summary['iterations']} iterations, converged={summary['converged']}, "
      f"{summary['n_points']} points -> {snakemake.output.points}")
