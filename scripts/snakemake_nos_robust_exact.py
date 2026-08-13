# SPDX-License-Identifier: MIT
"""Snakemake wrapper: nos_robust_exact (Tier 3b) of the staged NOS pipeline.

Realises the intersection's Chebyshev centre as a full network design: fixes
each dimension's aggregate capacity at the centre value and re-solves the
cost-minimising model under budget (scripts.near_optimal.realise_design),
mirroring the reference implementation's compute_robust_exact.py.

Note: unlike the reference implementation (which re-solves jointly across all
weather years to check feasibility everywhere at once), this realises against
a single reference network (the current-config NOS_NET_IN) -- a single-network
simplification, not a simultaneous multi-year feasibility check. Their
'conservative' / 'mean' / 'naive' heuristic capacity-allocation strategies
(scripts.vendor... BasisCapacities.scale/shift in the original) are not ported
here either. See rules/near_optimal_staged.smk.
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

with open(snakemake.input.centre) as f:
    centre_data = json.load(f)
if not centre_data["feasible_intersection"] or centre_data["centre"] is None:
    raise RuntimeError(
        "Intersection is infeasible (per-year near-optimal spaces do not overlap "
        f"at slack={mga['slack']}) — no robust design to realise. "
        f"nos_intersect summary: {centre_data}"
    )
centre = pd.Series(centre_data["centre"])

status, condition = nos.realise_design(
    n, dimensions, centre, slack=mga["slack"],
    n_flags=n_flags, re_alpha=re_alpha,
    solver_name=solver, solver_options=solver_options,
)
if status != "ok":
    raise RuntimeError(f"realise_design failed: {status} / {condition}")

n.export_to_netcdf(snakemake.output.network)
print(f"[nos_robust_exact] realised at centre {centre.to_dict()} -> {snakemake.output.network}")
