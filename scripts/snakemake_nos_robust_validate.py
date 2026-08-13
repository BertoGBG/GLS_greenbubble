# SPDX-License-Identifier: MIT
"""Snakemake wrapper: nos_robust_validate of the staged NOS pipeline.

Simultaneous multi-year feasibility check (Grochowicz et al.'s validate_robust /
solve_operations.py + summarise_feasibility.py): fixes nos_robust_exact's
realised design non-extendable and re-solves pure operations against each
robustness year's own cost-optimal network, with a load-shedding safety valve
so a shortfall is a graded curtailment number rather than an opaque solver
"infeasible". See scripts/near_optimal.py's validate_design_across_years and
scripts/vendor/near_opt_feasibility.py. See rules/near_optimal_staged.smk.
"""
from pathlib import Path
import sys

import pypsa

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import config as c
from scripts import near_optimal as nos
from scripts.solver_profiles import get_solver_options

mga = c.mga
years = list(mga["robustness"]["years"].keys())
year_network_paths = dict(zip(years, snakemake.input.years))

realised_network = pypsa.Network(snakemake.input.realised)

run_cfg = nos.load_run_config(snakemake.input.network) if mga["network_path"] else {}
n_flags = run_cfg.get("n_flags", c.n_flags)
re_alpha = run_cfg.get("max_RE_to_grid", c.max_RE_to_grid)

solver = c.optimization["solver"]
profile = c.optimization["solver_profile"]
solver_options = get_solver_options(solver, profile) if profile else None

summary = nos.validate_design_across_years(
    realised_network, year_network_paths,
    n_flags=n_flags, re_alpha=re_alpha,
    solver_name=solver, solver_options=solver_options,
)
summary.to_csv(snakemake.output.summary)
print(f"[nos_robust_validate] feasibility across {years} -> {snakemake.output.summary}")
print(summary.to_string())
