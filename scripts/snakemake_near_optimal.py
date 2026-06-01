# SPDX-License-Identifier: MIT
"""Snakemake wrapper: near-optimal space (NOS / MGA) exploration.

Consumes a solved ``*_OPT.nc`` network and the ``mga`` config block, then runs:
  * Tier 1 — per-technology capacity ranges (always)
  * Tier 2 — near-optimal hull (when mga.n_directions > 0)
  * Tier 3 — robustness across years (when mga.robustness.enabled)

Writes ranges.csv, points.csv, summary.json, plots and a .done sentinel under
``{OUTDIR}/{network}/nos/``.
"""
import json
from pathlib import Path

import pandas as pd
import pypsa

from scripts import config as c, parameters as p
from scripts.helpers import prepare_costs
from scripts.solver_profiles import get_solver_options
from scripts.technology_inputs import tech_inputs
from scripts import near_optimal as nos

# ---- folders -------------------------------------------------------------- #
nos_dir = Path(snakemake.output.summary).parent
plot_dir = nos_dir / "plots"
nos_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

mga = c.mga
slack = mga["slack"]
solver = c.optimization["solver"]
profile = c.optimization["solver_profile"]
solver_options = get_solver_options(solver, profile) if profile else None

# ---- load solved network + rebuild tech mapping --------------------------- #
n = pypsa.Network(snakemake.input.network)

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
comp_tech_map = nos.build_comp_tech_map(n, tech_costs.index)
stochastic = nos.is_stochastic(n)

# When exploring an external network (mga.network_path), use the parameters that
# *that* network was solved with (from its config_run.yaml) for the re-applied
# custom RE-to-grid constraint — so a config mismatch can't silently distort the
# near-optimal space. Falls back to the live config for the coupled case.
run_cfg = nos.load_run_config(snakemake.input.network) if mga["network_path"] else {}
nos_n_flags  = run_cfg.get("n_flags", c.n_flags)
nos_re_alpha = run_cfg.get("max_RE_to_grid", c.max_RE_to_grid)
if mga["network_path"]:
    print(f"[NOS] config source: {'config_run.yaml' if run_cfg else 'LIVE config (config_run.yaml not found!)'} "
          f"| n_flags={ {k:v for k,v in nos_n_flags.items() if v} } | max_RE_to_grid={nos_re_alpha}")

dimensions = nos.resolve_dimensions(
    n, mga["dimensions"], comp_tech_map, c.n_config.index
)
dim_keys = list(dimensions)
print(f"[NOS] {'STOCHASTIC' if stochastic else 'deterministic'} network — "
      f"exploring dimensions: {dim_keys}  (slack={slack}, solver={solver}/{profile})")

summary: dict = {
    "dimensions": dim_keys,
    "slack": slack,
    "solver": solver,
    "profile": profile,
    "stochastic": bool(stochastic),
    "c_opt": nos.optimal_objective(n),   # optimal objective = near-optimal budget anchor
}

# ---- Tier 1: ranges ------------------------------------------------------- #
ranges = nos.mga_ranges(
    n, dimensions, slack=slack, n_flags=nos_n_flags, re_alpha=nos_re_alpha,
    solver_name=solver, solver_options=solver_options,
)
ranges.to_csv(snakemake.output.ranges)
nos.plot_ranges(ranges, slack, plot_dir / "nos_ranges.png")
summary["must_have"] = ranges.index[ranges["must_have"]].tolist()
summary["must_avoid"] = ranges.index[ranges["must_avoid"]].tolist()
print("[NOS] Tier 1 ranges:\n", ranges.to_string())

# ---- Tier 2: hull --------------------------------------------------------- #
optimal_point = ranges["optimal"]   # cost-optimum projection (from Tier 1)

points = pd.DataFrame(columns=dim_keys)
if mga["n_directions"] > 0:
    # reload a clean optimal network: the Tier-1 loop left an MGA solution in `n`
    n = pypsa.Network(snakemake.input.network)
    result = nos.explore_hull(
        n, dimensions, slack=slack,
        n_directions=mga["n_directions"], sampling=mga["direction_sampling"],
        seed=mga["seed"], n_flags=nos_n_flags, re_alpha=nos_re_alpha,
        solver_name=solver, solver_options=solver_options,
    )
    result["slack"] = slack
    points = result["points"]
    points.to_csv(snakemake.output.points, index=False)
    nos.plot_hull_projections(result, optimal_point, plot_dir / "nos_hull.png")
    summary["n_points"] = int(len(points))
    summary["hull_volume"] = result["volume"]
    print(f"[NOS] Tier 2: {len(points)} extreme points, hull volume={result['volume']}")
else:
    points.to_csv(snakemake.output.points, index=False)

# ---- Tier 3: robustness across years -------------------------------------- #
rob = mga["robustness"]
if rob["enabled"] and stochastic:
    print("[NOS] Tier 3 (robustness across years) is SKIPPED on a stochastic network: "
          "scenarios already span multiple years/conditions, so intersecting across "
          "years would double-count the same uncertainty axis. Tiers 1–2 explore the "
          "near-optimal space of the shared first-stage investment under expected cost.")
    summary["robustness"] = {"skipped": "stochastic network — scenarios already span years"}
elif rob["enabled"] and rob["years"]:
    from scripts.create_stoch_scenarios import set_input_paths
    from scripts.preprocessing import prepare_all_inputs
    from scripts.prepare_network import network_dependencies, build_network
    from scripts.helpers import build_model_solve_network

    n_flags_OK = network_dependencies(c.n_flags)

    # Pass 1: build + solve each year cost-optimal (keep networks + dims + c_opt).
    year_nets: dict[str, object] = {}
    year_dims: dict[str, dict] = {}
    per_year_copt: dict[str, float] = {}
    for year, ydict in rob["years"].items():
        print(f"[NOS] robustness year {year}: building + solving cost-optimal …")
        set_input_paths(p, year)
        inputs = prepare_all_inputs(
            targets_dict      = c.targets_dict,
            CO2_cost          = ydict.get("CO2_cost", c.CO2_cost),
            CO2_cost_ref_year = ydict.get("CO2_cost_ref_year", c.CO2_cost_ref_year),
            max_RE_to_grid    = c.max_RE_to_grid,
        )
        ny = build_network(tech_costs, inputs, n_flags_OK, c.n_options, p)
        build_model_solve_network(
            ny, results_folder=str(nos_dir), solver=solver, profile=profile,
            n_config=c.n_config, collect_all_duals=False, return_model=False,
            n_name=f"nos_{year}",
        )
        ctm_y = nos.build_comp_tech_map(ny, tech_costs.index)
        year_nets[str(year)] = ny
        year_dims[str(year)] = nos.resolve_dimensions(ny, dim_keys, ctm_y, c.n_config.index)
        per_year_copt[str(year)] = nos.optimal_objective(ny)   # optimal objective c_opt(i)

    # Shared budget anchor c* across years (Grochowicz eq. 9) or per-year optimum.
    # cost_bound 'max' uses the highest (least-negative) optimal objective across years.
    c_star = max(per_year_copt.values()) if rob["cost_bound"] == "max" else None
    print(f"[NOS] robustness cost_bound={rob['cost_bound']}, c*={c_star}")

    # Pass 2: explore each year's near-optimal hull (shared bound if requested).
    per_year_points: dict[str, pd.DataFrame] = {}
    for year, ny in year_nets.items():
        res_y = nos.explore_hull(
            ny, year_dims[year], slack=slack,
            n_directions=mga["n_directions"], sampling=mga["direction_sampling"],
            seed=mga["seed"], n_flags=n_flags_OK,
            solver_name=solver, solver_options=solver_options,
            c_opt=c_star,
        )
        per_year_points[year] = res_y["points"]

    # intersection + Chebyshev centre
    hulls = [pts.to_numpy() for pts in per_year_points.values() if len(pts) >= len(dim_keys) + 1]
    cheb = nos.chebyshev_centre(hulls, keys=dim_keys) if len(hulls) >= 1 else {
        "centre": None, "radius": float("nan"), "feasible": False}
    summary["robustness"] = {
        "years": list(per_year_points),
        "c_opt_per_year": per_year_copt,
        "chebyshev_radius": cheb["radius"],
        "feasible_intersection": bool(cheb["feasible"]),
        "centre": (cheb["centre"].to_dict() if cheb["centre"] is not None else None),
    }
    nos.plot_robustness(per_year_points, cheb["centre"], dim_keys, plot_dir / "nos_robustness.png")
    print(f"[NOS] Tier 3: Chebyshev radius={cheb['radius']}, feasible={cheb['feasible']}")

# ---- write summary + sentinel --------------------------------------------- #
with open(snakemake.output.summary, "w") as fh:
    json.dump(summary, fh, indent=2, default=str)
Path(snakemake.output.done).touch()
print(f"[NOS] done → {nos_dir}")
