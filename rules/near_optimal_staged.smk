# rules/near_optimal_staged.smk
#
# Staged, per-tier / per-year alternative to `explore_near_optimal`
# (rules/solve.smk), mirroring the rule decomposition of the Grochowicz et
# al. (2023) reference pipeline (compute_optimum -> mga -> compute_near_opt
# -> intersect_near_optimal -> compute_robust_*):
# https://github.com/aleks-g/intersecting-near-opt-spaces
#
# Each stage is its own Snakemake rule with an intermediate file output, so
# e.g. re-running the intersection after tweaking the Chebyshev-centre logic
# does not require re-solving three years of networks. The monolithic
# `explore_near_optimal` rule is untouched and remains the default path;
# these rules are reached only by explicit target name, e.g.:
#
#   snakemake --cores 4 nos_robust_exact
#
# Tier 2 here always uses the *adaptive* sampler (scripts/near_optimal.py's
# explore_hull_adaptive, built on the vendored scripts/vendor/near_opt_geometry.py)
# rather than explore_near_optimal's fixed-count Halton sampling -- that is
# the other half of the near-optimal_dev3 work this rule set exists to expose.
#
# Requires mga.enabled and mga.robustness.enabled (reads the same `mga:`
# config block as explore_near_optimal; NOS_NET_IN / NOS_OUT_DIR / MGA_ENABLED
# are defined once in the main Snakefile).

NOS_YEARS = list(_mga_cfg.get("robustness", {}).get("years", {}).keys()) if MGA_ENABLED else []
NOS_STAGED_DIR = f"{NOS_OUT_DIR}/staged"


rule nos_seed:
    """Tier 1: cardinal +/- unit-axis solves -> per-technology ranges + seed points.

    Seeds the adaptive Tier-2 sampler exactly like the reference implementation's
    `mga.py` (cardinal directions first, before adaptive refinement).
    """
    input:
        network  = NOS_NET_IN,
        costs_eu = f"data/technology-data/outputs/costs_{YEAR_INVESTMENT}.csv",
    output:
        ranges = f"{NOS_STAGED_DIR}/ranges.csv",
        points = f"{NOS_STAGED_DIR}/seed_points.csv",
    log:
        "logs/nos_staged_seed.log",
    script:
        "../scripts/snakemake_nos_seed.py"


rule nos_hull_adaptive:
    """Tier 2: adaptive facet/Chebyshev-ball hull refinement on the current-config
    network, seeded from nos_seed. Uses explore_hull_adaptive (vendored sampler)."""
    input:
        network  = NOS_NET_IN,
        seed     = f"{NOS_STAGED_DIR}/seed_points.csv",
        costs_eu = f"data/technology-data/outputs/costs_{YEAR_INVESTMENT}.csv",
    output:
        points  = f"{NOS_STAGED_DIR}/hull_points.csv",
        summary = f"{NOS_STAGED_DIR}/hull_summary.json",
    log:
        "logs/nos_staged_hull_adaptive.log",
    script:
        "../scripts/snakemake_nos_hull_adaptive.py"


rule nos_year_optimum:
    """Per-year cost-optimal network for the robustness tier (wildcard: {year}).

    Mirrors the reference implementation's `compute_optimum.py`, one instance
    per `mga.robustness.years` entry instead of per weather year.
    """
    output:
        network = f"{NOS_STAGED_DIR}/years/{{year}}_optimum.nc",
        obj     = f"{NOS_STAGED_DIR}/years/{{year}}_obj.txt",
    log:
        "logs/nos_staged_year_optimum_{year}.log",
    script:
        "../scripts/snakemake_nos_year_optimum.py"


rule nos_cost_bound:
    """Shared budget anchor c* = max_i c_opt(i) across robustness years.

    Grochowicz et al. eq. 9 / their `calc_obj_bound.py`: c_bound = (1+eps)*max(objs).
    GreenBubble applies the (1+eps) part inside each year's budget constraint
    (_add_budget_constraint), so this rule only computes the shared max(objs)=c*
    that gets passed through as that budget's anchor.
    """
    input:
        objs = expand(f"{NOS_STAGED_DIR}/years/{{year}}_obj.txt", year=NOS_YEARS),
    output:
        bound = f"{NOS_STAGED_DIR}/cost_bound.txt",
    log:
        "logs/nos_staged_cost_bound.log",
    script:
        "../scripts/snakemake_nos_cost_bound.py"


rule nos_year_hull:
    """Adaptive Tier-2 hull for one robustness year, under the shared c* bound
    (wildcard: {year})."""
    input:
        network  = f"{NOS_STAGED_DIR}/years/{{year}}_optimum.nc",
        bound    = f"{NOS_STAGED_DIR}/cost_bound.txt",
        costs_eu = f"data/technology-data/outputs/costs_{YEAR_INVESTMENT}.csv",
    output:
        points = f"{NOS_STAGED_DIR}/years/{{year}}_hull_points.csv",
    log:
        "logs/nos_staged_year_hull_{year}.log",
    script:
        "../scripts/snakemake_nos_year_hull.py"


rule nos_intersect:
    """Tier 3a: intersect all robustness years' hulls; Chebyshev centre of the
    intersection (vendored scripts.vendor.near_opt_geometry, via
    scripts.near_optimal.chebyshev_centre / intersect_hulls)."""
    input:
        points = expand(f"{NOS_STAGED_DIR}/years/{{year}}_hull_points.csv", year=NOS_YEARS),
    output:
        centre       = f"{NOS_STAGED_DIR}/intersection_centre.json",
        intersection = f"{NOS_STAGED_DIR}/intersection_points.csv",
        plot         = f"{NOS_STAGED_DIR}/plots/nos_staged_robustness.png",
    log:
        "logs/nos_staged_intersect.log",
    script:
        "../scripts/snakemake_nos_intersect.py"


rule nos_robust_exact:
    """Tier 3b realisation -- 'exact' strategy (Grochowicz et al.'s
    compute_robust_exact): fix aggregate capacities at the intersection's
    Chebyshev centre and re-solve the cost-minimising model under budget, on
    the current-config (NOS_NET_IN) network.

    Note: unlike the reference implementation, this realises against a single
    reference network, not a simultaneous multi-year dispatch check -- the
    'conservative' / 'mean' / 'naive' heuristic allocation strategies and a
    multi-year feasibility check are not yet ported (see the guide's
    near-optimal_dev3 notes).
    """
    input:
        centre   = f"{NOS_STAGED_DIR}/intersection_centre.json",
        network  = NOS_NET_IN,
        costs_eu = f"data/technology-data/outputs/costs_{YEAR_INVESTMENT}.csv",
    output:
        network = f"{NOS_STAGED_DIR}/robust_exact.nc",
    log:
        "logs/nos_staged_robust_exact.log",
    script:
        "../scripts/snakemake_nos_robust_exact.py"
