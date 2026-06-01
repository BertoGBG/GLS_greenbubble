# rules/solve.smk
# Step 6 – solve optimisation (deterministic or stochastic)
#          also handles EVPI wait-and-see networks when configured


rule solve_network:
    """Solve the network; export OPT .nc and EVPI CSV (if stochastic EVPI enabled)."""
    input:
        network    = f"resources/{RUN_DIR}/{{network}}_PRE.nc",
        costs_eu   = f"data/technology-data/outputs/costs_{YEAR_INVESTMENT}.csv",
        comp_alloc = f"resources/{RUN_DIR}/{{network}}_comp_alloc.pkl",
    output:
        network = f"{OUTDIR}/{RUN_DIR}/networks/{{network}}_OPT.nc",
    log:
        f"logs/solve_{{network}}.log",
    wildcard_constraints:
        network = NETWORK_PATTERN,
    script:
        "../scripts/snakemake_solve.py"


rule explore_near_optimal:
    """Near-optimal space (MGA) exploration on a solved network.

    Tier 1 (ranges) always; Tier 2 (hull) when mga.n_directions > 0;
    Tier 3 (robustness) when mga.robustness.enabled (auto-skipped on stochastic
    networks). Gated into `rule all` only when config['mga']['enabled'] is true.

    Target network is NOS_NET_IN (resolved in the Snakefile): the current-config
    *_OPT.nc when mga.network_path is empty, else the given pre-solved network.
    Paths are concrete (no wildcards) since there is one NOS target per run.
    """
    input:
        network  = NOS_NET_IN,
        costs_eu = f"data/technology-data/outputs/costs_{YEAR_INVESTMENT}.csv",
    output:
        ranges  = f"{NOS_OUT_DIR}/ranges.csv",
        points  = f"{NOS_OUT_DIR}/points.csv",
        summary = f"{NOS_OUT_DIR}/summary.json",
        done    = f"{NOS_OUT_DIR}/.done",
    log:
        "logs/nos.log",
    script:
        "../scripts/snakemake_near_optimal.py"
