# rules/solve.smk
# Step 6 – solve optimisation (deterministic or stochastic)
#          also handles EVPI wait-and-see networks when configured


rule solve_network:
    """Solve the network; export OPT .nc and EVPI CSV (if stochastic EVPI enabled)."""
    input:
        network    = "resources/{network}/{network}_PRE.nc",
        costs_eu   = f"data/technology-data/outputs/costs_{YEAR_INVESTMENT}.csv",
        comp_alloc = "resources/{network}/{network}_comp_alloc.pkl",
    output:
        network = f"{OUTDIR}/{{network}}/networks/{{network}}_OPT.nc",
    log:
        f"logs/solve_{{network}}.log",
    wildcard_constraints:
        network = NETWORK_PATTERN,
    script:
        "../scripts/snakemake_solve.py"
