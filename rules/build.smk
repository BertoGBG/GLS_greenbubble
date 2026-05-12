# rules/build.smk
# Step 3 – assemble network input dictionary
# Step 4 – build PyPSA network (+ stochastic scenarios)


rule prepare_inputs:
    """Load and assemble all network inputs: timeseries, demands, efficiencies.
    Waits for all scenario years to be preprocessed before assembling.
    """
    input:
        done = expand("data/Inputs_{year}/.preprocessed", year=PREPROCESS_YEARS),
    output:
        inputs = f"resources/inputs_{YEAR}.pkl",
    log:
        f"logs/prepare_inputs_{YEAR}.log",
    script:
        "../scripts/snakemake_prepare_inputs.py"


rule build_network:
    """Build PyPSA network; add stochastic scenarios if configured. Saves PRE network."""
    input:
        costs_eu = f"data/technology-data/outputs/costs_{YEAR_EU}.csv",
        inputs   = f"resources/inputs_{YEAR}.pkl",
    output:
        network    = "resources/{network}_PRE.nc",
        comp_alloc = "resources/{network}_comp_alloc.pkl",
    params:
        plot_folder = f"{OUTDIR}/{{network}}/plots",
    log:
        "logs/build_network_{network}.log",
    wildcard_constraints:
        network = NETWORK_PATTERN,
    script:
        "../scripts/snakemake_build_network.py"
