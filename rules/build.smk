# rules/build.smk
# Step 3 – build technology-cost DataFrame
# Step 4 – assemble network input dictionary
# Step 5 – build PyPSA network (+ stochastic scenarios)


rule prepare_costs:
    """Build tech_costs DataFrame from retrieved cost CSV (EU/US blend by location)."""
    input:
        costs_eu = f"data/technology-data/outputs/costs_{YEAR_EU}.csv",
    output:
        tech_costs = "resources/tech_costs.pkl",
    log:
        "logs/prepare_costs.log",
    script:
        "../scripts/snakemake_prepare_costs.py"


rule prepare_inputs:
    """Load and assemble all network inputs: timeseries, demands, efficiencies."""
    input:
        done = f"data/Inputs_{YEAR}/.preprocessed",
    output:
        inputs = f"resources/inputs_{YEAR}.pkl",
    log:
        f"logs/prepare_inputs_{YEAR}.log",
    script:
        "../scripts/snakemake_prepare_inputs.py"


rule build_network:
    """Build PyPSA network; add stochastic scenarios if configured. Saves PRE network."""
    input:
        tech_costs = "resources/tech_costs.pkl",
        inputs     = f"resources/inputs_{YEAR}.pkl",
    output:
        network    = "resources/{network}_PRE.nc",
        comp_alloc = "resources/{network}_comp_alloc.pkl",
    log:
        "logs/build_network_{network}.log",
    wildcard_constraints:
        network = NETWORK_PATTERN,
    script:
        "../scripts/snakemake_build_network.py"
