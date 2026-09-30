# rules/retrieve.smk
# Step 1 – fetch remote technology-cost CSV
# Step 2 – download and preprocess energy market data (prices, CFs, demands)


rule retrieve_tech_data:
    """Download all technology-cost CSVs (2020-2050) from remote repo if not already up-to-date."""
    output:
        costs = expand("data/technology-data/outputs/costs_{year}.csv", year=TECH_DATA_YEARS),
    log:
        "logs/retrieve_tech_data.log",
    script:
        "../scripts/snakemake_retrieve_tech.py"


rule preprocess_inputs:
    """Download and preprocess energy-market input data for a given year.
    {folder} is the input-data folder, e.g. data/Inputs_2024 or data/GB/Inputs_2024
    (see preprocessed_marker in the Snakefile); the year is its last four digits.
    Called once per year (En_price_year + all stochastic scenario years).
    Re-trigger manually with --forcerun preprocess_inputs if data needs refreshing.
    """
    output:
        done = "{folder}/.preprocessed",
    log:
        "logs/preprocess_inputs_{folder}.log",
    wildcard_constraints:
        folder = r"data/(\w+/)?Inputs_\d{4}",
    script:
        "../scripts/snakemake_preprocess.py"
