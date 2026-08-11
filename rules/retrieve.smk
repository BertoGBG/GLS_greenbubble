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
    Called once per year (En_price_year + all stochastic scenario years).
    Re-trigger manually with --forcerun preprocess_inputs if data needs refreshing.

    costs_eu is only actually read when pypsa_eur_link.enabled (methanol
    price's CO2-cost markup, see scripts/pypsa_eur_link.py), but is listed
    unconditionally so the DAG dependency on retrieve_tech_data is always
    correct regardless of config.
    """
    input:
        costs_eu = f"data/technology-data/outputs/costs_{YEAR_INVESTMENT}.csv",
    output:
        done = "data/Inputs_{year}/.preprocessed",
    log:
        "logs/preprocess_inputs_{year}.log",
    wildcard_constraints:
        year = r"\d{4}",
    script:
        "../scripts/snakemake_preprocess.py"
