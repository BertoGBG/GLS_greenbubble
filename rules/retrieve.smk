# rules/retrieve.smk
# Step 1 – fetch remote technology-cost CSV
# Step 2 – download and preprocess energy market data (prices, CFs, demands)


rule retrieve_tech_data:
    """Download technology cost CSV from remote repo if not already up-to-date."""
    output:
        costs_eu = f"data/technology-data/outputs/costs_{YEAR_EU}.csv",
    log:
        f"logs/retrieve_tech_data_{YEAR_EU}.log",
    script:
        "../scripts/snakemake_retrieve_tech.py"


rule preprocess_inputs:
    """Download and preprocess energy-market input data for the run year.
    When config preprocess_flag is False the rule still creates the marker
    so downstream rules can proceed using existing CSVs in data/Inputs_{year}/.
    """
    output:
        done = f"data/Inputs_{YEAR}/.preprocessed",
    log:
        f"logs/preprocess_inputs_{YEAR}.log",
    script:
        "../scripts/snakemake_preprocess.py"
