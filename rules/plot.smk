# rules/plot.smk
# Step 7 – generate analysis plots and CSV exports


rule plot_results:
    """Run run_plot_and_export on the solved network; touch .done marker on completion."""
    input:
        network    = f"{OUTDIR}/{{network}}/networks/{{network}}_OPT.nc",
        comp_alloc = "resources/{network}_comp_alloc.pkl",
    output:
        done = f"{OUTDIR}/{{network}}/plots/.done",
    log:
        f"logs/plot_{{network}}.log",
    wildcard_constraints:
        network = NETWORK_PATTERN,
    script:
        "../scripts/snakemake_plot.py"
