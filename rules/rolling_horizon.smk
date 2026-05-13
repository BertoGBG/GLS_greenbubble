# rules/rolling_horizon.smk
# Dispatch-only rolling horizon optimisation on a fixed-capacity network.
# Only active when rolling_horizon.enabled: true in config.yaml.
# ruleorder: solve_rolling_horizon > solve_network resolves the ambiguity that
# arises because both rules produce {network}_OPT.nc for the _RH network name.

_rh = config.get("rolling_horizon", {}) or {}

if _rh.get("enabled", False):

    # Collect extra preprocessed-inputs dependency when rh_year differs from the
    # main optimisation year (YEAR is already preprocessed by the normal pipeline).
    _rh_year_raw = _rh.get("rh_year", None)
    _RH_YEAR = int(_rh_year_raw) if _rh_year_raw not in (None, "", "null") else YEAR
    _rh_extra_inputs = (
        [f"data/Inputs_{_RH_YEAR}/.preprocessed"]
        if _RH_YEAR != YEAR else []
    )

    rule solve_rolling_horizon:
        """Dispatch-only rolling horizon solve on a provided fixed-capacity network.

        When rh_year differs from En_price_year a fresh network is built from
        year-2 inputs; capacities are then transferred from the OPT network before
        the rolling-horizon dispatch runs.
        """
        input:
            network      = _rh["network_path"],
            costs_eu     = f"data/technology-data/outputs/costs_{YEAR_EU}.csv",
            extra_inputs = _rh_extra_inputs,
        output:
            network = f"{OUTDIR}/{{network}}/networks/{{network}}_OPT.nc",
        log:
            f"logs/rolling_horizon_{{network}}.log",
        wildcard_constraints:
            network = NETWORK_PATTERN,
        script:
            "../scripts/snakemake_rolling_horizon.py"


    rule plot_rolling_horizon:
        """Generate full plots and PF vs RH comparison for a rolling horizon result."""
        input:
            network    = f"{OUTDIR}/{{network}}/networks/{{network}}_OPT.nc",
            network_pf = _rh["network_path"],
        output:
            done = f"{OUTDIR}/{{network}}/plots_rh/.done",
        log:
            f"logs/plot_rolling_horizon_{{network}}.log",
        wildcard_constraints:
            network = NETWORK_PATTERN,
        script:
            "../scripts/snakemake_plot_rolling_horizon.py"

    # When RH is enabled both solve_network and solve_rolling_horizon match the
    # same output path ({network}_OPT.nc).  This directive tells Snakemake to
    # always prefer the RH rule so the DAG is unambiguous.
    ruleorder: solve_rolling_horizon > solve_network
