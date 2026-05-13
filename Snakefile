import re

configfile: "config/config.yaml"


# ---------------------------------------------------------------------------
# Network name helper — mirrors file_name_network() in scripts/helpers.py
# but reads purely from config so the full path is known before any rule runs.
# file_name_network() is kept in helpers.py for standalone greenbubble_main.py use.
# ---------------------------------------------------------------------------

def build_network_name(cfg):
    nf = cfg["n_flags"]
    flag_map = [
        ("biogas",       "B_"),
        ("central_heat", "H_"),
        ("renewables",   "RE_"),
        ("electrolysis", "H2_"),
        ("meoh",         "MEOH_"),
        ("methanation",  "METH_"),
        ("symbiosis",    "SN_"),
        ("storage",      "ST_"),
    ]
    prefix = "".join(sfx for key, sfx in flag_map if nf.get(key, False))

    co2    = int(cfg["CO2_cost"])
    driver = cfg["targets"]["driver"]
    target = "tD" if driver == "demand" else "tP"

    if driver == "demand":
        H2   = int(cfg["targets"]["demand_H2"])   // 1000
        MeOH = int(cfg["targets"]["demand_meoh"]) // 1000
        CH4  = int(cfg["targets"]["demand_CH4"])  // 1000
    else:
        # price-mode: use configured price targets as name components
        H2   = int(cfg["targets"]["price_H2"])
        MeOH = int(cfg["targets"]["price_meoh"])
        CH4  = cfg["targets"]["price_bioCH4"]
        CH4  = int(CH4) if isinstance(CH4, (int, float)) else 0

    year = cfg["En_price_year"]
    el   = cfg["max_RE_to_grid"]
    stch = "STC" if cfg["stochastic"]["stochastic"] else "DET"
    run  = cfg["run_name"]

    return f"{prefix}CO2_{co2}_{target}_H2_{H2}_MeOH_{MeOH}_CH4_{CH4}_{year}_El_{el}_{stch}_{run}"


NETWORK         = build_network_name(config)
NETWORK_PATTERN = re.escape(NETWORK)   # literal regex for wildcard_constraints
YEAR_EU         = config["year_eu"] if "year_eu" in config else config["year_EU"]
YEAR            = config["En_price_year"]
OUTDIR          = config["outputs_folder"]

# Years that need preprocessed inputs: always the main year + all scenario years if stochastic
_stoch_cfg = config["stochastic"]
if _stoch_cfg["stochastic"]:
    PREPROCESS_YEARS = sorted(set(
        [str(YEAR)] + [str(y) for y in _stoch_cfg["scenarios"].keys()]
    ))
else:
    PREPROCESS_YEARS = [str(YEAR)]


_rh_cfg    = config.get("rolling_horizon", {}) or {}
RH_ENABLED = bool(_rh_cfg.get("enabled", False))
_rh_year_raw = _rh_cfg.get("rh_year", None)
RH_YEAR    = int(_rh_year_raw) if _rh_year_raw not in (None, "", "null") else YEAR

if RH_ENABLED:
    NETWORK         = NETWORK + "_RH"
    NETWORK_PATTERN = re.escape(NETWORK)

if RH_ENABLED and RH_YEAR != YEAR:
    PREPROCESS_YEARS = sorted(set(list(PREPROCESS_YEARS) + [str(RH_YEAR)]))

onstart:
    # Check whether the remote technology-data CSV has changed since the last
    # download.  If the git blob SHA differs from the cached value, delete the
    # local file so Snakemake sees a missing output and re-runs retrieve_tech_data
    # automatically — without any --forcerun needed.
    import sys as _sys
    import os as _os
    _sys.path.insert(0, ".")
    from scripts.retrieve import fetch_remote_sha, get_cached_sha
    from scripts import parameters as _p

    _csv = f"data/technology-data/outputs/costs_{YEAR_EU}.csv"
    _fname = _os.path.basename(_csv)
    _remote = fetch_remote_sha(_p.technology_data_url, _fname)
    _cached = get_cached_sha(_csv)

    if _remote is None:
        print(f"[onstart] Could not reach GitHub API — skipping staleness check for {_fname}.")
    elif _remote != _cached:
        short_old = _cached[:8] if _cached else "none"
        print(f"[onstart] {_fname} changed on remote (SHA {short_old}... → {_remote[:8]}...). Removing local copy to force re-download.")
        if _os.path.exists(_csv):
            _os.remove(_csv)
        _sha_file = _csv + ".sha"
        if _os.path.exists(_sha_file):
            _os.remove(_sha_file)
    else:
        print(f"[onstart] {_fname} is up-to-date (SHA {_remote[:8]}...).")


include: "rules/retrieve.smk"
include: "rules/build.smk"
include: "rules/solve.smk"
include: "rules/plot.smk"
include: "rules/rolling_horizon.smk"


rule all:
    input:
        expand(
            "{outdir}/{network}/plots_rh/.done",
            outdir=OUTDIR, network=NETWORK,
        ) if RH_ENABLED else
        expand("{outdir}/{network}/plots/.done", outdir=OUTDIR, network=NETWORK),
    default_target: True
