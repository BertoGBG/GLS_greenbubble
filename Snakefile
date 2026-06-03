import re
from pathlib import Path

# Load base defaults first, then user override if it exists (mirrors pypsa-eur pattern).
# To customise a run: copy config/config.default.yaml → config/config.yaml and edit.
configfile: "config/config.default.yaml"

if Path("config/config.yaml").exists():
    configfile: "config/config.yaml"


# ---------------------------------------------------------------------------
# Network name helper — mirrors file_name_network() in scripts/helpers.py
# but reads purely from config so the full path is known before any rule runs.
#
# Format: {run_name}_{year}_{det|stc}_{res}
# Examples: tut1_price_2024_det_3h   tut4_stoch_2024_stc_3h
#
# Inspired by PyPSA-EUR: run_name carries the semantic label; year/mode/res
# add minimum disambiguation.  The full configuration is stored inside the
# network .nc file so nothing is lost.  This keeps folder names short enough
# to stay within Windows' 260-character path limit.
# ---------------------------------------------------------------------------

def build_network_name(cfg):
    run  = cfg["run_name"]
    year = cfg["En_price_year"]
    stch = "stc" if cfg["stochastic"]["stochastic"] else "det"
    resolution = (cfg.get("clustering") or {}).get("temporal", {}).get("resolution", False)
    res  = resolution if resolution else "1h"
    return f"{run}_{year}_{stch}_{res}"


NETWORK          = build_network_name(config)
NETWORK_PATTERN  = re.escape(NETWORK)   # literal regex for wildcard_constraints
YEAR_INVESTMENT  = config["year_investment"]
YEAR             = config["En_price_year"]
TECH_DATA_YEARS  = [2020, 2025, 2030, 2035, 2040, 2045, 2050]
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
    # Check whether any technology-data CSV has changed since the last download.
    # If the git blob SHA differs from the cached value, delete the local file so
    # Snakemake sees a missing output and re-runs retrieve_tech_data automatically.
    import sys as _sys
    import os as _os
    _sys.path.insert(0, ".")
    from scripts.retrieve import fetch_remote_sha, get_cached_sha
    from scripts import parameters as _p

    for _yr in TECH_DATA_YEARS:
        _csv = f"data/technology-data/outputs/costs_{_yr}.csv"
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
