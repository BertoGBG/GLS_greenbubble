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


include: "rules/retrieve.smk"
include: "rules/build.smk"
include: "rules/solve.smk"
include: "rules/plot.smk"


rule all:
    input:
        expand("{outdir}/{network}/plots/.done", outdir=OUTDIR, network=NETWORK),
    default_target: True
