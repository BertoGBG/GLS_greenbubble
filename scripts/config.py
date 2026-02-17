# LOGIC for results structure

# results_folder = network(s) filename
# inside results_folder:
# - networks: dir
# - plots : dir
# network(s) filename  is defined a combination of :
# 1) n_flags, CO2_cost, demand_H2, demand_CH4, demand_meoh, max_RE_to_grid, stochastic, En_price_year (automatic)
# 2) run_name  (set by the user)

# ------------------------------------
import pandas as pd
import yaml
from pathlib import Path

'''Load configuration '''
#  --- optimization ----

CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with CFG_PATH.open("r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f) or {}

# ---  network configuration ---
n_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "n_config.yaml"
with n_CFG_PATH.open("r", encoding="utf-8") as f:
    n_cfg = yaml.safe_load(f) or {}

n_cfg.pop("base", None)
n_config = pd.DataFrame.from_dict(n_cfg, orient="index").sort_index()

# --- options  ---
n_OPT_PATH = Path(__file__).resolve().parent.parent / "config" / "n_options.yaml"
with n_OPT_PATH.open("r", encoding="utf-8") as f:
    n_opt = yaml.safe_load(f) or {}

# --- plots  ---
PLOTS_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "plots_config.yaml"
with PLOTS_CFG_PATH.open("r", encoding="utf-8") as f:
    plt_config = yaml.safe_load(f) or {}

n_opt.pop("base", None)
n_options = pd.DataFrame.from_dict(n_opt, orient="index").sort_index()

# ------  Expose optimization variables with the same name in the model (retro-compatibility)
run_name                 = _cfg["run_name"]
CO2_cost                 = _cfg["CO2_cost"]

# Targets
targets_dict              = dict(_cfg["targets"])

max_RE_to_grid           = _cfg["max_RE_to_grid"]
En_price_year            = _cfg["En_price_year"]
preprocess_flag          = _cfg["preprocess_flag"]

latitude                 = _cfg["latitude"]
longitude                = _cfg["longitude"]

n_flags                  = dict(_cfg["n_flags"])
n_flags_opt              = dict(_cfg["n_flags_opt"])
outputs_folder           = _cfg["outputs_folder"]

if targets_dict['driver'] == 'demand':
    H2_profile_flag = False
    H2_delivery_frequency = 1
else:
    H2_profile_flag          = _cfg["H2_profile_flag"]
    H2_delivery_frequency    = _cfg["H2_delivery_frequency"]

CO2_cost_ref_year        = _cfg["CO2_cost_ref_year"]

rfnbos_dict              = dict(_cfg["rfnbos_dict"])

year_EU                  = _cfg["year_EU"]
USD_to_EUR               = _cfg["USD_to_EUR"]
DKK_Euro                 = _cfg["DKK_Euro"]
discount_rate            = _cfg["discount_rate"]

stochastic               = dict(_cfg["stochastic"])
if not stochastic['stochastic']:
    stochastic['EVPI'] = False

tariffs_dict             = dict(_cfg["tariffs_dict"])

# ---------------- Optimization config ----------------

_opt = _cfg.get("optimization", {}) or {}

optimization = {
    "solver": _opt.get("solver", None),
    "solver_profile": _opt.get("solver_profile", None),
    "overrides": _opt.get("overrides", None),
    "collect_all_duals": bool(_opt.get("collect_all_duals", False)),
    "return_model": bool(_opt.get("return_model", False)),
}

# Normalize common YAML -> Python edge cases
if isinstance(optimization["overrides"], str) and optimization["overrides"].lower() in {"none", "null", ""}:
    optimization["overrides"] = None

if isinstance(optimization["solver_profile"], str) and optimization["solver_profile"].strip() == "":
    optimization["solver_profile"] = None

# ---------------- Plotting/Export config ----------------

plot_cfg = plt_config["plotting"]
thresholds = plot_cfg["thresholds"]
items = plot_cfg["capacity_items"]
bus_list_mp = plot_cfg["bus_list_mp"]

# Replace symbolic thresholds (GEN_TH → numeric)
for it in items:
    th_key = it["th"]
    it["th"] = thresholds[th_key]

