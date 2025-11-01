# LOGIC for results structure

# results_folder = network(s) filename
# inside results_folder:
# - networks: dir
# - plots : dir
# network(s) filename  is defined a combination of :
# 1) n_flags, CO2_cost, demand_H2, demand_CH4, demand_meoh, el_DK1_sale_el_RFNBO, En_price_year (automatic)
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

n_opt.pop("base", None)
n_options = pd.DataFrame.from_dict(n_opt, orient="index").sort_index()


# ------  Expose optimization variables with the same name in the model (retro-compatibility)
run_name                 = _cfg["run_name"]
CO2_cost                 = _cfg["CO2_cost"]

flh_H2                   = _cfg["flh_H2"]
H2_output                = _cfg["H2_output"]
flh_Biogas               = _cfg["flh_Biogas"]
biogas_output            = _cfg["biogas_output"]
flh_meoh                 = _cfg["flh_meoh"]
meoh_output              = _cfg["meoh_output"]

# Derived demands (same formulas as before)
demand_H2                = flh_H2    * H2_output
demand_CH4               = flh_Biogas * biogas_output
demand_meoh              = flh_meoh  * meoh_output

el_DK1_sale_el_RFNBO     = _cfg["el_DK1_sale_el_RFNBO"]
En_price_year            = _cfg["En_price_year"]
preprocess_flag          = _cfg["preprocess_flag"]

latitude                 = _cfg["latitude"]
longitude                = _cfg["longitude"]

n_flags                  = dict(_cfg["n_flags"])      # stays a dict
n_flags_opt              = dict(_cfg["n_flags_opt"])      # stays a dict
outputs_folder           = _cfg["outputs_folder"]

H2_profile_flag          = _cfg["H2_profile_flag"]
H2_delivery_frequency    = _cfg["H2_delivery_frequency"]

CO2_cost_ref_year        = _cfg["CO2_cost_ref_year"]

rfnbos_dict              = dict(_cfg["rfnbos_dict"])  # stays a dict

year_EU                  = _cfg["year_EU"]
USD_to_EUR               = _cfg["USD_to_EUR"]
DKK_Euro                 = _cfg["DKK_Euro"]
discount_rate            = _cfg["discount_rate"]

share_bio_NG             = _cfg['share_bio_NG']

stochastic               = _cfg["stochastic"]
#--------------------------

