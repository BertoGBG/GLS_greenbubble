# SPDX-License-Identifier: MIT
"""Configuration loader: reads config files at import time.

Loading order (mirrors pypsa-eur):
  1. config/config.default.yaml          — committed base defaults
  2. config/config.yaml                  — user overrides (if present, gitignored)
  3. config/n_config.default.yaml        — network component defaults (includes options: section)
  4. config/n_config.yaml                — network overrides (if present, gitignored)
  5. config/plots_config.default.yaml    — plot defaults

All optimisation settings, technology flags, stochastic scenario definitions
and demand targets are exposed as module-level variables so that any script
can do::

    from scripts import config as c
    c.n_flags        # dict of active technology flags
    c.stochastic     # stochastic scenario config block
    c.optimization   # solver and profile settings

.. note::
   Modifying this module's variables at runtime (e.g., in Snakemake wrappers)
   affects all subsequent imports within the same process, which is the
   intended behaviour for EVPI wait-and-see runs.
"""

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

_CFG_DIR = Path(__file__).resolve().parent.parent / "config"



# Keys replaced wholesale on override rather than merged key-by-key: these
# represent a coherent, internally-consistent set keyed by scenario year
# (probabilities must sum to 1, and years must match across all three dicts).
# Deep-merging them would silently mix leftover default years into the
# user's override.
_REPLACE_WHOLESALE_KEYS = {"scenarios", "CO2_cost_s", "CO2_cost_ref_year_s"}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base in-place and return base."""
    for k, v in override.items():
        if k in _REPLACE_WHOLESALE_KEYS:
            base[k] = v
        elif k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _load_with_override(default_path: Path, user_path: Path) -> dict:
    """Load default YAML and deep-merge user override on top if it exists."""
    with default_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if user_path.exists():
        with user_path.open("r", encoding="utf-8") as f:
            _deep_merge(data, yaml.safe_load(f) or {})
    return data


# --- main run config ---
_cfg = _load_with_override(
    _CFG_DIR / "config.default.yaml",
    _CFG_DIR / "config.yaml",
)

# --- network component config (options: subsection holds former n_options.yaml) ---
_n_raw = _load_with_override(
    _CFG_DIR / "n_config.default.yaml",
    _CFG_DIR / "n_config.yaml",
)
_n_raw.pop("base", None)
_n_opt = _n_raw.pop("options", {})
_n_opt.pop("base", None)
n_config = pd.DataFrame.from_dict(_n_raw, orient="index").sort_index()
n_options = pd.DataFrame.from_dict(_n_opt, orient="index").sort_index()

# --- plots ---
plt_config = _load_with_override(
    _CFG_DIR / "plots_config.default.yaml",
    _CFG_DIR / "plots_config.yaml",
)

# ------  Expose optimization variables with the same name in the model (retro-compatibility)
run_name                 = _cfg["run_name"]
CO2_cost                 = _cfg["CO2_cost"]

# Targets
targets_dict              = dict(_cfg["targets"])

max_RE_to_grid           = _cfg["max_RE_to_grid"]
En_price_year            = _cfg["En_price_year"]

latitude                 = _cfg["latitude"]
longitude                = _cfg["longitude"]

n_flags                  = dict(_cfg["n_flags"])
n_flags_opt              = dict(_cfg["n_flags_opt"])
outputs_folder           = _cfg["outputs_folder"]

CO2_cost_ref_year        = _cfg["CO2_cost_ref_year"]

rfnbos_dict              = dict(_cfg["rfnbos_dict"])

year_investment          = _cfg["year_investment"]
_ap_raw                  = _cfg.get("amortization_period", None)
amortization_period      = None if (_ap_raw is None or str(_ap_raw).lower() in {"null", "none", ""}) else float(_ap_raw)
USD_to_EUR               = _cfg["USD_to_EUR"]
EUR_to_DKK                 = _cfg["EUR_to_DKK"]
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
    "zero_threshold_MW": float(_opt.get("zero_threshold_MW", 0.0)),
}

# Normalize common YAML -> Python edge cases
if isinstance(optimization["overrides"], str) and optimization["overrides"].lower() in {"none", "null", ""}:
    optimization["overrides"] = None

if isinstance(optimization["solver_profile"], str) and optimization["solver_profile"].strip() == "":
    optimization["solver_profile"] = None

# ---------------- Temporal clustering config ----------------

_cl = _cfg.get("clustering", {}) or {}
_cl_t = _cl.get("temporal", {}) or {}
clustering = {
    "temporal": {
        "resolution": _cl_t.get("resolution", False),
    }
}

# ---------------- Rolling horizon config ----------------

_rh = _cfg.get("rolling_horizon", {}) or {}
_rh_year_raw = _rh.get("rh_year", None)
rolling_horizon = {
    "enabled":      bool(_rh.get("enabled", False)),
    "horizon":      int(_rh.get("horizon", 168)),
    "overlap":      int(_rh.get("overlap", 24)),
    "network_path": _rh.get("network_path", None),
    "rh_year":      int(_rh_year_raw) if _rh_year_raw not in (None, "", "null") else None,
}

if rolling_horizon["enabled"] and not rolling_horizon["network_path"]:
    raise ValueError("rolling_horizon.network_path must be set when rolling_horizon.enabled is true.")

# ---------------- Near-optimal space (MGA) config ----------------

_mga = _cfg.get("mga", {}) or {}
_mga_rob = _mga.get("robustness", {}) or {}
_mga_adaptive = _mga.get("adaptive", {}) or {}
mga = {
    "enabled":            bool(_mga.get("enabled", False)),
    "network_path":       (_mga.get("network_path") or "").strip(),
    "dimensions":         list(_mga.get("dimensions", []) or []),
    "dimension_weight":   str(_mga.get("dimension_weight", "capacity")),
    "slack":              float(_mga.get("slack", 0.05)),
    "n_directions":       int(_mga.get("n_directions", 0)),
    "direction_sampling": str(_mga.get("direction_sampling", "halton")),
    "max_parallel":       int(_mga.get("max_parallel", 1)),
    "seed":               (None if _mga.get("seed", None) in (None, "", "null")
                           else int(_mga.get("seed"))),
    "robustness": {
        "enabled":    bool(_mga_rob.get("enabled", False)),
        "years":      dict(_mga_rob.get("years", {}) or {}),
        "cost_bound": str(_mga_rob.get("cost_bound", "max")),
    },
    # Adaptive Tier 2 (near-optimal_dev3's staged pipeline only — see
    # rules/near_optimal_staged.smk / scripts.near_optimal.explore_hull_adaptive).
    "adaptive": {
        "direction_method":    str(_mga_adaptive.get("direction_method", "maximal-centre-then-facets")),
        "direction_angle_sep": float(_mga_adaptive.get("direction_angle_sep", 15.0)),
        "angle_tolerance":     float(_mga_adaptive.get("angle_tolerance", 1.0)),
        "conv_method":         str(_mga_adaptive.get("conv_method", "volume")),
        "conv_eps":            float(_mga_adaptive.get("conv_eps", 2.0)),
        "conv_iter":           int(_mga_adaptive.get("conv_iter", 2)),
        "max_iter":            int(_mga_adaptive.get("max_iter", 20)),
    },
}

# ---------------- Plotting/Export config ----------------

plot_cfg = plt_config["plotting"]
items = plot_cfg["capacity_items"]
bus_list_mp = plot_cfg["bus_list_mp"]
carrier_colors = dict(plt_config.get("carrier_colors", {}))

