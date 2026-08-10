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
import warnings
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

# ---------------- PyPSA-Eur soft-link config ----------------
# See scripts/pypsa_eur_link.py and docs/pypsa_eur_link.rst for the extraction
# logic and design rationale. Minimal file footprint by design: only a solved
# PyPSA-Eur network (.nc) and its onshore-regions GeoJSON are required.

_pel = _cfg.get("pypsa_eur_link", {}) or {}
pypsa_eur_link = {
    "enabled":               bool(_pel.get("enabled", False)),
    "network_path":          _pel.get("network_path", None),
    "regions_path":          _pel.get("regions_path", None),
    # Optional run id. Kept distinct from En_price_year on purpose:
    # En_price_year remains the PyPSA-Eur planning-horizon year (meaningful
    # for future multi-year transition studies chaining several PyPSA-Eur
    # networks), while "id" lets more than one soft-linked scenario for the
    # same year (different network/config) coexist without collision. See
    # scripts/pypsa_eur_link.py:inputs_folder.
    "id":                    str(_pel.get("id", "") or ""),
    # "average" (energy-weighted scalar, matching options.DH.price-style flat
    # config fields) or "timeseries" (full marginal_price series) for the
    # "co2 stored" bus price GB sells captured CO2 into.
    "co2_stored_price_mode": _pel.get("co2_stored_price_mode", "average"),
    # Per-series overrides -- each defaults to true (fully soft-linked); set
    # any to false in config.yaml to keep config.yaml/n_config.yaml's own
    # value for that series instead of the PyPSA-Eur-extracted one.
    "override_co2_cost":            bool(_pel.get("override_co2_cost", True)),
    "override_solid_biomass_price": bool(_pel.get("override_solid_biomass_price", True)),
    "override_DH_price":            bool(_pel.get("override_DH_price", True)),
    "override_H2_price":            bool(_pel.get("override_H2_price", True)),
    "override_methanol_price":      bool(_pel.get("override_methanol_price", True)),
    "override_bioCH4_price":        bool(_pel.get("override_bioCH4_price", True)),
}

if pypsa_eur_link["enabled"]:
    if not pypsa_eur_link["network_path"]:
        raise ValueError("pypsa_eur_link.network_path must be set when pypsa_eur_link.enabled is true.")
    if not pypsa_eur_link["regions_path"]:
        raise ValueError("pypsa_eur_link.regions_path must be set when pypsa_eur_link.enabled is true.")
    if pypsa_eur_link["co2_stored_price_mode"] not in ("average", "timeseries"):
        raise ValueError("pypsa_eur_link.co2_stored_price_mode must be 'average' or 'timeseries'.")

    # A soft-linked run takes its economic environment (prices, CO2 cost) as
    # exogenous and fixed for the linked network's own planning horizon —
    # forcing every technology's own annuity length rather than an
    # independently-configured shortened recovery window keeps the payback
    # question ("does this clear its own capital cost under these prices")
    # well-posed. See economics-annuity / economics-payback in the docs.
    if amortization_period is not None:
        warnings.warn(
            f"pypsa_eur_link.enabled=true forces amortization_period to null "
            f"(config.yaml had {amortization_period!r}).",
            UserWarning, stacklevel=2,
        )
    amortization_period = None

    # clustering.temporal.resolution must be set to match the linked
    # network's own snapshot spacing, not left independently configurable —
    # resampling already-resampled data is lossy in a hard-to-reason-about
    # way. The exact numeric match is validated where the network is loaded
    # anyway (scripts/snakemake_preprocess.py's pypsa_eur_link branch), not
    # here: config.py avoids loading a (potentially large, sector-coupled)
    # PyPSA-Eur network just to check one field.
    if clustering["temporal"]["resolution"] is False:
        raise ValueError(
            "clustering.temporal.resolution must be explicitly set (matching the "
            "linked PyPSA-Eur network's own resolution, e.g. '4h') when "
            "pypsa_eur_link.enabled is true."
        )

    from scripts.pypsa_eur_link import inputs_folder, read_scalars
    _pel_folder = inputs_folder(En_price_year, True, pypsa_eur_link["id"])

    # Point DH's price profile at the soft-link's own extracted rural-heat
    # price when DH is enabled -- the existing use_ts_price code path in
    # prepare_network.py's add_symbiosis (n_options["DH"]["price profile"] ->
    # CSV) picks it up with zero changes there. Only overrides if the user
    # hasn't already set their own explicit price profile (and only if
    # override_DH_price is true).
    if pypsa_eur_link["override_DH_price"] and "DH" in n_options.index and bool(n_options.at["DH", "enable"]):
        _existing_profile = n_options.at["DH", "price profile"]
        if _existing_profile in (None, "", "null") or (
            isinstance(_existing_profile, float) and pd.isna(_existing_profile)
        ):
            # "price profile" defaults to all-null (float64 dtype) -- cast to
            # object first so assigning a path string doesn't hit pandas'
            # incompatible-dtype FutureWarning (a hard error in future versions).
            if n_options["price profile"].dtype != object:
                n_options["price profile"] = n_options["price profile"].astype(object)
            n_options.at["DH", "price profile"] = f"{_pel_folder}/DH_price_input.csv"

    # CO2_cost and the pellets/moist-biomass market price: these are plain
    # scalars (no CSV/profile mechanism to read from disk like the series
    # above), so they're persisted via a small JSON sidecar written by
    # write_softlink_inputs alongside the CSVs (inputs_folder(...)/
    # pypsa_eur_link_scalars.json) -- read here rather than loading the full
    # PyPSA-Eur network again just for two numbers. Absent on a fresh
    # checkout before preprocess_inputs has run once; falls back to
    # config.yaml's own defaults with a warning in that case, since this
    # module is also imported *by* that same preprocessing step.
    if pypsa_eur_link["override_co2_cost"] or pypsa_eur_link["override_solid_biomass_price"]:
        _pel_scalars = read_scalars(_pel_folder)
        if _pel_scalars is None:
            warnings.warn(
                f"pypsa_eur_link.enabled=true but {_pel_folder}/pypsa_eur_link_scalars.json "
                "does not exist yet -- run the preprocess_inputs step first. Falling back to "
                "config.yaml's own CO2_cost and/or pellets/moist-biomass market prices for now.",
                UserWarning, stacklevel=2,
            )
        else:
            if pypsa_eur_link["override_co2_cost"]:
                CO2_cost = float(_pel_scalars["co2_cost"])
                # CO2_cost_ref_year exists to correct *historical* electricity
                # prices that already embed a specific CO2 tax, when modelling
                # a different assumed tax than that reference year's
                # (helpers.py: mk_el_grid_price = el_grid_price +
                # CO2_emiss_El * (CO2_cost - CO2_cost_ref_year)). That concept
                # doesn't apply to a soft-linked price: it comes straight out
                # of PyPSA-Eur's own CO2-constrained solve, so there is no
                # separate "reference year" -- CO2_cost IS the rate already
                # embedded in it. Setting them equal makes the correction
                # term zero regardless of CO2_emiss_El's value (itself
                # written as 0, since GB has no soft-linked grid-mix CO2
                # intensity yet -- see write_softlink_inputs).
                CO2_cost_ref_year = CO2_cost
            if pypsa_eur_link["override_solid_biomass_price"]:
                _solid_biomass_price = float(_pel_scalars["price_solid_biomass"])
                for _market in ("pellets market", "moist biomass market"):
                    if _market in n_options.index and bool(n_options.at[_market, "enable"]):
                        n_options.at[_market, "price"] = _solid_biomass_price

# ---------------- Plotting/Export config ----------------

plot_cfg = plt_config["plotting"]
items = plot_cfg["capacity_items"]
bus_list_mp = plot_cfg["bus_list_mp"]
carrier_colors = dict(plt_config.get("carrier_colors", {}))

