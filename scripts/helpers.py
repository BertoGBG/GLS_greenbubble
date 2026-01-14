import pandas as pd
import numpy as np
import pypsatopo
from scripts.config import En_price_year, discount_rate, outputs_folder, CO2_cost_ref_year, max_RE_to_grid, targets_dict, run_name
from scripts import parameters as p
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import pickle as pkl
from scripts.solver_profiles import SOLVER_PROFILES
import yaml
from copy import deepcopy
import inspect, datetime as dt
import math
from pathlib import Path
import xarray as xr
import calendar
import os
import time


# -------NETWORK ----

def build_snapshots(year):
    start = f"{year}-01-01 00:00"
    end   = f"{year+1}-01-01 00:00"

    # tz-naive hourly index; end is excluded
    hours = pd.date_range(start, end, freq="h", inclusive="left")

    # optional: drop Feb 29 if your inputs don't have it
    if calendar.isleap(year):
        hours = hours[~((hours.month == 2) & (hours.day == 29))]

    return hours, start, end


# -------TECHNO-ECONOMIC DATA & ANNUITY
def annuity(n, r):
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20,0.05)*20 = 1.6"""

    if r > 0:
        return r / (1. - 1. / (1. + r) ** n)
    else:
        return 1 / n

def dict_to_costs_df(tech_inputs: dict, target_columns=None) -> pd.DataFrame:
    """
    Convert a dict like:
      {('technology','parameter'): {'value':..., 'unit':..., ...}, ...}
    into a MultiIndex DataFrame with index names ['technology','parameter'].

    Any missing target columns are added as NaN; extra keys are kept unless
    target_columns is provided (then we align to that set).
    """
    # Build DF from the inner dicts
    df_new = pd.DataFrame.from_dict(tech_inputs, orient='index')
    # Ensure a proper MultiIndex with names
    df_new.index = pd.MultiIndex.from_tuples(df_new.index, names=['technology', 'parameter'])

    # If you want to align exactly to the destination schema:
    if target_columns is not None:
        # add any missing columns from target
        for col in target_columns:
            if col not in df_new.columns:
                df_new[col] = np.nan
        # drop extra columns not in target (comment this if you want to keep them)
        df_new = df_new[target_columns]

    return df_new


def merge_into_costs(costs: pd.DataFrame, tech_inputs: dict, currency_year=None) -> pd.DataFrame:
    """
    Convert tech_inputs to DF and merge into 'costs':
      - overwrite existing rows for same (technology, parameter)
      - append brand new rows
    Optionally set currency_year for all new/updated rows.
    """
    target_cols = list(costs.columns)  # ['value','unit','source','further description','currency_year']
    df_new = dict_to_costs_df(tech_inputs, target_columns=target_cols)

    if currency_year is not None:
        df_new['currency_year'] = currency_year

    costs_out = costs.copy()

    # Append rows that are new (index not present in costs)
    new_idx = df_new.index.difference(costs_out.index)
    if len(new_idx) > 0:
        costs_out = pd.concat([costs_out, df_new.loc[new_idx]], axis=0)

    # Overwrite values for rows that already exist
    costs_out.update(df_new)

    # Ensure column order and types stay consistent
    costs_out = costs_out[target_cols]

    return costs_out


def prepare_costs(cost_path : str, tech_inputs: dict, USD_to_EUR: float , discount_rate : float, Nyears: int = 1, lifetime: int = 25):
    """ This function uses, data retrived form the technology catalogue and other sources and compiles a DF used in the model
    input:
    - cost_file. as downloaded from technology-data repository
    - tech_inputs. technical paramaters for various technolgies. usually stored in technology_inputs.py

    output: costs # DF with all cost used in the model"""

    # Nyear = nyear in the interval for myoptic optimization--> set to 1 for annual optimization

    # set all asset costs and other parameters

    costs_from_technology_data = pd.read_csv(cost_path, index_col=[0, 1]).sort_index()

    # add extra technologies and parameters
    if tech_inputs:
        costs= merge_into_costs(costs_from_technology_data, tech_inputs, currency_year=None)
    else:
        costs = costs_from_technology_data

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.loc[costs.unit.str.contains("USD"), "value"] *= USD_to_EUR

    # min_count=1 is important to generate NaNs which are then filled by fillna
    costs = costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)
    costs = costs.fillna({"CO2 intensity": 0,
                          "FOM": 0,
                          "VOM": 0,
                          "discount rate": discount_rate,
                          "efficiency": 1,
                          "fuel": 0,
                          "investment": 0,
                          "lifetime": lifetime
                          })
    annuity_factor = lambda v: annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100
    costs["fixed"] = [annuity_factor(v) * v["investment"] * Nyears for i, v in costs.iterrows()]
    return costs


def cost_add_technology(discount_rate, tech_costs, technology, investment, lifetime, FOM):
    '''function to calculate annualized fixed cost for any technology from inpits
    and adds it to the tech_costs dataframe '''
    annuity_factor = annuity(lifetime, discount_rate) + FOM / 100
    tech_costs.at[technology, "fixed"] = annuity_factor * investment
    tech_costs.at[technology, "lifetime"] = lifetime
    tech_costs.at[technology, "FOM"] = FOM
    tech_costs.at[technology, "investment"] = investment
    return tech_costs


def add_technology_cost(tech_costs, other_tech_costs):
    """Function that adds the tehcnology costs not present in the original cost file"""
    for idx in other_tech_costs.index.values:
        investment = other_tech_costs.at[idx, 'investment']
        FOM = other_tech_costs.at[idx, 'FOM']
        lifetime = other_tech_costs.at[idx, 'lifetime']
        cost_add_technology(discount_rate, tech_costs, idx, investment, lifetime, FOM)

    return tech_costs

def build_electricity_grid_price_w_tariff(Elspotprices):
    """this function creates the Electricity grid price including the all the tariffs
    Note that CO2 tax is added separately
    Tariff system valid for customer conected to 60kV grid via a 60/10kV transformer
    Tariff system in place from 2025"""

    # for tariff reference check the parameter file
    # Grid tariff are based on hour of the day, day of the week and season:
    # high tariff in summer + weekdays + 06:00 to 24.00
    # high tariff in winter + weekends + 06:00 to 24.00
    # high tariff in winter + weekdays + 21:00 to 24.00
    # peak tariff in winter + weekdays + 06:00 to 21.00
    # Low tariff the rest of the time

    summer_start = str(En_price_year) + '-04-01 00:00'  # '2019-04-01 00:00:00+00:00' # Monday
    summer_end = str(En_price_year) + '-10-01 00:00'  # '2019-10-01 00:00:00+00:00'
    winter_1 = pd.date_range(p.start_date , summer_start , freq='h')
    winter_1 = winter_1.drop(winter_1[-1])
    winter_2 = pd.date_range(summer_end , p.end_date, freq='h')
    winter_2 = winter_2.drop(winter_2[-1])
    winter = winter_1.append(winter_2)
    winter = winter[~((winter.month == 2) & (winter.day == 29))]
    summer = pd.date_range(summer_start, summer_end, freq='h')
    summer = summer.drop(summer[-1])

    peak_weekday = range(1, 6)
    peak_hours = range(7, 21 + 1)
    high_hours_weekday_winter = range(22, 24 + 1)
    high_hours_weekend_winter = range(7, 24 + 1)
    high_hours_weekday_summer = range(7, 24 + 1)

    # set the tariff in every hour equal to low and che
    el_grid_price = Elspotprices + p.el_transmission_tariff + p.el_system_tariff + p.el_afgift
    el_grid_sell_price = -Elspotprices + p.el_tariff_sell

    # assign tariff to hours
    for h in winter:
        day = h.weekday()
        hour = h.hour
        net_tariff = 0  # Default value

        if day in [5, 6]:  # weekends
            if hour in high_hours_weekend_winter:
                net_tariff = p.el_net_tariff_high
            else:
                net_tariff = p.el_net_tariff_low
        elif day in range(0, 5):  # weekdays
            if hour in peak_hours:
                net_tariff = p.el_net_tariff_peak
            elif hour in high_hours_weekday_winter:
                net_tariff = p.el_net_tariff_high
            else:
                net_tariff = p.el_net_tariff_low

        el_grid_price.loc[h, :] = el_grid_price.loc[h, :] + net_tariff

    for h in summer:
        day = h.weekday()
        hour = h.hour
        net_tariff = 0  # Default value

        if day in [5, 6]:  # weekends
            net_tariff = p.el_net_tariff_low
        elif day in range(0, 5):  # weekdays
            if hour in high_hours_weekday_summer:
                net_tariff = p.el_net_tariff_high
            else:
                net_tariff = p.el_net_tariff_low

        el_grid_price.loc[h, :] = el_grid_price.loc[h, :] + net_tariff

    return el_grid_price, el_grid_sell_price

def build_NG_grid_price_w_tariff(NG_price_year):
    ### function that adds NG TSO and DSO tariffs to the NG purchase price
    NG_grid_price = NG_price_year + p.NG_tso_tariff + p.NG_dso_tariff
    NG_sell_price = NG_price_year
    return NG_grid_price, NG_sell_price

def en_market_prices_w_CO2(inputs_dict, tech_costs, n_options):
    """Build market prices for electricity, natural gas, and district heating including CO₂ cost adjustments."""

    # --- 1. Base data from inputs_dict ---
    CO2_cost       = inputs_dict["CO2 cost"]
    CO2_emiss_El   = inputs_dict["CO2_emiss_El"]            # tCO2/MWh_el
    NG_price_year  = inputs_dict["NG_price_year"]           # €/MWh
    Elspotprices   = inputs_dict["Elspotprices"]            # Series or DataFrame

    # --- 2. Electricity grid prices (buy/sell) ---
    el_grid_price, el_grid_sell_price = build_electricity_grid_price_w_tariff(Elspotprices)

    # → FIX: both are DataFrames with one column, so flatten
    el_grid_price = el_grid_price.iloc[:, 0]
    el_grid_sell_price = el_grid_sell_price.iloc[:, 0]

    # --- 3. Apply CO₂ adjustment to electricity purchase price ---
    if isinstance(CO2_emiss_El, pd.DataFrame):
        CO2_emiss_El = CO2_emiss_El.iloc[:, 0]

    mk_el_grid_price = el_grid_price + CO2_emiss_El * (CO2_cost - CO2_cost_ref_year)
    mk_el_grid_sell_price = el_grid_sell_price.copy()

    # --- 4. Natural gas price (consumer pays CO₂ cost locally) ---
    co2_intensity_ng = tech_costs.at["gas", "CO2 intensity"]  # tCO₂/MWh_th

    NG_grid_price, NG_sell_price = build_NG_grid_price_w_tariff(NG_price_year)

    if isinstance(NG_grid_price, pd.DataFrame):
        NG_grid_price = NG_grid_price.iloc[:, 0]
    mk_NG_grid_price = NG_grid_price + co2_intensity_ng * (CO2_cost - CO2_cost_ref_year)
    mk_NG_grid_price = pd.Series(mk_NG_grid_price, index=el_grid_price.index)


    #if targets_dict.get("price_bioCH4", 0) == "NG_based":
    if isinstance(NG_sell_price, pd.DataFrame):
        NG_sell_price = NG_sell_price.iloc[:, 0]
    mk_NG_grid_sell_price = -1 * (NG_sell_price + co2_intensity_ng * (CO2_cost - CO2_cost_ref_year))
    mk_NG_grid_sell_price = pd.Series(mk_NG_grid_sell_price, index=el_grid_price.index)



    # --- 5. District heating price ---
    DH_price = pd.Series(
        -float(n_options.at["DH", "price"]),
        index=el_grid_price.index,
        name="DH_price",
    )

    # --- 6. Package results ---
    en_market_prices = {
        "el_grid_price": mk_el_grid_price,
        "el_grid_sell_price": mk_el_grid_sell_price,
        "NG_grid_price": mk_NG_grid_price,
        "bioCH4_grid_sell_price" : mk_NG_grid_sell_price ,
        "DH_price": DH_price,
    }

    return en_market_prices


# --- Add CUSTOM CONSTRAINTS ----

def add_el_grid_import_RFNBOs(inputs_dict, rfnbos_dict):
    """
    Return a 1-col DataFrame (0/1) limiting grid electricity for RFNBO compliance.

    inputs_dict must contain:
      - "Elspotprices": pd.Series or 1-col DataFrame
      - "CO2_emiss_El": pd.Series or 1-col DataFrame
    """

    Elspotprices   = inputs_dict["Elspotprices"]            # DataFrame
    CO2_emiss_El = inputs_dict['CO2_emiss_El']
    idx= Elspotprices.index

    # RFNBO rule
    limit = rfnbos_dict.get("limit", "unlimited")

    if limit == "unlimited":
        p_max_pu = pd.Series(1.0, index=idx)

    elif limit == "price":
        thr = rfnbos_dict["price_threshold"]
        p_max_pu = (Elspotprices <= thr).astype(float).fillna(0.0)
        p_max_pu = p_max_pu.iloc[:,0]

    elif limit == "emissions":
        thr = rfnbos_dict["emission_threshold"]
        p_max_pu = (CO2_emiss_El <= thr).astype(float).fillna(0.0)
        p_max_pu = p_max_pu.iloc[:,0]

    elif limit == "disconnected" :
        p_max_pu = pd.Series(0.0, index=idx)

    else:
        raise ValueError(f"Unknown RFNBO limit rule: {limit!r}")

    return p_max_pu

def add_custom_constraints_stores(n, n_config=None):
    """
    Enforce p_nom(link) <= alpha * e_nom(store) for selected charger/discharger links
    and their corresponding stores. Works whether variables are extendable or fixed.
    """
    m = n.model

    # ---- helpers to fetch variable slices or fall back to constants ----
    def _link_p_nom_expr(link_name):
        if "Link-p_nom" in m.variables:
            v = m.variables["Link-p_nom"]
            # v.dims typically: ('link',)
            if "link" in v.dims and link_name in v.coords["link"].values:
                return v.sel(link=link_name)
        # fall back to fixed parameter if the link exists
        if link_name in n.links.index:
            return float(n.links.at[link_name, "p_nom"])
        # otherwise skip
        return None

    def _store_e_nom_expr(store_name):
        if "Store-e_nom" in m.variables:
            v = m.variables["Store-e_nom"]
            # v.dims typically: ('store',)
            if "store" in v.dims and store_name in v.coords["store"].values:
                return v.sel(store=store_name)
        if store_name in n.stores.index:
            return float(n.stores.at[store_name, "e_nom"])
        return None

    def _add_bound(link_name, store_name, factor, tag):
        if not factor:
            return
        link_expr  = _link_p_nom_expr(link_name)
        store_expr = _store_e_nom_expr(store_name)
        if link_expr is None or store_expr is None:
            # Component missing or not part of the model; skip gracefully
            print(f"[custom-constraints] skip {tag}: missing "
                  f"{'link' if link_expr is None else 'store'} '{link_name if link_expr is None else store_name}'")
            return
        lhs = link_expr - float(factor) * store_expr
        cname = f"{tag}__{link_name}__le_{factor}__{store_name}"
        m.add_constraints(lhs <= 0, name=cname)

    # ---- read factors from config (if provided) ----
    def _cfg(at_tech, key, default=None):
        if n_config is None:
            return default
        try:
            return n_config.at[at_tech, key]
        except Exception:
            return default

    # ---- TES Concrete ----
    tes_store      = "TES Concrete storage"
    tes_charger    = "TES Concrete storage charger"
    tes_discharger = "TES Concrete storage discharger"
    if tes_store in n.stores.index:
        _add_bound(tes_charger,    tes_store, _cfg("TES concrete", "ramp limit up"),   "TES_concrete_charger_limit")
        _add_bound(tes_discharger, tes_store, _cfg("TES concrete", "ramp limit down"), "TES_concrete_discharger_limit")

    # ---- Water tank DH ----
    dh_store      = "Water tank DH storage"
    dh_charger    = "Water tank DH charger"
    dh_discharger = "Water tank DH discharger"
    if dh_store in n.stores.index:
        _add_bound(dh_charger,     dh_store, _cfg("TES DH", "ramp limit up"),   "Water_tank_DH_charger_limit")
        _add_bound(dh_discharger,  dh_store, _cfg("TES DH", "ramp limit down"), "Water_tank_DH_discharger_limit")

    # ---- Battery ----
    bat_store      = "battery"
    bat_charger    = "battery charger"
    bat_discharger = "battery discharger"
    if bat_store in n.stores.index:
        _add_bound(bat_charger,    bat_store, _cfg("battery", "ramp limit up"),   "battery_charger_limit")
        _add_bound(bat_discharger, bat_store, _cfg("battery", "ramp limit down"), "battery_discharger_limit")

def add_custom_constraint_max_annual_RE_sales(n, RE_PtX_links, RE_sell_store, k,
                                             name="Grid_RE_sell_cap_vs_RE_PtX_energy"):
    """
    Enforce an annual-energy coupling constraint:

        Store-e_nom[RE_sell_store]  <=  k * Σ_{t in snapshots} Σ_{l in RE_PtX_links} Link-p[t,l] * dt[t]

    Interpretation:
      - LHS is the optimized energy capacity (MWh) of the Store 'RE_sell_store'
      - RHS is k times the total (weighted) electricity sent through PtX links over the year (MWh)

    Inputs:
      n            : pypsa.Network (with linopy model in n.model)
      snapshots    : iterable / index of snapshots used in the solve
      RE_PtX_links : list[str] names of PtX Links
      RE_sell_store: str name of the Store (e.g. "Grid RE sell")
      k            : float scaling factor
      name         : constraint name in linopy
    """

    m = n.model
    snapshots = n.snapshots
    # --- get variables (handle Link-p vs Link-p0 like your old function) ---
    if "Store-e_nom" in m.variables:
        e_store = m.variables["Store-e_nom"]
    else:
        # some versions store them differently, but this is the usual
        e_store = m["Store-e_nom"]

    if "Link-p" in m.variables:
        p_l = m.variables["Link-p"]
    elif "Link-p0" in m.variables:
        p_l = m.variables["Link-p0"]
    else:
        raise KeyError("Neither 'Link-p' nor 'Link-p0' found in n.model.variables")

    # --- sanity: ensure names exist in the variable coords ---
    store_names = set(e_store.coords["name"].values)
    link_names  = set(p_l.coords["name"].values)

    if RE_sell_store not in store_names:
        raise ValueError(f"Store '{RE_sell_store}' not found in Store-e_nom decision variables.")

    lhs_existing = [l for l in RE_PtX_links if l in link_names]
    if len(lhs_existing) == 0:
        raise ValueError("None of RE_PtX_links exist in Link dispatch decision variables.")

    # --- select ---
    grid_sell_e_nom = e_store.sel(name=RE_sell_store)  # (MWh)

    # same snapshot set as the model
    snapshots = n.snapshots.intersection(snapshots)

    p_link = p_l.sel(snapshot=snapshots, name=lhs_existing)  # (MW)

    # --- weights as DataArray with dim 'snapshot' (robust) ---
    dt_s = n.snapshot_weightings["objective"].reindex(snapshots)
    dt = xr.DataArray(dt_s.to_numpy(), dims=("snapshot",), coords={"snapshot": snapshots})  # (h)

    # RHS (MWh)
    rhs = k * (p_link * dt).sum(("snapshot", "name"))

    m.add_constraints(grid_sell_e_nom <= rhs, name=name)
    return

# --- COHERENCY CHECKS ----
#TODO finalize coherency checks: CURRENLTY NOT EXECUTED

def _is_finite_number(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def _has_finite_cap(series, keys=("p_nom_max", "e_nom_max")):
    """True if any of the cap keys exist and are finite and > 0."""
    for k in keys:
        if k in series.index and _is_finite_number(series[k]) and float(series[k]) > 0:
            return True
    return False


def _is_finite_number(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def _has_finite_cap(series, keys=("p_nom_max", "e_nom_max")) -> bool:
    """
    True if any of the cap keys exist and are finite and > 0.
    Works on a pandas Series (row from n.links / n.stores / n.generators).
    """
    for k in keys:
        if k in series.index and _is_finite_number(series[k]) and float(series[k]) > 0:
            return True
    return False

def _infer_h2_grid_bus(
    n,
    targets_dict,
    demand_load_name="H2 grid",
    price_store_name="H2 delivery",
):
    """
    Infer the target (delivery) bus for H2 based on targets_dict['driver']:

    - driver == 'demand': bus where Load named demand_load_name is attached
    - driver == 'price' : bus where Store named price_store_name is attached

    Returns: (bus_name, warnings_list)
    Raises ValueError if it can't infer uniquely.
    """
    warnings = []
    driver = (targets_dict or {}).get("driver", None)

    if driver == "demand":
        if not hasattr(n, "loads") or len(n.loads) == 0:
            raise ValueError("Cannot infer h2_grid_bus: network has no loads.")
        if demand_load_name not in n.loads.index:
            # helpful debug: list candidates that look relevant
            candidates = [i for i in n.loads.index if "H2" in str(i)]
            raise ValueError(
                f"Cannot infer h2_grid_bus for driver='demand': Load '{demand_load_name}' not found. "
                f"H2-like load candidates: {candidates[:20]}"
            )
        bus = n.loads.at[demand_load_name, "bus"]
        return bus, warnings

    if driver == "price":
        if not hasattr(n, "stores") or len(n.stores) == 0:
            raise ValueError("Cannot infer h2_grid_bus: network has no stores.")
        if price_store_name not in n.stores.index:
            candidates = [i for i in n.stores.index if "H2" in str(i) or "delivery" in str(i).lower()]
            raise ValueError(
                f"Cannot infer h2_grid_bus for driver='price': Store '{price_store_name}' not found. "
                f"H2/delivery-like store candidates: {candidates[:20]}"
            )
        bus = n.stores.at[price_store_name, "bus"]
        return bus, warnings

def check_H2_target_coherency(
    n,
    targets_dict,
    h2_grid_bus=None,
    H2_demand_load_name = "H2 grid",
    H2_price_store_name="H2 delivery",
    electrolysis_H2_bus="H2",
    RE_bus="El3 bus",
    electrolyzer_carrier=("AC",),  # your convention: links with carrier 'AC' feed electrolysis_H2_bus
    re_carrier=("solar", "onwind", "offwind", "wind", "pv"),
    sale_price=None,              # negative = revenue term (per your convention)
    has_revenue_term=None,        # override/augment sale_price detection if revenue encoded elsewhere
    strict=False,
    rfnbos_dict=None,
):
    """
    Heuristic coherency checker for target-driven optimisation setups in PyPSA.

    Sign convention (as you stated):
      - costs are positive
      - selling prices / revenues appear as negative contributions in the objective

    Driver convention:
      targets_dict['driver'] in {'price', 'demand'} (others allowed but treated as 'generic').

    What is flagged:
      - Missing/uncapped delivery/sink structure around the target bus
      - Extendable electrolyser capacity without p_nom_max
      - Extendable RE capacity without p_nom_max (optionally on RE_bus)
      - Unconstrained grid import link DK1_to_{RE_bus} when rfnbos_dict['limit']=='unlimited'

    Notes:
      - For driver='price' with revenue incentive, issues are framed as "unbounded objective risk".
      - For driver='demand' (or non-revenue price driver), issues are framed as "ill-posed / inconsistent setup".

    Returns:
      {"ok": bool, "issues": list[str], "warnings": list[str]}
    """
    issues, warnings = [], []

    # --- Determine driver & risk wording ---------------------------------------------------------
    driver = (targets_dict or {}).get("driver", "generic")
    if driver not in ("price", "demand"):
        warnings.append(
            f"targets_dict['driver'] not in {{'price','demand'}} (got {driver!r}); proceeding with generic checks."
        )
        driver = "generic"

    # Detect whether there's an incentive that can push flow/capacity upwards in the objective
    # (negative value = revenue under your convention)
    negative_incentive = (sale_price is not None) and (sale_price < 0)

    if has_revenue_term is None:
        has_revenue_term = negative_incentive

    # Only "unbounded objective" is a primary concern if (price-driver AND revenue term exists)
    unboundedness_mode = (driver == "price") and bool(has_revenue_term)

    if unboundedness_mode:
        risk = "unbounded objective (revenue term can scale without bound)"
        ctx = "price-driven target with revenue incentive"
    elif driver == "demand":
        risk = "infeasible or inconsistent target setup"
        ctx = "demand-driven target"
    else:
        risk = "ill-posed or inconsistent target setup"
        ctx = "target-driven setup"

    # Helpful warning if user says driver=price but we can't see a revenue term
    if driver == "price" and not has_revenue_term:
        warnings.append(
            "targets_dict['driver']=='price' but no revenue term detected from sale_price/has_revenue_term. "
            "If revenue is encoded elsewhere (e.g., negative marginal_cost on a link/generator), "
            "set has_revenue_term=True to activate unboundedness-focused wording."
        )

    # --- Basic existence checks ------------------------------------------------------------------
    # --- infer h2_grid_bus if needed
    if h2_grid_bus is None:
        try:
            inferred_bus, w = _infer_h2_grid_bus(
                n,
                targets_dict,
                demand_load_name=H2_demand_load_name,
                price_store_name=H2_price_store_name,
            )
            h2_grid_bus = inferred_bus
            warnings.extend(w)
        except ValueError as e:
            issues.append(str(e))
            report = {"ok": False, "issues": issues, "warnings": warnings}
            if strict:
                raise ValueError("Target coherency check failed:\n- " + "\n- ".join(issues))
            return report

    if h2_grid_bus not in n.buses.index:
        issues.append(f"Target bus '{h2_grid_bus}' not found in n.buses.")
        report = {"ok": False, "issues": issues, "warnings": warnings}
        if strict:
            raise ValueError("\n".join(issues))
        return report

    # --- 1) Check there is a finite sink/delivery cap around the target bus ----------------------
    # Loads on the target bus (fixed p_set) do not "cap" in the objective sense, but they provide a defined sink.
    if hasattr(n, "loads") and len(n.loads) > 0:
        h2_loads = n.loads.index[n.loads.bus == h2_grid_bus]
    else:
        h2_loads = []

    if len(h2_loads) == 0:
        warnings.append(
            f"No Loads found on target bus '{h2_grid_bus}'. For {ctx}, ensure delivery/export/storage is well-defined "
            f"and appropriately bounded."
        )

    # Your convention: delivery links are those with bus1 == h2_grid_bus (deliver INTO the target bus).
    if hasattr(n, "links") and len(n.links) > 0:
        h2_in_links = n.links.index[n.links.bus1 == h2_grid_bus]
    else:
        h2_in_links = []

    has_capped_delivery = False
    if len(h2_in_links) > 0:
        for lid in h2_in_links:
            row = n.links.loc[lid]
            if bool(row.get("p_nom_extendable", False)):
                if _has_finite_cap(row, keys=("p_nom_max",)):
                    has_capped_delivery = True
                else:
                    issues.append(
                        f"Link '{lid}' delivers into '{h2_grid_bus}' and is extendable with no finite p_nom_max "
                        f"-> can lead to {risk} ({ctx})."
                    )
            else:
                if _is_finite_number(row.get("p_nom", np.nan)) and float(row.get("p_nom", 0.0)) > 0:
                    has_capped_delivery = True

    # Storage on the target bus
    if hasattr(n, "stores") and len(n.stores) > 0:
        h2_stores = n.stores.index[n.stores.bus == h2_grid_bus]
    else:
        h2_stores = []

    has_capped_storage = False
    for sid in h2_stores:
        row = n.stores.loc[sid]
        if bool(row.get("e_nom_extendable", False)):
            if _has_finite_cap(row, keys=("e_nom_max",)):
                has_capped_storage = True
            else:
                issues.append(
                    f"Store '{sid}' on '{h2_grid_bus}' is e_nom_extendable with no finite e_nom_max "
                    f"-> can allow unlimited accumulation and contribute to {risk} ({ctx})."
                )
        else:
            if _is_finite_number(row.get("e_nom", np.nan)) and float(row.get("e_nom", 0.0)) > 0:
                has_capped_storage = True

    if not (has_capped_delivery or has_capped_storage or len(h2_loads) > 0):
        issues.append(
            f"No obvious finite sink/cap structure on target bus '{h2_grid_bus}' "
            f"(no load, no capped delivery link, no capped store) -> common cause of {risk} ({ctx})."
        )

    # --- 2) Check electrolyser capacity is bounded -----------------------------------------------
    carriers = (electrolyzer_carrier,) if isinstance(electrolyzer_carrier, str) else tuple(electrolyzer_carrier)

    if hasattr(n, "links") and len(n.links) > 0:
        ely_links = n.links.index[(n.links.carrier.isin(carriers)) & (n.links.bus1 == electrolysis_H2_bus)]
    else:
        ely_links = []

    if len(ely_links) == 0:
        warnings.append(
            f"No electrolyser Links found with carriers {carriers} feeding '{electrolysis_H2_bus}'. "
            "If the target relies on electrolysis, check your carrier/bus conventions."
        )
    else:
        for lid in ely_links:
            row = n.links.loc[lid]
            if bool(row.get("p_nom_extendable", False)) and not _has_finite_cap(row, keys=("p_nom_max",)):
                issues.append(
                    f"Electrolyser link '{lid}' is extendable with no finite p_nom_max "
                    f"-> can contribute to {risk} ({ctx})."
                )
            if (not bool(row.get("p_nom_extendable", False))) and not (
                _is_finite_number(row.get("p_nom", np.nan)) and float(row.get("p_nom", 0.0)) > 0
            ):
                warnings.append(
                    f"Electrolyser link '{lid}' has no clear finite p_nom (or p_nom=0). Check setup."
                )

    # --- 3) Check RE electricity supply is bounded (if it can drive scaling) ----------------------
    re_carriers = (re_carrier,) if isinstance(re_carrier, str) else tuple(re_carrier)

    if not hasattr(n, "generators") or len(n.generators) == 0:
        warnings.append("No generators found in network; skipping RE cap checks.")
    else:
        gens = n.generators
        if RE_bus is not None:
            re_gens = gens.index[(gens.bus == RE_bus) & (gens.carrier.isin(re_carriers))]
        else:
            re_gens = gens.index[gens.carrier.isin(re_carriers)]

        if len(re_gens) == 0:
            warnings.append(f"No RE generators found for carriers {re_carriers} (RE_bus={RE_bus}).")
        else:
            for gid in re_gens:
                row = gens.loc[gid]
                if bool(row.get("p_nom_extendable", False)) and not _has_finite_cap(row, keys=("p_nom_max",)):
                    issues.append(
                        f"RE generator '{gid}' is extendable with no finite p_nom_max "
                        f"-> can contribute to {risk} ({ctx})."
                    )

    # --- 4) Check RFNBO grid import is not unconstrained -----------------------------------------
    if rfnbos_dict is not None and rfnbos_dict.get("limit", None) == "unlimited":
        link_rfnbos = f"DK1_to_{RE_bus}"

        if not hasattr(n, "links") or link_rfnbos not in n.links.index:
            issues.append(
                f"RFNBO grid import link '{link_rfnbos}' not found, but rfnbos_dict['limit']=='unlimited' "
                f"-> can break consistency of {ctx} and lead to {risk}."
            )
        else:
            row = n.links.loc[link_rfnbos]

            if bool(row.get("p_nom_extendable", False)):
                if not _has_finite_cap(row, keys=("p_nom_max",)):
                    issues.append(
                        f"RFNBO grid import link '{link_rfnbos}' is extendable with no p_nom_max "
                        f"while rfnbos limit is 'unlimited' -> allows unconstrained grid electricity use "
                        f"and can lead to {risk} ({ctx})."
                    )
            else:
                if not (_is_finite_number(row.get("p_nom", np.nan)) and float(row.get("p_nom", 0.0)) > 0):
                    issues.append(
                        f"RFNBO grid import link '{link_rfnbos}' has no finite p_nom while rfnbos limit is 'unlimited' "
                        f"-> can lead to {risk} ({ctx})."
                    )

    ok = len(issues) == 0
    report = {"ok": ok, "issues": issues, "warnings": warnings}

    if strict and not ok:
        raise ValueError("Target coherency check failed:\n- " + "\n- ".join(issues))

    return report

# --- OPTIMIZATION-----
def _apply_common_overrides(solver, opts, threads=None, time_limit=None):
    if threads is not None:
        key = "Threads" if solver == "gurobi" else "threads"
        opts[key] = int(threads)
    if time_limit is not None:
        key = "TimeLimit" if solver == "gurobi" else "time_limit"
        opts[key] = float(time_limit)

def solve_network(n, solver="gurobi", profile=None,
                  io_api="direct", time_limit=None, threads=None,
                  overrides=None, fallback_order=("highs",),
                  assign_all_duals=False, n_config=None,
                  return_model=True):
    """
    Solve PyPSA network using Linopy and return solver results and model.
    Returns: (status, condition, used_solver, used_options, model)
    """

    #--- HELP FUNCTIONS ----
    def _assign_duals(n):
        if not assign_all_duals:
            return
        if hasattr(n.optimize, "assign_duals"):
            try:
                n.optimize.assign_duals(assign_all_duals=True)
            except TypeError:
                n.optimize.assign_duals()
            return
        if hasattr(n, "model") and hasattr(n.model, "assign_duals"):
            n.model.assign_duals()
            return
        if hasattr(n.optimize, "read_solution"):
            try:
                n.optimize.read_solution(assign_all_duals=True)
            except TypeError:
                n.optimize.read_solution()
            return

    def _stamp_meta(n, status, condition, used_solver, used_opts):
        obj_val = float("nan")
        try:
            val = getattr(getattr(n, "model", None), "objective", None)
            if val is not None and getattr(val, "value", None) is not None:
                obj_val = float(val.value)
        except Exception:
            pass

        meta = dict(getattr(n, "meta", {}) or {})
        safe_opts = {k: (v if isinstance(v, (int, float, str)) else str(v))
                     for k, v in (used_opts or {}).items()}

        meta.update({
            "objective": None if math.isnan(obj_val) else obj_val,
            "opt_status": str(status) if status else None,
            "opt_termination": str(condition) if condition else None,
            "opt_solver": used_solver,
            "opt_options": safe_opts,
        })
        n.meta = meta
    #-----------------------

    solver = solver.lower()
    if profile is None:
        profile = "gurobi-default" if solver == "gurobi" else "highs-default"

    try:
        base = SOLVER_PROFILES[solver][profile]
    except KeyError:
        raise ValueError(f"Unknown profile '{profile}' for solver '{solver}'")

    opts = deepcopy(base)
    _apply_common_overrides(solver, opts, threads=threads, time_limit=time_limit)
    if overrides:
        opts.update(overrides)

    # 1) Try to build Linopy model, but skip if empty networks (or network without any cost)
    try:
        m = n.optimize.create_model()
    except ValueError as e:
        if "Objective function could not be created" in str(e):
            print("⚠️  No costed components found — skipping optimization, setting objective = 0.0")
            # Stamp metadata for consistency
            _stamp_meta(n, status="skipped", condition="no_costs", used_solver=None, used_opts=None)
            n.meta = getattr(n, "meta", {}) or {}
            n.meta["objective"] = 0.0
            print("Objective value manually set to 0.0 in metadata.")
            return "skipped", "no_costs", None, {}, None

        else:
            raise

    # 2) add constraints for stores (charging and discharging rates)
    add_custom_constraints_stores(n, n_config=n_config)

    # 3) add custom constraints RE sell vs PtX consumption
    ptx_elec_buses = ["El_H2", "El_meoh", "El_methanation"]

    # candidate links in the network (may be MultiIndex)
    cand = n.links.index[n.links.bus1.isin(ptx_elec_buses)]

    # convert to plain link names
    if hasattr(cand, "nlevels") and cand.nlevels > 1:
        # pick the level that corresponds to link names (usually last)
        cand_names = cand.get_level_values(-1)
    else:
        cand_names = cand

    import xarray as xr

    # --- links present in the model ---
    p_l = n.model.variables["Link-p"] if "Link-p" in n.model.variables else n.model.variables["Link-p0"]
    model_links = set(map(str, p_l.coords["name"].values))

    RE_PtX_links = sorted([str(l) for l in cand_names.unique() if str(l) in model_links])

    # --- store present in the model ---
    RE_sell_store = "Grid RE sell"

    e_store = n.model.variables["Store-e_nom"] if "Store-e_nom" in n.model.variables else n.model["Store-e_nom"]
    model_stores = set(map(str, e_store.coords["name"].values))

    store_exists = RE_sell_store in model_stores
    links_exist = len(RE_PtX_links) > 0

    if links_exist and store_exists:
        add_custom_constraint_max_annual_RE_sales(
            n,
            RE_PtX_links=RE_PtX_links,
            RE_sell_store=RE_sell_store,
            k=max_RE_to_grid,
            name="Grid_RE_sell_cap_vs_RE_PtX_energy",
        )
    else:
        print(
            f"Skipping annual RE sales constraint: "
            f"links_exist={links_exist} (n={len(RE_PtX_links)}), "
            f"store_exists={store_exists} ({RE_sell_store})"
        )

    # --- ensure solver logs are written ---
    log_dir = Path(outputs_folder) / "solver_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    if solver == "gurobi":
        opts.setdefault("solver_options", {})
        opts["solver_options"].setdefault("OutputFlag", 1)
        opts["solver_options"]["LogFile"] = str(log_dir / f"gurobi_{run_name}_{stamp}.log")

    # ---- main solver ----

    try:
        status, condition = n.optimize.solve_model(
            solver_name=solver,
            io_api=io_api,
            **opts,
        )

        _assign_duals(n)
        _stamp_meta(n, status, condition, solver, opts)

        if return_model:
            return status, condition, solver, opts, m  # 👈 return the model
        else:
            return status, condition, solver, opts

    except Exception as e:
        print(f"[WARN] {solver} failed: {e}")

    #### TEMP START
    # ---- main solver (debug wrapper) ----
    #import traceback

    #try:
    #    # 1) Solve
    #    status, condition = n.optimize.solve_model(
    #        solver_name=solver,
    #        io_api=io_api,
    #        **opts,
    #    )

        # 2) Assign duals (often where stochastic writeback breaks)
    #    try:
    #        _assign_duals(n)
    #    except Exception as e_duals:
    #        print(f"[WARN] dual assignment failed: {e_duals}")
    #        traceback.print_exc()
    #        raise  # re-raise so you see it's dual-related

    #    # 3) Stamp metadata (rarely the issue, but isolate it)
    #    try:
    #        _stamp_meta(n, status, condition, solver, opts)
    #    except Exception as e_meta:
    #        print(f"[WARN] stamping metadata failed: {e_meta}")
    #        traceback.print_exc()
    #        raise

    #    if return_model:
    #        return status, condition, solver, opts, m
    #    else:
    #        return status, condition, solver, opts

    #except Exception as e:
    #    print(f"[WARN] {solver} failed: {e}")
    #    traceback.print_exc()

    #    # --- OPTIONAL quick diagnostics (safe to leave on while debugging) ---
    #    try:
            # Check for duplicate columns in time-dependent result tables
    #        import pandas as pd

    #        def _check_df_cols(df, label):
    #            if isinstance(df, pd.DataFrame) and df.columns.has_duplicates:
    #                dups = df.columns[df.columns.duplicated()].unique()
    #                print(f"❌ duplicate columns in {label}: {list(dups)[:20]}")

    #        for comp in ["links_t", "generators_t", "buses_t", "loads_t", "stores_t"]:
    #            if hasattr(n, comp):
    #                t = getattr(n, comp)
    #                for field in ["p0", "p1", "p", "p_set", "marginal_price", "mu_upper", "mu_lower"]:
    #                    if hasattr(t, field):
    #                        _check_df_cols(getattr(t, field), f"{comp}.{field}")

            # Check model coords if model exists
    #        if hasattr(n, "model") and n.model is not None:
    #            m2 = n.model
    #            if hasattr(m2, "variables") and ("Link-p" in m2.variables or "Link-p0" in m2.variables):
    #                pL = m2.variables["Link-p"] if "Link-p" in m2.variables else m2.variables["Link-p0"]
    #                for dim in ["scenario", "snapshot", "name"]:
    #                    if dim in pL.coords:
    #                        idx = pL.coords[dim].to_index()
    #                        if not idx.is_unique:
    #                            print(f"❌ duplicate labels in model coord '{dim}':",
    #                                  idx[idx.duplicated()].unique()[:20])

    #    except Exception as e_diag:
    #        print(f"[WARN] diagnostics also failed: {e_diag}")
    #        traceback.print_exc()

        # Re-raise so fallback logic triggers (or so you can see failure clearly)
    #    raise

    ### TEMP END


    # ---- fallback ----
    for fb in fallback_order:
        fb = fb.lower()
        if fb not in SOLVER_PROFILES:
            continue
        fb_profile = next(iter(SOLVER_PROFILES[fb].keys()))
        fb_opts = deepcopy(SOLVER_PROFILES[fb][fb_profile])
        _apply_common_overrides(fb, fb_opts, threads=threads, time_limit=time_limit)
        try:
            print(f"Falling back to {fb} …")
            status, condition = n.optimize.solve_model(
                solver_name=fb,
                io_api=io_api,
                **fb_opts,
            )
            _assign_duals(n)
            _stamp_meta(n, status, condition, fb, fb_opts)
            if return_model:
                return status, condition, fb, fb_opts, m
            else:
                return status, condition, fb, fb_opts
        except Exception as e2:
            print(f"[WARN] {fb} fallback failed: {e2}")

    _stamp_meta(n, status=None, condition="failed", used_solver=solver, used_opts=opts)
    raise RuntimeError("All solver attempts failed.")


def optimal_network_only(n_opt):
    """function that removes unused: buses, links, stores, generators, storage_units and loads,
     from the plot of the optimal network"""
    n = n_opt

    idx_gen_zero = n.generators.p_nom_opt[n.generators.p_nom_opt == 0].index
    idx_lnk_zero = n.links.p_nom_opt[n.links.p_nom_opt == 0].index
    idx_str_zero = n.stores.e_nom_opt[n.stores.e_nom_opt == 0].index
    idx_stg_zero = n.storage_units.p_nom_opt[n.storage_units.p_nom_opt == 0].index

    for g in idx_gen_zero:
        n.remove('Generator', g)
    for l in idx_lnk_zero:
        n.remove('Link', l)
    for s in idx_str_zero:
        n.remove('Store', s)
    for su in idx_stg_zero:
        n.remove('StorageUnit', su)

    bus_ok = set(n.links.bus0.values) | set(n.links.bus1.values) | set(n.links.bus2.values) | set(
        n.links.bus3.values) | set(n.links.bus4.values) | set(n.generators.bus.values) | set(n.stores.bus.values) | set(
        n.storage_units.bus.values) | set(n.loads.bus.values)
    bus_zero = list(set(n.buses.index.values) - bus_ok)

    if len(bus_zero):
        for b in bus_zero:
            n.remove('Bus', b)
    return n


# ---- SAVE & EXPORT RESULTS
def file_name_network(n, n_flags, run_name, inputs_dict, targets_dict, En_price_year, stochastic):
    """
    Create a descriptive filename for a PyPSA network based on:
    - enabled technologies (n_flags)
    - CO2 price
    - target type (demand or price)
    - H2 / MeOH / CH4 targets
    - year and export cap
    """

    # ------------------
    # Basic inputs
    # ------------------
    CO2_c = int(inputs_dict["CO2 cost"])
    year = int(En_price_year)
    max_RE_to_grid = inputs_dict["max_RE_to_grid"]
    target = targets_dict["driver"]

    # ------------------
    # Helper functions
    # ------------------
    def annual_gwh(load_name):
        """Annual energy demand in GWh (approx)."""
        if load_name not in n.loads.index:
            return 0
        return int(n.loads_t.p_set[load_name].sum() // 1000)

    def mean_abs_marginal_cost(link_name):
        """Mean absolute marginal cost (supports time-varying costs)."""
        if link_name not in n.links.index:
            return 0

        if hasattr(n, "links_t") and hasattr(n.links_t, "marginal_cost"):
            if link_name in n.links_t.marginal_cost.columns:
                return int(abs(n.links_t.marginal_cost[link_name].mean()))

        return int(abs(n.links.at[link_name, "marginal_cost"]))

    # ------------------
    # Targets
    # ------------------
    if target == "demand":
        H2_t   = annual_gwh("H2 grid")
        MeOH_t = annual_gwh("Methanol")
        CH4_t  = annual_gwh("bioCH4")

    elif target == "price":
        H2_t   = mean_abs_marginal_cost("H2_to_delivery")
        MeOH_t = mean_abs_marginal_cost("Methanol_to_delivery")
        CH4_t  = mean_abs_marginal_cost("bioCH4_to_delivery")

    else:
        H2_t = MeOH_t = CH4_t = 0

    # ------------------
    # Stochastic
    # ------------------
    if stochastic:
        stch = 'STC'
    else:
        stch = 'DET'

    # ------------------
    # Technology flags
    # ------------------
    prefix = (
            n_flags.get("biogas", False) * "SB_" +
            n_flags.get("central_heat", False) * "CH_" +
            n_flags.get("renewables", False) * "RE_" +
            n_flags.get("electrolysis", False) * "H2_" +
            n_flags.get("meoh", False) * "meoh_" +
            n_flags.get("methanation", False) * "meth_" +
            n_flags.get("symbiosis", False) * "SN_" +
            n_flags.get("storage", False) * "ST_"
    )

    # ------------------
    # Filename
    # ------------------
    file_name = (
        f"{prefix}"
        f"CO2c{CO2_c}_"
        f"{target}_"
        f"{stch}_"
        f"H2{H2_t}_"
        f"MeOH{MeOH_t}_"
        f"CH4{CH4_t}_"
        f"{year}_"
        f"ElDK1_{max_RE_to_grid}_"
        f"{run_name}"
    )

    return file_name



def create_results_folders (network_name):

    # Create directories for saving files if not existing
    results_folder = create_folder_if_not_exists(outputs_folder, network_name)
    networks_folder = create_folder_if_not_exists(results_folder, 'networks')
    plots_folder = create_folder_if_not_exists(results_folder, 'plots')

    return networks_folder, plots_folder


def save_config (results_folder,c):
    # export configuration from config.py
    networks_folder = create_folder_if_not_exists(results_folder, 'networks')
    dump_params_module(c, dst_folder=networks_folder, filename="config_run.yaml",
                       dataframes_as="records")


def create_folder_if_not_exists(path, folder_name):
    # general function for storing plots
    folder_path = os.path.join(path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")
    return folder_path  # Return the full path of the folder


def export_print_network(n, n_flags, network_name, results_folder, suffix, model = None):
    """
        Export and optionally plot a PyPSA network and its Linopy model.

        Parameters
        ----------
        n : pypsa.Network
            The network to export.
        n_flags : dict
            Dict of boolean options ('print', 'export').
        network_name : str
            Base name for the exported files.
        results_folder : str
            Root folder for results.
        suffix : str
            String suffix (e.g., '_OPT', '_DET', etc.) added to filenames.
        model : linopy.Model, optional
            If provided, the Linopy model is saved as .nc in the same folder.

        Returns
        -------
        str or None
            Full path of the exported network file (if any).
        """

    networks_folder = create_folder_if_not_exists(results_folder, 'networks')

    full_path = None

    if n_flags.get('print'):
        filename = f"{network_name}{suffix}.svg"
        full_path = os.path.join(networks_folder, filename)
        pypsatopo.generate(n, file_output=full_path,
                           negative_efficiency=False, carrier_color=True)
        print(f"✅ PyPSA network plotted to: {full_path}")


    # --- Export network to NetCDF ---
    if n_flags.get("export"):
        filename_nc = f"{network_name}{suffix}.nc"
        nc_path = os.path.join(networks_folder, filename_nc)
        n.export_to_netcdf(nc_path)
        full_path = nc_path
        print(f"✅ Solved PyPSA network saved to: {nc_path}")

        # --- Optional model export ---
        if model is not None:
            try:
                model_filename = f"{network_name}{suffix}_model.nc"
                model_path = os.path.join(networks_folder, model_filename)
                model.to_netcdf(model_path)
                print(f"✅ Linopy model saved to: {model_path}")
            except Exception as e:
                print(f"[WARN] Could not export Linopy model: {e}")

    if full_path is None:
        return None

    return full_path


def save_network_comp_allocation (results_folder, network_comp_allocation):
    # save allocation of compeonts to each agent/plant in pkl file

    networks_folder = create_folder_if_not_exists(results_folder, 'networks')

    networks_folder = Path(networks_folder)
    with open(networks_folder / 'network_comp_allocation.pkl', 'wb') as f:
        pkl.dump(network_comp_allocation, f)

    return


def _to_basic(obj, dataframes_as="records"):
    """Convert common scientific Python objects to YAML-safe Python types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, (np.bool_,)):     return bool(obj)
    if isinstance(obj, (np.ndarray,)):   return obj.tolist()

    if isinstance(obj, pd.DataFrame):
        if dataframes_as == "records":
            return obj.to_dict(orient="records")
        if dataframes_as == "split":
            return obj.to_dict(orient="split")  # {index, columns, data}
        if dataframes_as == "columns":
            return obj.to_dict(orient="list")   # {col: [..], ...}
        if dataframes_as == "csv":
            return obj.to_csv(index=False)
        if dataframes_as == "summary":
            return {"__type__": "DataFrame",
                    "shape": list(obj.shape),
                    "columns": obj.columns.tolist()}
        return obj.to_dict(orient="records")

    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {k: _to_basic(v, dataframes_as) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_basic(v, dataframes_as) for v in obj]

    # Fallback for anything else
    return repr(obj)


def _extract_public_values(module):
    vals = {}
    for k, v in vars(module).items():
        if k.startswith("_"):                       # skip private/special
            continue
        if inspect.ismodule(v) or inspect.isfunction(v) or inspect.isclass(v):
            continue                                # skip callables/classes/modules
        vals[k] = v
    return vals


def dump_params_module(module, dst_folder, filename="params.yaml",
                       dataframes_as="records", sort_keys=False):
    """
    Dump the public contents of `module` to YAML in `dst_folder/filename`.
    dataframes_as: 'records' | 'split' | 'columns' | 'csv' | 'summary'
    """
    params = _extract_public_values(module)
    clean = _to_basic(params, dataframes_as=dataframes_as)

    dst = Path(dst_folder)
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / filename

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, sort_keys=sort_keys, allow_unicode=True)

    return path


# --- ANALYSIS AND PLOT ----

def get_capital_cost(n_opt):
    '''function to retrive annualized capital cost for the optimized network, for each genertor, store and link '''
    # loads do not have capital or marginal costs
    # generatars: marginal + capital cost
    # links: marginal + capital costs
    # stores: marginal (only production) + capital costs
    cc_stores = n_opt.stores.capital_cost * n_opt.stores.e_nom_opt
    cc_generators = n_opt.generators.capital_cost * n_opt.generators.p_nom_opt
    cc_links = n_opt.links.capital_cost * n_opt.links.p_nom_opt

    return cc_stores, cc_generators, cc_links


def get_marginal_cost(n_opt):
    """function to retrive marginal cost for the optimized network, for each genertor, store and link """

    # calculate the marginal cost for every store: note mc is applied only to power generated
    mc_store = []
    # stores with constant marginal costs
    df_marginal_cost_s = n_opt.stores.marginal_cost * n_opt.stores_t.p
    mc_store = df_marginal_cost_s.sum()
    # generators with variable marginal cost
    df_marginal_cost_s2 = n_opt.stores_t.marginal_cost * n_opt.stores_t.p[
        n_opt.stores_t.marginal_cost.columns.values]
    mc_store[
        n_opt.stores_t.marginal_cost.columns.values] = df_marginal_cost_s2.sum()

    mc_gen = []
    # generators with constant marginal costs
    df_marginal_cost_g = n_opt.generators.marginal_cost * n_opt.generators_t.p
    mc_gen = df_marginal_cost_g.sum()
    # generators with variable marginal cost
    df_marginal_cost_g2 = n_opt.generators_t.marginal_cost * n_opt.generators_t.p[
        n_opt.generators_t.marginal_cost.columns.values]
    mc_gen[
        n_opt.generators_t.marginal_cost.columns.values] = df_marginal_cost_g2.sum()

    mc_link = []
    # links with constant marginal cost
    df_marginal_cost_l = n_opt.links.marginal_cost * n_opt.links_t.p0
    mc_link = df_marginal_cost_l.sum()
    # links with variable marginal cost
    df_marginal_cost_l2 = n_opt.links_t.marginal_cost * n_opt.links_t.p0[
        n_opt.links_t.marginal_cost.columns.values]
    mc_link[n_opt.links_t.marginal_cost.columns.values] = df_marginal_cost_l2.sum()

    return mc_store, mc_gen, mc_link


def get_system_cost(n_opt):
    """function that retunr total capital, marginal and system cost"""
    # loads do not have capital or marginal costs
    # generatars: marginal + capital cost
    # links: marginal + capital costs
    # stores: marginal (only production) + capital costs

    # total capital cost
    cc_stores, cc_generators, cc_links = get_capital_cost(n_opt)
    tot_cc_stores = cc_stores.sum()
    tot_cc_generators = cc_generators.sum()
    tot_cc_links = cc_links.sum()
    tot_cc = [tot_cc_stores, tot_cc_generators, tot_cc_links]

    # Total marginal cost
    mc_store, mc_gen, mc_link = get_marginal_cost(n_opt)
    tot_mc_stores = mc_store.sum()
    tot_mc_generators = mc_gen.sum()
    tot_mc_links = mc_link.sum()
    tot_mc = [tot_mc_stores, tot_mc_generators, tot_mc_links]

    # total system cost
    tot_sc = np.sum(tot_cc) + np.sum(tot_mc)
    return tot_cc, tot_mc, tot_sc


def get_total_marginal_capital_cost_agents(n_opt, network_comp_allocation, plot_flag, folder):
    """Return dicts with total capital and marginal costs per agent and (optionally) plot one stacked bar."""
    cc_stores, cc_generators, cc_links = get_capital_cost(n_opt)
    mc_stores, mc_generators, mc_links = get_marginal_cost(n_opt)

    cc_tot_agent, mc_tot_agent = {}, {}

    for key in network_comp_allocation:
        agent_links_n_opt = list(set(network_comp_allocation[key]['links']).intersection(n_opt.links.index))
        agent_generators_n_opt = list(set(network_comp_allocation[key]['generators']).intersection(n_opt.generators.index))
        agent_stores_n_opt = list(set(network_comp_allocation[key]['stores']).intersection(n_opt.stores.index))

        # Sum safely even if lists are empty
        cc_tot_agent[key] = (
            cc_links.get(agent_links_n_opt, 0).sum()
            + cc_generators.get(agent_generators_n_opt, 0).sum()
            + cc_stores.get(agent_stores_n_opt, 0).sum()
        )
        mc_tot_agent[key] = (
            mc_links.get(agent_links_n_opt, 0).sum()
            + mc_generators.get(agent_generators_n_opt, 0).sum()
            + mc_stores.get(agent_stores_n_opt, 0).sum()
        )

    if plot_flag:
        cats = list(cc_tot_agent.keys())
        cats.sort(key=lambda c: abs(cc_tot_agent[c] + mc_tot_agent[c]), reverse=True)

        cmap = mpl.cm.get_cmap("tab20", len(cats))
        colors = {cat: cmap(i) for i, cat in enumerate(cats)}

        fig, ax = plt.subplots(figsize=(7, 6))

        x = 0
        bottom_pos = 0.0
        bottom_neg = 0.0

        def stack_segment(value, facecolor, hatch=None):
            nonlocal bottom_pos, bottom_neg
            if value == 0:
                return
            if value >= 0:
                ax.bar(x, value, bottom=bottom_pos, color=facecolor, edgecolor="black",
                       linewidth=0.6, hatch=hatch)
                bottom_pos += value
            else:
                ax.bar(x, value, bottom=bottom_neg, color=facecolor, edgecolor="black",
                       linewidth=0.6, hatch=hatch)
                bottom_neg += value

        # Build the single stacked bar
        for cat in cats:
            col = colors[cat]
            stack_segment(cc_tot_agent.get(cat, 0.0), facecolor=col, hatch=None)     # CAPEX (plain)
            stack_segment(mc_tot_agent.get(cat, 0.0), facecolor=col, hatch="///")    # Marginal (striped)

        # Cosmetics
        ax.set_xticks([x], ["Total system cost"])
        ax.set_ylabel("€/y")
        ax.set_title("Annualized Total system cost\nplain=CAPEX, striped=Marginal")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        total = bottom_pos + bottom_neg
        ax.text(x, bottom_pos if total >= 0 else bottom_neg, f"{total:,.0f}",
                ha="center", va="bottom" if total >= 0 else "top", fontsize=9)

        # Legends
        pattern_legend = [
            Patch(facecolor="white", edgecolor="black", label="Fixed cost (plain)"),
            Patch(facecolor="white", edgecolor="black", hatch="///", label="Operational cost (striped)")
        ]
        ax.legend(handles=pattern_legend, loc="upper left", frameon=True)

        cat_handles = [Patch(facecolor=colors[c], edgecolor="black", label=c) for c in cats]
        ax2 = ax.inset_axes([1.02, 0.0, 0.28, 1.0], transform=ax.transAxes)
        ax2.axis("off")
        ax2.legend(handles=cat_handles, title="Categories", loc="upper left", frameon=True)

        plt.tight_layout()
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / 'system_cost.png', dpi=300, bbox_inches="tight")
        plt.close(fig)

    return cc_tot_agent, mc_tot_agent


# good network

