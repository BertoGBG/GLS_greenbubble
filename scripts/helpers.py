import pandas as pd
import numpy as np
import pypsatopo
from scripts.config import En_price_year, n_flags, discount_rate, outputs_folder, CO2_cost_ref_year, max_RE_to_grid, tariffs_dict
import pickle as pkl
from scripts.solver_profiles import SOLVER_PROFILES
import yaml
from copy import deepcopy
import inspect, datetime as dt
from pathlib import Path
import xarray as xr
import calendar
import os
import time
import reverse_geocoder as rg
from contextlib import contextmanager
import logging
import re
import warnings
import pypsa


def assert_stochastic_schema_consistent(n, where=""):
    # For stochastic networks, every *_t DataFrame with (scenario,name) columns
    # must reference entries that exist in the corresponding component table index.
    checks = [
        ("buses",      getattr(getattr(n, "buses_t", None), "marginal_price", None)),
        ("generators", getattr(getattr(n, "generators_t", None), "p_max_pu", None)),
        ("generators", getattr(getattr(n, "generators_t", None), "p", None)),
        ("loads",      getattr(getattr(n, "loads_t", None), "p_set", None)),
        ("links",      getattr(getattr(n, "links_t", None), "marginal_cost", None)),
        ("links",      getattr(getattr(n, "links_t", None), "p_max_pu", None)),
    ]

    for comp, df in checks:
        if df is None or not hasattr(df, "columns"):
            continue
        if not isinstance(df.columns, pd.MultiIndex):
            continue

        comp_idx = getattr(getattr(n, comp), "index", None)
        if comp_idx is None:
            continue

        # Must be same key space: df.columns ⊆ component index
        missing = df.columns.difference(comp_idx)
        if len(missing):
            ex = missing[:10].tolist()
            raise RuntimeError(
                f"[{where}] {comp}_t has columns not in n.{comp}.index. "
                f"Example missing keys: {ex}\n"
                f"df.columns.names={df.columns.names}, n.{comp}.index.names={getattr(comp_idx,'names',None)}"
            )
####



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

def ensure_bus(n, name, *, carrier="Heat", unit="MW"):
    if name not in n.buses.index:
        n.add("Bus", name, carrier=carrier, unit=unit)


def ensure_carrier(n, name):
    if name not in n.carriers.index:
        n.add("Carrier", name)
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

    if target_columns is not None:
        # add any missing columns from target
        for col in target_columns:
            if col not in df_new.columns:
                df_new[col] = np.nan
        # drop extra columns not in target (comment this if you want to keep them)
        df_new = df_new[target_columns]

    return df_new


def is_eu_or_us(lat, lon):
    # small function to merge EU and US costs based on cordinares:
    EU_COUNTRIES = {
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
        "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"
    }
    res = rg.search((lat, lon), mode=1)[0]
    cc = res["cc"]   # ISO country code

    if cc == "US":
        return "US"
    elif cc in EU_COUNTRIES:
        return "EU"
    else:
        return "OTHER"


def merge_EU_US_tech_costs(tech_costs_EU, tech_costs_US, dict_tech_US_EU):
    tech_costs = tech_costs_EU.copy()
    key_list=[]
    for key in dict_tech_US_EU:
        if dict_tech_US_EU[key] != '':
            tech_costs.loc[key,:] = tech_costs_US.loc[key,:]
            key_list.append(key)
    print(key_list)

    return tech_costs


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


def read_costs(cost_path : str, tech_inputs: dict, USD_to_EUR: float , discount_rate : float, Nyears: int = 1, lifetime: int = 25):
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


def prepare_costs(latitude: float, longitude: float, tech_inputs: dict,
                  USD_to_EUR: float, discount_rate: float,
                  cost_path_EU: str, cost_path_US: str = None,
                  dict_tech_US_EU: dict = None):

    if is_eu_or_us(latitude, longitude) == 'EU':
        tech_costs = read_costs(
            cost_path=cost_path_EU,
            tech_inputs=tech_inputs,
            USD_to_EUR=USD_to_EUR,
            discount_rate=discount_rate
        )

    elif is_eu_or_us(latitude, longitude) == 'US' and (cost_path_US is not None):
        tech_costs_EU = read_costs(
            cost_path=cost_path_EU,
            tech_inputs=tech_inputs,
            USD_to_EUR=USD_to_EUR,
            discount_rate=discount_rate
        )
        tech_costs_US = read_costs(
            cost_path=cost_path_US,
            tech_inputs=tech_inputs,
            USD_to_EUR=USD_to_EUR,
            discount_rate=discount_rate
        )
        tech_costs = merge_EU_US_tech_costs(
            tech_costs_EU=tech_costs_EU,
            tech_costs_US=tech_costs_US,
            dict_tech_US_EU=dict_tech_US_EU
        )

    else:  # fallback on EU costs (for future modifications)
        tech_costs = read_costs(
            cost_path=cost_path_EU,
            tech_inputs=tech_inputs,
            USD_to_EUR=USD_to_EUR,
            discount_rate=discount_rate
        )

    return tech_costs


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


def build_electricity_grid_price_w_tariff(Elspotprices, En_price_year, tariffs_dict):
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
    start_date = str(En_price_year) + '-01-01 00:00'
    end_date = str(En_price_year) + '-12-31 23:00'
    winter_1 = pd.date_range(start_date , summer_start , freq='h')
    winter_1 = winter_1.drop(winter_1[-1])
    winter_2 = pd.date_range(summer_end , end_date, freq='h')
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
    el_grid_price = Elspotprices + tariffs_dict['el_transmission_tariff'] + tariffs_dict['el_system_tariff']  + tariffs_dict['el_afgift']
    el_grid_sell_price = -Elspotprices + tariffs_dict['el_tariff_sell']

    # assign tariff to hours
    for h in winter:
        day = h.weekday()
        hour = h.hour
        net_tariff = 0  # Default value

        if day in [5, 6]:  # weekends
            if hour in high_hours_weekend_winter:
                net_tariff = tariffs_dict['el_net_tariff_high']
            else:
                net_tariff = tariffs_dict['el_net_tariff_low']
        elif day in range(0, 5):  # weekdays
            if hour in peak_hours:
                net_tariff = tariffs_dict['el_net_tariff_peak']
            elif hour in high_hours_weekday_winter:
                net_tariff = tariffs_dict['el_net_tariff_high']
            else:
                net_tariff = tariffs_dict['el_net_tariff_low']

        el_grid_price.loc[h, :] = el_grid_price.loc[h, :] + net_tariff

    for h in summer:
        day = h.weekday()
        hour = h.hour
        net_tariff = 0  # Default value

        if day in [5, 6]:  # weekends
            net_tariff = tariffs_dict['el_net_tariff_low']
        elif day in range(0, 5):  # weekdays
            if hour in high_hours_weekday_summer:
                net_tariff = tariffs_dict['el_net_tariff_high']
            else:
                net_tariff = tariffs_dict['el_net_tariff_low']

        el_grid_price.loc[h, :] = el_grid_price.loc[h, :] + net_tariff

    return el_grid_price, el_grid_sell_price


def build_NG_grid_price_w_tariff(NG_price_year):
    ### function that adds NG TSO and DSO tariffs to the NG purchase price
    NG_grid_price = NG_price_year + tariffs_dict['NG_tso_tariff']  + tariffs_dict['NG_dso_tariff']
    NG_sell_price = NG_price_year
    return NG_grid_price, NG_sell_price


def en_market_prices_w_CO2(inputs_dict, tech_costs, n_options):
    """Build market prices for electricity, natural gas, and district heating including CO₂ cost adjustments."""

    # --- 1. Base data from inputs_dict ---
    CO2_cost       = inputs_dict["CO2 cost"]
    CO2_cost_ref_year = inputs_dict["CO2 cost ref year"]
    CO2_emiss_El   = inputs_dict["CO2_emiss_El"]            # tCO2/MWh_el
    NG_price_year  = inputs_dict["NG_price_year"]           # €/MWh
    Elspotprices   = inputs_dict["Elspotprices"]            # Series or DataFrame

    # --- 2. Electricity grid prices (buy/sell) ---
    el_grid_price, el_grid_sell_price = build_electricity_grid_price_w_tariff(Elspotprices, En_price_year, tariffs_dict)

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

# --- Build Reference Cost for Shapley Values on Demand driven Optimization


def add_market_import_fallback(n, targets_dict: dict, verbose=True):
    if not hasattr(n, "snapshots") or len(n.snapshots) == 0:
        raise RuntimeError("Network has no snapshots; cannot add time-series loads/imports.")

    specs = [
        dict(bus="H2",       load="H2 grid",  demand_key="demand_H2",  price_key="price_H2",     carrier="H2"),
        dict(bus="Methanol", load="Methanol", demand_key="demand_meoh", price_key="price_meoh",   carrier="Methanol"),
        dict(bus="bioCH4",   load="bioCH4",   demand_key="demand_CH4", price_key="price_bioCH4", carrier="gas"),
    ]

    def _to_float(x):
        try:
            return None if x is None else float(x)
        except Exception:
            return None

    def _ensure_carrier(car):
        if car and car not in n.carriers.index:
            n.add("Carrier", car)

    def uniform_load_series(annual_mwh: float) -> pd.Series:
        return pd.Series(annual_mwh / len(n.snapshots), index=n.snapshots)

    for s in specs:
        bus, load_name, car = s["bus"], s["load"], s["carrier"]
        demand = _to_float(targets_dict.get(s["demand_key"]))
        price  = _to_float(targets_dict.get(s["price_key"]))

        need_anything = (demand is not None and demand > 0) or (price is not None)
        if not need_anything:
            continue

        _ensure_carrier(car)

        # 1) Bus
        if bus not in n.buses.index:
            n.add("Bus", bus, carrier=car, unit="MW")
            if verbose:
                print(f"[market fallback] Added Bus '{bus}'")

        # 2) Load (only if no load on that bus)
        if demand is not None and demand > 0:
            has_load_on_bus = (len(n.loads.index) > 0) and (n.loads.bus == bus).any()

            if not has_load_on_bus and load_name not in n.loads.index:
                n.add("Load", load_name, bus=bus)  # carrier optional
                n.loads_t.p_set[load_name] = uniform_load_series(demand)
                print(f'load_{load_name} added')
                if verbose:
                    print(f"[market fallback] Added Load '{load_name}' on bus '{bus}' (annual {demand:g} MWh/y)")

        # 3) Market import generator
        if price is not None:
            gen_name = f"MARKET_IMPORT::{bus}"
            if gen_name not in n.generators.index:
                n.add(
                    "Generator",
                    gen_name,
                    bus=bus,
                    carrier=car,
                    p_nom_extendable=True,
                    marginal_cost=float(price),
                )

                if verbose:
                    print(f"[market fallback] Added {gen_name} at bus '{bus}' with MC={price:g} €/MWh")


# --- Add CUSTOM CONSTRAINTS ----

def add_custom_constraints_stores(n, m, n_config=None, case_insensitive=True):

    # -----------------------
    # helpers: canonical names
    # -----------------------
    def _canon(x):
        return x[-1] if isinstance(x, tuple) else x

    def _norm(s):
        s = str(s)
        return s.strip().casefold() if case_insensitive else s

    def _index_keys_matching(idx, base_name):
        """
        Return list of keys in idx whose canonical name matches base_name.
        Works for Index and MultiIndex.
        """
        target = _norm(base_name)
        keys = []
        for k in idx:
            if _norm(_canon(k)) == target:
                keys.append(k)
        return keys

    # -----------------------------------------
    # helpers: resolve selection key in variables
    # -----------------------------------------
    def _var_uses_tuple_names(var, dim="name"):
        """True if var.coords[dim] contains tuple-like entries."""
        if dim not in var.dims:
            return False
        vals = list(var.coords[dim].values)
        return any(isinstance(v, tuple) for v in vals)

    def _resolve_name_in_var(var, key, dim="name"):
        """
        - if variable uses tuple keys -> use full key (tuple)
        - else -> use canonical string
        """
        return key if _var_uses_tuple_names(var, dim=dim) else _canon(key)

    def _safe_sel(var, dim, key):
        """Select only if key exists in coord."""
        if dim not in var.dims:
            return None
        coord_vals = set(var.coords[dim].values)
        if key not in coord_vals:
            return None
        return var.sel({dim: key})

    # -----------------------
    # expressions for p_nom/e_nom
    # -----------------------
    def _link_p_nom_expr(link_key):
        if "Link-p_nom" in m.variables:
            v = m.variables["Link-p_nom"]
            k = _resolve_name_in_var(v, link_key, dim="name")
            out = _safe_sel(v, "name", k)
            if out is not None:
                return out

        # fallback to network data
        if link_key in n.links.index:
            val = n.links.at[link_key, "p_nom"]
            if pd.notna(val):
                return float(val)

        # fallback: try first matching canonical name
        matches = _index_keys_matching(n.links.index, _canon(link_key))
        if matches:
            val = n.links.at[matches[0], "p_nom"]
            if pd.notna(val):
                return float(val)

        return None

    def _store_e_nom_expr(store_key):
        if "Store-e_nom" in m.variables:
            v = m.variables["Store-e_nom"]
            k = _resolve_name_in_var(v, store_key, dim="name")
            out = _safe_sel(v, "name", k)
            if out is not None:
                return out

        # fallback to network data
        if store_key in n.stores.index:
            val = n.stores.at[store_key, "e_nom"]
            if pd.notna(val):
                return float(val)

        matches = _index_keys_matching(n.stores.index, _canon(store_key))
        if matches:
            val = n.stores.at[matches[0], "e_nom"]
            if pd.notna(val):
                return float(val)

        return None

    # -----------------------
    # config getter
    # -----------------------
    def _cfg(at_tech, key, default=None):
        if n_config is None:
            return default
        try:
            return n_config.at[at_tech, key]
        except Exception as e:
            print(f"⚠️ _cfg failed for at_tech={at_tech!r}, key={key!r}: {type(e).__name__}: {e}")
            return default

    # -----------------------
    # core constraint adder
    # -----------------------
    def _add_bound(link_base_name, store_base_name, factor, tag):

        if factor is None or pd.isna(factor):
            return
        try:
            factor = float(factor)
        except Exception:
            print(f"⚠️ [custom-constraints] skip {tag}: bad factor={factor!r}")
            return
        if factor == 0.0:
            return

        store_keys = _index_keys_matching(n.stores.index, store_base_name)
        link_keys  = _index_keys_matching(n.links.index,  link_base_name)

        if not store_keys or not link_keys:
            missing = []
            if not link_keys:
                missing.append(f"link '{link_base_name}'")
            if not store_keys:
                missing.append(f"store '{store_base_name}'")
            print(f"[custom-constraints] skip {tag}: missing " + " and ".join(missing))
            return

        # If your p_nom/e_nom variables are (name) only (as in the model printout),
        # then adding ONE constraint is sufficient (shared across scenarios).
        link_expr  = _link_p_nom_expr(link_keys[0])
        store_expr = _store_e_nom_expr(store_keys[0])

        if link_expr is None or store_expr is None:
            print(f"[custom-constraints] skip {tag}: could not build expressions "
                  f"(link_expr={link_expr is not None}, store_expr={store_expr is not None})")
            return

        cname = f"{tag}__{_canon(link_keys[0])}__le_{factor}__{_canon(store_keys[0])}"

        # prevent bool-constraint crash if both are constants
        if isinstance(link_expr, (int, float)) and isinstance(store_expr, (int, float)):
            lhs_val = float(link_expr) - factor * float(store_expr)
            if lhs_val > 0:
                print(f"⚠️ [custom-constraints] fixed bound violated (not added): {cname}")
            return

        lhs = link_expr - factor * store_expr

        print(
            f"[debug] {tag}: link_key={link_keys[0]} store_key={store_keys[0]} "
            f"factor={factor} "
            f"link_is_var={not isinstance(link_expr, (int, float))} "
            f"store_is_var={not isinstance(store_expr, (int, float))} "
            f"link_expr={type(link_expr)} store_expr={type(store_expr)}"
        )

        m.add_constraints(lhs <= 0, name=cname)
        print(f"[custom-constraints] added: {cname}")

    # -----------------------
    # constraints
    # -----------------------
    _add_bound("TES concrete charger", "TES Concrete storage",
               _cfg("TES concrete", "ramp limit up"), "TES_concrete_charger_limit")
    _add_bound("TES concrete discharger", "TES Concrete storage",
               _cfg("TES concrete", "ramp limit down"), "TES_concrete_discharger_limit")

    _add_bound("TES DH charger", "TES DH storage",
               _cfg("TES DH", "ramp limit up"), "Water_tank_DH_charger_limit")
    _add_bound("TES DH discharger", "TES DH storage",
               _cfg("TES DH", "ramp limit down"), "Water_tank_DH_discharger_limit")

    _add_bound("battery charger", "battery",
               _cfg("battery", "ramp limit up"), "battery_charger_limit")
    _add_bound("battery discharger", "battery",
               _cfg("battery", "ramp limit down"), "battery_discharger_limit")


# Max RE sales
def _default_included_agents(n_flags: dict) -> list[str]:
    """
    Reasonable default: include all enabled agents except ones that are typically
    'non-demand' / bookkeeping / external: renewables, symbiosis, storage, print, export.
    Adjust to your needs.
    """
    exclude = {"renewables", "symbiosis", "storage", "print", "export"}
    return [k for k, v in (n_flags or {}).items() if v and k not in exclude]


def _allowed_rhs_target_buses(n_flags: dict, include_agents: list[str] | None) -> set[str]:
    if include_agents is None:
        include_agents = _default_included_agents(n_flags)

    # buses are El_<agent> with NO suffix
    return {f"El_{a}" for a in include_agents}


def filter_consuming_links_by_counterparty_bus(
    n,
    bus: str,
    consuming_links: list,
    link_ports: dict,
    allowed_other_buses: set[str],
):
    """
    Keep only consuming links from `bus` that connect (anywhere) to one of `allowed_other_buses`.
    """
    if not consuming_links:
        return {}, []

    L = n.links
    bus_cols = [c for c in L.columns if c.startswith("bus")]

    kept_ports = {}
    for link in consuming_links:
        row = L.loc[link]

        # All buses on this link except the constrained bus
        other_buses = {row.get(c) for c in bus_cols}
        other_buses.discard(bus)

        if other_buses & allowed_other_buses:
            kept_ports[link] = link_ports[link]

    return kept_ports, sorted(kept_ports.keys())

def consuming_links_from_bus(n, bus: str):
    """
    Returns:
      - link_ports: dict {link_key: port_int}
      - links_list: sorted list of link_key
    Rule:
      consumes if bus0 == bus OR (bus<i> == bus and efficiency<i> < 0)
    """
    L = n.links
    if L.empty:
        return {}, []

    bus_cols = [c for c in L.columns if c.startswith("bus")]
    link_ports = {}

    for link, row in L.iterrows():
        matched_ports = []
        for bc in bus_cols:
            if row.get(bc) != bus:
                continue
            port = int(bc.replace("bus", ""))

            if port == 0:
                matched_ports.append(0)
            else:
                eff_col = "efficiency" if port == 1 else f"efficiency{port}"
                eff = row.get(eff_col, None)
                if eff is not None and pd.notna(eff) and float(eff) < 0:
                    matched_ports.append(port)

        if matched_ports:
            # prefer 0 if present, else smallest
            port = 0 if 0 in matched_ports else min(matched_ports)
            link_ports[link] = port

    return link_ports, sorted(link_ports.keys())


def _get_dt(n, snaps):
    """
    Return dt as an xr.DataArray with dim 'snapshot'.
    Prefers hour-weights columns; falls back to 1.0 per snapshot.
    """
    for col in ("generators", "stores"):
        if col in n.snapshot_weightings.columns:
            s = n.snapshot_weightings[col].reindex(snaps)
            return xr.DataArray(s.to_numpy(), dims=("snapshot",), coords={"snapshot": snaps})

    # fallback: hourly snapshots
    return xr.DataArray(
        1.0,
        dims=("snapshot",),
        coords={"snapshot": snaps},
    )


def _get_link_p_var(m, port: int):
    """Fetch Link power variable for a port (works across common PyPSA/Linopy variants)."""
    key = f"Link-p{port}"
    if key in m.variables:
        return m.variables[key]

    if port == 0:
        if "Link-p" in m.variables:
            return m.variables["Link-p"]
        if "Link-p0" in m.variables:
            return m.variables["Link-p0"]

    return None


def _canon_link_name(x):
    """If link keys are tuples (MultiIndex), return last level; else return itself."""
    return x[-1] if isinstance(x, tuple) else x


def _snapshots_by_scenario(n, scenario_level=0):
    snaps = n.snapshots
    if isinstance(snaps, pd.MultiIndex):
        scenarios = snaps.get_level_values(scenario_level).unique()
        return {s: snaps[snaps.get_level_values(scenario_level) == s] for s in scenarios}
    return {None: snaps}


def _resolve_name_in_var(var, link_key):
    """
    Return the correct 'name' key to use for .sel(name=...) in a given variable:
    - if variable uses tuple keys, use link_key (tuple)
    - else use canonical string (last level)
    """
    names = set(var.coords["name"].values)
    if any(isinstance(x, tuple) for x in names):
        return link_key
    else:
        return _canon_link_name(link_key)


def find_export_links(n, pattern: str, export_bus: str = "ElDK1 sell bus"):
    """
    Return list of link keys in n.links.index whose canonical name contains `pattern`.
    If export_bus is provided, also require any output bus (bus1, bus2, ...) equals export_bus.
    """
    keys = [k for k in n.links.index if pattern in _canon_link_name(k)]

    if export_bus is None:
        return keys

    out_bus_cols = [c for c in n.links.columns if c.startswith("bus") and c != "bus0"]
    if not out_bus_cols:
        return keys

    filtered = []
    for k in keys:
        row = n.links.loc[k]
        if row[out_bus_cols].eq(export_bus).any():
            filtered.append(k)

    return filtered


def add_max_RE_sales_constraint(
    n,
    m,
    bus: str = "El3 bus",
    export_pattern: str = "El3_to",
    export_bus: str = "ElDK1 sell bus",
    alpha: float = 0.2,
    name: str = "El3_export_fraction_of_RE",
    *,
    n_flags: dict | None = None,
    include_agents: list[str] | None = None,
    warn: bool = True,
):
    """
    Adds a custom PyPSA constraint:
    RE exported to the grid <= (alpha / (1 - alpha))  * RE consumed at the site

    Behavior:
      - If any prerequisite is missing, the function SKIPS (returns / continues),
        and optionally issues a warning. It never raises for missing elements.

      - Still validates alpha strictly (raises ValueError) because that's a user/config error.
    """

    def _skip(msg: str):
        if warn:
            warnings.warn(
                f"[add_max_RE_sales_constraint] {msg} (skipping constraint)",
                RuntimeWarning,
                stacklevel=2,
            )
        return  # do nothing

    alpha = float(alpha)
    if not (0.0 <= alpha):
        raise ValueError(f"alpha cannot be <0, got {alpha}")

    # --- Skip if bus not present (works for deterministic and MultiIndex buses) ---
    try:
        buses_index = n.buses.index
        if hasattr(buses_index, "get_level_values"):  # MultiIndex
            bus_names = set(buses_index.get_level_values(-1))
        else:
            bus_names = set(buses_index)
    except Exception:
        bus_names = set()
    if bus not in bus_names:
        return

    # --- consuming links from bus ---
    link_ports, consuming_links = consuming_links_from_bus(n, bus)
    if not consuming_links:
        return
    #print('consuming_links', consuming_links)

    # --- restrict RHS to only links connected to El_{agent} buses ---
    allowed_buses = _allowed_rhs_target_buses(n_flags or {}, include_agents)

    link_ports, consuming_links = filter_consuming_links_by_counterparty_bus(
        n,
        bus=bus,
        consuming_links=consuming_links,
        link_ports=link_ports,
        allowed_other_buses=allowed_buses,
    )
    if not consuming_links:
        return _skip(
            f"No consuming links from '{bus}' matched allowed agent buses {sorted(allowed_buses)}"
        )

    print('consuming_links', consuming_links)

    # --- export links ---
    export_keys = find_export_links(n, export_pattern, export_bus)
    if not export_keys:
        return _skip(f"No export links found matching '{export_pattern}' exporting to '{export_bus}'")
    #print('export_keys',export_keys)
    export_canon = set(_canon_link_name(k) for k in export_keys)

    snaps_by_scen = _snapshots_by_scenario(n, scenario_level=0)  # adjust if needed

    links_by_port = {}
    for l in consuming_links:
        links_by_port.setdefault(link_ports[l], []).append(l)

    any_constraint_added = False

    for scen, snaps in snaps_by_scen.items():
        dt = _get_dt(n, snaps)

        # RHS: drawn energy excluding export links
        rhs_terms = []
        for port, links in links_by_port.items():
            var = _get_link_p_var(m, int(port))
            if var is None:
                continue

            var_names = set(var.coords["name"].values)
            selected = []
            for l in links:
                if _canon_link_name(l) in export_canon:
                    continue
                lname = _resolve_name_in_var(var, l)
                if lname in var_names:
                    selected.append(lname)

            if not selected:
                continue

            p = var.sel(snapshot=snaps, name=selected)
            rhs_terms.append((p * dt).sum(("snapshot", "name")))
            print('rhs_terms', rhs_terms)

        if not rhs_terms:
            # skip this scenario, keep going
            if warn:
                warnings.warn(
                    f"[add_max_RE_sales_constraint] No RHS draw terms found for scenario={scen}. "
                    f"Check Link-p variables. (skipping scenario)",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue

        drawn_energy = sum(rhs_terms)
        rhs = (alpha / (1 - alpha)) * drawn_energy
        #print('drawn_energy', drawn_energy)
        #print('rhs', rhs)

        # LHS: export energy
        lhs_terms = []
        for ek in export_keys:
            eport = link_ports.get(ek, 0)
            var = _get_link_p_var(m, int(eport))
            if var is None:
                continue

            ek_name = _resolve_name_in_var(var, ek)
            if ek_name not in set(var.coords["name"].values):
                continue

            pexp = var.sel(snapshot=snaps, name=ek_name)
            lhs_terms.append((pexp * dt).sum("snapshot"))

        if not lhs_terms:
            # skip this scenario, keep going
            if warn:
                warnings.warn(
                    f"[add_max_RE_sales_constraint] No export term found for scenario={scen}. "
                    f"Check export link presence/variables. (skipping scenario)",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue

        export_energy = sum(lhs_terms)
        #print('export_energy', export_energy)
        #print('lhs_terms', lhs_terms)

        cname = name if scen is None else f"{name}__scen_{scen}"
        m.add_constraints(export_energy <= rhs, name=cname)
        any_constraint_added = True

    if (not any_constraint_added) and warn:
        warnings.warn(
            "[add_max_RE_sales_constraint] Constraint not added for any scenario "
            "(all scenarios skipped due to missing terms).",
            RuntimeWarning,
            stacklevel=2,
        )

# --- OPTIMIZATION-----
def _apply_common_overrides(solver, opts, threads=None, time_limit=None):
    if threads is not None:
        key = "Threads" if solver == "gurobi" else "threads"
        opts[key] = int(threads)
    if time_limit is not None:
        key = "TimeLimit" if solver == "gurobi" else "time_limit"
        opts[key] = float(time_limit)

@contextmanager
def capture_pypsa_unassigned_constraints():
    logger = logging.getLogger("pypsa.optimization.optimize")
    records = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    h = _Handler()
    logger.addHandler(h)
    try:
        yield records
    finally:
        logger.removeHandler(h)


def parse_unassigned_constraint_blocks(log_messages):
    blocks = []
    for msg in log_messages:
        if "shadow-prices of the constraints" in msg:
            m = re.search(r"constraints (.*) were not assigned", msg)
            if m:
                blocks.extend([x.strip() for x in m.group(1).split(",")])

    # unique, preserve order
    seen, out = set(), []
    for b in blocks:
        if b and b not in seen:
            seen.add(b)
            out.append(b)
    return out


def export_constraint_duals(n, patterns, outpath, compress=True):
    m = n.model
    keys = [k for k in m.constraints if any(k.startswith(p) for p in patterns)]

    data_vars = {}
    for k in keys:
        c = m.constraints[k]
        d = getattr(c, "dual", None)
        if d is None:
            continue

        if "name" in d.dims:
            if k.startswith("Generator-"):
                d = d.rename({"name": "generator"})
            elif k.startswith("Link-"):
                d = d.rename({"name": "link"})
            elif k.startswith("Store-"):
                d = d.rename({"name": "store"})
            elif k.startswith("Bus-"):
                d = d.rename({"name": "bus"})

        # Drop problematic coordinate called 'name'
        if "name" in d.coords and "name" not in d.dims:
            d = d.reset_coords("name", drop=True)

        data_vars[k] = d

    if not data_vars:
        raise ValueError("No matching constraint duals found to export.")

    ds = xr.Dataset(data_vars)

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if compress:
        encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        ds.to_netcdf(outpath, encoding=encoding)
    else:
        ds.to_netcdf(outpath)

    return ds


def build_model_solve_network(
    n,
    results_folder,
    solver="gurobi",
    profile="gurobi-default",   # <-- explicit default
    io_api="direct",
    n_config=None,
    time_limit=None,
    threads=None,
    overrides=None,
    collect_all_duals=False,
    compress_duals=True,
    return_model=True,
    n_name=None,
):
    """
    Build the linopy model and add all custom constraints.
    """

    def _normalize_component_index_names(n):
        for attr in ["carriers", "buses", "links", "generators", "loads", "stores"]:
            if hasattr(n, attr):
                df = getattr(n, attr)
                idx = getattr(df, "index", None)
                if idx is None:
                    continue
                if isinstance(idx, pd.MultiIndex):
                    continue
                if idx.name != "name":
                    df.index.name = "name"
        try:
            if getattr(n.snapshots, "name", None) not in (None, "snapshot"):
                n.snapshots.name = "snapshot"
        except Exception:
            pass

    def assert_unique_component_names(n):
        for comp_name in ["buses", "links", "generators", "loads", "stores", "carriers"]:
            df = getattr(n, comp_name, None)
            if df is None:
                continue
            idx = df.index
            if isinstance(idx, pd.MultiIndex):
                dup = idx[idx.duplicated()].unique()
                if len(dup):
                    raise ValueError(f"{comp_name} has duplicated MultiIndex entries, e.g. {list(dup[:10])}")
            else:
                dup = idx[idx.duplicated()].unique()
                if len(dup):
                    raise ValueError(f"{comp_name} has duplicated names: {list(dup[:20])}")

    _normalize_component_index_names(n)

    try:
        m = n.optimize.create_model()
        print("Model variables:", list(m.variables))

        if getattr(n, "model", None) is not m:
            n.model = m
    except ValueError as e:
        if "Objective function could not be created" in str(e):
            n.meta = getattr(n, "meta", {}) or {}
            n.meta["objective"] = 0.0
            n.meta["opt_status"] = "skipped"
            n.meta["opt_termination"] = "no_costs"
            return None
        raise

    # ----  custom constraints  ----

    add_max_RE_sales_constraint(
        n,
        m,
        bus="El3",
        export_pattern="El3_to",
        export_bus="ElDK1 sell bus",
        alpha=max_RE_to_grid,
        name="El3_export_fraction_of_total_RE",
        n_flags=n_flags,
        include_agents=["biogas", "electrolysis", "methanation", "meoh"],  # NOTE DO NOT INCLUDE CENTRAL HEAT
    )

    #add_custom_constraints_stores(n, m, n_config=n_config)

    assert_unique_component_names(n)

    """
    Solve model
    """
    solver = solver.lower()

    # If user passes profile=None, fall back to a sensible default
    if profile is None:
        profile = "gurobi-default" if solver == "gurobi" else "highs-default"

    # Validate profile exists (clear error)
    if solver not in SOLVER_PROFILES:
        raise ValueError(f"Unknown solver '{solver}'. Available: {list(SOLVER_PROFILES)}")
    if profile not in SOLVER_PROFILES[solver]:
        raise ValueError(
            f"Unknown profile '{profile}' for solver '{solver}'. "
            f"Available: {list(SOLVER_PROFILES[solver].keys())}"
        )

    base = SOLVER_PROFILES[solver][profile]

    opts = deepcopy(base)
    _apply_common_overrides(solver, opts, threads=threads, time_limit=time_limit)
    if overrides:
        opts.update(overrides)

    if n_name is None:
        n_name = time.strftime("%Y%m%d_%H%M%S")
    results_folder = Path(results_folder)
    dual_dir = results_folder / "duals"
    dual_dir.mkdir(parents=True, exist_ok=True)

    # ---- Solve + (optional) dual collection ----
    print("MODEL ID before solve:", id(n.model))

    with capture_pypsa_unassigned_constraints() as log_msgs:
        status, condition = n.optimize.solve_model(
            solver_name=solver,
            io_api=io_api,
            assign_all_duals=False,
            **opts,
        )
        print("MODEL ID after solve:", id(n.model))

        # primals
        try:
            n.optimize.assign_solution()
        except Exception:
            pass

        if collect_all_duals:
            try:
                n.optimize.assign_duals(assign_all_duals=True)
            except Exception as e:
                # keep going; we can still export from linopy
                print(f"⚠️ PyPSA dual writeback raised (continuing): {e}")

    # ---- Export only unassigned constraints ----
    unassigned = []
    dual_export_path = None

    if collect_all_duals:
        unassigned = parse_unassigned_constraint_blocks(log_msgs)

        custom = [k for k in n.model.constraints if "__" in k]

        # export both: PyPSA-unassigned + all custom constraints
        patterns = list(dict.fromkeys(unassigned + custom))  # unique, keep order

        if patterns:
            dual_export_path = dual_dir / f"duals_export_{n_name}.nc"
            ds = export_constraint_duals(
                n,
                patterns=patterns,
                outpath=dual_export_path,
                compress=compress_duals,
            )
            print("Exported dual blocks:", [k for k in ds.data_vars])

    # ---- meta breadcrumbs ----
    n.meta = getattr(n, "meta", {}) or {}
    n.meta.update({
        "opt_status": str(status),
        "opt_termination": str(condition),
        "opt_solver": solver,
        "collect_all_duals": bool(collect_all_duals),
        "unassigned_dual_blocks": unassigned,
        "unassigned_duals_path": None if dual_export_path is None else str(dual_export_path),
    })

    if return_model:
        return status, condition, solver, opts, getattr(n, "model", None)
    return status, condition, solver, opts


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

# ---- STOCHASTIC OPTIMIZATION

def _get_objective_value(n):
    if hasattr(n, "objective") and n.objective is not None:
        try:
            return float(n.objective)
        except Exception:
            pass

    m = getattr(n, "model", None)
    if m is not None and getattr(m, "objective", None) is not None:
        val = getattr(m.objective, "value", None)
        if val is not None:
            return float(val)

    meta = getattr(n, "meta", None) or {}
    if "objective" in meta and meta["objective"] is not None:
        return float(meta["objective"])

    raise AttributeError("Could not find objective value on network.")


def compare_objective(n_stoch, ws_networks, probs):
    """
    Returns a DataFrame ready to be written to CSV.

    Index:
      scenario labels + RP + E_WS + EVPI

    Columns:
      WS_objective, probability, prob_weighted
    """
    probs_s = pd.Series({str(k): float(v) for k, v in probs.items()})
    ws_s = {str(k): v for k, v in ws_networks.items()}

    z_rp = _get_objective_value(n_stoch)

    ws_obj = {}
    for s, n_ws in ws_s.items():
        if s not in probs_s.index:
            raise KeyError(f"Scenario {s} missing from probs.")
        ws_obj[s] = _get_objective_value(n_ws)

    ws_by_scenario = pd.Series(ws_obj).sort_index()

    df = pd.DataFrame({
        "WS_objective": ws_by_scenario,
        "probability": probs_s.reindex(ws_by_scenario.index),
    })

    df["prob_weighted"] = df["WS_objective"] * df["probability"]

    E_WS = df["prob_weighted"].sum()
    EVPI = z_rp - E_WS

    summary = pd.DataFrame(
        {
            "WS_objective": [z_rp, E_WS, EVPI],
        },
        index=["RP", "E_WS", "EVPI"],
    )

    summary[["probability", "prob_weighted"]] = np.nan

    return pd.concat([df, summary])


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
    if targets_dict["driver"] == 'price':
        target = 'tP'
    elif targets_dict["driver"] == 'demand':
        target = 'tD'


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
    if targets_dict["driver"] == 'demand':
        H2_t   = annual_gwh("H2 grid")
        MeOH_t = annual_gwh("Methanol")
        CH4_t  = annual_gwh("bioCH4")

    elif targets_dict["driver"] == 'price':
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
            n_flags.get("biogas", False) * "B_" +
            n_flags.get("central_heat", False) * "H_" +
            n_flags.get("renewables", False) * "RE_" +
            n_flags.get("electrolysis", False) * "H2_" +
            n_flags.get("meoh", False) * "MEOH_" +
            n_flags.get("methanation", False) * "METH_" +
            n_flags.get("symbiosis", False) * "SN_" +
            n_flags.get("storage", False) * "ST_"
    )

    # ------------------
    # Filename
    # ------------------
    file_name = (
        f"{prefix}"
        f"CO2_{CO2_c}_"
        f"{target}_"
        f"H2_{H2_t}_"
        f"MeOH_{MeOH_t}_"
        f"CH4_{CH4_t}_"
        f"{year}_"
        f"El_{max_RE_to_grid}_"
        f"{stch}_"
        f"{run_name}"
    )

    return file_name


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
    #else:
        # print(f"Folder already exists: {folder_path}")
    return folder_path  # Return the full path of the folder

def export_network(n, n_flags, network_name, networks_folder, suffix, model = None):
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
        suffix : str
            String suffix (e.g., '_OPT', '_DET', etc.) added to filenames.
        model : linopy.Model, optional
            If provided, the Linopy model is saved as .nc in the same folder.
        stochastic : bool

        Returns
        -------
        str or None
            Full path of the exported network file (if any).
        """

    if not n_flags.get("export", False):
        return None

    filename_nc = f"{network_name}{suffix}.nc"
    nc_path = os.path.join(networks_folder, filename_nc)

    n.export_to_netcdf(nc_path)
    print(f"✅ {suffix} PyPSA network saved to: {nc_path}")

    if model is not None:
        try:
            model_filename = f"{network_name}{suffix}_model.nc"
            model_path = os.path.join(networks_folder, model_filename)
            model.to_netcdf(model_path)
            print(f"✅ Linopy model saved to: {model_path}")
        except Exception as e:
            print(f"[WARN] Could not export Linopy model: {e}")

    return nc_path


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
