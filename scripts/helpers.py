import pandas as pd
import numpy as np
import pypsatopo
from scripts.config import En_price_year, discount_rate, outputs_folder, CO2_cost_ref_year, share_bio_NG, stochastic
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
import json
import calendar
import os


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

def en_market_prices_w_CO2(inputs_dict, tech_costs, n_options):
    """Build market prices for electricity, natural gas, and district heating including CO₂ cost adjustments."""

    # --- 1. Base data from inputs_dict ---
    CO2_cost       = inputs_dict["CO2 cost"]
    CO2_emiss_El   = inputs_dict["CO2_emiss_El"]           # tCO2/MWh_el
    NG_price_year  = inputs_dict["NG_price_year"]           # currency/MWh
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
    if isinstance(NG_price_year, pd.DataFrame):
        NG_price_year = NG_price_year.iloc[:, 0]

    co2_intensity_ng = tech_costs.at["gas", "CO2 intensity"]  # tCO₂/MWh_th
    mk_NG_grid_price = (
        NG_price_year + co2_intensity_ng * (1 - share_bio_NG) * (CO2_cost - CO2_cost_ref_year)
    )
    mk_NG_grid_price = pd.Series(mk_NG_grid_price, index=el_grid_price.index)

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
        "DH_price": DH_price,
    }

    return en_market_prices


# --- Add CUSTOM CONSTRAINTS ----
def _exists(name, index):
    return name in index

def _is_extendable_store(n, name):
    return _exists(name, n.stores.index) and bool(n.stores.at[name, "e_nom_extendable"])

def _is_extendable_link(n, name):
    return _exists(name, n.links.index) and bool(n.links.at[name, "p_nom_extendable"])


def add_custom_constraints(n, n_config=None):
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

    # 1) Build Linopy model
    m = n.optimize.create_model()

    # 2) add constraints for stores ( charging and discharging rates)
    add_custom_constraints(n, n_config=n_config)

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

def file_name_network(n, n_flags, run_name, inputs_dict):
    """function that automatically creates a file name give a network"""
    # the netwrok name includes: the agents included,  the demands variables H2_d, MeOH_d, CO2 cost, bioChar credits
    # and max fraction of electricity sold externally
    # example: Biogas_CHeat_RE_H2_MeOH_SymN_CO2c200_H2d297_MeOHd68
    CO2_cost = inputs_dict['CO2 cost']

    # loads
    if 'H2 grid' in n.loads.index.values:
        H2_d = int(n.loads_t.p_set['H2 grid'].sum() // 1000)  # yearly production of H2 in GWh
    else:
        H2_d = 0

    if 'Methanol' in n.loads.index.values:
        MeOH_d = int(n.loads_t.p_set['Methanol'].sum() // 1000)  # yearly production of MeOH in GWh
    else:
        MeOH_d = 0

    if 'bioCH4' in n.loads.index.values:
        bioCH4_d = int(n.loads_t.p_set['bioCH4'].sum() // 1000)  # yearly production of MeOH in GWh
    else:
        bioCH4_d = 0

    # CO2 tax
    CO2_c = int(CO2_cost)  # CO2 price in currency

    # year
    year = int(En_price_year)  # energy price year

    # max El to DK1
    el_DK1_sale_el_RFNBO = inputs_dict['el_DK1_sale_el_RFNBO']

    # agents
    file_name = n_flags['biogas'] * 'SB_' + n_flags['central_heat'] * 'CH_' + n_flags['renewables'] * 'RE_' + \
                n_flags['electrolysis'] * 'H2_' + n_flags['meoh'] * 'meoh_' + n_flags['methanation'] * 'meth_' + n_flags['symbiosis'] * 'SN_' + \
                n_flags['storage'] * 'ST_' + 'CO2c' + str(CO2_c) + '_' + 'H2d' + str(H2_d) + \
                '_' + 'MeOHd' + str(MeOH_d) + '_' + 'CH4d' + str(bioCH4_d) + '_' + str(year) + '_' + 'ElDK1' + '_' + str(el_DK1_sale_el_RFNBO) + '_' + run_name

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

    n_plot = n
    full_path = None

    if n_flags.get('print'):
        filename = f"{network_name}{suffix}.svg"
        full_path = os.path.join(networks_folder, filename)
        pypsatopo.generate(n_plot, file_output=full_path,
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

