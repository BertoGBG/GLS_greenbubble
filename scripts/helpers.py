import pandas as pd
import numpy as np
import pypsatopo
from config import En_price_year, discount_rate, outputs_folder
import parameters as p
import os
import inspect
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
import matplotlib as mpl
from matplotlib.patches import Patch
import pickle as pkl
from copy import deepcopy
from scripts.solver_profiles import SOLVER_PROFILES
import datetime as dt
import yaml

# -------NETWORK
def build_snapshots(En_price_year):
    start_day = str(En_price_year) + '-01-01'
    start_date = start_day + 'T00:00'  # keep the format 'YYYY-MM-DDThh:mm' when selecting start and end time
    end_day = str(En_price_year + 1) + '-01-01'
    end_date = end_day + 'T00:00'  # excludes form the data set

    hours_in_period = pd.date_range(start_date + 'Z', end_date + 'Z', freq='h')
    hours_in_period = hours_in_period.drop(hours_in_period[-1])

    # Check if it's a leap year
    def is_leap_year(year):
        return (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))

    if is_leap_year(En_price_year):
        # Remove all timestamps that fall on February 29
        hours_in_period = hours_in_period[~((hours_in_period.month == 2) & (hours_in_period.day == 29))]
    return hours_in_period, start_date, end_date

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
                  overrides=None, fallback_order=("highs",)):
    """
    Solve with Gurobi or HiGHS using named profiles.
    Returns: (status, condition, used_solver, used_options)
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

    n.optimize.create_model()

    # ---- primary attempt: pass options as **kwargs (correct) ----
    try:
        status, condition = n.optimize.solve_model(
            solver_name=solver,
            io_api=io_api,
            **opts,
        )
        return status, condition, solver, opts
    except Exception as e:
        print(f"[WARN] {solver} failed: {e}")

    # ---- fallbacks: try other solvers with their default profile ----
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
            return status, condition, fb, fb_opts
        except Exception as e2:
            print(f"[WARN] {fb} fallback failed: {e2}")

    raise RuntimeError("All solver attempts failed.")

#def solve_network(n, solver="gurobi"):
#    """Create and solve the Linopy model using gurobi; fall back to HiGHS if needed."""
#    n.optimize.create_model()
#    try:
#        n.optimize.solve_model(solver_name=solver)
#    except Exception as e:
#        print(f"[WARN] {solver} failed: {e}\nFalling back to HiGHS.")
#        n.optimize.solve_model(solver_name="highs")

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

    if 'methanation' in n.loads.index.values:
        Methanation_d = int(n.loads_t.p_set['methanation'].sum() // 1000)  # yearly production of MeOH in GWh
    else:
        Methanation_d = 0

    # CO2 tax
    CO2_c = int(CO2_cost)  # CO2 price in currency

    # year
    year = int(En_price_year)  # energy price year

    # max El to DK1
    el_DK1_sale_el_RFNBO = inputs_dict['el_DK1_sale_el_RFNBO']

    # agents
    file_name = n_flags['biogas'] * 'SB_' + n_flags['central_heat'] * 'CH_' + n_flags['renewables'] * 'RE_' + \
                n_flags['electrolysis'] * 'H2_' + n_flags['meoh'] * 'meoh_' + n_flags['methanation'] * 'meth_' + str(
        Methanation_d) + n_flags['symbiosis'] * 'SN_' + \
                n_flags['storage'] * 'ST_' + 'CO2c' + str(CO2_c) + '_' + 'H2d' + str(H2_d) + \
                '_' + 'MeOHd' + str(MeOH_d) + '_' + str(year) + '_' + 'El2DK1' + '_' + str(el_DK1_sale_el_RFNBO) + run_name

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
    dump_params_module(c, dst_folder=networks_folder, filename="config.yaml",
                       dataframes_as="records")


def export_print_network(n, n_flags, network_name, results_folder, suffix):
    # function that expoers a netowrk to a location and print the svg file from pypsa topo
    # n : pypsa network
    # n_flags: dict containig n_flags['print'] : bool and n_flags['export'] : bool
    # file_name : str
    # suffix : '_OPT' for postnetworks and '_PRE' for prenetworks

    # Create directories for saving files if not existing
    networks_folder = create_folder_if_not_exists(results_folder, 'networks')

    if suffix == '_OPT':
        n_plot = optimal_network_only(n)
    else:
        n_plot = n

    if n_flags['print']:
        filename = network_name + suffix + '.svg'
        full_path = os.path.join(networks_folder, filename)
        pypsatopo.generate(n_plot, file_output=full_path, negative_efficiency=False, carrier_color=True)
    if n_flags['export']:
        filename = network_name + suffix + '.nc'
        full_path = os.path.join(networks_folder, filename)
        n.export_to_netcdf(full_path)

    return

def save_network_comp_allocation (results_folder, network_comp_allocation):
    # save allocation of compeonts to each agent/plant in pkl file

    networks_folder = create_folder_if_not_exists(results_folder, 'networks')

    networks_folder = Path(networks_folder)
    with open(networks_folder / 'network_comp_allocation.pkl', 'wb') as f:
        pkl.dump(network_comp_allocation, f)

    return

# --- ANALYSIS AND PLOT ----

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

    summer_start = str(En_price_year) + '-04-01T00:00'  # '2019-04-01 00:00:00+00:00' # Monday
    summer_end = str(En_price_year) + '-10-01T00:00'  # '2019-10-01 00:00:00+00:00'
    winter_1 = pd.date_range(p.start_date + 'Z', summer_start + 'Z', freq='H')
    winter_1 = winter_1.drop(winter_1[-1])
    winter_2 = pd.date_range(summer_end + 'Z', p.end_date + 'Z', freq='H')
    winter_2 = winter_2.drop(winter_2[-1])
    winter = winter_1.append(winter_2)
    winter = winter[~((winter.month == 2) & (winter.day == 29))]
    summer = pd.date_range(summer_start + 'Z', summer_end + 'Z', freq='H')
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
            nonlocal bottom_pos, bottom_neg   # <-- fix
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


def reg_coef(x, y, label=None, color=None, hue=None, **kwargs):
    ''' function that calculates the pearson correlation conefficient (r) for plotting in PairGrid'''
    ax = plt.gca()
    r, p = pearsonr(x, y)
    ax.annotate('r = {:.2f}'.format(r), xy=(0.5, 0.5), xycoords='axes fraction', ha='center')
    ax.set_axis_off()
    return


def create_folder_if_not_exists(path, folder_name):
    # general function for storing plots
    folder_path = os.path.join(path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")
    return folder_path  # Return the full path of the folder


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
