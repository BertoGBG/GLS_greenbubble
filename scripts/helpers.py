import pandas as pd
import numpy as np
import pypsatopo
import parameters as p
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
import matplotlib as mpl
from matplotlib.patches import Patch

# -------TECHNO-ECONOMIC DATA & ANNUITY
def annuity(n, r):
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20,0.05)*20 = 1.6"""

    if r > 0:
        return r / (1. - 1. / (1. + r) ** n)
    else:
        return 1 / n

def prepare_costs(cost_file, USD_to_EUR, discount_rate, Nyears, lifetime):
    """ This function uses, data retrived form the technology catalogue and other sources and compiles a DF used in the model
    input: cost_file # csv
    output: costs # DF with all cost used in the model"""

    # Nyear = nyear in the interval for myoptic optimization--> set to 1 for annual optimization
    # set all asset costs and other parameters


    costs = pd.read_csv(cost_file, index_col=[0, 1]).sort_index()

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
    """Function that adds the tehcnology costs not presente din the original cost file"""

    for idx in other_tech_costs.index.values:
        investment = other_tech_costs.at[idx, 'investment']
        FOM = other_tech_costs.at[idx, 'FOM']
        lifetime = other_tech_costs.at[idx, 'lifetime']
        cost_add_technology(p.discount_rate, tech_costs, idx, investment, lifetime, FOM)

    return tech_costs


# --- OPTIMIZATION-----
def solve_network(net, solver="gurobi"):
    """Create and solve the Linopy model using gurobi; fall back to HiGHS if needed."""
    net.optimize.create_model()
    try:
        net.optimize.solve_model(solver_name=solver)
    except Exception as e:
        print(f"[WARN] {solver} failed: {e}\nFalling back to HiGHS.")
        net.optimize.solve_model(solver_name="highs")

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


def export_print_network(n, n_flags_opt, folder, file_name):
    # Define file name
    # export network and print layout using pypsatopo
    file_name = file_name + '_OPT'

    if n_flags_opt['print']:
        n_plot = optimal_network_only(n)
        filename = file_name + '_OPT.svg'
        full_path = os.path.join(folder, filename)
        pypsatopo.generate(n_plot, file_output=full_path, negative_efficiency=False, carrier_color=True)
    if n_flags_opt['export']:
        filename = file_name + '_OPT.nc'
        full_path = os.path.join(folder, filename)
        n.export_to_netcdf(full_path)
    return


# --- ANALAYSIS AND PLOT ----

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

    summer_start = str(p.En_price_year) + '-04-01T00:00'  # '2019-04-01 00:00:00+00:00' # Monday
    summer_end = str(p.En_price_year) + '-10-01T00:00'  # '2019-10-01 00:00:00+00:00'
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

