import pandas as pd
import numpy as np
import pypsa
import pypsatopo
import parameters as p
import os
import matplotlib.pyplot as plt
import math
import seaborn as sns
import re
from pathlib import Path
from scripts.preprocessing import en_market_prices_w_CO2, load_input_data
from scripts.helpers import (
    optimal_network_only,
    build_electricity_grid_price_w_tariff,
    get_system_cost,
    get_total_marginal_capital_cost_agents)


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

def shadow_prices_violinplot(
    n,
    inputs_dict,
    tech_costs,
    folder,
    handle_spikes="clip",      # 'clip' (per-series quantiles), 'iqr', or 'none'
    quantile_hi=0.98,          # per-series upper quantile
    quantile_lo=None,          # per-series lower quantile; defaults to 1-quantile_hi
    whisker=1.5,               # per-series IQR whisker if handle_spikes='iqr'
    floor_zero=False,          # set True to never show values < 0 after clipping
    note_text="dunkelflauten spikes clipped",
    median_color="crimson",
    median_linewidth=2.0
):
    """Violin plot of marginal (shadow) prices with **per-series** clipping only."""

    CO2_cost = inputs_dict['CO2 cost']
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    H2_d = 0
    meoh_d = 0
    fC_MeOH = 0
    if 'H2 grid' in n.loads.index:
        H2_d = int(n.loads_t.p_set['H2 grid'].sum() // 1000)  # GWh/y
    if 'Methanol' in n.loads.index:
        meoh_d = n.loads_t.p_set['Methanol'].sum()
        bioCH4_y_d = n.loads_t.p_set['bioCH4'].sum()
        CO2_MeOH_plant = 1 / n.links.efficiency['Methanol plant']
        bioCH4_CO2plant = n.links.efficiency['SkiveBiogas'] / n.links.efficiency2['SkiveBiogas']
        fC_MeOH = round((meoh_d * CO2_MeOH_plant) * bioCH4_CO2plant / bioCH4_y_d, 2)

    # Collect series (list of pd.Series)
    data, x_ticks_plot = [], []
    for b in n.buses_t.marginal_price.columns:
        if b == 'ElDK1 bus':
            continue
        s = pd.Series(n.buses_t.marginal_price[b], copy=False)
        if np.sum(s) != 0:
            data.append(s)
            x_ticks_plot.append(b)

    # Add grid price series (as Series)
    data.append(pd.Series(inputs_dict['Elspotprices'].squeeze()))
    x_ticks_plot.append('Elspotprices')
    data.append(pd.Series(en_market_prices['el_grid_price'].squeeze()))
    x_ticks_plot.append('ElDK1 (w/ tarif & CO2tax)')

    # --- PER-SERIES CLIPPING ---
    def clip_series_quantile(s: pd.Series) -> pd.Series:
        s = s.dropna()
        q_lo = quantile_lo if quantile_lo is not None else (1 - quantile_hi)
        lo, hi = s.quantile(q_lo), s.quantile(quantile_hi)
        if floor_zero:
            lo = max(lo, 0.0)
        return s.clip(lower=lo, upper=hi)

    def clip_series_iqr(s: pd.Series) -> pd.Series:
        s = s.dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - whisker * iqr, q3 + whisker * iqr
        if floor_zero:
            lo = max(lo, 0.0)
        return s.clip(lower=lo, upper=hi)

    if handle_spikes == "clip":
        data_vis = [clip_series_quantile(s) for s in data]
    elif handle_spikes == "iqr":
        data_vis = [clip_series_iqr(s) for s in data]
    else:
        data_vis = [s.dropna() for s in data]

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    vp = ax.violinplot(data_vis, showmeans=False, showmedians=True, showextrema=True)
    ax.set_xticks(range(1, len(x_ticks_plot) + 1), x_ticks_plot, rotation=90)
    ax.set_title('shadow prices in €/MWh or €/t (CO2)\nvariability during year')
    ax.grid(True)

    # Median styling
    vp['cmedians'].set_color(median_color)
    vp['cmedians'].set_linewidth(median_linewidth)

    # Info box (top-left)
    textstr = '\n'.join((
        r'CO2 tax (€/t)=%.0f' % (CO2_cost),
        r'H2 prod (GWh/y)=%.0f' % (H2_d),
        r'fC MeOH (frac. CO2_bg)=%.2f' % (fC_MeOH)))
    ax.text(0.05, 0.85, textstr, transform=ax.transAxes, fontsize=10, va='center',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # Note (top-right)
    if handle_spikes in ("clip", "iqr"):
        scope_note = f"{note_text}"
        if floor_zero:
            scope_note += "\n(floored at 0)"
        ax.text(0.98, 0.98, scope_note, transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.tight_layout()

    # Save
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / 'shd_prices_violin.png', dpi=300)
    plt.close(fig)
    return



""" PLOTS SINGLE OPTMIZATION """

def plot_duration_curve(ax, df_input, col_val):
    """plot duration curve from dataframe (df) with index being a DateTimeIndex
     col_val (str) indicate the name of the column with the value that must be plotted
     OUTPUTS:df_1_sorted
      for duration curve plt: x = df_1_sorted['duration'] and y =df_1_sorted[col_val]"""

    df_1 = df_input.copy()
    df_1['interval'] = 1  # time resolution of the index
    df_1_sorted = df_1.sort_values(by=[col_val], ascending=False)
    df_1_sorted['duration'] = df_1_sorted['interval'].cumsum()
    out = ax.plot(df_1_sorted['duration'], df_1_sorted[col_val])

    return out


def plot_El_Heat_prices(n_opt, inputs_dict, tech_costs, folder):
    """function that plots El and Heat prices in external grids and GLS"""

    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)
    el_grid_price_tariff, el_grid_sell_price_tariff = build_electricity_grid_price_w_tariff(inputs_dict['Elspotprices'])

    legend1 = ['DK1 price + tariff + CO2 tax', 'DK1 price + tariff', 'DK1 spotprice', 'GLS El price',
               'GLS El price for H2']
    legend2 = ['DK NG price + CO2 tax', 'DK NG price', 'GLS Heat MT price', 'GLS Heat DH price']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.plot(p.hours_in_period, en_market_prices['el_grid_price'])  # , label='DK1 price + tariff + CO2 tax')
    ax1.plot(p.hours_in_period, el_grid_price_tariff)  # , label='DK1 price + tariff')
    ax1.plot(p.hours_in_period, inputs_dict['Elspotprices'])  # , label='DK1 spotprice')
    # ax1.plot(p.hours_in_period, n_opt.buses_t.marginal_price['El2 bus'])#, label='GLS El price')
    # ax1.plot(p.hours_in_period, n_opt.buses_t.marginal_price['El3 bus'])#, label='GLS El price')
    ax1.set_ylabel('€/MWh')
    ax1.grid(True)
    ax1.legend(legend1)
    ax1.set_title('El prices time series')
    ax1.tick_params(axis='x', rotation=45)

    plot_duration_curve(ax2, pd.DataFrame(en_market_prices['el_grid_price']), 'SpotPrice EUR')
    plot_duration_curve(ax2, el_grid_price_tariff, 'SpotPrice EUR')
    plot_duration_curve(ax2, inputs_dict['Elspotprices'], 'SpotPrice EUR')
    # plot_duration_curve(ax2,pd.DataFrame(n_opt.buses_t.marginal_price['El2 bus']),'El2 bus')
    # plot_duration_curve(ax2,pd.DataFrame(n_opt.buses_t.marginal_price['El3 bus']),'El3 bus')
    ax2.set_ylabel('€/MWh')
    ax2.set_xlabel('h/y')
    ax2.legend(legend1)
    ax2.grid(True)
    ax2.set_title('El prices duration curve')

    ax3.plot(p.hours_in_period, en_market_prices['NG_grid_price'])  # , label='DK NG price + CO2 tax')
    ax3.plot(p.hours_in_period, inputs_dict['NG_price_year'])  # , label='DK NG price')
    # ax3.plot(p.hours_in_period, n_opt.buses_t.marginal_price['Heat MT'])#, label='GLS Heat MT price')
    # ax3.plot(p.hours_in_period, n_opt.buses_t.marginal_price['Heat DH'])#, label='GLS Heat DH price')
    ax3.set_ylabel('€/MWh')
    ax3.grid(True)
    ax3.legend(legend2)
    ax3.set_title('Heat prices time series')
    ax3.tick_params(axis='x', rotation=45)

    plot_duration_curve(ax4, pd.DataFrame(en_market_prices['NG_grid_price']), 'Neutral gas price EUR/MWh')
    plot_duration_curve(ax4, inputs_dict['NG_price_year'], 'Neutral gas price EUR/MWh')
    # plot_duration_curve(ax4,pd.DataFrame(n_opt.buses_t.marginal_price['Heat MT']),'Heat MT')
    # plot_duration_curve(ax4,pd.DataFrame(n_opt.buses_t.marginal_price['Heat DH']),'Heat DH')
    ax4.set_ylabel('€/MWh')
    ax4.set_xlabel('h/y')
    ax4.legend(legend2)
    ax4.grid(True)
    ax4.set_title('Heat prices durnation curves')


    # Save
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / 'el_heat_prices.png', dpi=300)
    plt.close(fig)
    return



def plot_bus_list_shadow_prices(
    n_opt,
    bus_list,
    legend,
    start_date,
    end_date,
    folder,
    handle_spikes="clip",   # 'clip' (default), 'limit', 'log', or 'none'
    quantile=0.95           # e.g., 0.95 or 0.99 depending on how extreme the spikes are
):
    """
    Plots shadow prices for the buses involved in the production of H2 and MeOH
    over the period [start_date, end_date]. Optionally suppresses large spikes so the
    main signal is visible.

    handle_spikes:
        - 'clip'  : cap values above the quantile threshold before plotting
        - 'limit' : keep data as-is but set y-axis upper limit to the threshold
        - 'log'   : plot on a log y-scale (no clipping/limiting)
        - 'none'  : no spike handling

    quantile: high quantile used to determine the spike threshold.
    """

    # date format: 'YYYY-MM-DDTHH:MM'
    time_ok = p.hours_in_period[(p.hours_in_period >= start_date) & (p.hours_in_period <= end_date)]

    # Pull once; filter time and only needed buses
    df_data = n_opt.buses_t.marginal_price.copy()
    df_plot = df_data.loc[time_ok, bus_list].copy()

    # Compute a common threshold across all selected series in the window
    # (so all lines are treated consistently)
    threshold = df_plot.stack().quantile(quantile)

    # Prepare the version to visualize depending on the chosen method
    if handle_spikes == "clip":
        df_vis = df_plot.clip(upper=threshold)
    else:
        df_vis = df_plot  # unchanged

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    # --- Time series ---
    for b in bus_list:
        ax1.plot(df_vis[b], label=b)

    ax1.set_ylabel('€/MWh or €/(t/h)')
    ax1.grid(True)
    ax1.set_title('time series')

    # Apply axis handling
    if handle_spikes == "limit":
        ax1.set_ylim(bottom=None, top=threshold)
    elif handle_spikes == "log":
        ax1.set_yscale('log')

    # Use user-supplied legend labels if provided; otherwise default to bus names
    if legend:
        ax1.legend(legend)
    else:
        ax1.legend(bus_list)

    ax1.text(0.02, 0.95, "dunkelflauten spikes clipped",
             transform=ax1.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    # --- Duration curves ---
    # If your plot_duration_curve expects a DataFrame and a column name,
    # pass the same df_vis so duration curves are drawn from clipped/limited data
    for b in bus_list:
        plot_duration_curve(ax2, df_vis, b)

    ax2.set_ylabel('€/MWh or €/(t/h)')
    ax2.set_xlabel('h/y')
    ax2.grid(True)
    ax2.set_title('duration curves')
    if legend:
        ax2.legend(legend)
    else:
        ax2.legend(bus_list)

    # Save
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    plt.show()
    fig.savefig(folder / 't_int_shadow_prices.png', dpi=300)
    plt.close(fig)


def save_opt_capacity_components(n_opt, network_comp_allocation, file_path):
    """function that creates and saves as png a DF with all the components in the optimal nework, including their
    capacities and annualized capital costs"""

    df_opt_componets = pd.DataFrame()

    agent_list_cost = []

    for key in network_comp_allocation:
        df_agent = pd.DataFrame(data=0, index=[],
                                columns=['Fixed cost (€/y)', 'capacity', 'component', 'reference inlet', 'unit',
                                         'agent'])

        agent_list_cost.append(key)
        agent_generators_n_opt = list(
            set(network_comp_allocation[key]['generators']).intersection(set(n_opt.generators.index)))
        agent_links_n_opt = list(set(network_comp_allocation[key]['links']).intersection(set(n_opt.links.index)))
        agent_stores_n_opt = list(set(network_comp_allocation[key]['stores']).intersection(set(n_opt.stores.index)))

        for g in agent_generators_n_opt:
            df_agent.at[g, 'Fixed cost (€/y)'] = n_opt.generators.p_nom_opt[g] * n_opt.generators.capital_cost[g]
            df_agent.at[g, 'capacity'] = n_opt.generators.p_nom_opt[g]
            df_agent.at[g, 'reference inlet'] = n_opt.generators.bus[g]
            df_agent.at[g, 'unit'] = n_opt.buses.unit[n_opt.generators.bus[g]]
            df_agent.at[g, 'plant'] = key
            df_agent.at[g, 'component'] = 'generator'

        for l in agent_links_n_opt:
            df_agent.at[l, 'Fixed cost (€/y)'] = n_opt.links.capital_cost[l] * n_opt.links.p_nom_opt[l]
            df_agent.at[l, 'capacity'] = n_opt.links.p_nom_opt[l]
            df_agent.at[l, 'reference inlet'] = n_opt.links.bus0[l]
            df_agent.at[l, 'unit'] = n_opt.buses.unit[n_opt.links.bus0[l]]
            df_agent.at[l, 'plant'] = key
            df_agent.at[l, 'component'] = 'link'

        for s in agent_stores_n_opt:
            df_agent.at[s, 'Fixed cost (€/y)'] = n_opt.stores.capital_cost[s] * n_opt.stores.e_nom_opt[s]
            df_agent.at[s, 'capacity'] = n_opt.stores.e_nom_opt[s]
            df_agent.at[s, 'reference inlet'] = n_opt.stores.bus[s]
            df_agent.at[s, 'unit'] = n_opt.buses.unit[n_opt.stores.bus[s]]
            df_agent.at[s, 'plant'] = key
            df_agent.at[s, 'component'] = 'store'

        df_opt_componets = pd.concat([df_opt_componets, df_agent])

    "save to csv"
    df_opt_componets.to_csv(file_path + '.csv')

    return df_opt_componets


def plot_heat_map_single_comp(df_time_serie, ax=None, label_freq_days=14, vmin=0, vmax=None, title=None):
    """
    Draw a heat map on the given Axes:
      x-axis: every day of the year (data columns)
      y-axis: hour of day (0–23)
      labels every `label_freq_days` days.
    """
    if ax is None:
        ax = plt.gca()

    col_name = str(df_time_serie.columns.values.squeeze())

    df = df_time_serie.copy()
    df["day_of_year"] = df.index.dayofyear
    df["hour_of_day"] = df.index.hour
    df["date"] = df.index.date

    heat_df = df.pivot_table(index="hour_of_day",
                             columns="day_of_year",
                             values=col_name,
                             aggfunc="mean")

    # Plot on provided axes (no plt.figure/plt.show here)
    hm = sns.heatmap(heat_df, cmap="YlGn", vmin=vmin, vmax=vmax, cbar=True, ax=ax)

    # Titles & labels
    ax.set_title(title or col_name, fontsize=10)
    ax.set_ylabel("Hour")

    # Ticks every N days with date labels
    all_days = heat_df.columns
    tick_positions = [i for i in range(len(all_days)) if i % label_freq_days == 0]
    tick_labels = [df[df["day_of_year"] == d]["date"].iloc[0].strftime("%b-%d")
                   for d in all_days[tick_positions]]
    ax.set_xticks([pos + 0.5 for pos in tick_positions])
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    # Fewer y ticks (optional: 0, 6, 12, 18)
    ax.set_yticks([0.5, 6.5, 12.5, 18.5])
    ax.set_yticklabels(["00", "06", "12", "18"])

def heat_map_CF(network_opt, key_comp_dict, folder):
    # Filter components that exist
    heat_map_comp_list = {
        'generators': list(set(key_comp_dict.get('generators', [])).intersection(network_opt.generators.index)),
        'links':      list(set(key_comp_dict.get('links', [])).intersection(network_opt.links.index)),
        'stores':     list(set(key_comp_dict.get('stores', [])).intersection(network_opt.stores.index)),
    }

    # Build DF (columns = components)
    df_cf_comp_ts = pd.DataFrame(index=network_opt.snapshots)

    # Avoid division by zero; use where denom>0
    for g in heat_map_comp_list['generators']:
        denom = network_opt.generators.p_nom_opt[g]
        series = network_opt.generators_t.p[g] / denom if denom and denom != 0 else 0.0
        df_cf_comp_ts[g] = series

    for l in heat_map_comp_list['links']:
        denom = network_opt.links.p_nom_opt[l]
        series = network_opt.links_t.p0[l] / denom if denom and denom != 0 else 0.0
        df_cf_comp_ts[l] = series

    for s in heat_map_comp_list['stores']:
        denom = network_opt.stores.e_nom_opt[s]
        series = network_opt.stores_t.e[s] / denom if denom and denom != 0 else 0.0
        df_cf_comp_ts[s] = series

    # Drop all-NA columns (just in case)
    df_cf_comp_ts = df_cf_comp_ts.dropna(axis=1, how="all")

    if df_cf_comp_ts.shape[1] == 0:
        print("No matching components with time series found.")
        return

    n_comp = df_cf_comp_ts.shape[1]
    n_rows = math.ceil(n_comp ** 0.5)
    n_cols = n_rows  # square-ish grid

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 2.8*n_rows))
    axes = np.atleast_2d(axes)  # ensure 2D

    fig.suptitle('Capacity Factors – daily × hourly patterns', fontsize=14)

    # Optional common color scale across all subplots (0..1 for CF)
    vmin, vmax = 0.0, 1.0

    # Plot each component into its subplot
    for k, comp in enumerate(df_cf_comp_ts.columns):
        r, c = divmod(k, n_cols)
        ax = axes[r, c]
        df_time_serie = pd.DataFrame(df_cf_comp_ts[comp])
        plot_heat_map_single_comp(
            df_time_serie,
            ax=ax,
            label_freq_days=30,
            vmin=vmin, vmax=vmax,
            title=comp
        )

    # Hide any unused subplots
    for k in range(n_comp, n_rows*n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].axis("off")

    plt.subplots_adjust(hspace=0.6, wspace=0.4)
    # Do NOT call plt.show() for batch saving
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / 'heat_map.png', dpi=300, bbox_inches="tight")
    plt.close(fig)



# -------- SENSITIVITY ANALYSIS

def results_df_plot_build(data_folder, dataset_flags, results_flags, network_comp_allocation, capacity_list):
    '''Function that reads all optimization runs in data_folder, selected by dataset_flags and
    import the variables selected in results_flags to the dataframe df_results.
    The dictionary results_plot contains the names of the variables with units to be used in the plots,
     it uses the same keys of results_flags'''

    _, GL_eff, _, _, bioCH4_prod, _, _, _, _, _, _, _ = load_input_data()

    ''' Build list of files to import '''
    name_files = []
    for f in os.listdir(data_folder):
        if f.endswith('.nc'):

            # check if CO2 cost in the file is among the selected ones
            m = re.search('CO2c(\d+)', f)
            m_co2 = int(m.group(1))

            # check if H2 demand in the file is among the selected ones
            m = re.search('H2d(\d+)', f)
            m_h2 = int(m.group(1)) * 1000 / p.H2_output

            # check if fC_MeOH in the file is among the selected ones
            m = re.search('MeOHd(\d+)', f)
            MeOH_y_d = int(m.group(1)) * 1000  # demand in GWh/y
            bioCH4_y_d = bioCH4_prod.values.sum()
            CO2_MeOH_plant = 1 / GL_eff.at['Methanol', 'Methanol plant']  # bus0 = CO2, bus1 = Methanol
            bioCH4_CO2plant = GL_eff.at['bioCH4', 'SkiveBiogas'] / GL_eff.at['CO2 pure', 'SkiveBiogas']
            fC_MeOH_value = round((MeOH_y_d * CO2_MeOH_plant) * bioCH4_CO2plant / bioCH4_y_d, 2)
            fC_MeOH = min(dataset_flags['fC_MeOH'], key=lambda x: abs(x - fC_MeOH_value))

            # check if DH in the file is among the selected ones
            cond_DH = np.sign(f.find('DH') + 1) in dataset_flags['DH']

            # check if En_year in the file is among the selected ones
            y_list = [str(y) for y in dataset_flags['En_year_price']]
            cond_En_y = any(ele in f for ele in y_list)

            # check if bioChar in the file name is among selected ones
            cond_bioChar = np.sign(f.find('bCh') + 1) in dataset_flags['bioChar']

            # check if el_DK1_sale_el_RFNBO in the file name is among selected ones
            m = re.search('El2DK1_(\d+(\.\d*)?)', f)
            m_El2DK1 = float(m.group(1))

            # append files names to file list
            if (m_co2 in dataset_flags['CO2_cost']) and (m_h2 in dataset_flags['d_H2']) and (
                    fC_MeOH in dataset_flags['fC_MeOH']) and cond_DH and cond_En_y and (
                    m_El2DK1 in dataset_flags['el_DK1_sale_el_RFNBO']):
                name_files.append(f)

    ''' Units per variable '''
    results_units = {
        'CO2_cost': '(€/t)',  # input parameter
        'fC_MeOH': '(% CO2 sep)',  # input parameter
        'd_H2': '(GWh/y)',  # input parameter
        'En_year_price': '(y)',  # input parameter
        'DH': '(-)',  # input parameter
        'el_DK1_sale_el_RFNBO': '(% El PtX)',  # input parameter
        'bioChar': '(-)',  # input parameter
        'DH_y': '(GWh/y)',  # output variable
        'RE_y': '(GWh/y)',  # output variable
        'MeOH_y': '(GWh/y)',  # output variable
        'mu_H2': '(€/MWh)',  # output variable
        'mu_MeOH': '(€/MWh)',  # output variable
        'mu_el_GLS': '(€/MWh)',  # output variable
        'mu_heat_MT': '(€/MWh)',  # output variable
        'mu_heat_DH': '(€/MWh)',  # output variable
        'mu_heat_LT': '(€/MWh)',  # output variable
        'mu_CO2': '(€/t)',  # output variable
        'mu_bioCH4': '(€/MWh)',  # output variable
        'H2_sales': '(€/y)',  # output variable
        'MeOH_sales': '(€/y)',  # output variable
        'RE_sales': '(€/y)',  # output variable
        'DH_sales': '(€/y)',  # output variable
        'BECS_sales': '(€/y)',  # output variable
        'bioCH4_sales': '(€/y)',  # output variable
        'tot_sys_cost': '(€/y)',  # output variable
        'tot_cap_cost': '(€/y)',  # output variable
        'tot_mar_cost': '(€/y)',  # output variable
    }
    agent_dict3 = dict((key + '_cc', '(€/y)') for key in network_comp_allocation)
    agent_dict4 = dict((key + '_mc', '(€/y)') for key in network_comp_allocation)
    results_units.update(agent_dict3)
    results_units.update(agent_dict4)

    '''Plot name per variable '''
    results_plot_name = {
        'CO2_cost': 'CO2 tax',
        'fC_MeOH': 'fCO2 to MeOH',
        'd_H2': 'H2 to Grid',
        'En_year_price': 'Energy year',
        'DH': 'DH',
        'el_DK1_sale_el_RFNBO': 'max RE to grid',
        'bioChar': 'biochar credits',
        'DH_y': 'DH production',
        'RE_y': 'RE production',
        'MeOH_y': 'MeOH prod',
        'mu_H2': r"$\lambda$" + ' H2',
        'mu_MeOH': r"$\lambda$" + ' MeOH',
        'mu_el_GLS': r"$\lambda$" + ' El GLS',
        'mu_heat_MT': r"$\lambda$" + ' heat MT',
        'mu_heat_DH': r"$\lambda$" + ' heat DH',
        'mu_heat_LT': r"$\lambda$" + ' heat LT',
        'mu_CO2': r"$\lambda$" + ' CO2',
        'mu_bioCH4': r"$\lambda$" + ' bioCH4',
        'H2_sales': 'H2_sales',
        'MeOH_sales': 'MeOH_sales',
        'RE_sales': 'RE sales',
        'DH_sales': 'DH sales',
        'bioCH4_sales': 'bioCH4 sales',
        'BECS_sales': 'biochar sales',
        'tot_sys_cost': 'tot system cost',
        'tot_cap_cost': 'tot capital cost',
        'tot_mar_cost': 'tot marginal cost',
        'external_grids_cc': 'external grids cap. cost',
        'SkiveBiogas_cc': 'SkiveBiogas cap. cost',
        'renewables_cc': 'renewables cap. cost',
        'electrolyzer_cc': 'electrolyzer cap. cost',
        'meoh_cc': 'MeOH cap. cost',
        'central_heat_cc': 'central heating cap. cost',
        'symbiosis_net_cc': 'symbiosis net cap. cost',
        'DH_cc': 'DH cap. cost',
        'external_grids_mc': 'external grids mar. cost',
        'SkiveBiogas_mc': 'SkiveBiogas mar. cost',
        'renewables_mc': 'renewables mar. cost',
        'electrolyzer_mc': 'electrolyzer mar. cost',
        'meoh_mc': 'MeOH mar. cost',
        'central_heat_mc': 'central heating mar. cost',
        'symbiosis_net_mc': 'symbiosis net mar. cost',
        'DH_mc': 'DH mar. cost',
    }

    ''' Build dictionary with plot names and units'''
    results_plot = {}
    for key in results_flags:
        if results_flags[key]:
            results_plot[key] = results_plot_name[key] + '\n' + results_units[key]
            # results_plot[key] = results_plot_name[key] + results_units[key]

    '''Load networks and retrive variables'''
    # define Results Data Frame
    results_columns = []
    for key in results_flags:
        if results_flags[key]:
            results_columns.append(key)

    # Data frame for results
    df_results = pd.DataFrame(0, index=name_files, columns=results_columns)

    # Load results according to
    for name in name_files:
        # import network
        n_name = 'n_' + 'name'  # network name
        n_name = pypsa.Network(os.path.join(data_folder, name))

        # Independent parameters
        if results_flags['CO2_cost']:
            m = re.search('CO2c(\d+)', name)
            df_results.at[name, 'CO2_cost'] = int(m.group(1))

        if results_flags['CO2_cost']:
            MeOH_y_d = n_name.loads_t.p_set['Methanol'].sum()
            bioCH4_y_d = n_name.loads_t.p_set['bioCH4'].sum()
            CO2_MeOH_plant = 1 / n_name.links.efficiency['Methanol plant']  # bus0 = CO2, bus1 = Methanol
            bioCH4_CO2plant = n_name.links.efficiency['SkiveBiogas'] / n_name.links.efficiency2[
                'SkiveBiogas']  # bus0 = biomass, bus1= bioCH4, bus2=CO2
            fC_MeOH = round((MeOH_y_d * CO2_MeOH_plant) * bioCH4_CO2plant / bioCH4_y_d, 2)
            df_results.at[name, 'fC_MeOH'] = fC_MeOH

        if results_flags['d_H2']:
            m = re.search('H2d(\d+)', name)
            df_results.at[name, 'd_H2'] = int(m.group(1))

        if results_flags['En_year_price']:
            if '2019' in name:
                df_results.at[name, 'En_year_price'] = 2019
            elif '2022' in name:
                df_results.at[name, 'En_year_price'] = 2022

        if results_flags['DH']:
            if 'DH' in name:
                df_results.at[name, 'DH'] = 1
            else:
                df_results.at[name, 'DH'] = 0

        if results_flags['el_DK1_sale_el_RFNBO']:
            m = re.search('El2DK1_(\d+(\.\d*)?)', name)
            m_El2DK1 = float(m.group(1))
            df_results.at[name, 'el_DK1_sale_el_RFNBO'] = m_El2DK1

        if results_flags['bioChar']:
            if 'bCh' in name:
                df_results.at[name, 'bioChar'] = 1
                df_results.at[name, 'BECS_sales'] = (
                        n_name.links_t.p0['biochar credits'] * n_name.links.marginal_cost['biochar credits']).sum()
            else:
                df_results.at[name, 'bioChar'] = 0
                df_results.at[name, 'BECS_sales'] = 0

        # Output variables
        # DH y production
        if results_flags['DH_y']:
            if 'DH' in name:
                df_results.at[name, 'DH_y'] = int(n_name.links_t.p0['DH GL_to_DH grid'].sum() // 1000)
            else:
                df_results.at[name, 'DH_y'] = 0
        if results_flags['DH_sales']:
            if 'DH' in name:
                df_results.at[name, 'DH_sales'] = df_results.at[name, 'DH_y'] * np.mean(
                    n_name.links.marginal_cost['DH GL_to_DH grid'])
            else:
                df_results.at[name, 'DH_sales'] = 0

        if results_flags['RE_y']:
            df_results.at[name, 'RE_y'] = int(n_name.links_t.p0['El3_to_DK1'].sum() // 1000)
        if results_flags['RE_sales']:
            df_results.at[name, 'RE_sales'] = int(
                (n_name.links_t.p0['El3_to_DK1'] * n_name.links_t.marginal_cost['El3_to_DK1']).sum())

        if results_flags['MeOH_y']:
            df_results.at[name, 'MeOH_y'] = int(n_name.loads_t.p_set['Methanol'].sum() // 1000)

        if results_flags['mu_H2']:
            m = re.search('H2d(\d+)', name)
            d_H2 = int(m.group(1))
            if d_H2 == 0:
                df_results.at[name, 'mu_H2'] = np.mean(n_name.buses_t.marginal_price['H2_meoh'])  # * p.lhv_h2 /1000
            else:
                df_results.at[name, 'mu_H2'] = np.mean(
                    n_name.buses_t.marginal_price['H2 delivery'])  # * p.lhv_h2 / 1000
            df_results.at[name, 'H2_sales'] = df_results.at[name, 'mu_H2'] * df_results.at[name, 'd_H2']

        if results_flags['mu_MeOH']:
            df_results.at[name, 'mu_MeOH'] = np.mean(n_name.buses_t.marginal_price['Methanol'])  # * p.lhv_meoh
            df_results.at[name, 'MeOH_sales'] = df_results.at[name, 'mu_MeOH'] * df_results.at[name, 'MeOH_y']

        # bioCH4 revenues (NOTE: it is a DELTA from standard operation)
        if results_flags['mu_bioCH4']:
            df_results.at[name, 'mu_bioCH4'] = np.mean(n_name.buses_t.marginal_price['bioCH4'])
            df_results.at[name, 'bioCH4_sales'] = bioCH4_y_d * df_results.at[name, 'mu_bioCH4']

        # El GL cost
        if results_flags['mu_el_GLS']:
            el_GL_bus = 'El2 bus'  # chose the representative El bus for a company at GLS. Note MeOH is connected to El3 bus
            df_results.at[name, 'mu_el_GLS'] = np.mean(n_name.buses_t.marginal_price['El2 bus'])

        # Heat MT GL cost
        if results_flags['mu_heat_MT']:
            if 'Heat MT' in n_name.buses.index.values:
                df_results.at[name, 'mu_heat_MT'] = np.mean(n_name.buses_t.marginal_price['Heat MT'])
            else:
                df_results.at[name, 'mu_heat_MT'] = 0

        # Heat DH GL cost
        if results_flags['mu_heat_DH']:
            if 'Heat DH' in n_name.buses.index.values:
                df_results.at[name, 'mu_heat_DH'] = np.mean(n_name.buses_t.marginal_price['Heat DH'])
            else:
                df_results.at[name, 'mu_heat_DH'] = 0

        # Heat LT GL cost
        if results_flags['mu_heat_DH']:
            if 'Heat LT' in n_name.buses.index.values:
                df_results.at[name, 'mu_heat_LT'] = np.mean(n_name.buses_t.marginal_price['Heat LT'])
            else:
                df_results.at[name, 'mu_heat_LT'] = 0

        # CO2 price GL sold by Biogas plant (LP)
        if results_flags['mu_CO2']:
            df_results.at[name, 'mu_CO2'] = np.mean(n_name.buses_t.marginal_price['CO2 sep'])

        # total system , capital and marginal cost
        tot_cc, tot_mc, tot_sc = get_system_cost(n_name)
        if results_flags['tot_sys_cost']:
            df_results.at[name, 'tot_sys_cost'] = np.sum(tot_sc)
        if results_flags['tot_cap_cost']:
            df_results.at[name, 'tot_cap_cost'] = np.sum(tot_cc)
        if results_flags['tot_mar_cost']:
            df_results.at[name, 'tot_mar_cost'] = np.sum(tot_mc)

        # total capital and marginal costs, by agents
        cc_tot_agent, mc_tot_agent = get_total_marginal_capital_cost_agents(n_name, network_comp_allocation, False)
        for key in cc_tot_agent:
            if results_flags[key + '_cc']:
                df_results.at[name, key + '_cc'] = cc_tot_agent[key]
            if results_flags[key + '_mc']:
                df_results.at[name, key + '_mc'] = mc_tot_agent[key]

        # import capacities
        # capacity_list = ['solar', 'onshorewind', 'Electrolyzer', 'El3_to_El2', 'El3_to_DK1', 'Methanol plant', 'SkyClean', 'Heat pump', 'CO2 compressor', 'H2 compressor', 'battery', 'Heat DH storage', 'Concrete Heat MT storage', 'H2 HP', 'CO2 pure HP', 'CO2 Liq']

        for c in capacity_list:

            if c == 'Methanol plant':
                f = GL_eff.at['Methanol', 'Methanol plant']
            elif c == 'H2 compressor':
                f = p.el_comp_H2
            elif c == 'CO2 compressor':
                f = p.el_comp_CO2
            else:
                f = 1

            if c in n_name.generators.p_nom_opt.index:
                df_results.at[name, c] = n_name.generators.p_nom_opt[c] * f
            elif c in n_name.links.p_nom_opt.index:
                df_results.at[name, c] = n_name.links.p_nom_opt[c] * f
            elif c in n_name.stores.e_nom_opt.index:
                df_results.at[name, c] = n_name.stores.e_nom_opt[c] * f
            else:
                df_results.at[name, c] = 0

    return df_results, results_plot
