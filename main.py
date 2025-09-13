#!/usr/bin/env python3
# main.py

# Python modules
from pathlib import Path
import pickle as pkl
import numpy as np
import pandas as pd

# (Optional) use a non-interactive backend so it works on servers/CLI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#  Local modules
import parameters as p
from scripts.prepare_network import network_dependencies, build_network, file_name_network
from scripts.preprocessing import pre_processing_all_inputs
from scripts.retrieve import retrieve_technology_data
from scripts.helpers import prepare_costs, add_technology_cost, solve_network, create_folder_if_not_exists, export_print_network, get_total_marginal_capital_cost_agents
from scripts.plots import shadow_prices_violinplot, plot_El_Heat_prices, plot_bus_list_shadow_prices, save_opt_capacity_components, heat_map_CF

# ----

def main():

    # ---- Network flags and dependency checks
    n_flags = p.n_flags
    n_flags_OK = network_dependencies(n_flags)

    # ---- Preprocess inputs
    inputs_dict = pre_processing_all_inputs(n_flags_OK=n_flags_OK, flh_H2=p.flh_H2,
                                            f_max_MeOH_y_demand=p.f_max_MeOH_y_demand,
                                            f_max_Methanation_y_demand=p.f_max_Methanation_y_demand,
                                            CO2_cost=p.CO2_cost, el_DK1_sale_el_RFNBO=p.el_DK1_sale_el_RFNBO,
                                            preprocess_flag=p.preprocess_flag)

    # ---- Tech costs
    retrieve_technology_data(p.cost_file, p.cost_folder, p.technology_data_url)
    tech_costs = prepare_costs(p.cost_folder + '/' + p.cost_file, p.USD_to_EUR, p.discount_rate, 1, p.lifetime)
    add_technology_cost(tech_costs, p.other_tech_costs)

    # ---- Build network
    network = build_network(tech_costs, inputs_dict, n_flags_OK)

    # ---- Optimize
    n_flags_opt = {'print': True, 'export': True, 'plot': True}
    solve_network(network, solver="gurobi")  # or "highs" if you prefer

    # ---- Export/print
    network_opt = network.copy()
    file_name = file_name_network(network_opt, n_flags, inputs_dict)
    results_folder = create_folder_if_not_exists(p.print_folder_Opt, file_name)
    networks_folder = create_folder_if_not_exists(results_folder, 'networks')
    export_print_network(network_opt, n_flags_opt, folder=networks_folder, file_name=file_name)

    # ---- Save component allocation
    network_comp_allocation = network.network_comp_allocation
    networks_folder = Path(networks_folder)
    with open(networks_folder / 'network_comp_allocation.pkl', 'wb') as f:
        pkl.dump(network_comp_allocation, f)

    # ---- Plotting
    if n_flags_opt['plot']:
        plots_folder = create_folder_if_not_exists(results_folder, 'plots')

        # Costs by plant
        cc_tot_agent, mc_tot_agent = get_total_marginal_capital_cost_agents(
            network_opt, network_comp_allocation, True, plots_folder
        )

        # Violin of shadow prices (with clipping + note box from your updated function)
        shadow_prices_violinplot(
            network_opt, inputs_dict, tech_costs, plots_folder,
            handle_spikes="clip", quantile_hi=0.98, quantile_lo=0.02
        )

        # Internal electricity & heat prices
        plot_El_Heat_prices(network_opt, inputs_dict, tech_costs, plots_folder)

        # Demo Partial time series for shadow prices
        d_start = f"{p.En_price_year}-01-01"
        d_end = f"{p.En_price_year}-03-31"

        bus_list = ['El3 bus', 'H2 delivery', 'Heat LT', 'Methanol', 'H2_distribution',
                    'CO2_distribution', 'El2 bus']
        legend = ['El internal', 'LCOE H2 ', 'Heat LT ', 'LCOE MeOH', 'CO2 internal']

        plot_bus_list_shadow_prices(
            network_opt, bus_list, legend, d_start, d_end, plots_folder,
            handle_spikes='clip', quantile=0.95
        )

        # Save optimal capacities
        file_path = Path(plots_folder) / 'table_capacities'
        df_opt_components = save_opt_capacity_components(network_opt, network_comp_allocation, str(file_path))

        # Heat map demo
        key_comp_dict = {
            'generators': ['onshorewind', 'solar'],
            'links': ['DK1_to_El3', 'El3_to_DK1', 'Electrolyzer', 'Methanol plant'],
            'stores': ['H2 HP', 'CO2 pure HP', 'battery']
        }
        heat_map_CF(network_opt, key_comp_dict, plots_folder)

    print("Done.")

if __name__ == "__main__":
    main()
