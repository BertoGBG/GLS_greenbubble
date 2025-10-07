#!/usr/bin/env python3
# main.py

# Python modules
from pathlib import Path
import numpy as np
import pandas as pd

# (Optional) use a non-interactive backend so it works on servers/CLI
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

#  Local modules
import config as c
import parameters as p
from scripts.technology_inputs import tech_inputs
from scripts.prepare_network import network_dependencies, build_network
from scripts.preprocessing import pre_processing_all_inputs
from scripts.retrieve import retrieve_technology_data
from scripts.helpers import prepare_costs, solve_network, export_print_network, file_name_network, save_network_comp_allocation, create_folder_if_not_exists, save_config
from scripts.plots import single_opt_plots

# ----

#def main():

# ---- Network flags and dependency checks
n_flags = c.n_flags
n_flags_OK = network_dependencies(n_flags)

# ---- Tech costs
retrieve_technology_data(p.cost_path, p.technology_data_url)
tech_costs = prepare_costs(cost_path = p.cost_path,
                           tech_inputs= tech_inputs,
                           USD_to_EUR= c.USD_to_EUR,
                           discount_rate=c.discount_rate)

# ---- Preprocess inputs
inputs_dict = pre_processing_all_inputs(n_flags_OK = n_flags_OK,
                                        demand_H2 = c.demand_H2,
                                        demand_meoh = c.demand_meoh,
                                        demand_CH4 = c.demand_CH4,
                                        CO2_cost = c.CO2_cost,
                                        el_DK1_sale_el_RFNBO = c.el_DK1_sale_el_RFNBO,
                                        tech_costs  = tech_costs,
                                        preprocess_flag = c.preprocess_flag)

# ---- Build network
network = build_network(tech_costs, inputs_dict, n_flags_OK)

# ---Create results folders
network_name = file_name_network(n=network, n_flags=n_flags, run_name=c.run_name, inputs_dict= inputs_dict) # creates network name for saving results
results_folder = create_folder_if_not_exists(c.outputs_folder, network_name) # Based on n_flags and demands

# ---- Export and print prenetwork
export_print_network(network, n_flags, network_name=network_name, results_folder=results_folder, suffix ='_PRE')
print('network built')

# ---- Optimize
n_flags_opt = {'print': True, 'export': True, 'plot': True}
solve_network(network, solver="gurobi", profile="gurobi-default")# or "highs" (select solver options from solver_profiles_old.py)

# ---- Export and print postnetwork and configuration
network_opt = network.copy()
export_print_network(network_opt, n_flags_opt, network_name=network_name, results_folder=results_folder, suffix ='_OPT')
save_config (results_folder,c)
print('network optimized & saved')
print('results saved in' + results_folder)

# ---- Save component allocation
network_comp_allocation = network.network_comp_allocation
save_network_comp_allocation(results_folder, network_comp_allocation)

# ---- Plotting and saving figures
if n_flags_opt['plot']:
    single_opt_plots(network_opt, network_comp_allocation, inputs_dict, tech_costs, results_folder)
    print('plotting done')

print("Done.")

#if __name__ == "__main__":
#    main()
