#!/usr/bin/env python3
# greenbubble_main.py

# use a non-interactive backend so it works on servers/CLI
import matplotlib
from dask.dataframe.methods import assign

matplotlib.use("Agg")
from pathlib import Path

#  Local modules
import scripts.config as c
from scripts import parameters as p
from scripts.technology_inputs import tech_inputs
from scripts.prepare_network import network_dependencies, build_network
from scripts.preprocessing import prepare_all_inputs
from scripts.retrieve import retrieve_technology_data
from scripts.helpers import prepare_costs, solve_network, export_print_network, file_name_network, save_network_comp_allocation, create_folder_if_not_exists, save_config
from scripts.plots import single_opt_plots

# ----

def main(n_flags=None, run_name=None, outputs_folder=None):

    # config modification for Shapley Values
    if n_flags is not None: c.n_flags = n_flags
    if run_name is not None: c.run_name = run_name
    if outputs_folder is not None: c.outputs_folder = outputs_folder

    # ---- Network flags and dependency checks
    n_flags = c.n_flags
    print('Network agents')
    print(n_flags)
    n_flags_opt = c.n_flags_opt
    n_flags_OK = network_dependencies(n_flags)

    # ---- Tech costs
    retrieve_technology_data(p.cost_path, p.technology_data_url)
    tech_costs = prepare_costs(cost_path = p.cost_path,
                               tech_inputs= tech_inputs,
                               USD_to_EUR= c.USD_to_EUR,
                               discount_rate=c.discount_rate)

    # ---- Preprocess inputs
    inputs_dict = prepare_all_inputs(n_flags_OK = n_flags_OK,
                                            demand_H2 = c.demand_H2,
                                            demand_meoh = c.demand_meoh,
                                            demand_CH4 = c.demand_CH4,
                                            CO2_cost = c.CO2_cost,
                                            el_DK1_sale_el_RFNBO = c.el_DK1_sale_el_RFNBO,
                                            tech_costs  = tech_costs,
                                            preprocess_flag = c.preprocess_flag)

    # ---- Build network
    print('building the network')
    network = build_network(tech_costs, inputs_dict, n_flags_OK, c.n_options, p)

    # ---Create results folders
    network_name = file_name_network(n=network, n_flags=n_flags, run_name=c.run_name, inputs_dict= inputs_dict) # creates network name for saving results
    results_folder = create_folder_if_not_exists(c.outputs_folder, network_name) # Based on n_flags and demands

    # ---- Stochastic
    if c.stochastic:
        from scripts.create_stoch_scenarios import create_scenarios, scenarios, CO2_cost_s, share_bio_NG_s
        create_scenarios(network, scenarios, CO2_cost_s, share_bio_NG_s, n_flags_OK, tech_costs)
        results_folder = create_folder_if_not_exists(results_folder , 'stochastic')
        n_flags['print'] = False
        n_flags_opt['print'] = False

    # ---- Export and print prenetwork
    export_print_network(n=network, n_flags=n_flags, network_name=network_name, results_folder=results_folder, suffix ='_PRE')
    print('network built')

    # Optimize
    solve_network(network,
                  solver="gurobi",  # or "highs"
                  profile="gurobi-default",  # select solver options from solver_profiles_old.py
                  overrides= None,# {"DualReductions": 0},
                  assign_all_duals=False,  #  set to False for stochastic optimization
                  return_model=True)

    # ---- Export and print postnetwork and configuration
    export_print_network(n=network, n_flags = n_flags_opt, network_name=network_name, results_folder=results_folder, suffix ='_OPT')

    # save the config used for this run
    save_config (results_folder,c)

    # ---- Save component allocation
    network_comp_allocation = network.network_comp_allocation
    save_network_comp_allocation(results_folder, network_comp_allocation)

    # ---- Plotting and saving figures # TODO custom plotting with pypsa 1.0 not tested. for the moment use standard Pypsa plotting functions
    #if c.n_flags_opt['plot']:
    #    single_opt_plots(network, network_comp_allocation, inputs_dict, tech_costs, results_folder)
    #    print('plotting not done')

if __name__ == "__main__":

    main()
