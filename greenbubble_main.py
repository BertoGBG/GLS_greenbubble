#!/usr/bin/env python3
# greenbubble_main.py

# use a non-interactive backend so it works on servers/CLI
import pypsa

#matplotlib.use("Agg")
import pandas as pd
from pathlib import Path


#  Local modules
import scripts.config as c
from scripts import parameters as p
from scripts.technology_inputs import tech_inputs
from scripts.prepare_network import network_dependencies, build_network
from scripts.preprocessing import prepare_all_inputs
from scripts.retrieve import retrieve_technology_data
from scripts.helpers import (prepare_costs,
                             build_model_solve_network,
                             export_network,
                             file_name_network,
                             save_network_comp_allocation,
                             create_folder_if_not_exists,
                             save_config,
                             compare_objective,
                             assert_stochastic_schema_consistent,
                             )
from scripts.plots import run_plot_and_export, print_network


# ---- MAIN ----

def main(n_flags=None, run_name=None, outputs_folder=None):

    # config modification for Coalitional Game Theory
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

    # build cost DF based on coordinates (EU or US)
    tech_costs= prepare_costs(latitude = c.latitude,
                              longitude = c.longitude,
                              tech_inputs = tech_inputs,
                              USD_to_EUR = c.USD_to_EUR ,
                              discount_rate = c.discount_rate,
                              cost_path_EU = p.cost_path,
                              cost_path_US = p.cost_path_US,
                              dict_tech_US_EU = p.dict_tech_US_EU)

    # ---- Preprocess inputs
    inputs_dict = prepare_all_inputs(targets_dict = c.targets_dict,
                                     CO2_cost = c.CO2_cost,
                                     CO2_cost_ref_year = c.CO2_cost_ref_year,
                                     max_RE_to_grid = c.max_RE_to_grid,
                                     preprocess_flag = c.preprocess_flag)

    # ---- Build network (deterministic)
    print('building the network')
    n = build_network(tech_costs, inputs_dict, n_flags_OK, c.n_options, p)
    n.consistency_check()

    # ---Create results folders
    network_name = file_name_network(n=n, n_flags=n_flags, run_name=c.run_name, inputs_dict= inputs_dict, targets_dict=c.targets_dict, En_price_year=c.En_price_year, stochastic=c.stochastic['stochastic']) # creates network name for saving results
    analysis_folder = create_folder_if_not_exists(c.outputs_folder, network_name)  # Based on n_flags and demands

    # dicts for running a series of networks
    networks_dict = {}
    n_names_dict = {}

    # ---- Stochastic ----
    if c.stochastic['stochastic']:

        from scripts.create_stoch_scenarios import create_scenarios, set_input_paths ,scenarios, CO2_cost_s, CO2_cost_ref_year_s

        # create Stochastic Network with scenarios (modifies permanently the network)
        create_scenarios(n, scenarios, CO2_cost_s, CO2_cost_ref_year_s, n_flags_OK, tech_costs)
        assert_stochastic_schema_consistent(n, where="after create_scenarios")

        # Extend the dictionaries for networks and names
        networks_dict['stoch'] = n
        n_names_dict['stoch'] = network_name

        if c.stochastic['EVPI']:
            # creates deterministic networks for every scenario to calc Expected Value of Perfect Info
            def network_ws_scenarios(c, p, tech_costs, n_flags_OK, scenarios, CO2_cost_s):
                meta = {}  # store scenario settings

                for year, prob in scenarios.items():
                    CO2_cost = CO2_cost_s[year]
                    set_input_paths(p, year)

                    inputs_dict = prepare_all_inputs(
                        targets_dict=c.targets_dict,
                        CO2_cost=CO2_cost,
                        CO2_cost_ref_year=c.CO2_cost_ref_year,
                        max_RE_to_grid=c.max_RE_to_grid,
                        preprocess_flag=False,
                    )

                    n_det = build_network(tech_costs, inputs_dict, n_flags_OK, c.n_options, p)
                    n_name = file_name_network(n=n_det, n_flags=n_flags, run_name=c.run_name, inputs_dict=inputs_dict,
                                                     targets_dict=c.targets_dict, En_price_year=year,
                                                     stochastic=False)

                    networks_dict[year] = n_det
                    n_names_dict[year] = n_name
                    meta[year] = {"En_price_year": year, "CO2_cost": CO2_cost, "prob": prob}
                return networks_dict, n_names_dict, meta

            # Create Wait and See (WS) networks (det) per scenario
            networks_dict, n_names_dict, meta = network_ws_scenarios(
                c=c, p=p, tech_costs=tech_costs, n_flags_OK=n_flags_OK,
                scenarios=scenarios, CO2_cost_s=CO2_cost_s
            )

        # update result folder for stochastic
        results_folder = create_folder_if_not_exists(analysis_folder , 'stochastic')
        plot_folder = create_folder_if_not_exists(results_folder, 'plots')
        csv_folder = create_folder_if_not_exists(results_folder, 'csv')
        networks_folder = create_folder_if_not_exists(results_folder, "networks")

        n_flags['print'] = False
        n_flags_opt['print'] = False

    else:
        # ---- Deterministic Network only
        networks_dict['network'] = n
        n_names_dict['network'] = network_name
        results_folder = analysis_folder
        plot_folder = create_folder_if_not_exists(results_folder, 'plots')
        csv_folder = create_folder_if_not_exists(results_folder, 'csv')
        networks_folder = create_folder_if_not_exists(results_folder, "networks")

    # ---- Export and print pre-network
    nc_path = export_network(n, n_flags_opt, network_name, networks_folder, "_PRE")
    print_network(
        n=n,
        n_flags=n_flags_opt,
        nc_path=nc_path,
        network_name=network_name,
        suffix="_OPT",
        plot_folder=plot_folder,
        is_stochastic=c.stochastic["stochastic"],
    )
    print('network built')

    # ---- Build the model(s) and Solve network(s)
    for key, net in networks_dict.items():
        name = n_names_dict[key]
        print('solving : ', name)

        #assert_stochastic_schema_consistent(n, where="before build_model_solve_network")
        status, condition, used_solver, used_opts, model = build_model_solve_network(
            net,
            results_folder=results_folder,
            solver=c.optimization["solver"],
            profile=c.optimization["solver_profile"],
            n_config=c.n_config,
            overrides=c.optimization["overrides"],
            collect_all_duals=c.optimization["collect_all_duals"],
            return_model=c.optimization["return_model"],
            n_name=name,
        )

        networks_dict[key] = net  # ensure dict holds the solved network instance

        nc_path = export_network(net, n_flags_opt, name, networks_folder, "_OPT", model=model)
        print_path = print_network(
            n=net,
            n_flags=n_flags_opt,
            nc_path=nc_path,
            network_name=name,
            suffix="_OPT",
            plot_folder=plot_folder,
            is_stochastic=c.stochastic["stochastic"],
        )

        save_config(results_folder, c)

        network_comp_allocation = n.network_comp_allocation
        save_network_comp_allocation(results_folder, network_comp_allocation)


    if c.stochastic['EVPI']:
        df_evpi = compare_objective(networks_dict["stoch"],
                                    {k: v for k, v in networks_dict.items() if k != "stoch"},
                                    scenarios)

        df_evpi.to_csv(Path(csv_folder) / "EVPI.csv")

        print(df_evpi)

    # ---- Plotting and exporting csv ------
    if c.n_flags_opt['plot']:

        # resolve items threshold strings -> numbers
        for it in c.items:
            if isinstance(it.get("th"), str):
                it["th"] = float(c.thresholds[it["th"]])

        failures = run_plot_and_export(
            n=n,
            c=c,
            csv_folder=csv_folder,
            plot_folder=plot_folder,
            thresholds=c.thresholds,
            items=c.items,
            bus_list_mp=c.bus_list_mp,
            network_comp_allocation=network_comp_allocation,
            scenarios=scenarios if c.stochastic["EVPI"] else None,
            networks_dict=networks_dict if c.stochastic["EVPI"] else None,
        )

        print('plotting done')

if __name__ == "__main__":

    main()