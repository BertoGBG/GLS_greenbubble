from functions import *

'''configuration parameters'''
CO2_cost = 150  # €/t (CO2_cost)
flh_H2 = 4000  # set hydrogen demand, flh for 100 MW plant (flh_H2)
f_max_MeOH_y_demand = 0.9  # % of CO2 from biogas upgrading converted to MeOG (sets MeOH demand)
el_DK1_sale_el_RFNBO = 0.1  # max electricity during the year that can be sold to ElDK1 (unit: fraction of El for RFNBOs)

'''Input the network configuration'''
n_flags = {'SkiveBiogas': True,
           'central_heat': True,
           'renewables': True,
           'electrolyzer': True,
           'meoh': True,
           'symbiosis_net': True,
           'DH': True,
           'bioChar' : True,            # True if biochar credits have value (== to CO2 tax)
           'print': False,              # saves svg of network before optimization
           'export': False}             # saves network before optimization

''' check dependecies for correct optimization '''
n_flags_OK = network_dependencies(n_flags)

'''Pre_process of all input data'''
# if preprocess_flag is False the input data are loaded from csv files, if True the input data are downloaded
# and saved as CSV files
preprocess_flag = False #
# adjust H2 and MeOH demand based on n_flags_OK
flh_H2_OK, f_max_MeOH_y_demand_OK = n_flags_to_preprocess (n_flags_OK, flh_H2, f_max_MeOH_y_demand)
# pre-process all inputs
inputs_dict = pre_processing_all_inputs(flh_H2_OK, f_max_MeOH_y_demand_OK, CO2_cost, el_DK1_sale_el_RFNBO, preprocess_flag)

'''Technology Data cost and add extra technology costs'''
# retrive technology-data repository
retrieve_technology_data(p.cost_file, p.cost_folder, p.technology_data_url)
tech_costs = prepare_costs(p.cost_folder + '/' + p.cost_file, p.USD_to_EUR, p.discount_rate, 1, p.lifetime)
add_technology_cost(tech_costs, p.other_tech_costs)

''' Build the PyPSA network'''
network = build_network(tech_costs, inputs_dict, n_flags_OK)

'''save network_comp_allocation as pkl file'''
network_comp_allocation = network.network_comp_allocation
with open(p.print_folder_Opt+'network_comp_allocation.pkl', 'wb') as f:
    pkl.dump(network_comp_allocation, f)

''' Optimization of the network'''
# Optimization with gurobi, file name automatic from network composition and input variables
n_flags_opt = {'print': True,  # saves svg of the topology
               'export': True,
               'plot' : True} # saves network file

# solve network
network.optimize.create_model()  # Create the Linopy model
network.optimize.solve_model(solver_name= 'gurobi')   # Solve using Gurobi solver
# network.optimize.solve_model(solver_name="highs")  # using free solver Highs

# export and print network
network_opt= network.copy()
export_print_network(network_opt, n_flags_opt, n_flags, inputs_dict)

#####-----------------------------------
''' Plot results - single optimization '''
if n_flags_opt['plot']:

    # create plots' folder
    base_path = p.print_folder_Opt  # Change this to your desired path
    file_name = file_name_network(network_opt, n_flags, inputs_dict)
    folder_name = file_name
    plots_folder = create_folder_if_not_exists(base_path, folder_name)

    # plt marginal and capital cost by plant #
    cc_tot_agent, mc_tot_agent = get_total_marginal_capital_cost_agents(network_opt, network_comp_allocation, True, plots_folder)

    # plt violin shadow prices
    shadow_prices_violinplot(network_opt,inputs_dict, tech_costs, plots_folder)

    # plt internal El and heat prices
    plot_El_Heat_prices(network_opt, inputs_dict, tech_costs, plots_folder)

    # plt partial timeseries for shadow prices -  set start and end in:
    # date format : '2022-01-01T00:00'
    d_start = str(p.En_price_year)+'-01-01'
    d_end = str(p.En_price_year)+'-03-31'

    bus_list1=['El3 bus','H2 delivery','Heat LT']
    legend1 = ['El to H2', 'H2 grid', 'GLS Heat LT ']
    bus_list2=['Methanol','H2_meoh','El2 bus','CO2_meoh','Heat MT_Methanol plant','Heat DH_Methanol plant']
    legend2 = ['MeOH prod cost', 'H2 to MeOH', 'El to MeOH', 'CO2 to MeOH', 'Heat MT to MeOH', 'Heat DH from MeOH']

    plot_bus_list_shadow_prices(network_opt, bus_list1, legend1, d_start, d_end, plots_folder)
    plot_bus_list_shadow_prices(network_opt, bus_list2, legend2, d_start, d_end, plots_folder)

    # print and save list of plants optimal capacities as .csv
    file_path= plots_folder + 'table_capacities'
    df_opt_componets = save_opt_capacity_components(network_opt, network_comp_allocation, file_path)

    # plot optimal operation as heat map
    # Select components to plot (if present in the solution)
    key_comp_dict={'generators':['onshorewind', 'solar', 'Straw Pellets'],
                   'links' : ['DK1_to_El3', 'El3_to_DK1', 'Electrolyzer', 'Methanol plant', 'H2 compressor', 'CO2 compressor', 'H2grid_to_meoh' , 'CO2 sep to atm' ,'SkyClean', 'El boiler', 'NG boiler', 'NG boiler extra', 'Pellets boiler', 'Heat pump'],
                   'stores': ['H2 HP', 'CO2 pure HP', 'CO2 Liq', 'battery', 'Water tank DH storage', 'Concrete Heat MT storage' , 'Digest DM']}


    heat_map_CF(network_opt, key_comp_dict, plots_folder)