from functions_GL_paper_V3 import *

'''sensitivity analysis parameters'''
CO2_cost = 150  # €/t (CO2_cost)
flh_H2 = 4000  # flh at 100 MW (flh_H2)
f_max_MeOH_y_demand = 0.85  # % of CO2 from (f_max_MeOH_y_demand)

'''retrieve Technology Data cost and add extra technology costs'''
tech_costs = prepare_costs(p.cost_file, p.USD_to_EUR, p.discount_rate, 1, p.lifetime)
tech_costs = add_technology_cost(tech_costs)

'''Pre_process of all input data'''
# if preprocess_flag is False the input data are loaded from csv files, if True the input data are downloaded
# from internet, saved as CSV files and loaded
preprocess_flag = True
inputs_dict = pre_processing_all_inputs(flh_H2, f_max_MeOH_y_demand, CO2_cost, preprocess_flag)

'''Build the network based on agents'''
n_flags = {'SkiveBiogas': True,
           'central_heat': True,
           'renewables': True,
           'electrolyzer': True,
           'meoh': True,
           'symbiosis_net': True,
           'DH': True,
           'print': False,
           'export': False}

''' check dependecies for correct optimization '''
n_flags_OK = network_dependencies(n_flags)

''' Build the PyPSA network'''
network = build_PyPSA_network_H2d_bioCH4d_MeOHd_V1(tech_costs, inputs_dict, n_flags_OK)

''' Optimization of the network'''
# Optimization with gurobi, file name automatic form network composition and input variables
n_flags_opt = {'print': True,  # saves svg of the topology
               'export': True} # saves network file

network_opt = OLPF_network_gurobi(network, n_flags_opt, n_flags, inputs_dict)

''' Save other Results'''
# get system cost (allocation by network topology)
network_comp_allocation  = network.network_comp_allocation

with open(p.print_folder_Opt+'network_comp_allocation.pkl', 'wb') as f:
    pkl.dump(network_comp_allocation, f)

''' Plots'''
# plt marginal and capital cost by agent
cc_tot_agent, mc_tot_agent = get_total_marginal_capital_cost_agents(network_opt, network_comp_allocation, True)

# plt shadow prices
shadow_prices_violinplot(network_opt,inputs_dict, tech_costs)

# El and heat prices
plot_El_Heat_prices(network_opt, inputs_dict, tech_costs)

# H2 and MeOH prices
bus_list1=['El3 bus','H2 delivery','Heat LT']
legend1 = ['El to H2', 'H2 grid', 'GLS Heat LT ']
bus_list2=['Methanol','H2_meoh','El_meoh','CO2_meoh','Heat MT_Methanol plant','Heat DH_Methanol plant']
legend2 = ['MeOH prod cost', 'H2 to MeOH', 'El to MeOH', 'CO2 to MeOH', 'Heat MT to MeOH', 'Heat DH from MeOH']

# date format : '2022-01-01T00:00'
start_hour = 'T00:00'
d_start = str(p.En_price_year)+'-01-01' + start_hour
d_end = str(p.En_price_year)+'-05-01' + start_hour

plot_bus_list_shadow_prices(network_opt, bus_list1, legend1, d_start, d_end)
plot_bus_list_shadow_prices(network_opt, bus_list2, legend2, d_start, d_end)

# print and save list of components at png
df_opt_componets = print_opt_components_table(network_opt, network_comp_allocation, True )

# heat mapt of weekly utilization of key plants

# Dict with components that if present in the network_opt are plotted
key_comp_dict={'generators':['onshorewind', 'solar', 'Straw Pellets'],
               'links' : ['DK1_to_El3', 'El3_to_DK1', 'Electrolyzer', 'Methanol plant', 'H2 compressor', 'CO2 compressor', 'H2grid_to_meoh' , 'CO2 sep to atm' ,'SkyClean', 'El boiler', 'NG boiler', 'NG boiler extra', 'Pellets boiler', 'Heat pump'],
               'stores': ['H2 HP', 'CO2 pure HP', 'CO2 Liq', 'battery', 'Water tank DH storage', 'Concrete Heat MT storage' , 'Digest DM']}

heat_map_CF(network_opt, key_comp_dict)



