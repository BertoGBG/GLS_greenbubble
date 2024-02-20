""" Main for sensitivity analysis"""
from functions_GL_paper_V3 import *

'''sensitivity analysis parameters'''
CO2_cost_list = [0, 150, 250]  # €/t (CO2_cost)
H2_demand_list = [0, 4000]  # flh at 100 MW (flh_H2)
MeOH_rec_list = [0.8, 0.85, 0.9, 0.95, 0.99]  # % of CO2 from (f_max_MeOH_y_demand)
DH_flag_list = [True]  # true false

'''Folder for results optimized networks'''
# Set in parameters file

'''retrieve Technology Data cost and add extra technology costs'''
tech_costs = prepare_costs(p.cost_file, p.USD_to_EUR, p.discount_rate, 1, p.lifetime)
tech_costs = add_technology_cost(tech_costs)

'''Build the network based on agents'''
for ia in CO2_cost_list:
    CO2_cost = ia
    for ib in H2_demand_list:
        flh_H2 = ib
        for ic in MeOH_rec_list:
            f_max_MeOH_y_demand = ic
            for ie in DH_flag_list:

                '''Pre_process of all input data'''
                # This function takes all the variables in the sensitivity analysis except the n_flags
                # NOTE: all pre-processing must be done before this analysis
                preprocess_flag = False
                inputs_dict = pre_processing_all_inputs(flh_H2, f_max_MeOH_y_demand, CO2_cost, preprocess_flag)

                n_flags = {'SkiveBiogas': True,
                           'central_heat': True,
                           'renewables': True,
                           'electrolyzer': True,
                           'meoh': True,
                           'symbiosis_net': True,
                           'DH': ie,
                           'print': False,
                           'export': True}

                ''' check dependecies for correct optimization '''
                n_flags_OK = network_dependencies(n_flags)

                ''' Build the PyPSA network'''
                network = build_PyPSA_network_H2d_bioCH4d_MeOHd_V1(tech_costs, inputs_dict,
                                                                                            n_flags_OK)
                print(ia)
                print(ib)
                print(ic)
                print(ie)

                ''' Optimization of the network'''
                # Optimization with gurobi, file name automatic form network composition and input variables
                n_flags_opt = {'print': False,
                               'export': True}
                network_opt = OLPF_network_gurobi(network, n_flags_opt, n_flags, inputs_dict)
                network_opt.network_comp_allocation = network.network_comp_allocation

                del network
                del network_opt
                del inputs_dict
                del n_flags
                del n_flags_opt
