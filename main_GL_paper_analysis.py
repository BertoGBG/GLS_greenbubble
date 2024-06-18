""" Main for sensitivity analysis"""
from functions_GL_paper_V3 import *

'''sensitivity analysis parameters'''
# NOTE 1 : all pre-processing must be done before this analysis --> prep processing flag set to False
# NOTE 2 : energy year price must be changed in parameters

CO2_cost_list = p.CO2_cost_list  # €/t (CO2_cost)
H2_demand_list = p.H2_demand_list  # flh at 100 MW (flh_H2)
MeOH_rec_list = p.MeOH_rec_list  # % of CO2 from biogas upgrading recovered to MeOh (f_max_MeOH_y_demand)
DH_flag_list = p.DH_flag_list  # district heating
bioCh_credits_list = p.bioCh_credits_list # biochar credits (value equal to CO2 tax)
el_DK1_sale_el_RFNBO_list = p.el_DK1_sale_el_RFNBO_list  # max electricity sold to grid as fraction of PtX consumption

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
            for id in bioCh_credits_list:
                for ie in DH_flag_list:
                    for ig in el_DK1_sale_el_RFNBO_list:
                        el_DK1_sale_el_RFNBO = ig
                        '''Pre_process of all input data'''
                        # This function takes all the variables in the sensitivity analysis except the n_flags
                        preprocess_flag = False
                        inputs_dict = pre_processing_all_inputs(flh_H2, f_max_MeOH_y_demand, CO2_cost, el_DK1_sale_el_RFNBO,
                                                                preprocess_flag)

                        n_flags = {'SkiveBiogas': True,
                                   'central_heat': True,
                                   'renewables': True,
                                   'electrolyzer': True,
                                   'meoh': True,
                                   'symbiosis_net': True,
                                   'DH': ie,
                                   'bioChar': id,
                                   'print': False,
                                   'export': False}

                        ''' check dependecies for correct optimization '''
                        n_flags_OK = network_dependencies(n_flags)

                        ''' Build the PyPSA network'''
                        network = build_PyPSA_network_H2d_bioCH4d_MeOHd_V1(tech_costs, inputs_dict,
                                                                                                    n_flags_OK)
                        print(ia)
                        print(ib)
                        print(ic)
                        print(id)
                        print(ie)
                        print(ig)

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
