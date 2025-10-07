
# LOGIC for results structure

# results_folder = network(s) filename
# inside results_folder:
# - networks: dir
# - plots : dir
# network(s) filename  is defined a combination of :
# 1) n_flags, CO2_cost, demand_H2, demand_CH4, demand_meoh, el_DK1_sale_el_RFNBO, En_price_year (automatic)
# 2) run_name  (set by the user)

# ------------------------------------
import pandas as pd
import yaml

'''run name'''
run_name = 'full_test' # define a run_name representative of the Network configuration

'''configuration parameters'''
CO2_cost = 150  # €/t (CO2_cost)

'''Annual Demands'''
# Hydrogen demand for the grid - set as equivalent plant : demand_H2 = flh_H2 * H2_output
flh_H2 = 4000  #  h/y
H2_output= 68 # MW H2 <-> 100 MW el
demand_H2 = flh_H2 * H2_output # MWh /y

# biomethane demand for the grid, set as equivalent plant : demand_CH4 = flh_Biogas * CH4_output
flh_Biogas = 8760 #  h/y
biogas_output = 27 # MW CH4 (Biogas Skive c.a. 27 MW ch4)
demand_CH4 = flh_Biogas * biogas_output   # MWh /y

# Methanol demand set as equivalent plant : demand_meoh = flh_meoh * meoh_output
flh_meoh = 2000 # h/y
meoh_output = 8 # MW
demand_meoh = flh_meoh * meoh_output # MWh/y

'''Sales of Electricity '''
el_DK1_sale_el_RFNBO = 0.3  # max electricity during the year that can be sold to ElDK1 (unit: fraction of El for RFNBOs)

'''Energy & Weather year'''
En_price_year = 2019  # # Year for historical Energy prices
preprocess_flag = False # : bool -> False the input data are loaded from csv files, True : the input data are downloaded from sources

'''Location'''
latitude = 56.566 # Skive (DK)
longitude = 9.033 # Skive (DK)

'''Input the network configuration'''
n_flags = {'biogas': True,
           'central_heat': True,
           'renewables': True,
           'electrolysis': True,
           'meoh': True,
           'methanation': True,
           'symbiosis': True,
           'storage' : True,
           'print': True,               # saves svg of network before optimization
           'export': False}             # export network before optimization


#folder for saving the outputs of single network analysis
outputs_folder = 'outputs/single_analysis/'

# --------------------------------------
''' Demand Flexibility (H2 and MeOH) '''
# H2 grid demand - Options:
H2_profile_flag = True  # 'True': the demand follows NG demand profile: 'False' it is flexible delivery during the period  (see frequency)
H2_delivery_frequency = 52  # 1: Single delivery at the end of the 'Year'. 12 : 'Month', 52: 'Week'

'''Others'''
# CO2 tax - to be coupled with historical el prices if they include CO2 tax (future)
CO2_cost_ref_year = 0  # €/ton (CO2 tax in the reference year of energy prices)

# --------------------------------------
""" RFNBOs : post 2030: EU rules for renewable el for H2"""
rfnbos_dict= {'limit' : 'emissions', # it can be set to 'emissions', 'price' or 'None' (RFNBOs legislation not active)
              'price_threshold' : 20 , # (Eur/MWh) : electricity is renewable if price is below 20€/MWh
              'emission_threshold' : 18 * 3.6 / 1000} # (gCO2e/MJ) --> tCO2e/MWh

# --------------------------------------
'''ECONOMICS AND COST ASSUMPTIONS'''
'''Technology Data Economic Parameters'''
year_EU = 2030  # investment year
USD_to_EUR = 1.00
DKK_Euro = 7.46  #
discount_rate = 0.07  #

#--------------------------------------
'''Network configuration '''
# Load network configuration
with open("n_config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.pop("base", None)
n_config = pd.DataFrame.from_dict(cfg, orient="index").sort_index()

# Minimum Capacity installed
cap_nom_min = {'TES concrete' : 1.5, # MWh
               'TES DH' : 5, # MWh
               }

# minimum load (can be deactivated fully)
p_min_pu ={'meoh' : 0.15,
           'electrolysis' : 0.0,
           'cat methanation' : 0.15,
           'biomethanation' : 0.0,
           }

# ramp up limit
ramp_limit_up ={'meoh' : 1/12,
                'electrolysis' : 1,
                'cat methanation' : 1/12,
                'biomethanation' : 1,
                'TES DH' : 1/50,
                'TES concrete' : 1/10}

# ramp down limit
ramp_limit_down = { 'meoh' : 1/12,
                    'electrolysis' : 1,
                    'cat methanation' : 1/12,
                    'biomethanation' : 1,
                    'TES DH' : ramp_limit_up['TES DH'],
                    'TES concrete' : ramp_limit_up['TES concrete']}

# standing loss for energy storage
standing_loss = {'TES concrete' : 0.02,
                 'TES DH' : 0.02,
                 }

# other options
n_options = {
    'DH' : True, # add sales of DH heat to teh external market - ONLY WITH n_flags['symbiosis']
    'biochar credits' : True, # biochar is rewarded for CO2 stored at same value of CO2 tax - ONLY WITH n_flags['central_heat']
    # 'CO2 Liq credits' : True, # Liquid CO2 (sequestred) is sold and rewarded e at same value of CO2 tax  - # TODO add to the model
    'pellets market' : False, # purchase of pellets from external market - ONLY WITH n_flags['central_heat']
    'moist biomass market' : False,  # purchase of chips (moist biomass) - ONLY WITH n_flags['central_heat']
    'symbiosis El transformer' : False, # True: the internal el grid is on two buses(El3 for variable RE, and H2) and (El2 for teh rest). if True they have different voltage and adds the cost for a transformer. - ONLY WITH n_flags['symbiosis']
    'pellets annual max': 20000, # MWh/y max cap fo consumption (risk of unbounded optimization if inf and biochar credits : true)
    'moist biomass annual max': 20000, # MWh/y max cap fo consumption (risk of unbounded optimization if inf and biochar credits : true)
    'Dig biomass annual max': 400000, # t/y of digestible biomass (manure)
}

