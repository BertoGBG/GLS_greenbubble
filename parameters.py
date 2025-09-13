import pandas as pd
import os
# ------------------------------------
'''configuration parameters'''
CO2_cost = 150  # €/t (CO2_cost)
'''Demands'''
flh_H2 = 4000  # set hydrogen demand, flh for 100 MW plant (flh_H2)
f_max_MeOH_y_demand = 0.45  # % of CO2 from biogas upgrading converted to MeOG (sets MeOH demand)
f_max_Methanation_y_demand = 0.45 # # % of CO2 from biogas upgrading converted to extra biomethane (sets biomethanation demand)

'''Sales of Electricity '''
el_DK1_sale_el_RFNBO = 0.1  # max electricity during the year that can be sold to ElDK1 (unit: fraction of El for RFNBOs)

'''Energy & Weather year'''
En_price_year = 2021  # # Year for historical Energy prices

'''Input the network configuration'''
n_flags = {'SkiveBiogas': True,
           'central_heat': True,
           'renewables': True,
           'electrolyzer': True,
           'meoh': True,
           'symbiosis_net': True,
           'DH': True,
           'methanation': True,
           'bioChar' : True,            # True if biochar credits have value (== to CO2 tax)
           'print': False,              # saves svg of network before optimization
           'export': False}             # saves network before optimization

# if preprocess_flag is False the input data are loaded from csv files, if True the input data are downloaded
# and saved as CSV files
preprocess_flag = True #

# --------------------------------------
''' Demand Flexibility (H2 and MeOH'''
# MeOH demand
MeOH_delivery_frequency = 1  # 1: Single delivery at the end of the 'Year'. 12 : 'Month', 52: 'Week'

# H2 demand - Options:
# 1) Single delivery end of the year (one_delivery = True) --> maximum flexibility
# 2) Monthly delivery - follow NG demand (DK) profile  --> medium flexibility
# 3) Weekly delivery - Follow NG demand (DK) profile --> minimum flexibility (especially with profile = True)
H2_profile_flag = False  # 'True': the demand follows NG demand profile: 'False' it is constant during year
H2_delivery_frequency = 1  # 1: Single delivery at the end of the 'Year'. 12 : 'Month', 52: 'Week'
H2_output= 68 # MW H2 <-> 100 MW el

'''Others'''
# CO2 tax - to be coupled with historical el prices if they include CO2 tax (future)
CO2_cost_ref_year = 0  # €/ton (CO2 tax in the reference year of energy prices)

# Biogas plants
f_FLH_Biogas = 4 / 5  # fraction of maximum capacity that the Biogas plant is operated (only for GLS purposes)

#---------------------------------------
"""Retrive data"""
# token to download  factors from Renewable Ninjas
# obtain your own token from : https://www.renewables.ninja/documentation/api
RN_token = ""  #
entsoe_api = ""   #
latitude = 56.566 # Skive (DK)
longitude = 9.033 # Skive (DK)

""" Crete an external NG demand"""
NG_demand_year = 2019 # year for NG demand

# --------------------------------------
'''CSV files as input to the model'''
folder_model_inputs='data' # folder where csv files for model input are saved after the pre-processing
folder_data= 'data/' + 'Inputs_' + str(En_price_year)
os.makedirs(folder_data, exist_ok=True)  # Create the folder if it doesn't exist

GL_input_file = folder_model_inputs + '/GreenLab_Input_file.xlsx'
El_price_input_file = folder_data + '/Elspotprices_input.csv'
CO2emis_input_file = folder_data + '/CO2emis_input.csv'
El_external_demand_input_file = folder_data + '/El_demand_input.csv'
NG_price_year_input_file = folder_data + '/NG_price_year_input.csv'
NG_demand_input_file = folder_data + '/NG_demand_DK_input.csv'
Methanol_demand_input_file = folder_data + '/Methanol_demand_GL_max_input.csv'
DH_external_demand_input_file = folder_data + '/DH_external_demand_input.csv'
CF_wind_input_file = folder_data + '/CF_wind.csv'
CF_solar_input_file = folder_data + '/CF_solar.csv'
bioCH4_prod_input_file = folder_data + '/bioCH4_demand.csv'
H2_demand_input_file = folder_data + '/H2_demand_input.csv'
NG_price_data_folder = folder_model_inputs + '/NG_price_year_2019'
DH_data_folder = folder_model_inputs + '/DH_weather_data'  # prices in currency/kWh

''' export and print folder for optimized networks'''
# folders for single network analysis
print_folder_NOpt = 'outputs/single_analysis/'
print_folder_Opt = 'outputs/single_analysis/'

# --------------------------------------
'''ECONOMICS AND COST ASSUMPTIONS'''
'''Technology Data Economic Parameters'''
#technology_data_url = "https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/"
technology_data_url = "https://raw.githubusercontent.com/BertoGBG/technology-data/master/outputs/"
year_EU = 2030  # investment year
cost_folder = "data/technology-data/outputs"
cost_file = "costs_" + str(year_EU) + ".csv"
USD_to_EUR = 1
DKK_Euro = 7.46  #
discount_rate = 0.07  #
Nyears = 1  # for myopic optimization (not used but needed in some functions)
lifetime = 25  #

'''set Currency of Cost Optimization: DKK or EUR'''
currency = 'EUR'
if currency == 'DKK':
    currency_multiplier = DKK_Euro
elif currency == 'EUR':
    currency_multiplier = 1

# --------------------------------------
""" post 2030: EU rules for renewable el for H2' (RFNBOs) """
rfnbos_dict= {'limit' : 'emissions', # it can be set to 'emissions', 'price' or 'None' (RFNBOs legislation not active)
              'price_threshold' : 20 * currency_multiplier, # (Eur/MWh) : electricity is renewable if price is below 20€/MWh
              'emission_threshold' : 18 * 3.6 / 1000} # (gCO2e/MJ) --> tCO2e/MWh
# --------------------------------------
''' Constants'''
FLH_y = 8760  # full load hours equivalent  in a year for MeOH
lhv_meoh= 5.54  # kWh/kg = MWh/ton
lhv_h2= 33.33 # MWh/t
lhv_ch4 = 13.9 # MWh/t
lhv_straw_pellets = 14.5/3.6 # MWh/t
lhv_dig_pellets = 16/3.6 # MWh/t
density_H2_1atm = 0.0827 # kg/m3
density_CO2_1atm = 1.98 # kg/m3
density_CH4_1atm = 0.716 # kg/m3

# --------------------------------------
''' PARAMETERS FOR REVTRIEVING AND PRE PROCESSING'''
'''Time Period in DK'''
start_day= str(En_price_year)+'-01-01'
start_date = start_day+'T00:00' # keep the format 'YYYY-MM-DDThh:mm' when selecting start and end time
end_day= str(En_price_year+1)+'-01-01'
end_date= end_day+'T00:00' # excludes form the data set

hours_in_period = pd.date_range(start_date + 'Z', end_date + 'Z', freq='h')
hours_in_period = hours_in_period.drop(hours_in_period[-1])

# Check if it's a leap year
def is_leap_year(year):
    return (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
if is_leap_year(En_price_year):
    # Remove all timestamps that fall on February 29
    hours_in_period = hours_in_period[~((hours_in_period.month == 2) & (hours_in_period.day == 29))]

'''Define reference empty data frame'''
ref_col_name = 'ref col'
ref_df = pd.DataFrame(index=hours_in_period, columns=[ref_col_name])
ref_df[ref_col_name] = 0

'''set area to DK1 (for data pre-processing, where applicable)'''
filter_area = r'filter={"PriceArea":"DK1"}' # for energidata
bidding_zone = 'DK_1' # for entsoe

# --------------------------------
'''District heating assumptions'''
DH_Skive_Capacity = 59  # MW district heating capacity in Skive
DH_Tamb_min = -15  # minimum outdoor temp --> maximum Capacity Factor
DH_Tamb_max = 18  # maximum outdoor temp--> capacity Factor = 0

# --------------------------------------
''' ASSUMPTIONS ON ENERGY PRICES'''
''' Biogenic Feedstocks '''
Straw_pellets_price = 250  # (€/t)
Dig_biomass_price = 0  # (€/t) (Manure) Set to 0 as only the Delta in bioCH4 prod costs are considered.

'''District Heating price'''
DH_price = 400 / DKK_Euro * currency_multiplier  #

'''Fossil Methanol'''
methanol_price_2023 = 360  # €/ton
CO2_intensity_MeOH_life = 110 / 1000000 * 3600 # (110 gCO2/MJ meoh) --> tCO2e/MWh

'''Electricity tariffs'''
# Purchased Electricity
# TSO and state tariff
el_transmission_tariff = 7.4 / 100 * 1000 / DKK_Euro * currency_multiplier  # from energinet inputs in Ore/kWh DKK/MWh
el_system_tariff = 5.1 / 100 * 1000 / DKK_Euro * currency_multiplier  # from energinet inputs in Ore/kWh DKK/MWh
el_afgift = 76.1 / 100 * 1000 / DKK_Euro * currency_multiplier

# DSO Tariff -  for 60/10kV transformer (A_low customer)
el_net_tariff_low = 1.5 / 100 * 1000 / DKK_Euro * currency_multiplier  # currency/MWh
el_net_tariff_high = 4.49 / 100 * 1000 / DKK_Euro * currency_multiplier
el_net_tariff_peak = 8.98 / 100 * 1000 / DKK_Euro * currency_multiplier

# Selling tariff
el_tariff_sell = ((0.9 + 0.16) / 100 * 1000) / DKK_Euro * currency_multiplier  # (Ore/kWh) *100/1000 = DKK
# / MWH includes transmission and system tariff

# H2 grid tarif
H2_grid_purchase = False # enables purchasing of H2 from external grid
H2_tariff = 0.04 * 1000 /lhv_h2 # (€/kg) * 10000 / MWh/t

#--------------------------------------
'''Technology inputs'''
# General Capacity expansion limits
cap_nom_max = {'onwind' : float("inf"),
               'solar': float("inf"),
               'battery' : float("inf"),
               'CO2 HP' : float("inf"),
               'pyrolysis' : float("inf")
               }

# CO2 liquefaction - internal data source (BCE-AU)
CO2_Liq_dict ={'el_demand' : 0.061, # MWh/t CO2
                'heat_out' : 0.166, # # water heat 80 C from refrigeration cycle
                'cost_factor' : 1,
                'e_nom_max' : float("inf"),
                'annualized_evaparator_cost' : 3765 } #k€/(t/h)/y

# CO2 HP cylinders storage (source: AU Foulum)
CO2_HP_dict= {'e_nom_max' : float('inf'),
              'el_extra' : 0.01, # MWh/t CO2
              'investment' : 77000, # €/t  includes control systems
              'FOM' : 1.0, # % inv/y
              'lifetime' : 25} # years

# CO2 compressor
CO2_comp_dict ={'el_demand' : 0.096, # MWe/(t/h)
                'heat_out' : 0.096 * 0.2/0.7, # MWth/(t/h) available at 135-80 C
                'cost_factor' : 1}
# H2 compressor
H2_comp_dict ={'el_demand' : 0.340 / lhv_h2, # MWe/(t/h)
                'heat_out' : 0.340 / lhv_h2 * 0.2/0.7, # MWth/(t/h) available at 135-80 C
                'cost_factor' : 1}
# H2 vessels storage
H2_storage_dict = {'el_extra' : H2_comp_dict['el_demand'] * 0.2, #  additional compression for storage MWh/MWh2
                   'e_nom_max' : float("inf"),
                   'cost_factor' : 1}

# Methanol (CO2 hydrogenation)
meoh_dict = {
    "ramp_limit_up": 1/12,
    "ramp_limit_down": 1/12,
    "p_min_pu": 0.15,
    "cost_factor" : 1}

# electrolysis
electrolysis_dict = {
    "ramp_limit_up": 1,
    "ramp_limit_down": 1,
    "p_min_pu": 0,
    "cost_factor" : 1}

# biomethanation
biometh_dict = {
    "ramp_limit_up": 1,
    "ramp_limit_down": 1,
    "p_min_pu": 0,
    "cost_factor" : 1,
    "active" : True } # NOTE  at least one between biometh or catmeth must be active if  n_flags['methanation] is True

# catalytic methanation
catmeth_dict = {
    "ramp_limit_up": 1/12,
    "ramp_limit_down": 1/12,
    "p_min_pu": 0.20,
    "cost_factor" : 1,
    "active" : False} # NOTE  at least one between biometh or catmeth must be active if  n_flags['methanation] is True

# HT heat storage
TES_conc_dict = {
    "standing_loss" : 0.02, # per unit of heat stored per time step
    "cost_factor" : 1.5, #   assumption due to reference cost based on 100 MWh
    "active" : True ,
    "e_nom_min" : 1.5,
    "e_nom_max" : float('inf')} # MWh (not technically relevant below this size)

# DH heat storage
TES_DH_dict = {
    "standing_loss" : 0.002, # per unit of heat stored per time step
    "cost_factor" : 1, #   assumption due to reference cost based on 100 MWh
    "active" : True ,
    "e_nom_max" : float('inf')} # MWh (not technically relevant below this size)

# biogas storage
biogas_storage_dict ={'lifetime': 15,
                      'investment' : 16750,
                      'FOM' : 0,
                      'e_nom_max' : 200} # € /MWh CH4 (biogas at 60%v)


# Estimated lenght of Local H2, CO2 and Pressurized Hot Water
dist_H2_pipe = 1  # km Estimated piping distance in the site--> for cost estimation
dist_CO2_pipe = 1  # km Estimated piping distance in the site--> for cost estimation
dist_PWH_pipe = 5  # km Estimated piping distance in the site--> for cost estimation
capital_cost_PHW = 25000 * currency_multiplier  # €/MW/km
heat_loss_PHW = 0.02  # MW/MW

''' Technologies not included in technology-data with source: DEA '''
DH_HEX_inv = 100000  # €/MW
DH_HEX_FOM = 0.05  # (%inv/Y)
DH_HEX_lifetime = 25  # (Y)
CO2_pipeline_inv = 130000  # €/(t/h)/km
CO2_pipeline_FOM = 20/CO2_pipeline_inv * 100  # €/(t/h)/km / year
CO2_pipeline_lifetime = 40 # years
H2_pipeline_inv = 3800  # €/MW/km
H2_pipeline_FOM = 0.27/H2_pipeline_inv *100  #
H2_pipeline_lifetime = 40  # years


other_DEA_technologies = ['DH heat exchanger', 'CO2_pipeline_gas', 'H2_pipeline_gas', 'biogas storage']
data_dict = {
    "investment": [DH_HEX_inv, CO2_pipeline_inv, H2_pipeline_inv, biogas_storage_dict['investment']],
    "FOM": [DH_HEX_FOM, CO2_pipeline_FOM, H2_pipeline_FOM, biogas_storage_dict['FOM']],
    "lifetime": [DH_HEX_lifetime, CO2_pipeline_lifetime, H2_pipeline_lifetime, biogas_storage_dict['lifetime'] ],
    "VOM": [0.00, 0.00, 0.00, 0.00]
}
other_tech_costs = pd.DataFrame(data_dict, index=other_DEA_technologies)


# --------------------------------------
''' ENERGY INPUTS soruces'''
# NG prices source: # https://api.energidataservice.dk/dataset
# EL prices source: # https://api.energidataservice.dk/dataset
# El emissions source: # https://api.energidataservice.dk/dataset
# DH capacity source: https://ens.dk/sites/ens.dk/files/Statistik/denmarks_heat_supply_2020_eng.pdf
# Weather data Skive source: https://www.dmi.dk/friedata/observationer/
# NG demand in DK source : source: https://www.energidataservice.dk/tso-gas/Gasflow
# Wind Capacity factor source : https://www.renewables.ninja/documentation/api
# Solar Capacity factor source : https://www.renewables.ninja/documentation/api
# El demand DK1 https://data.open-power-system-data.org/time_series/
# CO2 tax DK source: https://www.pwc.dk/da/artikler/2022/06/co2-afgift-realitet.html#:~:text=Afgiften%20for%20kvoteomfattede%20virksomheder%20udg%C3%B8r,2030%20(2022%2Dsatser).
# EL TSO tariff : https://energinet.dk/el/elmarkedet/tariffer/aktuelle-tariffer/
# EL DSO Tariff : https://n1.dk/priser-og-vilkaar/timetariffer
# MeOH fossil price: https://www.methanol.org/wp-content/uploads/2022/01/CARBON-FOOTPRINT-OF-METHANOL-PAPER_1-31-22.pdf
