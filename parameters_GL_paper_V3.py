import numpy as np
import pandas as pd

# --------------------------------------
''' MAIN SENSITIVITY PARAMETERS'''
# MeOH demand
MeOH_one_delivery = True  # True means one single delivery at the end of the year, otherwise deliveries are done weekly
# f_max_MeOH_y_demand = 0.7 # fraction of CO2 recovered from the biogas upgrading

# H2 demand
h2_demand_flag = 'constant'  # 'profile'
H2_one_delivery = True  # True means one single delivery at the end of the year, otherwise deliveries are done monthly
H2_output= 68 # MW H2 <-> 100 MW el
# flh_H2 = 0 # full load hours of a 100 MWe plant

# CO2 tax
# CO2_cost = 0 # Euros/ton
CO2_cost_ref_year = 0  # €/ton (CO2 tax for electricity production in year of energy prices)

# Energy prices
En_price_year= 2019 # alternative : 2022

# Biogas plants
f_FLH_Biogas = 4 / 5  # fraction of maximum capacity that the Biogas plant is operated

# Ratio between the el_DK1 demand and the RE used in the RFNBOs production
el_DK1_sale_el_RFNBO = 0.10 # must be >0  (MWh_y/MWh_y)

# --------------------------------------
'''CSV files as input to the model'''
folder_model_inputs='data' # folder where csv files for model input are saved in pre-processing

GL_input_file = folder_model_inputs + '/GreenLab_Input_file.xlsx'
El_price_input_file = folder_model_inputs + '/Inputs_' + str(En_price_year) + '/Elspotprices_input.csv'
CO2emis_input_file = folder_model_inputs + '/Inputs_' + str(En_price_year) + '/CO2emis_input.csv'
El_external_demand_input_file = folder_model_inputs + '/Inputs_' + str(En_price_year) + '/El_demand_input.csv'
NG_price_year_input_file = folder_model_inputs + '/Inputs_' + str(En_price_year) + '/NG_price_year_input.csv'
NG_demand_input_file = folder_model_inputs + '/Inputs_' + str(En_price_year) + '/NG_demand_DK_input.csv'
Methanol_demand_max_input_file = folder_model_inputs + '/Inputs_' + str(En_price_year) + '/Methanol_demand_GL_max_input.csv'
DH_external_demand_input_file = folder_model_inputs + '/Inputs_' + str(En_price_year) + '/DH_external_demand_input.csv'
CF_wind_input_file = folder_model_inputs + '/Inputs_' + str(En_price_year) + '/CF_wind.csv'
CF_solar_input_file = folder_model_inputs + '/Inputs_' + str(En_price_year) + '/CF_solar.csv'
bioCH4_prod_input_file = folder_model_inputs + '/Inputs_' + str(En_price_year) + '/bioCH4_demand.csv'
H2_demand_input_file = folder_model_inputs + '/Inputs_' + str(En_price_year) + '/H2_demand_input.csv'
NG_price_data_folder = folder_model_inputs + '/NG_price_year_2019'
DH_data_folder = folder_model_inputs + '/DH_weather_data'  # prices in currency/kWh

''' print folder for optimized networks'''
# folders for single network analysis
print_folder_NOpt = 'outputs/single_analysis/'
print_folder_Opt = 'outputs/single_analysis/'

# Folders for sensitivity analysis
# print_folder_Opt = 'outputs/sensitivity_analysis/Results_OptNetworkV2/'
# print_folder_NOpt = 'outputs/sensitivity_analysis/testing_NotOptNetworkV2/'

# --------------------------------------
'''ECONOMICS AND COST ASSUMPTIONS'''
'''Technology Data Economic Parameters'''
year_EU = 2030  # invesment year -> MUST be set > 2030
cost_file = "technology-data-master/outputs/costs_2030.csv"
USD_to_EUR = 1
DKK_Euro = 7.46  # ratio avg 2019
discount_rate = 0.07  #
Nyears = 1  # for myoptic optimization (parameter not used but needed for some functions)
lifetime = 25  # of the plant

'''SET Currency of Cost Optimization: DKK or EUR'''
currency = 'EUR'
if currency == 'DKK':
    currency_multiplier = DKK_Euro
elif currency == 'EUR':
    currency_multiplier = 1

# --------------------------------------
''' CONSTANTS'''
FLH_y = 8760  # full load hours in a year
lhv_meoh= 5.54  # kWh/kg = MWh/ton
lhv_h2= 33.33 # MWh/t
lhv_straw_pellets = 14.5/3.6 # MWh/t
lhv_dig_pellets = 16/3.6 # MWh/t

# --------------------------------------
''' PARAMETERS FOR PRE PROCESSING'''
'''Time Period in DK'''
start_day= str(En_price_year)+'-01-01'
start_date = start_day+'T00:00' # keep the format 'YYYY-MM-DDThh:mm' when selecting start and end time
end_day= str(En_price_year+1)+'-01-01'
end_date= end_day+'T00:00' # excludes form the data set

hours_in_period = pd.date_range(start_date + 'Z', end_date + 'Z', freq='H')
hours_in_period = hours_in_period.drop(hours_in_period[-1])

'''Define reference empty data frame'''
ref_col_name = 'ref col'
ref_df = pd.DataFrame(index=hours_in_period, columns=[ref_col_name])
ref_df[ref_col_name] = 0

'''set area to DK1 (for data pre-processing, where applicable)'''
filter_area = r'{"PriceArea":"DK1"}'

# --------------------------------
'''District heating assumptions'''
DH_Skive_Capacity = 59  # MW district heating capacity in Skive
DH_Tamb_min = -15  # minimum outdoor temp --> maximum Capacity Factor
DH_Tamb_max = 18  # maximum outdoor temp--> capacity Factor = 0

# --------------------------------------
''' ASSUMPTIONS ON ENERGY PRICES'''
''' Biogenic Feedstocks '''
Straw_pellets_price = 380 # €/t
Biomass_price = 0  # (€/t) Set to 0 as only the Delta in bioCH4 prod costs are considered.

'''District Heating price'''
DH_price = 400 / DKK_Euro * currency_multiplier  #

'''Electricity tariffs'''
#
# Purchased Electricity
# TOS and state tariff
el_transmission_tariff = 7.4 / 100 * 1000 / DKK_Euro * currency_multiplier  # from energinet inputs in Ore/kWh DKK/MWh
el_system_tariff = 5.1 / 100 * 1000 / DKK_Euro * currency_multiplier  # from energinet inputs in Ore/kWh DKK/MWh
el_afgift = 76.1 / 100 * 1000 / DKK_Euro * currency_multiplier
# DSO Tariff
# tariff valid for 60/10kV transformer (A_low customer)
el_net_tariff_low = 1.5 / 100 * 1000 / DKK_Euro * currency_multiplier  # currency/MWh
el_net_tariff_high = 4.49 / 100 * 1000 / DKK_Euro * currency_multiplier
el_net_tariff_peak = 8.98 / 100 * 1000 / DKK_Euro * currency_multiplier

# Selling tariff
el_tariff_sell = ((
                              0.9 + 0.16) / 100 * 1000) / DKK_Euro * currency_multiplier  # (Ore/kWh) *100/1000 = DKK
# / MWH includes transmission and system tariff

'''post 2030: EU rules for renewable el for H2'''
EU_renew_el_price_limit = 20 * currency_multiplier  # (Eur/MWh) : electricity is renewable if price is below 20€/MWh
EU_renew_el_emission_limit = 18 * 3.6 / 1000  # (gCO2e/MJ) --> tCO2e/MWh

# --------------------------------------
''' CAPACITY EXPANSION LIMITS'''
electrolyzer_p_nom_max = float("inf") #
p_nom_max_wind = float("inf") #
p_nom_max_solar = float("inf") #
battery_max_cap = float('inf')  # MWh battery storage capacity on site
e_nom_max_H2_HP = float('inf')  # float('inf') # MWh # it can be max 5t according to information from suppliers
e_nom_max_CO2_HP = float('inf')  # t_CO2 # HP CO2 storage
p_nom_max_skyclean = 50 # MW pellets input according to Stiesdal

'''Heat network expansion'''
e_nom_max_Heat_DH_storage = float('inf')  # MWH
ramp_limit_up_Heat_DH_storage = 1/2  # assumptions but it could be increased by larger heat exchanger
ramp_limit_down_Heat_DH_storage = 1/2 # assumptions but it could be increased by larger heat exchanger
e_nom_min_Heat_MT_storage = 1.5 # MWH
e_nom_max_Heat_MT_storage = float('inf')
ramp_limit_up_Heat_MT_storage = 1/6
ramp_limit_down_Heat_MT_storage = 1/6

#--------------------------------------
'''Technology inputs'''
# Compressors
el_comp_CO2 = 0.096  # MWe/(t/h)
el_comp_H2 = 0.340  # MWe/MWh2
heat_comp_CO2 = el_comp_CO2 * 0.2/0.7  # MWth/(t/h) available at 135-80 C
heat_comp_H2 = el_comp_H2 * 0.2/0.7  # MWth/MWh2 available at 135-80 C
CO2_comp_inv = 1516 # kEuro/(t/h)
CO2_comp_FOM = 4.0 # (%inv/Y)
CO2_comp_lifetime = 15 # years

# CO2 liquefaction
El_CO2_liq = 0.061 # MWh/t CO2
Heat_CO2_liq_DH = 0.166 # water heat 80 C from refrigeration cycle
CO2_evap_annualized_cost= 3765 # k€/(t/h)/y from internal information (Foulum) of rental of 16 bara CO2 evaporator #TODO CHECK cost
# H2 cylinders storage
El_H2_storage_add = 0.30 # MWh/MWh2

# CO2 cylinders storage
El_CO2_storage_add = 0.01 # MWh/t CO2
ro_H2_80bar = 6.3112 # density at 20 C (supercritical) in kg/m3 NIST Chemistry WebBook
CO2_cylinders_inv = 77000 # €/t  includes control systems

# Ramp up limits (added assumptions)
ramp_limit_up_MeOH = 1 / 48  # 48 hours needed to reach full operation # NEEDS A REFERENCE
ramp_limit_down_MeOH = 1 / 48
p_min_pu_meoh=0.2

# electrolysis
ramp_limit_up_electrolyzer = 1/2   # 2 hour needed to reach full power
ramp_limit_down_electrolyzer = 1/2

# Local H2, CO2 and Pressurized Hot Water
dist_H2_pipe = 1  # km Estimated piping distance in the site--> for cost estimation
dist_CO2_pipe = 1  # km Estimated piping distance in the site--> for cost estimation
dist_PWH_pipe = 5  # km Estimated piping distance in the site--> for cost estimation
capital_cost_PHW = 25000 * currency_multiplier  # €/MW/km
heat_loss_PHW = 0.02  # MW/MW

''' Technologies not included in Catalogue '''
DH_HEX_inv = 100000 # €/MW
DH_HEX_FOM = 0.05 # (%inv/Y)
DH_HEX_lifetime = 25 # (Y)
CO2_pipeline_inv = 130000 # €/(t/h)/km
CO2_pipeline_FOM = 20/CO2_pipeline_inv *100 # €/(t/h)/km / year
CO2_pipeline_lifetime = 50 # years
H2_pipeline_inv = 3800 # €/MW/km
H2_pipeline_FOM = 0.27/H2_pipeline_inv *100  #
H2_pipeline_lifetime = 50  # years
Pyrolysis_inv = 1.5 * 10**6  # € MWlhv from DEA tehcnology catalogue for renewable fuels
Pyrolysis_FOM = 0.048/1.5  # %inv/y
Pyrolysis_lifetime = 25  #
Electrolysis_small_inv = 875 *10**3 # €/MW 1-10MW
Electrolysis_small_FOM = 4 # %inv/y
Electrolysis_small_lifetime = 25 # Y
Electrolysis_large_inv = 550 * 10**3 # €/MWe 100MW
Electrolysis_large_FOM = 4 # %inv/y
Electrolysis_large_lifetime = 25# Y

# --------------------------------------
''' ENERGY INPUTS soruces'''
# NG prices source: # https://api.energidataservice.dk/dataset --> compiled with NG prices and year
# EL prices source: # https://api.energidataservice.dk/dataset --> compiled with EL prices and year
# El emissions source: # https://api.energidataservice.dk/dataset --> compiled with El emissions and year
# DH capacity source: https://ens.dk/sites/ens.dk/files/Statistik/denmarks_heat_supply_2020_eng.pdf
# Weather data Skive source: https://www.dmi.dk/friedata/observationer/
# NG demand in DK source (2019): source: https://www.energidataservice.dk/tso-gas/Gasflow
# Wind Capacity factor source (2017):
# Solar Capacity factor source (2017):
# El demand DK1 https://data.open-power-system-data.org/time_series/
# CO2 tax DK source: https://www.pwc.dk/da/artikler/2022/06/co2-afgift-realitet.html#:~:text=Afgiften%20for%20kvoteomfattede%20virksomheder%20udg%C3%B8r,2030%20(2022%2Dsatser).
# EL TSO tariff : https://energinet.dk/el/elmarkedet/tariffer/aktuelle-tariffer/
# EL DSO Tariff