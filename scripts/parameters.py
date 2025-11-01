import pandas as pd
import os
from scripts.config import DKK_Euro, En_price_year, year_EU
from scripts.helpers import build_snapshots
# --------------------------------------
''' Constants'''
FLH_y = 8760  # full load hours equivalent  in a year for MeOH
lhv_h2 = 33.33 # MWh/t
# NOTE physical properties defined in: scripts.technology_inputs
# --------------------------------------
''' PARAMETERS FOR RETRIEVING AND PRE PROCESSING'''

# Retrieve Technology-data
technology_data_url = "https://raw.githubusercontent.com/BertoGBG/technology-data/py-isa/outputs/"
cost_folder = "data/technology-data/outputs"
cost_file = "costs_" + str(year_EU) + ".csv"
cost_path = os.path.join(cost_folder, cost_file)

# token to download  factors from Renewable Ninjas
# obtain your own token from : https://www.renewables.ninja/documentation/api
RN_token = ""  #
entsoe_api = "" #

""" Crete an external NG demand"""
NG_demand_year = 2019 # year for NG demand

'''Build snapshots Time Period in DK'''
hours_in_period, start_date, end_date = build_snapshots (En_price_year)

'''Define reference empty data frame''' #  used in preprocessing
ref_col_name = 'ref col'
ref_df = pd.DataFrame(index=hours_in_period, columns=[ref_col_name])
ref_df[ref_col_name] = 0

'''set area to DK1 (for data pre-processing, where applicable)'''
filter_area = r'filter={"PriceArea":"DK1"}' # for energidata
bidding_zone = 'DK_1' # for entsoe

''' District heating external demand '''
# source: https://ens.dk/sites/ens.dk/files/Statistik/denmarks_heat_supply_2020_eng.pdf
DH_Skive_Capacity = 60  # MW
DH_Tamb_min = -15  # minimum outdoor temp --> maximum Capacity Factor
DH_Tamb_max = 18  # maximum outdoor temp--> capacity Factor = 0

# --------------------------------------
''' ASSUMPTIONS ON ENERGY TARIFFS'''
'''Electricity tariffs'''
# Purchased Electricity
# TSO and state tariff
el_transmission_tariff = 7.4 / 100 * 1000 / DKK_Euro   # from energinet inputs in Ore/kWh DKK/MWh
el_system_tariff = 5.1 / 100 * 1000 / DKK_Euro   # from energinet inputs in Ore/kWh DKK/MWh
el_afgift = 76.1 / 100 * 1000 / DKK_Euro

# DSO Tariff -  for 60/10kV transformer (A_low customer)
el_net_tariff_low = 1.5 / 100 * 1000 / DKK_Euro   # currency/MWh
el_net_tariff_high = 4.49 / 100 * 1000 / DKK_Euro
el_net_tariff_peak = 8.98 / 100 * 1000 / DKK_Euro

# Selling tariff
el_tariff_sell = ((0.9 + 0.16) / 100 * 1000) / DKK_Euro  # (Ore/kWh) *100/1000 = DKK
# / MWH includes transmission and system tariff

# H2 grid tariff
H2_grid_purchase = False # enables purchasing of H2 from external grid
H2_tariff = 0.04 * 1000 / lhv_h2 # symbiosis_n.at['H2 production', 'LHV'] # (€/kg) * 10000 / MWh/t

# --------------------------------------
'''Location of CSV files as input to the model'''
# retrieve data from to these folder and files AND loads these csv files in the preprocessign for the network

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
Methanation_demand_input_file = folder_data + '/Methanation_demand_GL_max_input.csv'
DH_external_demand_input_file = folder_data + '/DH_external_demand_input.csv'
CF_wind_input_file = folder_data + '/CF_wind.csv'
CF_solar_input_file = folder_data + '/CF_solar.csv'
bioCH4_prod_input_file = folder_data + '/bioCH4_demand.csv'
H2_demand_input_file = folder_data + '/H2_demand_input.csv'
NG_price_data_folder = folder_model_inputs + '/NG_price_year_2019'
DH_data_folder = folder_model_inputs + '/DH_weather_data'  # prices in currency/kWh


# --------------------------------------
''' PREPROCESSING: INPUTS sources'''
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
