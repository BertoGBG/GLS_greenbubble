# SPDX-License-Identifier: MIT
"""Global parameters and file-path constants for the GreenBubble model.

This module centralises every scalar constant, API token, and input/output
file path used across the pre-processing, network-building and optimisation
scripts.  It is imported as ``p`` throughout the codebase::

    from scripts import parameters as p
    p.RN_token          # Renewables.ninja API token
    p.El_price_input_file  # path to the electricity price CSV

.. note::
   File paths are derived from ``config.yaml`` (via :mod:`scripts.config`)
   and the project location coordinates.  EU locations write to
   ``data/Inputs_{year}/``; US/California locations write to
   ``data/California/Inputs_{year}/``.
"""

import pandas as pd
import os

from pathlib import Path
from scripts.config import En_price_year, year_investment, latitude, longitude, EUR_to_DKK
from scripts.helpers import build_snapshots, is_eu_or_us


def _load_dotenv(path: str = ".env") -> None:
    """Load key=value pairs from a .env file into os.environ (no-op if missing)."""
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


_load_dotenv()

# --------------------------------------
''' Constants'''
FLH_y = 8760  # full load hours equivalent  in a year for MeOH
lhv_h2 = 33.33 # MWh/t
lhv_NG = 0.010 # MWh/Nm3
# NOTE physical properties defined in: scripts.technology_inputs
# --------------------------------------
''' PARAMETERS FOR RETRIEVING AND PRE PROCESSING'''

# Retrieve Technology-data
technology_data_url = "https://raw.githubusercontent.com/BertoGBG/technology-data/pypsa-eur_AA/outputs/"
cost_folder = "data/technology-data/outputs"
cost_file = "costs_" + str(year_investment) + ".csv"
cost_path = os.path.join(cost_folder, cost_file)

# Stored US data
cost_file_US = "costs_" + str(year_investment) + '_US' + ".csv"
cost_path_US = os.path.join(cost_folder, cost_file)

# API tokens — loaded from environment variables (or .env file in project root).
# Copy .env.example to .env and fill in your own tokens.  Never commit .env.
# Obtain tokens at:
#   Renewables.ninja : https://www.renewables.ninja/documentation/api
#   ENTSO-E          : https://transparency.entsoe.eu/usrm/user/createPublicUser
#   EIA (US)         : https://www.eia.gov/opendata/register.php
RN_token   = os.environ.get("RN_TOKEN", "")
entsoe_api = os.environ.get("ENTSOE_API_KEY", "")
eia_api    = os.environ.get("EIA_API_KEY", "")



'''Build snapshots Time Period in DK'''
hours_in_period, start_date, end_date = build_snapshots(En_price_year)

'''Define reference empty data frame''' #  used in preprocessing
ref_col_name = 'ref col'
ref_df = pd.DataFrame(index=hours_in_period, columns=[ref_col_name])
ref_df[ref_col_name] = 0

'''set area to DK1 (for data pre-processing, where applicable)'''
filter_area = r'filter={"PriceArea":"DK1"}'  # for energidata
price_area = 'DK1'
bidding_zone = 'DK_1'  # for entsoe
currency = 'EUR'        # currency for el spot price column (SpotPriceEUR)

''' District heating external demand '''
# source: https://ens.dk/sites/ens.dk/files/Statistik/denmarks_heat_supply_2020_eng.pdf
DH_Skive_Capacity = 60  # MW
DH_Tamb_min = -15  # minimum outdoor temp --> maximum Capacity Factor
DH_Tamb_max = 18  # maximum outdoor temp--> capacity Factor = 0

# --------------------------------------
'''Location of CSV files as input to the model'''
# retrieve data from to these folder and files AND loads these csv files in the preprocessing for the network

folder_model_inputs='data' # folder where csv files for model input are saved after the pre-processing
if is_eu_or_us(latitude,longitude)  == 'EU':
    folder_data= 'data/' + 'Inputs_' + str(En_price_year)
elif is_eu_or_us(latitude,longitude)  == 'US':
    folder_data= 'data/California/' + 'Inputs_' + str(En_price_year)

os.makedirs(folder_data, exist_ok=True)  # Create the folder if it doesn't exist

GL_input_file = folder_model_inputs + '/GreenLab_Input_file.xlsx'
El_price_input_file = folder_data + '/Elspotprices_input.csv'
CO2emis_input_file = folder_data + '/CO2emis_input.csv'
NG_price_year_input_file = folder_data + '/NG_price_year_input.csv'
DH_external_demand_input_file = 'data/common/DH_external_demand_input.csv'
CF_wind_input_file = folder_data + '/CF_wind.csv'
CF_solar_input_file = folder_data + '/CF_solar.csv'
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

# --------------------------
'''Tolerances to avoid free-energy loops in model'''
loop_tol = 5e-6

# --------------------------
"""mapping to US costs """
dict_tech_US_EU ={"DH heat exchanger" : '',
                  "electricity grid connection": 'electricity grid connection',
                  'gas boiler steam' : 'gas boiler steam',
                  'electric boiler steam' : 'electric boiler steam',
                  'NG grid connection': '',
                  "biomass belt dryer" : '',
                  "CO2 storage tank": '',
                  "CO2 liquefaction small": '',
                  "CO2 storage tank small": '',
                  "hydrogen storage compressor" : '',
                  "CO2 industrial compressor": '',
                  'CH4 (g) fill compressor station': '',
                  "hydrogen storage tank type 1" : "hydrogen storage tank type 1",
                  "CO2 storage cylinders" : '',
                  "battery inverter" : 'battery inverter',
                  "battery storage" : '',
                  "Concrete-charger" : 'Concrete-charger',
                  "Concrete-discharger" : 'Concrete-discharger',
                  "Concrete-store" : 'Concrete-store',
                  "central water tank storage" : 'central water tank storage',
                  'industrial heat pump medium temperature' : 'industrial heat pump medium temperature',
                  'biogas upgrading': 'biogas upgrading',
                  'centrifugal dewatering' : '',
                  'biogas' : 'biogas',
                  'biogas storage' : '',
                  'biogas engine': '',
                  'onwind' : 'onwind',
                  'solar' : 'solar',
                  'AEC large' : 'AEC large',
                  'AEC small' : 'AEC small',
                  'PEMEC large' : 'PEMEC large',
                  'PEMEC small' : 'PEMEC small',
                  'SOEC large' : 'SOEC large',
                  'SOEC small' : 'SOEC small',
                  "methanolisation" : 'methanolisation',
                  "biomethanation" : '',
                  "biogas plus hydrogen" : 'biogas plus hydrogen',
                  "biochar pyrolysis" : 'biochar pyrolysis',
                  "biomass boiler" : 'biomass boiler',
                  "central gas boiler" : 'central gas boiler',
                  "electric boiler steam" : 'electric boiler steam',
                  "CO2 gas pipe" : '',
                  "H2 pipe" : '',
                  }
