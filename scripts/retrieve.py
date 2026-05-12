# SPDX-License-Identifier: MIT
"""Data retrieval utilities for external energy-market APIs.

This module provides functions to download time-series data from three
external sources used by the GreenBubble preprocessing pipeline:

* **Energi Data Service** (``api.energidataservice.dk``) — Danish electricity
  spot prices, CO₂ emission intensities and natural gas prices.
* **Renewables.ninja** — hourly wind and solar capacity-factor profiles for
  any geographic location (MERRA-2 reanalysis data).
* **ENTSO-E Transparency Platform** — historical electricity demand for
  European bidding zones.
* **EIA / CAISO** — US-side electricity data (California runs).

All functions return :class:`pandas.DataFrame` objects indexed on UTC-naive
hourly timestamps matching the model snapshot index built by
:func:`scripts.helpers.build_snapshots`.

.. note::
   API tokens for Renewables.ninja and ENTSO-E are stored in
   :mod:`scripts.parameters`.  Obtain your own tokens before running the
   pipeline on a new machine.
"""

import pandas as pd
import numpy as np
import requests
from scripts import parameters as p
import os
from io import StringIO
import json
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta
import pytz
import hashlib
import urllib.parse


# ------ INPUTS PRE-PROCESSING ----
def GL_inputs_to_eff(GL_inputs):
    ''' function that reads csv file with GreenLab energy and material flows for each plant and calculates
     efficiencies for multilinks in the network'''

    # NOTE: (-) refers to energy or material flow CONSUMED by the plant
    #      (+) refers to energy or material flow PRODUCED by the plant
    # Calculates Efficiencies for MultiLinks
    GL_eff = GL_inputs
    GL_eff = GL_eff.drop(columns='Bus Unit')  # drops not relevant columns
    GL_eff = GL_eff.drop(index='bus0')
    # bus-to-bus efficiency set with bus0 as reference (normalized)
    for j in list(GL_eff.columns.values):
        bus0_prc = GL_inputs.loc['bus0', j]
        bus0_val = GL_inputs.loc[bus0_prc, j]
        GL_eff.loc[:, j] = GL_eff.loc[:, j] / -bus0_val
        GL_eff[GL_eff == 0] = np.nan

    return GL_eff


def balance_bioCH4_MeOH_demand_GL():
    ''' function preprocesses the GreenLab site input data creting MeOH and bioCH4 demands'''

    '''Load GreenLab inputs'''
    GL_inputs = pd.read_excel(p.GL_input_file, sheet_name='Overview_2', index_col=0)
    GL_eff = GL_inputs_to_eff(GL_inputs)

    '''bioCH4 production ('demand')'''
    bioCH4_prod = p.ref_df.copy()
    bioCH4_prod = bioCH4_prod.rename(columns={p.ref_col_name: 'bioCH4 demand MWh'})
    bioCH4_prod['bioCH4 demand MWh'] = np.abs(
        GL_inputs.loc["bioCH4", 'SkiveBiogas']) * p.f_FLH_Biogas  # MWh Yearly demand delivered
    bioCH4_prod.to_csv(p.bioCH4_prod_input_file, sep=';')  # MWh/h

    """Methanol demand"""
    # maximum of MeOH (yearly) demand compatible with CO2 produced from the biogas plant
    Methanol_demand_y_max = np.abs(GL_eff.at['Methanol', 'Methanol plant']) * np.abs(
        GL_inputs.at['CO2 pure', 'SkiveBiogas']) * p.f_FLH_Biogas * p.FLH_y  # Max MWh MeOH Yearly delivered

    # Create Randomized weekly delivery
    # Time series demand (hourly)
    f_delivery = 24 * 365 // p.MeOH_delivery_frequency  # frequency of delivery in (h)
    n_delivery = len(p.hours_in_period) // f_delivery
    # Delivery constant amount profile
    q_delivery = Methanol_demand_y_max / n_delivery
    empty_v = np.zeros(len(p.hours_in_period))
    delivery = pd.DataFrame({'a': empty_v})
    Methanol_demand = p.ref_df.copy()
    Methanol_demand.rename(columns={p.ref_col_name: 'Methanol demand MWh'}, inplace=True)

    for i in range(n_delivery):
        delivery_ind = (i + 1) * f_delivery - 10  # Delivery at 14:00
        #delivery.iloc[delivery_ind] = q_delivery[i]
        delivery.iloc[delivery_ind] = q_delivery

    Methanol_demand['Methanol demand MWh'] = delivery['a'].values

    Methanol_demand.to_csv(p.Methanol_demand_input_file, sep=';')  # t/h

    return


def load_input_data():
    """Load csv files and prepare Input Data to GL network"""
    GL_inputs = pd.read_excel(p.GL_input_file, sheet_name='Overview_2', index_col=0)
    GL_eff = GL_inputs_to_eff(GL_inputs)
    Elspotprices = pd.read_csv(p.El_price_input_file, sep=';', index_col=0)  # currency/MWh
    Elspotprices = Elspotprices.set_axis(p.hours_in_period)
    CO2_emiss_El = pd.read_csv(p.CO2emis_input_file, sep=';', index_col=0)  # kg/MWh CO2
    CO2_emiss_El = CO2_emiss_El.set_axis(p.hours_in_period)
    bioCH4_prod = pd.read_csv(p.bioCH4_prod_input_file, sep=';', index_col=0)  # MWh/h y
    bioCH4_prod = bioCH4_prod.set_axis(p.hours_in_period)
    CF_wind = pd.read_csv(p.CF_wind_input_file, sep=';', index_col=0)  # MWh/h y
    CF_wind = CF_wind.set_axis(p.hours_in_period)
    CF_solar = pd.read_csv(p.CF_solar_input_file, sep=';', index_col=0)  # MWh/h y
    CF_solar = CF_solar.set_axis(p.hours_in_period)
    NG_price_year = pd.read_csv(p.NG_price_year_input_file, sep=';', index_col=0)  # MWh/h y
    NG_price_year = NG_price_year.set_axis(p.hours_in_period)
    Methanol_demand_max = pd.read_csv(p.Methanol_demand_input_file, sep=';', index_col=0)  # MWh/h y Methanol
    Methanol_demand_max = Methanol_demand_max.set_axis(p.hours_in_period)
    #NG_demand_DK = pd.read_csv(p.NG_demand_input_file, sep=';', index_col=0)  # currency/MWh
    #NG_demand_DK = NG_demand_DK.set_axis(p.hours_in_period) # different time scale
    #El_demand_DK1 = pd.read_csv(p.El_external_demand_input_file, sep=';', index_col=0)  # currency/MWh
    #El_demand_DK1 = El_demand_DK1.set_axis(p.hours_in_period)
    DH_external_demand = pd.read_csv(p.DH_external_demand_input_file, sep=';', index_col=0)  # currency/MWh
    DH_external_demand = DH_external_demand.set_axis(p.hours_in_period)
    #return GL_inputs, GL_eff, Elspotprices, CO2_emiss_El, bioCH4_prod, CF_wind, CF_solar, NG_price_year, Methanol_demand_max, NG_demand_DK, El_demand_DK1, DH_external_demand

    return GL_inputs, GL_eff, Elspotprices, CO2_emiss_El, bioCH4_prod, CF_wind, CF_solar, NG_price_year, Methanol_demand_max, DH_external_demand


# ----- EXTERNAL ENERGY MARKETS

def remove_feb_29(df):
    # Function to remove February 29 if it's a leap year, works on df and series
    # Check if the year is a leap year
    if any((df.index.month == 2) & (df.index.day == 29)):
        # Remove rows where the date is February 29
        df = df[~((df.index.month == 2) & (df.index.day == 29))]
    return df


def download_energidata(dataset_name, start_date, end_date, sort_val, filter_area):
    """Download a dataset from the Energi Data Service REST API.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset endpoint, e.g. ``"Elspotprices"`` or
        ``"DeclarationGridEmission"``.
    start_date : str
        Start of the requested period in ``"YYYY-MM-DD HH:MM"`` format.
        Spaces are replaced with ``T`` before sending the request.
    end_date : str
        End of the requested period, same format as *start_date*.
    sort_val : str
        Pre-encoded sort parameter string, e.g. ``"sort=HourDK%20asc"``.
        The function URL-decodes this before passing it to *requests* to
        avoid double-encoding.
    filter_area : str
        Filter expression string, e.g. ``r'filter={"PriceArea":"DK1"}'``.
        String values are automatically wrapped in arrays as required by
        the API (``{"PriceArea": ["DK1"]}``).

    Returns
    -------
    pandas.DataFrame
        Flat DataFrame of all records returned by the API (``result["records"]``
        normalised with :func:`pandas.json_normalize`).

    Raises
    ------
    requests.HTTPError
        If the API returns a non-2xx status code.
    """
    url = f"https://api.energidataservice.dk/dataset/{dataset_name}"
    # API expects T separator: "2023-01-01T00:00" not "2023-01-01 00:00"
    params = {
        "start": str(start_date).replace(" ", "T"),
        "end": str(end_date).replace(" ", "T"),
    }

    if sort_val:
        # sort_val may already be %-encoded (e.g. "sort=HourDK%20asc") — decode first
        # so requests doesn't double-encode it
        params["sort"] = urllib.parse.unquote(sort_val.replace("sort=", "", 1))

    if filter_area:
        # filter_area is e.g. r'filter={"PriceArea":"DK1"}' — extract the JSON part
        filter_json_str = filter_area.split("filter=", 1)[1]
        filter_dict = json.loads(filter_json_str)
        # API expects array values: {"PriceArea": ["DK1"]}
        params["filter"] = json.dumps({k: [v] if isinstance(v, str) else v
                                       for k, v in filter_dict.items()})

    response = requests.get(url=url, params=params, headers={"Accept": "application/json"})
    response.raise_for_status()
    result = response.json()
    records = result.get('records', [])
    return pd.json_normalize(records)


def retrieve_renewable_capacity_factors(token, start_date, end_date, latitude, longitude):
    """Retrieve hourly wind and solar capacity factors from the Renewables.ninja API.

    Uses MERRA-2 reanalysis data.  Solar tilt is estimated from latitude via
    ``tilt = 0.87 * lat + 3.1``; wind is modelled as a Vestas V80 2000 kW
    turbine at 100 m hub height.

    Parameters
    ----------
    token : str
        Personal API token for Renewables.ninja (see
        ``https://www.renewables.ninja/documentation/api``).
    start_date : str
        First day of the period in ``"YYYY-MM-DD"`` format.
    end_date : str
        Last day of the period in ``"YYYY-MM-DD"`` format.
    latitude : float
        Site latitude in decimal degrees.
    longitude : float
        Site longitude in decimal degrees.

    Returns
    -------
    CF_solar : pandas.DataFrame
        Hourly solar capacity factors with column ``"CF solar"``.
    CF_wind : pandas.DataFrame
        Hourly wind capacity factors with column ``"CF wind"``.

    Raises
    ------
    requests.HTTPError
        If either API request fails (e.g. 429 rate-limit after retries).
    """
    api_base = 'https://www.renewables.ninja/api/'
    s = requests.session()
    s.headers = {'Authorization': 'Token ' + token}

    # Solar PV request
    url = api_base + 'data/pv'
    optimal_tilt = latitude * 0.87 + 3.1  #  simple optimal tilt expression

    args = {
        'lat': latitude,
        'lon': longitude,
        'date_from': start_date,
        'date_to': end_date,
        'dataset': 'merra2',
        'capacity': 1.0,
        'system_loss': 0.1,
        'tracking': 0,
        'tilt': optimal_tilt,
        'azim': 180,
        'format': 'json'
    }

    r = s.get(url, params=args)
    r.raise_for_status()  # Raise an error if request fails
    parsed_response = json.loads(r.text)
    CF_solar = pd.read_json(StringIO(json.dumps(parsed_response['data'])), orient='index')
    CF_solar.rename(columns={CF_solar.columns.values[0] : 'CF solar'}, inplace=True)

    # Wind power request
    url = api_base + 'data/wind'
    args = {
        'lat': latitude,
        'lon': longitude,
        'date_from': start_date,
        'date_to': end_date,
        'capacity': 1.0,
        'height': 100,
        'turbine': 'Vestas V80 2000',
        'format': 'json'
    }

    r = s.get(url, params=args)
    r.raise_for_status()
    parsed_response = json.loads(r.text)
    CF_wind = pd.read_json(StringIO(json.dumps(parsed_response['data'])), orient='index')
    CF_wind.rename(columns={CF_wind.columns.values[0] : 'CF wind'}, inplace=True)

    return CF_solar, CF_wind


def retrive_entsoe_el_demand(API_KEY, start_day, end_day, country_code):
    """Retrieve historical electricity demand from the ENTSO-E Transparency Platform.

    .. note::
       Country codes are listed at
       ``https://github.com/EnergieID/entsoe-py/blob/master/entsoe/mappings.py``.

    Parameters
    ----------
    API_KEY : str
        ENTSO-E API key (stored in :mod:`scripts.parameters` as ``entsoe_api``).
    start_day : str
        Start date in ``"YYYY-MM-DD"`` format.
    end_day : str
        End date in ``"YYYY-MM-DD"`` format.
    country_code : str
        ENTSO-E bidding-zone code, e.g. ``"DK_1"`` for Western Denmark.

    Returns
    -------
    pandas.Series
        Hourly electricity load in MW, UTC-localised index.
    """
    # NOTE: list of country codes available here: https://github.com/EnergieID/entsoe-py/blob/master/entsoe/mappings.py

    client = EntsoePandasClient(api_key= API_KEY)

    start = pd.Timestamp(start_day, tz='Europe/Brussels')
    end = pd.Timestamp(end_day, tz='Europe/Brussels')

    ts = client.query_load(country_code, start=start, end=end)

    return ts


def pre_processing_energy_data():
    """ function that preprocess all the energy input data and saves in
    NOTE:Some data are not always used depending on the network configuration
    Prices from DK are downlaoded in DKK"""

    '''El spot prices DK1 - input DKK/MWh or EUR/MWh'''
    dataset_name = 'Elspotprices'
    sort_val = 'sort=HourDK%20asc'
    #filter_area = r'filter={"PriceArea":"DK1"}'
    Elspotprices_data = download_energidata(dataset_name, p.start_date, p.end_date, sort_val, p.filter_area)
    Elspotprices = Elspotprices_data[['HourDK', 'SpotPrice' + p.currency]].copy()
    Elspotprices.rename(columns={'SpotPrice' + p.currency: 'SpotPrice ' + p.currency}, inplace=True)
    Elspotprices['HourDK'] = pd.to_datetime(Elspotprices['HourDK'])
    Elspotprices.set_index('HourDK', inplace=True)
    Elspotprices = remove_feb_29(Elspotprices)
    Elspotprices.index.name = None
    Elspotprices.to_csv(p.El_price_input_file, sep=';')  # currency/MWh

    '''CO2 emission from El Grid DK1'''
    # DeclarationEmissionHour was removed from the API; DeclarationGridEmission covers all years
    CO2emis_data = download_energidata('DeclarationGridEmission', p.start_date, p.end_date,
                                       'sort=HourDK%20asc', p.filter_area)
    CO2_emiss_El = CO2emis_data.query("FuelAllocationMethod == '125%'")[['HourDK', 'CO2PerkWh']].copy()

    CO2_emiss_El['CO2PerkWh'] = CO2_emiss_El['CO2PerkWh'] / 1000  # t/MWh
    CO2_emiss_El.rename(columns={'CO2PerkWh': 'CO2PerMWh'}, inplace=True)
    CO2_emiss_El['HourDK'] = pd.to_datetime(CO2_emiss_El['HourDK'])
    CO2_emiss_El.set_index('HourDK', inplace=True)
    CO2_emiss_El = remove_feb_29(CO2_emiss_El)
    CO2_emiss_El.to_csv(p.CO2emis_input_file, sep=';')  # kg/MWh

    '''El Demand DK1'''
    #El_demand_DK1 = retrive_entsoe_el_demand(p.entsoe_api, p.start_date.replace("-",""), p.end_date.replace("-",""), p.bidding_zone)
    # source https://data.open-power-system-data.org/time_series/
    # El_demand_DK1 = pd.read_csv('data/time_series_60min_singleindex_filtered_DK1_2019.csv', index_col=0,
    #                            usecols=['cet_cest_timestamp', 'DK_1_load_actual_entsoe_transparency'])
    #El_demand_DK1.rename(columns={'Actual Load': 'DK_1_load_actual_entsoe_transparency MWh'},
    #                     inplace=True)
    #El_demand_DK1 = remove_feb_29(El_demand_DK1)
    #El_demand_DK1 = El_demand_DK1.set_axis(p.hours_in_period)
    #El_demand_DK1.to_csv(p.El_external_demand_input_file, sep=';')  # MWh/h

    # NG prices depending on the year
    ''' NG prices prices in DKK/kWh or EUR/kWH'''
    if p.En_price_year <= 2022:
        # due to different structure of Energinet dataset for the year 2019 and 2022
        dataset_name = 'GasMonthlyNeutralPrice'
        sort_val = 'sort=Month%20ASC'
        filter_area = ''
        NG_price_year = download_energidata(dataset_name, p.start_date, p.end_date, sort_val, filter_area)
        NG_price_col_name = 'Neutral gas price ' + 'EUR' + '/MWh'
        NG_price_year.rename(columns={'MonthlyNeutralGasPriceDKK_kWh': NG_price_col_name}, inplace=True)
        NG_price_year.rename(columns={'Month': 'HourDK'}, inplace=True)
        NG_price_year['HourDK'] = pd.to_datetime(NG_price_year['HourDK'])
        NG_price_year['HourDK'] = pd.to_datetime(NG_price_year['HourDK']).dt.tz_localize(None)
        NG_price_year.set_index('HourDK', inplace=True)
        NG_price_year[NG_price_col_name] = NG_price_year[NG_price_col_name] * 1000 / p.EUR_to_DKK  # coversion to €/MWh
        last_rows3 = pd.DataFrame(
            {'HourDK': p.hours_in_period[-1:len(p.hours_in_period)], NG_price_col_name: NG_price_year.iloc[-1, 0]})
        last_rows3.set_index('HourDK', inplace=True)
        NG_price_year = pd.concat([NG_price_year, last_rows3])
        NG_price_year = NG_price_year.asfreq('h', method='ffill')

    elif p.En_price_year > 2022:
        # due to different structure of Energinet dataset for the year 2019 and 2022
        dataset_name = 'GasDailyBalancingPrice'
        sort_val = 'sort=GasDay%20ASC'
        filter_area = ''

        THE_daily_NG_prices = download_energidata(dataset_name, p.start_date, p.end_date, sort_val, filter_area)
        THE_daily_NG_prices['THE_NG_pricesEUR_MWh'] = THE_daily_NG_prices['THEPriceDKK_kWh'] * 1000 / \
                                                      THE_daily_NG_prices['ExchangeRateEUR_DKK'] * 100
        THE_daily_NG_prices.rename(columns={'GasDay': 'HourDK'}, inplace=True)
        THE_daily_NG_prices['HourDK'] = pd.to_datetime(THE_daily_NG_prices['HourDK'])
        THE_daily_NG_prices['HourDK'] = pd.to_datetime(THE_daily_NG_prices['HourDK']).dt.tz_localize(None)
        THE_daily_NG_prices.set_index('HourDK', inplace=True)
        last_rows3 = pd.DataFrame(
            {'HourDK': p.hours_in_period[-1:len(p.hours_in_period)], 'THE_NG_pricesEUR_MWh': THE_daily_NG_prices.iloc[-1, 0]})
        last_rows3.set_index('HourDK', inplace=True)
        THE_daily_NG_prices = pd.concat([THE_daily_NG_prices, last_rows3])
        THE_daily_NG_prices = THE_daily_NG_prices.asfreq('h', method='ffill')
        NG_price_year = THE_daily_NG_prices[['THE_NG_pricesEUR_MWh']].copy()

    NG_price_year = remove_feb_29(NG_price_year)
    NG_price_year.to_csv(p.NG_price_year_input_file, sep=';')  # €/MWh

    '''  Estimated NG Demand DK '''
    # source: https://www.energidataservice.dk/tso-gas/Gasflow
    # used to create a profile for H2 demand - if required.
    dataset_name = 'Gasflow'
    sort_val = 'sort=GasDay'
    filter_area = ''
    start_date = str(p.NG_demand_year) + p.start_date[4:]
    end_date = str(p.NG_demand_year+1) + p.end_date[4:]
    NG_demand_DK_data = download_energidata(dataset_name, start_date, end_date, sort_val, filter_area)
    NG_demand_DK = NG_demand_DK_data[['GasDay', 'KWhToDenmark']].copy()
    NG_demand_DK['KWhToDenmark'] = NG_demand_DK['KWhToDenmark'] / -1000  # kWh-> MWh
    NG_demand_DK.rename(columns={'KWhToDenmark': 'NG Demand DK MWh'}, inplace=True)
    NG_demand_DK['GasDay'] = pd.to_datetime(NG_demand_DK['GasDay'])
    NG_demand_DK['GasDay'] = pd.to_datetime(NG_demand_DK['GasDay']).dt.tz_localize(None)
    NG_demand_DK.set_index('GasDay', inplace=True)
    NG_demand_DK = remove_feb_29(NG_demand_DK)
    NG_demand_DK.to_csv(p.NG_demand_input_file, sep=';')  # €/MWh

    '''District heating data'''
    # Download weather data near Skive (Mejrup)
    # https://www.dmi.dk/friedata/observationer/
    data_folder = p.DH_data_folder  # prices in currency/kWh
    name_files = os.listdir(data_folder)
    DH_Skive = pd.DataFrame()

    for name in name_files:
        df_temp_2 = pd.read_csv(os.path.join(data_folder, name), sep=';', usecols=['DateTime', 'Middeltemperatur'])
        DH_Skive = pd.concat([DH_Skive, df_temp_2])

    DH_Skive = DH_Skive.drop_duplicates(subset='DateTime', keep='first')
    DH_Skive = DH_Skive.sort_values(by=['DateTime'], ascending=True)
    DH_Skive['DateTime'] = pd.to_datetime(DH_Skive['DateTime'])
    DH_Skive['DateTime'] = pd.to_datetime(DH_Skive['DateTime'].dt.strftime("%Y-%m-%d %H:%M:%S+00:00"))
    hours_in_2019 = pd.date_range('2019-01-01T00:00' + 'Z', '2020-01-01T00:00' + 'Z', freq='h')
    hours_in_2019 = hours_in_2019.drop(hours_in_2019[-1])
    DH_Skive = DH_Skive.set_index("DateTime").reindex(hours_in_2019)

    DH_Skive_Capacity = 59  # MW
    # source: https://ens.dk/sites/ens.dk/files/Statistik/denmarks_heat_supply_2020_eng.pdf
    DH_Tamb_min = -15  # minimum outdoor temp --> maximum Capacity Factor
    DH_Tamb_max = 18  # maximum outdoor temp--> capacity Factor = 0
    CF_DH = (DH_Tamb_max - DH_Skive['Middeltemperatur'].values) / (DH_Tamb_max - DH_Tamb_min)
    CF_DH[CF_DH < 0] = 0
    DH_Skive['Capacity Factor DH'] = CF_DH
    # adjust for base load in summer months due to sanitary water
    # assumption: mean heat load in January/July = 6 (from Aarhus data).
    DH_CFmean_Jan = np.mean(DH_Skive.loc['2019-01', 'Capacity Factor DH'])
    DH_CFbase_load = DH_CFmean_Jan / 4
    DH_Skive['Capacity Factor DH'] = DH_Skive['Capacity Factor DH'] + DH_CFbase_load
    DH_Skive['DH demand MWh'] = DH_Skive[
                                    'Capacity Factor DH'] * DH_Skive_Capacity  # estimated demand for DH in Skive municipality
    DH_Skive = remove_feb_29(DH_Skive)
    DH_Skive = DH_Skive.set_axis(p.hours_in_period)
    DH_Skive.to_csv(p.DH_external_demand_input_file, sep=';')  # MWh/h

    '''Onshore Wind and Solar Capacity Factors'''
    # Download CF for wind and solar corresponding to the energy year
    CF_solar, CF_wind = retrieve_renewable_capacity_factors(p.RN_token, p.hours_in_period[0].strftime('%Y-%m-%d'), p.hours_in_period[-1].strftime('%Y-%m-%d'), p.latitude, p.longitude)
    CF_wind = remove_feb_29(CF_wind)
    CF_solar = remove_feb_29(CF_solar)
    CF_wind.to_csv(p.CF_wind_input_file, sep=';')  # kg/MWh
    CF_solar.to_csv(p.CF_solar_input_file, sep=';')  # kg/MWh

    return


# ---- Technology data

def _raw_url_to_api_url(base_url, file_name):
    """Convert a raw.githubusercontent.com URL to a GitHub Contents API URL."""
    raw_prefix = "https://raw.githubusercontent.com/"
    if not base_url.startswith(raw_prefix):
        return None
    parts = base_url[len(raw_prefix):].split("/", 3)  # owner, repo, branch, path_prefix
    owner, repo, branch = parts[0], parts[1], parts[2]
    path_prefix = parts[3] if len(parts) > 3 else ""
    return f"https://api.github.com/repos/{owner}/{repo}/contents/{path_prefix}{file_name}?ref={branch}"


def fetch_remote_sha(base_url, file_name):
    """Return the git blob SHA for *file_name* in the remote GitHub repo.

    Parameters
    ----------
    base_url : str
        Raw GitHub URL prefix (``https://raw.githubusercontent.com/...``).
    file_name : str
        Bare file name, e.g. ``"costs_2030.csv"``.

    Returns
    -------
    str or None
        The blob SHA string, or ``None`` if the API is unreachable.
    """
    api_url = _raw_url_to_api_url(base_url, file_name)
    if api_url is None:
        return None
    try:
        resp = requests.get(api_url, headers={"Accept": "application/vnd.github+json"}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("sha")
    except requests.exceptions.RequestException as e:
        print(f"[fetch_remote_sha] Could not reach GitHub API: {e}")
        return None


def get_cached_sha(local_file_path):
    """Return the SHA stored in ``<local_file_path>.sha``, or ``None``."""
    sha_path = str(local_file_path) + ".sha"
    if os.path.exists(sha_path):
        with open(sha_path) as f:
            return f.read().strip()
    return None


def _git_blob_sha(content: bytes) -> str:
    """Compute the git blob SHA1 for raw file bytes (matches GitHub's blob SHA)."""
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def retrieve_technology_data(local_file_path, base_url, max_retries: int = 3):
    """Download a cost CSV from the remote technology-data repo.

    Downloads and verifies the git blob SHA of the content against the GitHub
    Contents API.  Retries up to *max_retries* times if the CDN serves stale
    content (blob SHA mismatch).  Writes the verified SHA to
    ``<local_file_path>.sha`` so the Snakefile ``onstart`` block can detect
    future changes without downloading.

    Parameters
    ----------
    local_file_path : str
        Destination path for the downloaded CSV.
    base_url : str
        Raw GitHub URL prefix, e.g.
        ``"https://raw.githubusercontent.com/BertoGBG/technology-data/pypsa-eur_AA/outputs/"``.
    max_retries : int
        Number of download attempts before giving up on SHA verification.

    Returns
    -------
    str or None
        Path to the downloaded file, or ``None`` on network error.
    """
    local_file_path = str(local_file_path)
    local_folder = os.path.dirname(local_file_path)
    file_name = os.path.basename(local_file_path)
    sha_cache_path = local_file_path + ".sha"

    os.makedirs(local_folder, exist_ok=True)

    remote_sha = fetch_remote_sha(base_url, file_name)
    file_url = base_url + file_name

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                file_url,
                stream=True,
                timeout=30,
                headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
            )
            response.raise_for_status()
            content = response.content

            # Verify content against the git blob SHA from the API.
            # raw.githubusercontent.com can serve CDN-cached stale content
            # right after a push, causing us to store old data with a new SHA.
            if remote_sha:
                actual_sha = _git_blob_sha(content)
                if actual_sha != remote_sha:
                    print(
                        f"[retrieve] Attempt {attempt}/{max_retries}: "
                        f"SHA mismatch for {file_name} "
                        f"(expected {remote_sha[:8]}..., got {actual_sha[:8]}...). "
                        f"CDN likely served stale content — retrying."
                    )
                    if attempt < max_retries:
                        continue
                    print(f"[retrieve] Giving up SHA verification after {max_retries} attempts. Saving anyway.")

            with open(local_file_path, "wb") as fh:
                fh.write(content)
            if remote_sha:
                with open(sha_cache_path, "w") as fh:
                    fh.write(remote_sha)
            print(f"Technology-data downloaded: {file_name} (SHA: {remote_sha or 'unknown'})")
            return local_file_path

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {file_name} (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                return None
