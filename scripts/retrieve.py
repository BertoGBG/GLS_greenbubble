import pandas as pd
import numpy as np
import requests
import parameters as p
import os
from io import StringIO
import json
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta
import pytz
import hashlib


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
    NG_demand_DK = pd.read_csv(p.NG_demand_input_file, sep=';', index_col=0)  # currency/MWh
    #NG_demand_DK = NG_demand_DK.set_axis(p.hours_in_period) # different time scale
    El_demand_DK1 = pd.read_csv(p.El_external_demand_input_file, sep=';', index_col=0)  # currency/MWh
    El_demand_DK1 = El_demand_DK1.set_axis(p.hours_in_period)
    DH_external_demand = pd.read_csv(p.DH_external_demand_input_file, sep=';', index_col=0)  # currency/MWh
    DH_external_demand = DH_external_demand.set_axis(p.hours_in_period)

    return GL_inputs, GL_eff, Elspotprices, CO2_emiss_El, bioCH4_prod, CF_wind, CF_solar, NG_price_year, Methanol_demand_max, NG_demand_DK, El_demand_DK1, DH_external_demand


# ---- DEMANDS for H2, MeOH and El_DK1_GLS

def preprocess_H2_grid_demand(H2_size, flh_H2, NG_demand_DK, profile_flag, n):
    """
    Calculate H2 demand distribution over a given number of intervals (n),
    ensuring deliveries align with the last hour of each interval.

    Parameters:
    - H2_size: Hydrogen capacity size
    - flh_H2: Full load hours of H2 system
    - NG_demand_DK: DataFrame containing natural gas demand data
    - col_name: Column name for storing H2 demand
    - profile_flag: Boolean flag for profile-based allocation
    - n: Number of intervals (default: 12 for months, 52 for weeks, 1 for single year-end delivery)

    Returns:
    - H2_demand_y: DataFrame aligned with p.ref_df, with deliveries at correct timestamps
    """

    # Initialize output DataFrame with the same structure and index as p.ref_df
    H2_demand_y = p.ref_df.copy()
    col_name= 'H2_demand_MWh'
    H2_demand_y.rename(columns={'ref col': col_name}, inplace=True)
    H2_demand_y[col_name] = 0

    # Convert start_date and end_date from ISO 8601 format
    timezone = pytz.utc  # keeping UTC timestamps
    start_date = datetime.strptime(p.start_date, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone)
    end_date = datetime.strptime(p.end_date, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone)

    # NG_demand_DK align timestamp
    NG_demand_DK_2 = NG_demand_DK.copy()
    NG_demand_DK_2.index = pd.to_datetime(NG_demand_DK_2.index)
    NG_demand_DK_2.index = NG_demand_DK_2.index.map(lambda x: x.replace(year=start_date.year))

    # Determine the time step based on n (monthly or weekly)
    if n == 12:
        step = timedelta(days=30)  # Approximate monthly step
    elif n == 52:
        step = timedelta(weeks=1)  # Weekly step
    elif n == 1:
        step = end_date - start_date  # Single delivery at the end of the year
    else:
        raise ValueError("Invalid value for n. Use 1 (yearly), 12 (monthly), or 52 (weekly).")

    # Generate delivery timestamps
    delivery_dates = []
    current_time = start_date

    for i in range(n):
        # Calculate next delivery time
        if n == 1:
            next_time = end_date  # One delivery at year-end
        else:
            next_time = (current_time + step).replace(hour=23, minute=0, second=0)  # Last hour of the interval

        if next_time > end_date or i == n - 1:  # Ensure last delivery is exactly at year-end
            next_time = end_date.replace(hour=23, minute=0, second=0)

        # Convert to UTC datetime
        next_time = next_time.astimezone(pytz.utc)

        # Find the last available hour within the reference DataFrame index
        valid_times = H2_demand_y.index[H2_demand_y.index <= next_time]
        if valid_times.empty:
            continue
        last_hour = valid_times[-1]  # Ensures delivery at the last available hour

        delivery_dates.append(last_hour)
        current_time = next_time  # Move to next interval start

    # Assign H2 demand values at the correct timestamps
    for i in range(len(delivery_dates)):
        end_time = delivery_dates[i]
        st_time = delivery_dates[i - 1] if i > 0 else start_date  # Ensure first interval starts from start_date

        if profile_flag:
            # Compute H2_val based only on NG demand within the current interval
            period_data = NG_demand_DK_2.loc[st_time:end_time, :].values
            total_demand = np.sum(NG_demand_DK_2.loc[start_date:end_date, :].values)  # Total demand for normalization

            if total_demand > 0:  # Avoid division by zero
                H2_val = np.sum(period_data) / total_demand * H2_size * flh_H2
            else:
                H2_val = 0  # If there's no demand data, keep it zero
        else:
            H2_val = H2_size * flh_H2 / n  # Equal division among intervals

        # Assign H2 demand value at the correct timestamp
        H2_demand_y.at[end_time, col_name] = H2_val

    H2_demand_y.to_csv(p.H2_demand_input_file, sep=';')

    return H2_demand_y

# ----- EXTERNAL ENERGY MARKETS

def remove_feb_29(df):
    # Function to remove February 29 if it's a leap year, works on df and series
    # Check if the year is a leap year
    if any((df.index.month == 2) & (df.index.day == 29)):
        # Remove rows where the date is February 29
        df = df[~((df.index.month == 2) & (df.index.day == 29))]
    return df


def download_energidata(dataset_name, start_date, end_date, sort_val, filter_area):
    """ function that download energy data from energidataservice.dk and returns a dataframe"""
    # start_date and end_data in the format '2019-01-01'
    if filter_area != '':
        URL = 'https://api.energidataservice.dk/dataset/%s?start=%s&end=%s&%s&%s' % (
            dataset_name, start_date, end_date, sort_val, filter_area)
    elif filter_area == '':
        URL = 'https://api.energidataservice.dk/dataset/%s?start=%s&end=%s&%s' % (
            dataset_name, start_date, end_date, sort_val)

    response = requests.get(url=URL)
    result = response.json()
    records = result.get('records', [])
    downloaded_df = pd.json_normalize(records)
    return downloaded_df


def retrieve_renewable_capacity_factors(token, start_date, end_date, latitude, longitude):
    """Retrieve capacity factors for wind and solar (fixed mount) from Renewable Ninjas based on latitude and longitude.
    documentation: https://www.renewables.ninja/documentation/api"""
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
    """function that retrives historical el demand with hourly resolution from a specific bidding zone"""
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
    Elspotprices['HourDK'] = pd.to_datetime(Elspotprices['HourDK'], infer_datetime_format=True)
    Elspotprices.set_index('HourDK', inplace=True)
    Elspotprices = remove_feb_29(Elspotprices)
    Elspotprices.index.name = None
    Elspotprices.to_csv(p.El_price_input_file, sep=';')  # currency/MWh

    '''CO2 emission from El Grid DK1'''
    sort_val = 'sort=HourDK%20asc'
    # filter_area = r'filter={"PriceArea":"DK1"}'
    if p.En_price_year <= 2022:
        dataset_name = 'DeclarationEmissionHour'
        CO2emis_data = download_energidata(dataset_name, p.start_date, p.end_date, sort_val,
                                           p.filter_area)  # g/kWh = kg/MWh
        CO2_emiss_El = CO2emis_data[['HourDK', 'CO2PerkWh']].copy()

    elif p.En_price_year > 2022:
        dataset_name = 'DeclarationGridEmission'
        CO2emis_data = download_energidata(dataset_name, p.start_date, p.end_date, sort_val,
                                           p.filter_area)  # g/kWh = kg/MWh
        CO2_emiss_El = CO2emis_data.query("FuelAllocationMethod == '125%'")[['HourDK', 'CO2PerkWh']].copy()

    CO2_emiss_El['CO2PerkWh'] = CO2_emiss_El['CO2PerkWh'] / 1000  # t/MWh
    CO2_emiss_El.rename(columns={'CO2PerkWh': 'CO2PerMWh'}, inplace=True)
    CO2_emiss_El['HourDK'] = pd.to_datetime(CO2_emiss_El['HourDK'], infer_datetime_format=True)
    CO2_emiss_El.set_index('HourDK', inplace=True)
    CO2_emiss_El = remove_feb_29(CO2_emiss_El)
    CO2_emiss_El.to_csv(p.CO2emis_input_file, sep=';')  # kg/MWh

    '''El Demand DK1'''
    El_demand_DK1 = retrive_entsoe_el_demand(p.entsoe_api, p.start_date.replace("-",""), p.end_date.replace("-",""), p.bidding_zone)
    # source https://data.open-power-system-data.org/time_series/
    # El_demand_DK1 = pd.read_csv('data/time_series_60min_singleindex_filtered_DK1_2019.csv', index_col=0,
    #                            usecols=['cet_cest_timestamp', 'DK_1_load_actual_entsoe_transparency'])
    El_demand_DK1.rename(columns={'Actual Load': 'DK_1_load_actual_entsoe_transparency MWh'},
                         inplace=True)
    El_demand_DK1 = remove_feb_29(El_demand_DK1)
    El_demand_DK1 = El_demand_DK1.set_axis(p.hours_in_period)
    El_demand_DK1.to_csv(p.El_external_demand_input_file, sep=';')  # MWh/h

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
        NG_price_year['HourDK'] = pd.to_datetime(NG_price_year['HourDK'].dt.strftime("%Y-%m-%d %H:%M:%S+00:00"))
        NG_price_year.set_index('HourDK', inplace=True)
        NG_price_year[NG_price_col_name] = NG_price_year[NG_price_col_name] * 1000 / p.DKK_Euro  # coversion to €/MWh
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
        THE_daily_NG_prices['HourDK'] = pd.to_datetime(THE_daily_NG_prices['HourDK'].dt.strftime("%Y-%m-%d %H:%M:%S+00:00"))
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
    NG_demand_DK['GasDay'] = pd.to_datetime(NG_demand_DK['GasDay'].dt.strftime("%Y-%m-%d %H:%M:%S+00:00"))
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
    hours_in_2019 = pd.date_range('2019-01-01T00:00' + 'Z', '2020-01-01T00:00' + 'Z', freq='H')
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


def build_electricity_grid_price_w_tariff(Elspotprices):
    """this function creates the Electricity grid price including the all the tariffs
    Note that CO2 tax is added separately
    Tariff system valid for customer conected to 60kV grid via a 60/10kV transformer
    Tariff system in place from 2025"""

    # for tariff reference check the parameter file
    # Grid tariff are based on hour of the day, day of the week and season:
    # high tariff in summer + weekdays + 06:00 to 24.00
    # high tariff in winter + weekends + 06:00 to 24.00
    # high tariff in winter + weekdays + 21:00 to 24.00
    # peak tariff in winter + weekdays + 06:00 to 21.00
    # Low tariff the rest of the time

    summer_start = str(p.En_price_year) + '-04-01T00:00'  # '2019-04-01 00:00:00+00:00' # Monday
    summer_end = str(p.En_price_year) + '-10-01T00:00'  # '2019-10-01 00:00:00+00:00'
    winter_1 = pd.date_range(p.start_date + 'Z', summer_start + 'Z', freq='H')
    winter_1 = winter_1.drop(winter_1[-1])
    winter_2 = pd.date_range(summer_end + 'Z', p.end_date + 'Z', freq='H')
    winter_2 = winter_2.drop(winter_2[-1])
    winter = winter_1.append(winter_2)
    winter = winter[~((winter.month == 2) & (winter.day == 29))]
    summer = pd.date_range(summer_start + 'Z', summer_end + 'Z', freq='H')
    summer = summer.drop(summer[-1])

    peak_weekday = range(1, 6)
    peak_hours = range(7, 21 + 1)
    high_hours_weekday_winter = range(22, 24 + 1)
    high_hours_weekend_winter = range(7, 24 + 1)
    high_hours_weekday_summer = range(7, 24 + 1)

    # set the tariff in every hour equal to low and che
    el_grid_price = Elspotprices + p.el_transmission_tariff + p.el_system_tariff + p.el_afgift
    el_grid_sell_price = -Elspotprices + p.el_tariff_sell

    # assign tariff to hours
    for h in winter:
        day = h.weekday()
        hour = h.hour
        net_tariff = 0  # Default value

        if day in [5, 6]:  # weekends
            if hour in high_hours_weekend_winter:
                net_tariff = p.el_net_tariff_high
            else:
                net_tariff = p.el_net_tariff_low
        elif day in range(0, 5):  # weekdays
            if hour in peak_hours:
                net_tariff = p.el_net_tariff_peak
            elif hour in high_hours_weekday_winter:
                net_tariff = p.el_net_tariff_high
            else:
                net_tariff = p.el_net_tariff_low

        el_grid_price.loc[h, :] = el_grid_price.loc[h, :] + net_tariff

    for h in summer:
        day = h.weekday()
        hour = h.hour
        net_tariff = 0  # Default value

        if day in [5, 6]:  # weekends
            net_tariff = p.el_net_tariff_low
        elif day in range(0, 5):  # weekdays
            if hour in high_hours_weekday_summer:
                net_tariff = p.el_net_tariff_high
            else:
                net_tariff = p.el_net_tariff_low

        el_grid_price.loc[h, :] = el_grid_price.loc[h, :] + net_tariff

    return el_grid_price, el_grid_sell_price


# ---- Technology data
def retrieve_technology_data(file_name, local_folder, base_url):
    """
    Downloads a specific .CSV cost file from the PyPSA technology-data GitHub repository
    and saves it in a specified local folder. If the file already exists locally, it checks
    if the remote file is different before downloading.

    Parameters:
    - file_name (str): The name of the CSV file to download (e.g., "costs.csv").
    - local_folder (str): The local directory where the file will be saved.

    Returns:
    - str: Path to the downloaded file if successful, None if skipped.
    """

    # GitHub raw file URL
    # base_url = "https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/"
    file_url = base_url + file_name

    # Create the local folder if it does not exist
    os.makedirs(local_folder, exist_ok=True)

    # Local file path
    local_file_path = os.path.join(local_folder, file_name)

    # Function to compute file hash
    def compute_file_hash(file_path):
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    # Function to get GitHub file hash
    def get_github_file_hash(url):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            hasher = hashlib.sha256()
            for chunk in response.iter_content(chunk_size=8192):
                hasher.update(chunk)
            return hasher.hexdigest()
        except requests.exceptions.RequestException as e:
            print(f" Error checking GitHub file hash: {e}")
            return None

    # Check if file exists locally
    if os.path.exists(local_file_path):
        local_hash = compute_file_hash(local_file_path)
        github_hash = get_github_file_hash(file_url)

        if github_hash and local_hash == github_hash:
            print(f"{file_name} is already up-to-date. Skipping download.")
            return None  # File is unchanged, no need to download

    # Download the file
    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        with open(local_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Technology-data updated: {file_name}")
        return local_file_path

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {file_name}: {e}")
        return None
