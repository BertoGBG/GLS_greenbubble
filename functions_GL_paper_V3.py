import pandas as pd
import numpy as np
import requests
import pypsa
import linopy
import pypsatopo
import parameters_GL_paper_V3 as p
import os
import sys
import matplotlib.pyplot as plt
import pickle as pkl
import math
import itertools
import bisect
import dataframe_image as dfi
import seaborn as sns
import re


# -------TECHNO-ECONOMIC DATA & ANNUITY
def annuity(n, r):
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20,0.05)*20 = 1.6"""

    if r > 0:
        return r / (1. - 1. / (1. + r) ** n)
    else:
        return 1 / n

def prepare_costs(cost_file, USD_to_EUR, discount_rate, Nyears, lifetime):
    """ This function uses, data retrived form the technology catalogue and other sources and compiles a DF used in the model
    input: cost_file # csv
    output: costs # DF with all cost used in the model"""

    # Nyear = nyear in the interval for myoptic optimization--> set to 1 for annual optimization
    # set all asset costs and other parameters
    costs = pd.read_csv(cost_file, index_col=[0, 1]).sort_index()

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.loc[costs.unit.str.contains("USD"), "value"] *= USD_to_EUR

    # min_count=1 is important to generate NaNs which are then filled by fillna
    costs = costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)
    costs = costs.fillna({"CO2 intensity": 0,
                          "FOM": 0,
                          "VOM": 0,
                          "discount rate": discount_rate,
                          "efficiency": 1,
                          "fuel": 0,
                          "investment": 0,
                          "lifetime": lifetime
                          })
    annuity_factor = lambda v: annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100
    costs["fixed"] = [annuity_factor(v) * v["investment"] * Nyears for i, v in costs.iterrows()]
    return costs

def cost_add_technology(discount_rate, tech_costs, technology, investment, lifetime, FOM):
    '''function to calculate annualized fixed cost for any technology from inpits
    and adds it to the tech_costs dataframe '''
    annuity_factor = annuity(lifetime, discount_rate) + FOM / 100
    tech_costs.at[technology, "fixed"] = annuity_factor * investment
    tech_costs.at[technology, "lifetime"] = lifetime
    tech_costs.at[technology, "FOM"] = FOM
    tech_costs.at[technology, "investment"] = investment
    return tech_costs

def add_technology_cost(tech_costs):
    """Function that adds the tehcnology costs not presente din the original cost file"""

    cost_add_technology(p.discount_rate, tech_costs, 'CO2 storage cylinders', p.CO2_cylinders_inv, tech_costs.at['CO2 storage tank', 'lifetime'], tech_costs.at['CO2 storage tank', 'FOM'])
    cost_add_technology(p.discount_rate, tech_costs, 'CO2_pipeline_gas', p.CO2_pipeline_inv, p.CO2_pipeline_lifetime, p.CO2_pipeline_FOM)
    cost_add_technology(p.discount_rate, tech_costs, 'H2_pipeline_gas', p.H2_pipeline_inv, p.H2_pipeline_lifetime, p.H2_pipeline_FOM)
    cost_add_technology(p.discount_rate, tech_costs, 'CO2_compressor', p.CO2_comp_inv, p.CO2_comp_lifetime, p.CO2_comp_FOM)
    cost_add_technology(p.discount_rate, tech_costs, 'DH heat exchanger', p.DH_HEX_inv, p.DH_HEX_lifetime, p.DH_HEX_FOM)
    cost_add_technology(p.discount_rate, tech_costs, 'Slow pyrolysis', p.Pyrolysis_inv, p.Pyrolysis_lifetime, p.Pyrolysis_FOM)
    cost_add_technology(p.discount_rate, tech_costs, 'Electrolysis small', p.Electrolysis_small_inv, p.Electrolysis_small_lifetime, p.Electrolysis_small_FOM)
    cost_add_technology(p.discount_rate, tech_costs, 'Electrolysis large', p.Electrolysis_large_inv, p.Electrolysis_large_lifetime, p.Electrolysis_large_FOM)

    return tech_costs


# ------ INPUTS PRE-PROCESSING
'''GreenLab Inputs'''
def GL_inputs_to_eff(GL_inputs):
    ''' function that reads csv file with GreenLab energy and material flows for each plant and calcualtes the
     efficiencies for multilinks in the network'''

    #NOTE: (-) refers to energy or material flow CONSUMED by the plant
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
    ''' function preprocesses the GreenLab site input data'''

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
    f_delivery = 24 * 7  # frequency of delivery in (h)
    n_delivery = len(p.hours_in_period) // f_delivery
    # Delivery amount profile
    KK = np.random.uniform(low=0.0, high=1.0, size=n_delivery)
    q_delivery = KK * Methanol_demand_y_max / np.sum(KK)  # quantity for each delivery
    empty_v = np.zeros(len(p.hours_in_period))
    delivery = pd.DataFrame({'a': empty_v})
    Methanol_demand_max = p.ref_df.copy()
    Methanol_demand_max.rename(columns={p.ref_col_name: 'Methanol demand MWh'}, inplace=True)

    for i in range(n_delivery):
        delivery_ind = (i + 1) * f_delivery - 10  # Delivery at 14:00
        delivery.iloc[delivery_ind] = q_delivery[i]
    Methanol_demand_max['Methanol demand MWh'] = delivery['a'].values
    Methanol_demand_max.to_csv(p.Methanol_demand_max_input_file, sep=';')  # t/h
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
    Methanol_demand_max = pd.read_csv(p.Methanol_demand_max_input_file, sep=';', index_col=0)  # MWh/h y Methanol
    Methanol_demand_max = Methanol_demand_max.set_axis(p.hours_in_period)
    NG_demand_DK = pd.read_csv(p.NG_demand_input_file, sep=';', index_col=0)  # currency/MWh
    NG_demand_DK = NG_demand_DK.set_axis(p.hours_in_period)
    El_demand_DK1 = pd.read_csv(p.El_external_demand_input_file, sep=';', index_col=0)  # currency/MWh
    El_demand_DK1 = El_demand_DK1.set_axis(p.hours_in_period)
    DH_external_demand = pd.read_csv(p.DH_external_demand_input_file, sep=';', index_col=0)  # currency/MWh
    DH_external_demand = DH_external_demand.set_axis(p.hours_in_period)

    return GL_inputs, GL_eff, Elspotprices, CO2_emiss_El, bioCH4_prod, CF_wind, CF_solar, NG_price_year, Methanol_demand_max, NG_demand_DK, El_demand_DK1, DH_external_demand


'''Create demands for H2, MeOH and El_DK1_GLS'''
def preprocess_H2_grid_demand(flh_H2, H2_size, NG_demand_DK, flag_one_delivery):
    ''' Creates the H2 demand from the grid '''
    # flh_H2 = target full load hours from the plant
    # H2 size = plant size in MW of H2 produced
    # IF: flag_one_delivery == True -> single delivery at the end of the year
    # IF: flag_one_delivery == False -> monthly delivery follows the same profile of the NG demand

    H2_demand_y = p.ref_df.copy()
    col_name = 'H2_demand' + ' MWh/month'
    H2_demand_y.rename(columns={p.ref_col_name: col_name}, inplace=True)
    h2_demand_flag = p.h2_demand_flag

    n = 12  # numebr of intervals (months)
    for i in range(1, n + 1):
        if i < 9:
            st_time1 = p.start_date[0:5] + '0%d' + p.start_date[7:]
            end_time1 = p.start_date[0:5] + '0%d' + p.start_date[7:]
            st_time = st_time1 % i
            end_time = end_time1 % (i + 1)
            if h2_demand_flag == 'profile':
                H2_val = np.sum(NG_demand_DK.loc[st_time:end_time, :].values) / np.sum(
                    NG_demand_DK.values) * H2_size * flh_H2
                H2_demand_y.at[end_time, col_name] = H2_val
            else:
                H2_demand_y.at[end_time, col_name] = H2_size * flh_H2 / n
        elif i == 9:
            st_time1 = p.start_date[0:5] + '0%d' + p.start_date[7:]
            end_time1 = p.start_date[0:5] + '%d' + p.start_date[7:]
            st_time = st_time1 % i
            end_time = end_time1 % (i + 1)
            if h2_demand_flag == 'profile':
                H2_val = np.sum(NG_demand_DK.loc[st_time:end_time, :].values) / np.sum(
                    NG_demand_DK.values) * H2_size * flh_H2
                H2_demand_y.at[end_time, col_name] = H2_val
            else:
                H2_demand_y.at[end_time, col_name] = H2_size * flh_H2 / n
        elif i == n:
            st_time1 = p.start_date[0:5] + '%d' + p.start_date[7:]
            st_time = st_time1 % i
            end_time = p.end_date
            if h2_demand_flag == 'profile':
                H2_val = np.sum(NG_demand_DK.loc[st_time:end_time, :].values) / np.sum(
                    NG_demand_DK.values) * H2_size * flh_H2
                H2_demand_y.at[end_time, col_name] = H2_val
            else:
                H2_demand_y.at[end_time, col_name] = H2_size * flh_H2 / n
        else:
            st_time1 = p.start_date[0:5] + '%d' + p.start_date[7:]
            end_time1 = p.start_date[0:5] + '%d' + p.start_date[7:]
            st_time = st_time1 % i
            end_time = end_time1 % (i + 1)
            if h2_demand_flag == 'profile':
                H2_val = np.sum(NG_demand_DK.loc[st_time:end_time, :].values) / np.sum(
                    NG_demand_DK.values) * H2_size * flh_H2
                H2_demand_y.at[end_time, col_name] = H2_val
            else:
                H2_demand_y.at[end_time, col_name] = H2_size * flh_H2 / n

    if flag_one_delivery:
        delivery = H2_demand_y.iloc[:, 0].sum()
        H2_demand_y = p.ref_df.copy()
        H2_demand_y.iloc[-1, 0] = delivery

    H2_demand_y.to_csv(p.H2_demand_input_file, sep=';')  # currency/ton

    return H2_demand_y

def preprocess_methanol_demand(Methanol_demand_max, f_max_MeOH_y_demand, flag_one_delivery):
    ''' function that builds a Methnaol demand (load) for the GL network.
    The load can be on weekly basis or one single delivery at the end of the year'''

    # if: flag_one_delivery= True => aggregates methanol demand to one single time step in the end of the year.
    # f_max_MeOH_y_demand = fraction (0-1) of maximum MeOH production compatible with yearly prod of CO2 from biogas plant
    # if p.MeOH_set_demand_input == True:
    if flag_one_delivery == True:  # yearly demand
        Methanol_input_demand = Methanol_demand_max.copy()
        Methanol_input_demand["Methanol demand MWh"] = 0
        Methanol_input_demand.iloc[-1, :] = np.sum(Methanol_demand_max.values) * f_max_MeOH_y_demand
    elif flag_one_delivery == False:  # weekly demand
        Methanol_input_demand = Methanol_demand_max * f_max_MeOH_y_demand

    return Methanol_input_demand

'''Energy Market data'''
def download_energidata(dataset_name, start_date, end_date, sort_val, filter_area):
    ''' function that download energy data from energidataservice.dk and returns a dataframe'''
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

def pre_processing_energy_data():
    ''' function that preprocess all the energy input data and saves in
    NOTE:Some data are not always used depending on the network configuration
    Prices from DK are downlaoded in DKK'''

    '''El spot prices DK1 - input DKK/MWh or EUR/MWh'''
    dataset_name = 'Elspotprices'
    sort_val = 'sort=HourDK%20asc'
    filter_area = r'filter={"PriceArea":"DK1"}'
    Elspotprices_data = download_energidata(dataset_name, p.start_date, p.end_date, sort_val, filter_area)
    Elspotprices = Elspotprices_data[['HourDK', 'SpotPrice' + p.currency]].copy()
    Elspotprices.rename(columns={'SpotPrice' + p.currency: 'SpotPrice ' + p.currency}, inplace=True)
    Elspotprices['HourDK'] = pd.to_datetime(Elspotprices['HourDK'], infer_datetime_format=True)
    Elspotprices.set_index('HourDK', inplace=True)
    Elspotprices.index.name = None
    Elspotprices.to_csv(p.El_price_input_file, sep=';')  # currency/MWh

    '''CO2 emission from El Grid DK1'''
    dataset_name = 'DeclarationEmissionHour'
    sort_val = 'sort=HourDK%20asc'
    filter_area = r'filter={"PriceArea":"DK1"}'
    CO2emis_data = download_energidata(dataset_name, p.start_date, p.end_date, sort_val, filter_area)  # g/kWh = kg/MWh
    CO2_emiss_El = CO2emis_data[['HourDK', 'CO2PerkWh']].copy()
    CO2_emiss_El['CO2PerkWh'] = CO2_emiss_El['CO2PerkWh'] / 1000  # t/MWh
    CO2_emiss_El.rename(columns={'CO2PerkWh': 'CO2PerMWh'}, inplace=True)
    CO2_emiss_El['HourDK'] = pd.to_datetime(CO2_emiss_El['HourDK'], infer_datetime_format=True)
    CO2_emiss_El.set_index('HourDK', inplace=True)
    CO2_emiss_El.index.name = None
    CO2_emiss_El.to_csv(p.CO2emis_input_file, sep=';')  # kg/MWh

    '''El Demand DK1'''
    # source https://data.open-power-system-data.org/time_series/
    El_demand_DK1 = pd.read_csv('data/time_series_60min_singleindex_filtered_DK1_2019.csv', index_col=0,
                                usecols=['cet_cest_timestamp', 'DK_1_load_actual_entsoe_transparency'])
    El_demand_DK1.rename(columns={'DK_1_load_actual_entsoe_transparency': 'DK_1_load_actual_entsoe_transparency MWh'},
                         inplace=True)
    El_demand_DK1 = El_demand_DK1.set_axis(p.hours_in_period)
    El_demand_DK1 = El_demand_DK1
    El_demand_DK1.to_csv(p.El_external_demand_input_file, sep=';')  # MWh/h

    # NG prices depending on the year
    ''' NG prices prices in DKK/kWh or EUR/kWH'''

    if p.En_price_year == 2019:
        # due to different structure of Energinet dataset for the year 2019 and 2022
        data_folder = p.NG_price_data_folder  # prices in DKK/kWh or EUR/kWH
        name_files = os.listdir(data_folder)
        NG_price_year = pd.DataFrame()
        NG_price_col_name = 'Neutral gas price ' + 'DKK' + '/kWh'
        NG_price_col_name_new = 'Neutral gas price ' + 'EUR' + '/MWh'

        for name in name_files:
            # print(name)
            df_temp = pd.read_csv(os.path.join(data_folder, name),
                                  skiprows=[0, 1, 2], sep=';', index_col=0,
                                  usecols=['Delivery date', NG_price_col_name])
            df_temp.index = pd.to_datetime(df_temp.index, dayfirst=True).strftime("%Y-%m-%dT%H:%M:%S+000")

            if df_temp[NG_price_col_name].dtype != 'float64':
                df_temp[NG_price_col_name] = df_temp[NG_price_col_name].str.replace(',', '.').astype(float)
            NG_price_year = pd.concat([NG_price_year, df_temp])

        NG_price_year = NG_price_year.sort_index()
        NG_price_year[NG_price_col_name] = NG_price_year[NG_price_col_name] * 1000 / p.DKK_Euro # converts to MwH and Euro
        NG_price_year.rename(columns={NG_price_col_name: NG_price_col_name_new}, inplace=True)
        NG_price_year.index.rename('HourDK', inplace=True)
        NG_price_year.index.name = None

        # Up-sampling to hour resolution
        NG_price_year.index = pd.to_datetime(NG_price_year.index)
        NG_price_year = NG_price_year.asfreq('H', method='ffill')
        # add last 23h
        last_rows_time = p.hours_in_period[-23:len(p.hours_in_period)]
        last_rows_val = np.ones(len(last_rows_time)) * NG_price_year.iloc[-1, 0]
        last_rows = pd.DataFrame({'Delivery date': last_rows_time, NG_price_col_name_new: last_rows_val})
        last_rows = last_rows.set_index('Delivery date')
        last_rows.index = pd.to_datetime(last_rows.index)
        NG_price_year = pd.concat([NG_price_year, last_rows])
        NG_price_year.to_csv(p.NG_price_year_input_file, sep=';')  # €/MWh

    elif p.En_price_year == 2022:
        # due to different structure of Energinet dataset for the year 2019 and 2022
        dataset_name = 'GasMonthlyNeutralPrice'
        sort_val = 'sort=Month%20ASC'
        filter_area=''
        NG_price_year = download_energidata(dataset_name, p.start_date, p.end_date, sort_val, filter_area)

        NG_price_col_name='Neutral gas price ' + 'EUR' + '/MWh'
        NG_price_year.rename(columns={'MonthlyNeutralGasPriceDKK_kWh': NG_price_col_name}, inplace=True)
        NG_price_year.rename(columns={'Month': 'HourDK'}, inplace=True)
        NG_price_year['HourDK'] = pd.to_datetime(NG_price_year['HourDK'])
        NG_price_year['HourDK'] = pd.to_datetime(NG_price_year['HourDK'].dt.strftime("%Y-%m-%d %H:%M:%S+00:00"))
        NG_price_year.set_index('HourDK', inplace=True)
        NG_price_year[NG_price_col_name] = NG_price_year[NG_price_col_name] * 1000 / p.DKK_Euro # coversion to €/MWh
        last_rows3 = pd.DataFrame({'HourDK': p.hours_in_period[-1:len(p.hours_in_period)], NG_price_col_name: NG_price_year.iloc[-1,0]})
        last_rows3.set_index('HourDK', inplace=True)
        NG_price_year = pd.concat([NG_price_year, last_rows3])
        NG_price_year = NG_price_year.asfreq('H', method='ffill')
        NG_price_year.to_csv(p.NG_price_year_input_file, sep=';')  # €/MWh


    ''' NG Demand (Consumption) DK '''
    # source: https://www.energidataservice.dk/tso-gas/Gasflow
    dataset_name = 'Gasflow'
    sort_val = 'sort=GasDay'
    filter_area = ''
    NG_demand_DK_data = download_energidata(dataset_name, p.start_date, p.end_date, sort_val, filter_area)
    NG_demand_DK = NG_demand_DK_data[['GasDay', 'KWhToDenmark']].copy()
    NG_demand_DK['KWhToDenmark'] = NG_demand_DK['KWhToDenmark'] / -1000  # kWh-> MWh
    NG_demand_DK.rename(columns={'KWhToDenmark': 'NG Demand DK MWh'}, inplace=True)
    NG_demand_DK['GasDay'] = pd.to_datetime(NG_demand_DK['GasDay'], infer_datetime_format=True)
    NG_demand_DK.set_index('GasDay', inplace=True)
    NG_demand_DK = NG_demand_DK.asfreq('H', method='ffill')
    NG_demand_DK.rename_axis(index='HourDK', inplace=True)

    # add last 23h
    last_rows_time2 = p.hours_in_period[-23:len(p.hours_in_period)]
    last_rows_val2 = np.ones(len(last_rows_time2)) * NG_demand_DK.iloc[-1, 0]
    last_rows2 = pd.DataFrame({'HourDK': last_rows_time2, 'NG Demand DK MWh': last_rows_val2})
    last_rows2 = last_rows2.set_index('HourDK')
    last_rows2.index = pd.to_datetime(last_rows2.index)
    NG_demand_DK = pd.concat([NG_demand_DK, last_rows2])
    NG_demand_DK['NG Demand DK MWh'] = NG_demand_DK['NG Demand DK MWh'] / 24
    NG_demand_DK = NG_demand_DK.set_index(p.hours_in_period)
    NG_demand_DK.to_csv(p.NG_demand_input_file, sep=';')  # MWh/h

    '''District heating data'''
    # Download weather data near Skive (Mejrup)
    # https://www.dmi.dk/friedata/observationer/
    data_folder = p.DH_data_folder  # prices in currency/kWh
    name_files = os.listdir(data_folder)
    DH_Skive = pd.DataFrame()

    for name in name_files:
        df_temp_2 = pd.read_csv(os.path.join(data_folder, name), sep=';', usecols=['DateTime', 'Middeltemperatur'])
        DH_Skive = pd.concat([DH_Skive, df_temp_2])
        # print(name)

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
    DH_Skive.to_csv(p.DH_external_demand_input_file, sep=';')  # MWh/h
    DH_Skive = DH_Skive.set_axis(p.hours_in_period)

    '''Onshore Wind Capacity Factor DK1'''
    # from 2017
    hours_in_2017 = pd.date_range('2017-01-01T00:00Z', '2017-12-31T23:00Z', freq='H')
    df_onshorewind = pd.read_csv('data/onshore_wind_1979-2017.csv', sep=';', index_col=0)
    df_onshorewind.index = pd.to_datetime(df_onshorewind.index)
    CF_wind = pd.DataFrame(df_onshorewind['DNK'][[hour.strftime("%Y-%m-%d %H:%M:%S+00:00") for hour in hours_in_2017]])
    CF_wind.rename(columns={'DNK': 'Capacity Factor Wind Onshore DK1'}, inplace=True)
    CF_wind = CF_wind.set_axis(p.hours_in_period)
    CF_wind.to_csv(p.CF_wind_input_file, sep=';')  # kg/MWh

    '''Solar Capacity Factor DK'''
    # add solar PV generator
    df_solar = pd.read_csv('data/pv_optimal.csv', sep=';', index_col=0)
    df_solar.index = pd.to_datetime(df_solar.index)
    CF_solar = pd.DataFrame(df_solar['DNK'][[hour.strftime("%Y-%m-%dT%H:%M:%S+00:00") for hour in hours_in_2017]])
    CF_solar.rename(columns={'DNK': 'Capacity Factor Solar DK'}, inplace=True)
    CF_solar = CF_solar.set_axis(p.hours_in_period)
    CF_solar.to_csv(p.CF_solar_input_file, sep=';')  # kg/MWh

    return

def build_electricity_grid_price_w_tariff(Elspotprices):
    '''this function creates the Electricity grid price including the all the tariffs
    Note that CO2 tax is added separately
    Tariff system valid for customer conected to 60kV grid via a 60/10kV transformer
    Tariff system in place from 2025'''

    # for tariff reference check the parameter file
    # Grid tariff are based on hour of the day, day of the week and season:
    # high tariff in summer + weekdays + 06:00 to 24.00
    # high tariff in winter + weekends + 06:00 to 24.00
    # high tariff in winter + weekdays + 21:00 to 24.00
    # peak tariff in winter + weekdays + 06:00 to 21.00
    # Low tariff the rest of the time

    summer_start = str(p.En_price_year) + '-04-01T00:00'  # '2019-04-01 00:00:00+00:00' # Monday
    summer_end = str(p.En_price_year) +'-10-01T00:00'  # '2019-10-01 00:00:00+00:00'
    winter_1 = pd.date_range(p.start_date + 'Z', summer_start + 'Z', freq='H')
    winter_1 = winter_1.drop(winter_1[-1])
    winter_2 = pd.date_range(summer_end + 'Z', p.end_date + 'Z', freq='H')
    winter_2 = winter_2.drop(winter_2[-1])
    winter = winter_1.append(winter_2)
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
        if day in [6, 7]:  # weekends
            if hour in high_hours_weekend_winter:
                net_tariff = p.el_net_tariff_high
            else:
                net_tariff = p.el_net_tariff_low
        elif day in range(1, 6):  # weekdays
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
        if day in [6, 7]:  # weekends
            net_tariff = p.el_net_tariff_low
        elif day in range(1, 6):  # weekdays
            if hour in high_hours_weekday_summer:
                net_tariff = p.el_net_tariff_high
            else:
                net_tariff = p.el_net_tariff_low
        el_grid_price.loc[h, :] = el_grid_price.loc[h, :] + net_tariff

    return el_grid_price, el_grid_sell_price


''' Pre-processing for PyPSA network '''
def pre_processing_all_inputs(flh_H2, f_max_MeOH_y_demand, CO2_cost, preprocess_flag):
    # functions calling all other functions and build inputs dictionary to the model
    # inputs_dict contains all in
    '''Run preprocessing of energy data and build input file OR read input files saved'''
    if preprocess_flag:
        pre_processing_energy_data() # download + preprocessing + save to CSV
        balance_bioCH4_MeOH_demand_GL() # Read CSV GL + create CSV with bioCH4 and MeOH max demands

    # load the inputs form CSV files
    GL_inputs, GL_eff, Elspotprices, CO2_emiss_El, bioCH4_prod, CF_wind, CF_solar, NG_price_year, Methanol_demand_max, NG_demand_DK, El_demand_DK1, DH_external_demand = load_input_data()

    ''' create H2 grid demand'''
    # H2_input_demand = preprocess_H2_grid_demand(flh_H2, np.abs(GL_inputs.at['H2', 'GreenHyScale']), NG_demand_DK,
    #                                            p.H2_one_delivery)

    H2_input_demand = preprocess_H2_grid_demand(flh_H2, p.H2_output, NG_demand_DK,
                                                p.H2_one_delivery)

    ''' create  Methanol demand'''
    Methanol_input_demand = preprocess_methanol_demand(Methanol_demand_max, f_max_MeOH_y_demand, p.MeOH_one_delivery)

    """ return the yearly el demand for the DK1 which is avaibale sale of RE form GLS,
    it is estimated in proportion to the El in GLS needed for producing RFNBOs """

    # Guess of  the RE EL demand yearly in GLS based on H2 and MeOH demand
    El_d_H2 = np.abs(
        H2_input_demand.values.sum() / GL_eff.at['H2', 'GreenHyScale'])  # yearly electricity demand for H2 demand
    El_d_MeOH = np.abs(Methanol_input_demand.values.sum() * (
                (GL_eff.at['H2', 'Methanol plant'] / GL_eff.at['Methanol', 'Methanol plant']) * (
                    p.el_comp_H2 + 1 / GL_eff.at['H2', 'GreenHyScale']) + p.el_comp_CO2 / GL_eff.at[
                    'Methanol', 'Methanol plant']))
    El_d_y_guess_GLS = El_d_H2 + El_d_MeOH  # MWh el for H2 and MeOH

    # Assign a ratio between the RE consumed for RFNBO production at the GLS and the Max which can be sold to DK1
    el_DK1_sale_el_RFNBO = p.el_DK1_sale_el_RFNBO
    if el_DK1_sale_el_RFNBO <0:
        el_DK1_sale_el_RFNBO = 0
        print('Warning: ElDK1 demand set = 0')

    El_d_y_DK1 = El_d_y_guess_GLS * el_DK1_sale_el_RFNBO

    # Distribute the available demand according to the time series of DK1 demand
    El_demand_DK1.iloc[:, 0] = El_demand_DK1.iloc[:, 0] * (
                El_d_y_DK1 / len(p.hours_in_period)) / El_demand_DK1.values.mean()

    inputs_dict = {'GL_inputs': GL_inputs,
                   'GL_eff': GL_eff,
                   'Elspotprices': Elspotprices,
                   'CO2_emiss_El': CO2_emiss_El,
                   'bioCH4_demand': bioCH4_prod,
                   'CF_wind': CF_wind,
                   'CF_solar': CF_solar,
                   'NG_price_year': NG_price_year,
                   'Methanol_input_demand': Methanol_input_demand,
                   'NG_demand_DK': NG_demand_DK,
                   'El_demand_DK1': El_demand_DK1,
                   'DH_external_demand': DH_external_demand,
                   'H2_input_demand': H2_input_demand,
                   'CO2 cost': CO2_cost}

    return inputs_dict

def en_market_prices_w_CO2(inputs_dict, tech_costs):
    " Returns the market price of commodities adjusted for CO2 tax --> including CO2 tax there needed!"
    " returns input currency"
    CO2_cost = inputs_dict['CO2 cost']
    CO2_emiss_El = inputs_dict['CO2_emiss_El']
    NG_price_year = inputs_dict['NG_price_year']
    Elspotprices = inputs_dict['Elspotprices']
    GL_eff = inputs_dict['GL_eff']

    el_grid_price, el_grid_sell_price = build_electricity_grid_price_w_tariff(Elspotprices)  #

    # Market prices of energy commodities purchased on the market, including CO2 tax
    # adjust el price for difference in CO2 tax
    mk_el_grid_price = el_grid_price + (np.array(CO2_emiss_El) * (CO2_cost - p.CO2_cost_ref_year))  # currency / MWh
    mk_el_grid_sell_price = el_grid_sell_price  # NOTE selling prices are negative in the model

    # NG grid price uneffected by CO2 tax (paid locally by the consumer)
    mk_NG_grid_price = NG_price_year + tech_costs.at['gas','CO2 intensity'] * (CO2_cost - p.CO2_cost_ref_year) # currency / MWH

    # District heating price
    DH_price = p.ref_df.copy()
    DH_price.iloc[:, 0] = -p.DH_price

    en_market_prices = {'el_grid_price': np.squeeze(mk_el_grid_price),
                        'el_grid_sell_price': np.squeeze(mk_el_grid_sell_price),
                        'NG_grid_price': np.squeeze(mk_NG_grid_price),
                        #'bioCH4_grid_sell_price': np.squeeze(mk_bioCH4_grid_sell_price),
                        'DH_price': np.squeeze(DH_price)
                        }

    return en_market_prices


# -----CONSTRAINT on GRID ELECTRICITY RFNBOs---------------
def p_max_pu_EU_renewable_el(Elspotprices, CO2_emiss_El):
    ''' function that enables power from the grid tp be used for H2 production according to EU rules:
    1) price below limit, 2) emissionintensity below limit'''

    idx_renw_el_p = Elspotprices[Elspotprices.values <= p.EU_renew_el_price_limit].index
    idx_renw_el_em = CO2_emiss_El[CO2_emiss_El.values <= p.EU_renew_el_emission_limit].index
    p_max_pu_renew_el_price = pd.DataFrame(data=0, index=p.hours_in_period, columns=['p_max_pu el price'])
    p_max_pu_renew_em = pd.DataFrame(data=0, index=p.hours_in_period, columns=['p_max_pu emiss limit'])
    p_max_pu_renew_el_price.loc[idx_renw_el_p, 'p_max_pu el price'] = 1
    p_max_pu_renew_em.loc[idx_renw_el_em, 'p_max_pu emiss limit'] = 1

    return p_max_pu_renew_el_price, p_max_pu_renew_em


def add_link_El_grid_to_H2(n, inputs_dict, tech_costs):
    """ sets condition for use of electricity formthe grid - depending on the year_EU and the legislation
    it is limiting the use of electricity form the grid after 2030 withouth installaiton of additional renewables"""

    Elspotprices = inputs_dict['Elspotprices']
    CO2_emiss_El = inputs_dict['CO2_emiss_El']
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    # Grid to H2 availability
    p_max_pu_renew_el_price, p_max_pu_renew_em = p_max_pu_EU_renewable_el(Elspotprices, CO2_emiss_El)

    # Link for use fo electricity from the grid to produce H2
    p_max_pu_grid_to_h2 = p.ref_df.copy()
    p_max_pu_grid_to_h2.iloc[:, 0] = p_max_pu_renew_el_price.iloc[:, 0]
    capex_DK1_to_h2 = 0  # because RE peak sold is expected to be higher than peak consumption from grid

    n.add('Link',
          'DK1_to_El3',
          bus0="ElDK1 bus",
          bus1="El3 bus",
          efficiency=1,
          p_nom_extendable=True,
          p_max_pu=p_max_pu_grid_to_h2.iloc[:, 0],
          capital_cost=capex_DK1_to_h2,
          marginal_cost=en_market_prices['el_grid_price'])

    return n



# ------- BUILD PYPSA NETWORK-------------

def network_dependencies(n_flags):
    """Check if all required dependedvies are satisfied when building the network based on n_flags dictionary in main,
    modifies n_flag dict """
    n_flags_OK = n_flags.copy()

    # SkiveBiogas : NO dependencies
    n_flags_OK['SkiveBiogas'] = n_flags['SkiveBiogas']

    # renewables : NO Dependencies
    n_flags_OK['renewables'] = n_flags['renewables']

    # H2 production Dependencies
    n_flags_OK['electrolyzer'] = n_flags['electrolyzer']

    # MeOH production Dependencies
    if n_flags['meoh'] and n_flags_OK['electrolyzer'] and n_flags['SkiveBiogas'] and n_flags['symbiosis_net']:
        n_flags_OK['meoh'] = True
    else:
        n_flags_OK['meoh'] = False

    # Symbiosis net : NO Dependencies (but layout depends on the other n_flags_OK)
    n_flags_OK['symbiosis_net'] = n_flags['symbiosis_net']

    # Central heating Dependencies
    if n_flags['central_heat'] and n_flags['symbiosis_net']:
        n_flags_OK['central_heat'] = True
    else:
        n_flags_OK['central_heat'] = False

    # DH Dependencies ( option for heat recovery form MeOH available)
    if n_flags['DH'] and n_flags['symbiosis_net']:
        n_flags_OK['DH'] = True
    else:
        n_flags_OK['DH'] = False

    return n_flags_OK

def override_components_mlinks():
    """function required by PyPSA for overrinng link component to multiple connecitons (multilink)
    the model can take up to 5 additional buses (7 in total) but can be extended"""

    override_component_attrs = pypsa.descriptors.Dict(
        {k: v.copy() for k, v in pypsa.components.component_attrs.items()})
    override_component_attrs["Link"].loc["bus2"] = ["string", np.nan, np.nan, "2nd bus", "Input (optional)"]
    override_component_attrs["Link"].loc["bus3"] = ["string", np.nan, np.nan, "3rd bus", "Input (optional)"]
    override_component_attrs["Link"].loc["bus4"] = ["string", np.nan, np.nan, "4th bus", "Input (optional)"]
    override_component_attrs["Link"].loc["bus5"] = ["string", np.nan, np.nan, "5th bus", "Input (optional)"]
    override_component_attrs["Link"].loc["bus6"] = ["string", np.nan, np.nan, "6th bus", "Input (optional)"]

    override_component_attrs["Link"].loc["efficiency2"] = ["static or series", "per unit", 1., "2nd bus efficiency",
                                                           "Input (optional)"]
    override_component_attrs["Link"].loc["efficiency3"] = ["static or series", "per unit", 1., "3rd bus efficiency",
                                                           "Input (optional)"]
    override_component_attrs["Link"].loc["efficiency4"] = ["static or series", "per unit", 1., "4th bus efficiency",
                                                           "Input (optional)"]
    override_component_attrs["Link"].loc["efficiency5"] = ["static or series", "per unit", 1., "5th bus efficiency",
                                                           "Input (optional)"]
    override_component_attrs["Link"].loc["efficiency6"] = ["static or series", "per unit", 1., "6th bus efficiency",
                                                           "Input (optional)"]

    override_component_attrs["Link"].loc["p2"] = ["series", "MW", 0., "2nd bus output", "Output"]
    override_component_attrs["Link"].loc["p3"] = ["series", "MW", 0., "3rd bus output", "Output"]
    override_component_attrs["Link"].loc["p4"] = ["series", "MW", 0., "4th bus output", "Output"]
    override_component_attrs["Link"].loc["p5"] = ["series", "MW", 0., "5th bus output", "Output"]
    override_component_attrs["Link"].loc["p6"] = ["series", "MW", 0., "6th bus output", "Output"]

    return override_component_attrs

def add_local_heat_connections(n, heat_bus_list, GL_eff, plant_name, n_flags, tech_costs):
    '''function that creates local heat buses for each plant.
    heat leaving the plant can be rejected to the ambient for free.
    heat required by the plant can be supplied by symbiosys net ar added heating technologies
    '''

    new_buses = ['', '', '']

    for i in range(len(heat_bus_list)):
        b = heat_bus_list[i] # symbiosys net bus
        if not math.isnan(GL_eff.loc[b, plant_name]):
            sign_eff = np.sign(
                GL_eff.loc[b, plant_name])  # negative is consumed by  the agent, positive is produced by the agent

            # add local bus (input)
            bus_name = b + '_' + plant_name
            new_buses[i] = bus_name

            n.add('Bus',bus_name,carrier='Heat', unit='MW')

            # for heat rejection add connection to Heat amb (cooling included in plant cost)
            if sign_eff > 0:
                link_name = b + '_' + plant_name + '_amb'
                n.add('Link',
                      link_name,
                      bus0=bus_name,
                      bus1='Heat amb',
                      efficiency=1,
                      p_nom_extendable=True)

            # if symbiosys net is available, enable connection with heat grids and add cost (bidirectional)
            if n_flags['symbiosis_net']:
                if b not in n.buses.index.values:
                    n.add('Bus',b, carrier='Heat', unit='MW')
                link_name = b + '_' + plant_name

                if sign_eff > 0:
                    bus0 = bus_name
                    bus1 = b
                elif sign_eff < 0:
                    bus0 = b
                    bus1 = bus_name

                n.add('Link', link_name,
                      bus0=bus0,
                      bus1=bus1,
                      efficiency=1,
                      p_min_pu=-1,
                      p_nom_extendable=True,
                      capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * p.currency_multiplier)

    return n, new_buses

def add_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs):
    '''function that adds El connections for the plant
    one connection to the DK1 grid.
    one connection to the El2 bus if symbiosys net is active'''

    # ------ Create Local El bus
    n.add('Bus',local_EL_bus ,carrier='AC', unit='MW')

    # -------EL connections------------
    link_name1 = 'DK1_to_' + local_EL_bus

    # direct grid connection
    n.add("Link",
          link_name1,
          bus0="ElDK1 bus",
          bus1=local_EL_bus,  # 'El_biogas',
          efficiency=1,
          marginal_cost=en_market_prices['el_grid_price'],
          capital_cost=tech_costs.at[
                           'electricity grid connection', 'fixed'] * p.currency_multiplier,
          p_nom_extendable=True)

    # internal el connection
    if n_flags['symbiosis_net']:
        if 'El2 bus' not in n.buses.index.values:
            n.add('Bus','El2 bus', carrier='AC', unit='MW')

        link_name2 = 'El2_to_' + local_EL_bus
        n.add("Link",
              link_name2,
              bus0="El2 bus",
              bus1=local_EL_bus,  # 'El_biogas',
              efficiency=1,
              p_nom_extendable=True)
    return n

def add_local_boilers(n, local_EL_bus, local_heat_bus, plant_name, tech_costs,en_market_prices):
    """function that add a local El boiler and NG boiler for plants requiring heating but not connected to the sybiosys net.
    both boilers need connections to local buses"""

    # additional NG boiler
    n.add("Link",
          "NG boiler" + plant_name,
          bus0="NG",
          bus1=local_heat_bus,
          efficiency=tech_costs.at['central gas boiler', 'efficiency'],
          p_nom_extendable=True,
          capital_cost=tech_costs.at['central gas boiler', 'fixed'] * p.currency_multiplier,
          marginal_cost=en_market_prices['NG_grid_price'] +
                        tech_costs.at['gas boiler steam', 'VOM'] * p.currency_multiplier)

    # additional El boiler
    n.add('Link',
          'El boiler',
          bus0=local_EL_bus,
          bus1=local_heat_bus,
          efficiency=tech_costs.at['electric boiler steam', 'efficiency'],
          capital_cost=tech_costs.at['electric boiler steam', 'fixed'] * p.currency_multiplier,
          marginal_cost=tech_costs.at['electric boiler steam', 'VOM'] * p.currency_multiplier,
          p_nom_extendable=True)

    return n

def add_external_grids(network, inputs_dict, n_flags):
    '''function building the external grids and loads according to n_flgas dict,
    this function DOES NOT allocate capital or marginal costs to any component'''

    '''-----BASE NETWORK STRUCTURE - INDEPENDENT ON CONFIGURATION --------'''
    ''' these components do not have allocated capital costs'''

    bus_list = ['ElDK1 bus', 'Heat amb', 'NG']
    carrier_list = ['AC', 'Heat', 'gas']
    unit_list = ['MW', 'MW', 'MW']
    add_buses = list(set(bus_list) - set(network.buses.index.values))
    idx_add = [bus_list.index(i) for i in add_buses]

    # take a status of the network before adding componets
    n0_links = network.links.index.values
    n0_generators = network.generators.index.values
    n0_loads = network.loads.index.values
    n0_stores = network.stores.index.values
    n0_buses = network.buses.index.values

    if add_buses:
        network.madd('Bus', add_buses, carrier=[carrier_list[i] for i in idx_add], unit=[unit_list[i] for i in idx_add])

    # -----------Electricity Grid and connection DK1-----------
    # Load simulating the DK1 grid load
    El_demand_DK1 = inputs_dict['El_demand_DK1']
    network.add("Load",
                "Grid Load",
                bus="ElDK1 bus",
                p_set=El_demand_DK1.iloc[:, 0])  #

    # generator simulating  all the generators in DK1
    network.add("Generator",
                "Grid gen",
                bus="ElDK1 bus",
                p_nom_extendable=True)

    # ----------ambient heat sink --------------------
    # add waste heat to ambient if not present already
    network.add("Store",
                "Heat amb",
                bus="Heat amb",
                e_nom_extendable=True,
                e_nom_min=0,
                e_nom_max=float("inf"),  # Total emission limit
                e_cyclic=False)

    # ----------NG source in local distrubtion------
    network.add("Generator",
                "NG grid",
                bus="NG",
                p_nom_extendable=True)

    # --------------District heating-------------------
    if n_flags['DH']:
        DH_external_demand = inputs_dict['DH_external_demand']
        network.add('Bus', 'DH grid', carrier='Heat', unit='MW')

        # External DH grid
        network.add('Load',
                    'DH load',
                    bus='DH grid',
                    p_set=DH_external_demand['DH demand MWh'])

        network.add("Generator",
                    "DH gen",
                    bus="DH grid",
                    p_nom_extendable=True)

    # new componets
    new_links = list(set(network.links.index.values) - set(n0_links))
    new_generators = list(set(network.generators.index.values) - set(n0_generators))
    new_loads = list(set(network.loads.index.values) - set(n0_loads))
    new_stores = list(set(network.stores.index.values) - set(n0_stores))
    new_buses = list(set(network.buses.index.values) - set(n0_buses))
    new_components = {'links': new_links,
                      'generators': new_generators,
                      'loads': new_loads,
                      'stores': new_stores,
                      'buses': bus_list}

    return network, new_components

def add_biogas(n, n_flags, inputs_dict, tech_costs):
    '''fucntion that add the biogas plant to the network and all the dependecies if not preset in the network yet'''

    bioCH4_demand = inputs_dict['bioCH4_demand']
    GL_eff = inputs_dict['GL_eff']
    GL_inputs = inputs_dict['GL_inputs']
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    # take a status of the network before adding componets
    n0_links = n.links.index.values
    n0_generators = n.generators.index.values
    n0_loads = n.loads.index.values
    n0_stores = n.stores.index.values
    n0_buses = n.buses.index.values

    bus_list = ['Biomass', 'Digest DM', 'ElDK1 bus', 'bioCH4', 'NG', 'CO2 sep', 'CO2 pure atm']
    carrier_list = ['Biomass','Digest DM', 'AC', 'gas', 'gas', 'CO2 pure', 'CO2 pure']
    unit_list = ['MW', 't/h', 'MW', 'MW', 'MW', 't/h', 't/h']

    if n_flags['SkiveBiogas']:
        # add required buses if not in the network
        add_buses = list(set(bus_list) - set(n.buses.index.values))
        idx_add = [bus_list.index(i) for i in add_buses]

        if add_buses:
            n.madd('Bus', add_buses, carrier=[carrier_list[i] for i in idx_add], unit=[unit_list[i] for i in idx_add])

        # ----------External BioMethane Load---------------------
        n.add("Load",
              "bioCH4",
              bus="bioCH4",
              p_set=bioCH4_demand['bioCH4 demand MWh'])

        # ------- Biomass generator -------
        n.add("Generator",
              "Biomass",
              bus="Biomass",
              p_nom_extendable=True)

        # ------- add EL connections------------
        local_EL_bus='El_biogas'
        n = add_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

        # -----add local heat connections
        plant_name='SkiveBiogas'
        heat_bus_list = ["Heat MT", "Heat DH", "Heat LT"]
        n, new_heat_buses = add_local_heat_connections(n, heat_bus_list, GL_eff, plant_name, n_flags, tech_costs)

        # -----Biogas plant --------------
        # NOTE 1: OPERATES STEADY STATE DUE TO CONSTANT DEMAND
        # NOTE 2: REFERENCE in the study is that standard operation of the Biogas plant has a cost = 0
        # Hence there is an opportunity for revenue by NOT using the NG boiler and Grid electricity.
        # In the calculation the plant is allocated this "Revenue" as marginal cost (every hour).
        # If in the optimal solution the Heat and El required are only from NG and Grid then the opportunity revenue canceled


        NG_opportunity_revenue = -(en_market_prices['NG_grid_price'] * np.abs(
            GL_eff.loc["Heat MT", "SkiveBiogas"]) / tech_costs.at[
                                     'gas boiler steam', 'efficiency']) # €/(t_biomass)

        EL_opportunity_revenue = -(en_market_prices['el_grid_price'] * np.abs(
            GL_eff.loc["El2 bus", "SkiveBiogas"])) # €/(t_biomass))

        n.add("Link",
              "SkiveBiogas",
              bus0="Biomass",
              bus1="bioCH4",
              bus2="CO2 sep",
              bus3=new_heat_buses[0], #"Heat MT",
              bus4=local_EL_bus, # 'El_biogas',
              bus5='Digest DM',
              bus6=new_heat_buses[2], #"Heat LT",
              efficiency=GL_eff.loc["bioCH4", "SkiveBiogas"],
              efficiency2=GL_eff.loc["CO2 pure", "SkiveBiogas"],
              efficiency3=GL_eff.loc["Heat MT", "SkiveBiogas"],
              efficiency4=GL_eff.loc["El2 bus", "SkiveBiogas"],
              efficiency5=GL_eff.loc["DM digestate", "SkiveBiogas"],
              efficiency6=GL_eff.loc["Heat LT", "SkiveBiogas"],
              p_nom=np.abs(GL_inputs.loc["Biomass", 'SkiveBiogas']),
              marginal_cost=(p.Biomass_price + NG_opportunity_revenue + EL_opportunity_revenue) * p.currency_multiplier,
              p_nom_extendable=False)

        # DM digestate  store
        n.add("Store",
                    "Digestate",
                    bus="Digest DM",
                    e_nom_extendable=True,
                    e_nom_min=0,
                    e_nom_max=float("inf"),
                    e_cyclic=False,
                    capital_cost=0)

        # ---------NG boiler--------------
        # Existing NG boiler (which can supply heat to the symbiosys net)

        n.add("Link",
              "NG boiler",
              bus0="NG",
              bus1=new_heat_buses[0],
              efficiency=tech_costs.at['gas boiler steam', 'efficiency'],
              p_nom=np.abs(
                  GL_inputs.loc['Heat MT', 'SkiveBiogas'] / tech_costs.at['gas boiler steam', 'efficiency']),
              marginal_cost= en_market_prices['NG_grid_price'] +
                            tech_costs.at['gas boiler steam', 'VOM'] * p.currency_multiplier,
              p_nom_extendable=False)

        # enables existing NG boiler to supply heat to the symbiosys network
        if n_flags['symbiosis_net']:
            n.links.p_min_pu.at[new_heat_buses[0]] = -1

        # -----------infinite Store of biogenic CO2 (venting to ATM)
        n.add("Store",
              "CO2 biogenic out",
              bus="CO2 pure atm",
              e_nom_extendable=True,
              e_nom_min=0,
              e_nom_max=float("inf"),
              e_cyclic=False,
              marginal_cost=0,
              capital_cost=0)

        n.add("Link",
              "CO2 sep to atm",
              bus0="CO2 sep",
              bus1="CO2 pure atm",
              efficiency=1,
              p_nom_extendable=True)

        # new components
        new_links = list(set(n.links.index.values) - set(n0_links))
        new_generators = list(set(n.generators.index.values) - set(n0_generators))
        new_loads = list(set(n.loads.index.values) - set(n0_loads))
        new_stores = list(set(n.stores.index.values) - set(n0_stores))
        new_buses = list(set(n.buses.index.values) - set(n0_buses))
        new_components = {'links': new_links,
                          'generators': new_generators,
                          'loads': new_loads,
                          'stores': new_stores,
                          'buses': bus_list}
    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components

def add_renewables(n, n_flags, inputs_dict, tech_costs):
    """function that add Renewable generation (wind and PV) to the model
    adds connection to the external electricity grid"""

    CF_wind = inputs_dict['CF_wind']
    CF_solar = inputs_dict['CF_solar']
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    # take a status of the network before adding componets
    n0_links = n.links.index.values
    n0_generators = n.generators.index.values
    n0_loads = n.loads.index.values
    n0_stores = n.stores.index.values
    n0_buses = n.buses.index.values

    bus_list = ['El3 bus', 'ElDK1 bus']
    carrier_list = ['AC', 'AC', 'AC']
    unit_list = ['MW', 'MW', 'MW']

    if n_flags['renewables']:
        # add required buses if not in the network
        add_buses = list(set(bus_list) - set(n.buses.index.values))
        idx_add = [bus_list.index(i) for i in add_buses]
        if add_buses:
            n.madd('Bus', add_buses, carrier=[carrier_list[i] for i in idx_add], unit=[unit_list[i] for i in idx_add])

        # Add onshore wind generators
        n.add("Carrier", "onshorewind")
        n.add("Generator",
              "onshorewind",
              bus="El3 bus",
              p_nom_max=p.p_nom_max_wind,
              p_nom_extendable=True,
              carrier="onshorewind",
              capital_cost=tech_costs.at['onwind', 'fixed'] * p.currency_multiplier,
              marginal_cost=tech_costs.at['onwind', 'VOM'] * p.currency_multiplier,
              p_max_pu=CF_wind['Capacity Factor Wind Onshore DK1'])

        # add PV utility generators
        n.add("Carrier", "solar")
        n.add("Generator",
              "solar",
              bus="El3 bus",
              p_nom_max=p.p_nom_max_solar,
              p_nom_extendable=True,
              carrier="solar",
              capital_cost=tech_costs.at['solar', 'fixed'] * p.currency_multiplier,
              marginal_cost=tech_costs.at['solar', 'VOM'] * p.currency_multiplier,
              p_max_pu=CF_solar['Capacity Factor Solar DK'])

        # add link to sell power to the external El grid
        n.add("Link",
              "El3_to_DK1",
              bus0="El3 bus",
              bus1="ElDK1 bus",
              efficiency=1,
              marginal_cost=en_market_prices['el_grid_sell_price'],
              p_nom_extendable=True,
              capital_cost=tech_costs.at[
                                     'electricity grid connection', 'fixed'] * p.currency_multiplier)

        # new componets
        new_links = list(set(n.links.index.values) - set(n0_links))
        new_generators = list(set(n.generators.index.values) - set(n0_generators))
        new_loads = list(set(n.loads.index.values) - set(n0_loads))
        new_stores = list(set(n.stores.index.values) - set(n0_stores))
        new_buses = list(set(n.buses.index.values) - set(n0_buses))
        new_components = {'links': new_links,
                          'generators': new_generators,
                          'loads': new_loads,
                          'stores': new_stores,
                          'buses': bus_list}
    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components

def add_electrolysis(n, n_flags, inputs_dict, tech_costs):
    GL_eff = inputs_dict['GL_eff']
    H2_input_demand = inputs_dict['H2_input_demand']

    # take a status of the network before adding componets
    n0_links = n.links.index.values
    n0_generators = n.generators.index.values
    n0_loads = n.loads.index.values
    n0_stores = n.stores.index.values
    n0_buses = n.buses.index.values

    # Grid to H2 availability
    # p_max_pu_renew_el_price, p_max_pu_renew_em = p_max_pu_EU_renewable_el(Elspotprices, CO2_emiss_El)
    bus_list = ['El3 bus', 'H2', 'H2 delivery', 'Heat amb']
    carrier_list = ['AC', 'H2', 'H2', 'Heat']
    unit_list = ['MW', 'MW', 'MW', 'MW']

    if n_flags['electrolyzer']:
        # add required buses if not in the network
        add_buses = list(set(bus_list) - set(n.buses.index.values))
        idx_add = [bus_list.index(i) for i in add_buses]
        if add_buses:
            n.madd('Bus', add_buses, carrier=[carrier_list[i] for i in idx_add], unit=[unit_list[i] for i in idx_add])

        # ---------- conditions for use of electricity form the grid without additional RE----
        n = add_link_El_grid_to_H2(n, inputs_dict, tech_costs)

        # -----add local heat connections
        plant_name='GreenHyScale'
        heat_bus_list = ['Heat MT', "Heat DH", "Heat LT"]
        n, new_heat_buses = add_local_heat_connections(n, heat_bus_list, GL_eff, plant_name, n_flags, tech_costs)

        # -----------Electrolyzer------------------
        # cost_electrolysis dependent on scale (grid ot MeOH only)
        if H2_input_demand.iloc[:, 0].sum() > 0:
            electrolysis_cost= tech_costs.at['Electrolysis large', 'fixed'] * p.currency_multiplier
        else:
            electrolysis_cost = tech_costs.at['Electrolysis small', 'fixed'] * p.currency_multiplier

        n.add("Link",
              "Electrolyzer",
              bus0="El3 bus",
              bus1="H2",
              bus2=new_heat_buses[2],
              efficiency=GL_eff.at['H2', 'GreenHyScale'],
              efficiency2=GL_eff.at['Heat LT', 'GreenHyScale'],
              capital_cost=electrolysis_cost,
              marginal_cost= 0,
              p_nom_extendable=True,
              ramp_limit_up=p.ramp_limit_up_electrolyzer,
              ramp_limit_down=p.ramp_limit_down_electrolyzer)

        # ------------H2 Grid for selling H2 (flexible delivery) -------
        n.add("Load",
              "H2 grid",
              bus="H2 delivery",
              p_set=H2_input_demand.iloc[:, 0])

        # bidirectional link for supply or pickup of H2 from the grid
        n.add('Link',
              'H2_to_delivery',
              bus0='H2',
              bus1='H2 delivery',
              efficiency=1,
              p_nom_extendable=True)

        # infinite store capacity for H2 grid allowing flexible production
        n.add("Store",
              "H2 delivery",
              bus="H2 delivery",
              e_nom_extendable=True,
              e_cyclic=True)

        # new components
        new_links = list(set(n.links.index.values) - set(n0_links))
        new_generators = list(set(n.generators.index.values) - set(n0_generators))
        new_loads = list(set(n.loads.index.values) - set(n0_loads))
        new_stores = list(set(n.stores.index.values) - set(n0_stores))
        new_buses = list(set(n.buses.index.values) - set(n0_buses))
        new_components = {'links': new_links,
                          'generators': new_generators,
                          'loads': new_loads,
                          'stores': new_stores,
                          'buses': bus_list}
    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components

def add_meoh(n, n_flags, inputs_dict, tech_costs):
    ''' function installing required MeOH facilities
    MeOH system can be supplied with own electolyzer but does not have a CO2 source
    To enable CO2 trade is NEEDED the symbiosis net and the source (Biogas)'''

    # if electrolyser not available in the configuration. it will be installed to fulfill MeOH demand
    GL_eff = inputs_dict['GL_eff']
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)
    Methanol_input_demand = inputs_dict['Methanol_input_demand']
    H2_input_demand = inputs_dict['H2_input_demand']

    # take a status of the network before adding componets
    n0_links = n.links.index.values
    n0_generators = n.generators.index.values
    n0_loads = n.loads.index.values
    n0_stores = n.stores.index.values
    n0_buses = n.buses.index.values

    bus_list = ['ElDK1 bus', 'H2_meoh', 'H2 HP', 'CO2_meoh', 'CO2 pure HP',
                'Methanol', 'Heat amb']
    carrier_list = ['AC',  'H2', 'H2', 'CO2 pure', 'CO2 pure', 'Methanol', 'Heat' ]
    unit_list = ['MW', 'MW', 'MW', 't/h', 't/h', 'MW', 'MW']

    if n_flags['meoh']:
        # add required buses if not in the network
        add_buses = list(set(bus_list) - set(n.buses.index.values))
        idx_add = [bus_list.index(i) for i in add_buses]
        if add_buses:
            n.madd('Bus', add_buses, carrier=[carrier_list[i] for i in idx_add], unit=[unit_list[i] for i in idx_add])

        n.add('Store',
              'Methanol prod',
              bus='Methanol',
              e_nom_extendable=True,
              e_nom_min=0,
              e_nom_max=float("inf"),
              e_cyclic=True,
              marginal_cost=0,
              capital_cost=0)

        # ----------MeOH deliver infinite storage-------
        n.add("Load",
              "Methanol",
              bus="Methanol",
              p_set=Methanol_input_demand.iloc[:, 0])

        # ------- add EL connections------------
        local_EL_bus='El_meoh'
        n = add_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

        #--------add H2 grid connection if available -----
        if n_flags['electrolyzer']:
            if H2_input_demand.iloc[:, 0].sum() > 0:                    # external H2 demand
                n.add("Link",
                      "H2grid_to_meoh",
                      bus0="H2 delivery",
                      bus1='H2_meoh',
                      efficiency=1,
                      p_nom_extendable=True,
                      capital_cost=tech_costs.at[
                                       'H2_pipeline_gas', "fixed"] * p.dist_H2_pipe * p.currency_multiplier)

        # -----------H2 compressor -----------------------
        n.add('Bus', 'H2 comp heat', carrier='Heat',unit='MW')
        n.add("Link",
              "H2 compressor",
              bus0="H2_meoh",
              bus1="H2 HP",
              bus2= local_EL_bus,
              bus3='H2 comp heat',
              efficiency=1,
              efficiency2=-1 * p.el_comp_H2,
              efficiency3 = 1 * p.heat_comp_H2,
              p_nom_extendable=True,
              capital_cost=tech_costs.at['hydrogen storage compressor', 'fixed'] * p.currency_multiplier,
              marginal_cost=tech_costs.at['hydrogen storage compressor', 'VOM'] * p.currency_multiplier)

        n.add('Link',
              'H2 comp heat rejection',
              bus0= 'H2 comp heat',
              bus1= 'Heat amb',
              efficiency= 1,
              p_nom_extendable= True)

        # -----------CO2 compressor -----------------------
        n.add('Bus', 'CO2 comp heat', carrier='Heat',unit='MW')
        n.add("Link",
              "CO2 compressor",
              bus0="CO2_meoh",
              bus1="CO2 pure HP",
              bus2=local_EL_bus,
              bus3='CO2 comp heat',
              efficiency= 1,
              efficiency2=-1 * p.el_comp_CO2,
              efficiency3= 1 * p.heat_comp_CO2,
              p_nom_extendable=True,
              capital_cost=tech_costs.at['CO2_compressor', "fixed"] * p.currency_multiplier)

        n.add('Link',
              'CO2 comp heat rejection',
              bus0= 'CO2 comp heat',
              bus1= 'Heat amb',
              efficiency= 1,
              p_nom_extendable= True)

        # ----------METHANOL PLANT---------
        # add local heat connections
        plant_name='Methanol plant'
        heat_bus_list = ['Heat MT', "Heat DH", "Heat LT"]
        n, new_heat_buses = add_local_heat_connections(n,heat_bus_list, GL_eff, plant_name, n_flags, tech_costs)

        if not n_flags['central_heat']:
            add_local_boilers(n, local_EL_bus, new_heat_buses[0], plant_name, tech_costs, en_market_prices)

        n.add("Link",
              "Methanol plant",
              bus0="CO2 pure HP",
              bus1="Methanol",
              bus2="H2 HP",
              bus3=local_EL_bus,
              bus4=new_heat_buses[0],
              bus5=new_heat_buses[1],
              efficiency=GL_eff.loc["Methanol", "Methanol plant"],
              efficiency2=GL_eff.loc["H2", "Methanol plant"],
              efficiency3=GL_eff.loc["El2 bus", "Methanol plant"],
              efficiency4=GL_eff.at['Heat MT', 'Methanol plant'],
              efficiency5=GL_eff.at['Heat DH', 'Methanol plant'],
              p_nom_extendable=True,
              capital_cost=tech_costs.at['methanolisation', "fixed"] * p.currency_multiplier,
              ramp_limit_up=p.ramp_limit_up_MeOH,
              ramp_limit_down=p.ramp_limit_down_MeOH)

        # -----------H2 HP storage cylinders ---------------
        # H2 compressed local HP Storage
        n.add('Bus', 'H2 storage',carrier='H2',unit='MW')

        n.add('Link',
              'H2 storage send',
              bus0= 'H2 HP',
              bus1= 'H2 storage',
              bus2= local_EL_bus,
              efficiency= 1,
              efficiency2= -1 * p.El_H2_storage_add,
              p_nom_extendable=True)

        n.add('Link',
              'H2 storage return',
              bus0= 'H2 storage',
              bus1= 'H2 HP',
              efficiency= 1,
              p_nom_extendable=True)

        n.add("Store",
              "H2 HP",
              bus="H2 storage",
              e_nom_extendable=True,
              capital_cost=tech_costs.at['hydrogen storage tank type 1', 'fixed'] * p.currency_multiplier,
              marginal_cost= tech_costs.at['hydrogen storage tank type 1', 'VOM'] * p.currency_multiplier,
              e_nom_max=p.e_nom_max_H2_HP,
              e_cyclic=True)

        # -----------CO2 HP storage cylinders ---------------
        n.add('Bus', 'CO2 storage', carrier='CO2', unit='t/h')
        n.add('Link',
              'CO2 storage send',
              bus0='CO2 pure HP',
              bus1='CO2 storage',
              bus2=local_EL_bus,
              efficiency=1,
              efficiency2=-1 * p.El_CO2_storage_add,
              p_nom_extendable=True)

        n.add('Link',
              'CO2 storage return',
              bus0='CO2 storage',
              bus1='CO2 pure HP',
              efficiency=1,
              p_nom_extendable=True)

        n.add("Store",
              "CO2 pure HP",
              bus="CO2 storage",
              e_nom_extendable=True,
              capital_cost=tech_costs.at[
                               'CO2 storage cylinders', 'fixed'] * p.currency_multiplier,
              e_nom_max=p.e_nom_max_CO2_HP,
              e_cyclic=True)

        # -----------CO2 Storage liquefaction--------------------
        n.add('Bus', 'CO2 liq storage', carrier='CO2 pure', unit='t/h')
        n.add('Bus', 'CO2 liq heat LT', carrier='Heat', unit='MW')

        n.add('Link',
              'CO2 liq send',
              bus0='CO2_meoh',
              bus1='CO2 liq storage',
              bus2=local_EL_bus,
              bus3='CO2 liq heat LT',
              efficiency=1,
              efficiency2=-1 * p.El_CO2_liq,
              efficiency3=p.Heat_CO2_liq_DH,
              capital_cost=tech_costs.at['CO2 liquefaction', 'fixed'] * p.currency_multiplier,
              p_nom_extendable=True)

        n.add('Link',
              'CO2 liq return',
              bus0='CO2 liq storage',
              bus1='CO2_meoh',
              capital_cost=p.CO2_evap_annualized_cost * p.currency_multiplier,
              efficiency=1,
              p_nom_extendable=True)

        if n_flags['symbiosis_net']:
            if 'Heat LT' not in n.buses.index.values:
                n.add('Bus', 'Heat LT', carrier='Heat', unit='MW')
            n.add('Link',
                  'CO2 liq heat integration',
                  bus0='CO2 liq heat LT',
                  bus1='Heat LT',
                  efficiency=1,
                  capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * p.currency_multiplier,
                  p_nom_extendable=True)

        n.add('Link',
              'CO2 liq heat rejection',
              bus0='CO2 liq heat LT',
              bus1='Heat amb',
              efficiency=1,
              p_nom_extendable=True)

        n.add("Store",
              "CO2 Liq",
              bus="CO2 liq storage",
              e_nom_extendable=True,
              capital_cost=tech_costs.at[
                               'CO2 storage tank', 'fixed'] * p.currency_multiplier,
              marginal_cost=tech_costs.at[
                                'CO2 storage tank', 'VOM'] * p.currency_multiplier,
              e_initial=0,
              e_cyclic=True)


        # new components
        new_links = list(set(n.links.index.values) - set(n0_links))
        new_generators = list(set(n.generators.index.values) - set(n0_generators))
        new_loads = list(set(n.loads.index.values) - set(n0_loads))
        new_stores = list(set(n.stores.index.values) - set(n0_stores))
        new_buses = list(set(n.buses.index.values) - set(n0_buses))
        new_components = {'links': new_links,
                          'generators': new_generators,
                          'loads': new_loads,
                          'stores': new_stores,
                          'buses': bus_list}
    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components
def add_central_heat_MT(n, n_flags, inputs_dict, tech_costs):
    '''this function adds expansion capacity for heating technology'''

    GL_eff = inputs_dict['GL_eff']
    GL_inputs = inputs_dict['GL_inputs']
    CO2_cost = inputs_dict['CO2 cost']
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    # take a status of the network before adding componets
    n0_links = n.links.index.values
    n0_generators = n.generators.index.values
    n0_loads = n.loads.index.values
    n0_stores = n.stores.index.values
    n0_buses = n.buses.index.values

    bus_list = ['Straw Pellets', 'ElDK1 bus', 'NG']
    carrier_list = ['Straw Pellets', 'AC',  'NG' ]
    unit_list = ['t/h', 'MW', 'MW']

    if n_flags['central_heat']:
        # add required buses if not in the network
        add_buses = list(set(bus_list) - set(n.buses.index.values))
        idx_add = [bus_list.index(i) for i in add_buses]
        if add_buses:
            n.madd('Bus', add_buses, carrier=[carrier_list[i] for i in idx_add], unit=[unit_list[i] for i in idx_add])

        # ------- add EL connections------------
        local_EL_bus='El_C_heat'
        n = add_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

        # ------- add Heat MT bus ------
        if n_flags['symbiosis_net']:
            if 'Heat MT' not in n.buses.index.values:
                n.add('Bus', 'Heat MT', carrier='Heat', unit='MW')

        # ---------SkyClean---------
        n.add("Generator",
              "Straw Pellets",
              bus="Straw Pellets",
              p_nom_extendable=True,
              marginal_cost=p.Straw_pellets_price * p.currency_multiplier)

        # link converting straw pellets (t(h) to equivalent Digestate pellets (t/h) for Skyclean
        n.add("Link",
              "Straw to Skyclean",
              bus0="Straw Pellets",
              bus1="Digest DM",
              bus2='local_EL_bus',
              efficiency= p.lhv_straw_pellets / p.lhv_dig_pellets,
              efficiency2= -GL_eff.at['El2 bus', 'SkyClean'] * p.lhv_straw_pellets / p.lhv_dig_pellets,
              p_nom_extendable=True) # TRICK: electricity in Skyclean is moslty for pelletization, hence it is balanced (produced for free) in this link when pellets are purchased

        n.add("Link",
              "SkyClean",
              bus0='Digest DM',
              bus1='Heat MT',
              bus2=local_EL_bus,
              efficiency=GL_eff.at['Heat MT', 'SkyClean'],
              efficiency2=GL_eff.at['El2 bus', 'SkyClean'],
              marginal_cost=(tech_costs.at[
                  'biomass HOP', 'VOM'] + GL_eff.at['CO2e bus', 'SkyClean'] * CO2_cost)* p.currency_multiplier, # REWARD FOR NEGATIVE EMISSIONS
              p_nom_extendable=True,
              p_nom_max=p.p_nom_max_skyclean / p.lhv_dig_pellets,
              capital_cost=tech_costs.at['Slow pyrolysis', "fixed"] * p.currency_multiplier) #

        #------ OPTIONAL BIOMASS BOILER (no biochar)-------
        n.add("Link",
              "Pellets boiler",
              bus0='Digest DM',
              bus1="Heat MT",
              bus2=local_EL_bus,
              efficiency=tech_costs.at['biomass HOP', 'efficiency'] * p.lhv_dig_pellets,
              efficiency2=GL_eff.at['El2 bus', 'SkyClean'],
              marginal_cost=(tech_costs.at[
                  'biomass HOP', 'VOM'] ) * p.currency_multiplier,
              p_nom_extendable=True,
              capital_cost=tech_costs.at['biomass HOP', 'fixed'] * p.currency_multiplier)

        # additional NG boiler
        n.add("Link",
              "NG boiler extra",
              bus0="NG",
              bus1="Heat MT",
              efficiency=tech_costs.at['central gas boiler', 'efficiency'],
              p_nom_extendable=True,
              capital_cost=tech_costs.at['central gas boiler', 'fixed'] * p.currency_multiplier,
              marginal_cost=en_market_prices['NG_grid_price'] +
                            tech_costs.at['gas boiler steam', 'VOM'] * p.currency_multiplier)

        # additional El boiler
        n.add('Link',
              'El boiler',
              bus0=local_EL_bus,
              bus1='Heat MT',
              efficiency=tech_costs.at['electric boiler steam', 'efficiency'],
              capital_cost=tech_costs.at['electric boiler steam', 'fixed'] * p.currency_multiplier,
              marginal_cost=tech_costs.at['electric boiler steam', 'VOM'] * p.currency_multiplier,
              p_nom_extendable=True)

        # new componets
        new_links = list(set(n.links.index.values) - set(n0_links))
        new_generators = list(set(n.generators.index.values) - set(n0_generators))
        new_loads = list(set(n.loads.index.values) - set(n0_loads))
        new_stores = list(set(n.stores.index.values) - set(n0_stores))
        new_buses = list(set(n.buses.index.values) - set(n0_buses))
        new_components = {'links': new_links,
                          'generators': new_generators,
                          'loads': new_loads,
                          'stores': new_stores,
                          'buses': bus_list}
    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components

def add_symbiosis(n, n_flags, inputs_dict, tech_costs):
    '''this function builds the simbiosys net with: Buses, Links, Storeges
     The services includes: RE, Heat MT, H2, CO2, connection to DH'''

    GL_inputs = inputs_dict['GL_inputs']

    # take a status of the network before adding componets
    n0_links = n.links.index.values
    n0_generators = n.generators.index.values
    n0_loads = n.loads.index.values
    n0_stores = n.stores.index.values
    n0_buses = n.buses.index.values

    bus_list = ['El2 bus', 'Heat MT', 'Heat LT', 'Heat DH', 'battery', 'Heat DH storage', 'Heat MT storage', 'H2', 'H2_meoh',
                'CO2 sep', 'CO2_meoh','Heat DH']
    carrier_list = ['AC', 'AC', 'Heat', 'Heat', 'Heat', 'battery', 'Heat', 'Heat', 'H2', 'H2', 'H2', 'H2', 'CO2 pure',
                    'CO2 pure','Heat']
    unit_list = ['MW', 'MW', 'MW', 'MW', 'MW', 'MW', 'MW', 'MW', 'MW', 'MW', 'MW', 'MW', 't/h', 't/h', 'MW']

    if n_flags['symbiosis_net']:
        # add required buses if not in the network
        add_buses = list(set(bus_list) - set(n.buses.index.values))
        idx_add = [bus_list.index(i) for i in add_buses]
        if add_buses:
            n.madd('Bus', add_buses, carrier=[carrier_list[i] for i in idx_add], unit=[unit_list[i] for i in idx_add])

        # Link for trading of RE in the park----------------
        if n_flags['renewables']:
            if 'El3 bus' not in n.buses.index.values:
                n.add('Bus', 'El3 bus', carrier='AC', unit='MW')
            n.add("Link",
                  "El3_to_El2",
                  bus0="El3 bus",
                  bus1="El2 bus",
                  efficiency=1,
                  capital_cost=tech_costs.at[
                                   'electricity grid connection', 'fixed'] * p.currency_multiplier,
                  p_nom_extendable=True)

        # Add battery as storage. Note time resolution = 1h, hence battery max C-rate (ch  & dch) is 1
        n.add("Store",
              "battery",
              bus="battery",
              e_cyclic=True,
              e_nom_extendable=True,
              e_nom_max=p.battery_max_cap,
              capital_cost=tech_costs.at["battery storage", 'fixed'] * p.currency_multiplier)  #

        n.add("Link",
              "battery charger",
              bus0="El2 bus",
              bus1="battery",
              efficiency=tech_costs.at["battery inverter", 'efficiency'],
              p_nom_extendable=True,
              capital_cost=tech_costs.at[
                               "battery inverter", 'fixed'] * p.currency_multiplier)  # cost added only on one of the links
        n.add("Link",
              "battery discharger",
              bus0="battery",
              bus1="El2 bus",
              efficiency=tech_costs.at["battery inverter", 'efficiency'],
              p_nom_extendable=True)

        # ------- Trading of  H2 (35 bars)---------------
        n.add("Link",
              "H2_distrib",
              bus0="H2",
              bus1="H2_meoh",
              efficiency=1,
              p_nom_extendable=True,
              capital_cost=tech_costs.at[
                               'H2_pipeline_gas', "fixed"] * p.dist_H2_pipe * p.currency_multiplier)

        # -------- Trading of CO2 (LP)-----
        n.add("Link",
              "CO2_distrib",
              bus0="CO2 sep",
              bus1="CO2_meoh",
              efficiency=1,
              p_nom_extendable=True,
              capital_cost=tech_costs.at['CO2_pipeline_gas', "fixed"] * p.dist_CO2_pipe * p.currency_multiplier)

        # -------- HEAT NETWORKS---------------
        # MT Heat to ambient (additional heat exchanger)
        if 'Heat_MT_to_amb' not in n.links.index.values:
            n.add("Link",
                  "Heat_MT_to_amb",
                  bus0="Heat MT",
                  bus1='Heat amb',
                  efficiency = 1,
                  p_nom_extendable=True,
                  capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * p.currency_multiplier)

        # DH heat to ambient
        if 'Heat_DH_to_amb' not in n.links.index.values:
            n.add("Link",
                  "Heat_DH_to_amb",
                  bus0="Heat DH",
                  bus1='Heat amb',
                  efficiency=1,
                  p_nom_extendable=True,
                  capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * p.currency_multiplier)

        # LT heat to ambient
        if 'Heat_LT_to_amb' not in n.links.index.values:
            n.add("Link",
                  "Heat_LT_to_amb",
                  bus0="Heat LT",
                  bus1='Heat amb',
                  efficiency=1,
                  p_nom_extendable=True,
                  capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * p.currency_multiplier)


        # HEAT INTEGRATION (heat cascade) - HEX
        n.add("Link",
              "Heat_MT_to_DH",
              bus0="Heat MT",
              bus1='Heat DH',
              efficiency=1,
              p_nom_extendable=True,
              capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * p.currency_multiplier)

        n.add("Link",
              "Heat_MT_to_LT",
              bus0="Heat MT",
              bus1='Heat LT',
              efficiency=1,
              p_nom_extendable=True,
              capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * p.currency_multiplier)

        n.add("Link",
              "Heat_DH_to_LT",
              bus0="Heat DH",
              bus1='Heat LT',
              efficiency=1,
              p_nom_extendable=True,
              capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * p.currency_multiplier)

        # TODO introduce piping cost for Mt, DH, LT

        # Thermal energy storage
        # water tank on Heat DH
        n.add('Store',
              'Water tank DH storage',
              bus='Heat DH storage',
              e_nom_extendable=True,
              e_nom_min=0,
              e_nom_max=p.e_nom_max_Heat_DH_storage,
              e_cyclic=True,
              capital_cost=tech_costs.at['central water tank storage', 'fixed'] * p.currency_multiplier)

        n.add("Link",
              "Heat DH storage charger",
              bus0="Heat DH",
              bus1="Heat DH storage",
              p_nom_extendable=True,
              capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * p.currency_multiplier)

        n.add("Link",
              "Heat storage discharger",
              bus0="Heat DH storage",
              bus1="Heat DH",
              p_nom_extendable=True)

        # Concrete Heat storage on HEat MT
        n.add('Store',
              'Concrete Heat MT storage',
              bus='Heat MT storage',
              e_nom_extendable=True,
              e_nom_min=0,
              e_nom_max= p.e_nom_max_Heat_MT_storage,
              e_cyclic=True,
              capital_cost=tech_costs.at['Concrete-store','fixed'] * p.currency_multiplier)

        n.add("Link",
              "Heat MT storage charger",
              bus0="Heat MT",
              bus1="Heat MT storage",
              p_nom_extendable=True,
              capital_cost=tech_costs.at['Concrete-charger','fixed'] * p.currency_multiplier)

        n.add("Link",
              "Heat MT storage discharger",
              bus0="Heat MT storage",
              bus1="Heat MT",
              p_nom_extendable=True,
              capital_cost=tech_costs.at['Concrete-discharger','fixed'] * p.currency_multiplier)


        # new componets
        new_links = list(set(n.links.index.values) - set(n0_links))
        new_generators = list(set(n.generators.index.values) - set(n0_generators))
        new_loads = list(set(n.loads.index.values) - set(n0_loads))
        new_stores = list(set(n.stores.index.values) - set(n0_stores))
        new_buses = list(set(n.buses.index.values) - set(n0_buses))
        new_components = {'links': new_links,
                          'generators': new_generators,
                          'loads': new_loads,
                          'stores': new_stores,
                          'buses': bus_list}
    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components

def add_DH(n, n_flags, inputs_dict, tech_costs):
    '''function that adds DH infrastruture in the park and grid outside'''
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    # take a status of the network before adding componets
    n0_links = n.links.index.values
    n0_generators = n.generators.index.values
    n0_loads = n.loads.index.values
    n0_stores = n.stores.index.values
    n0_buses = n.buses.index.values

    bus_list = ['ElDK1 bus','Heat DH','DH grid','DH GL']
    carrier_list = ['AC', 'Heat', 'Heat','Heat', ]
    unit_list = ['MW', 'MW', 'MW', 'MW']

    # options for DH if selected
    if n_flags['DH']:
        # add required buses if not in the network

        add_buses = list(set(bus_list) - set(n.buses.index.values))
        idx_add = [bus_list.index(i) for i in add_buses]
        if add_buses:
            n.madd('Bus', add_buses, carrier=[carrier_list[i] for i in idx_add], unit=[unit_list[i] for i in idx_add])

        # ------- add EL connections------------
        local_EL_bus='El_DH'
        n = add_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

        # Heat pump for increasing LT heat temperature to DH temperature
        n.add('Link',
              'Heat pump',
              bus0=local_EL_bus,
              bus1='DH GL',
              bus2='Heat LT',
              efficiency=tech_costs.at['industrial heat pump medium temperature', 'efficiency'],
              efficiency2=-(tech_costs.at['industrial heat pump medium temperature', 'efficiency'] - 1),
              capital_cost=tech_costs.at[
                               'industrial heat pump medium temperature', 'fixed'] * p.currency_multiplier,
              marginal_cost=tech_costs.at['industrial heat pump medium temperature', 'VOM'] * p.currency_multiplier,
              p_nom_extendable=True)

        # Link for sale of DH
        n.add('Link',
              'DH GL_to_DH grid',
              bus0='DH GL',
              bus1='DH grid',
              efficiency=1,
              p_nom_extendable=True,
              marginal_cost=-p.DH_price)

        # new componets
        new_links = list(set(n.links.index.values) - set(n0_links))
        new_generators = list(set(n.generators.index.values) - set(n0_generators))
        new_loads = list(set(n.loads.index.values) - set(n0_loads))
        new_stores = list(set(n.stores.index.values) - set(n0_stores))
        new_buses = list(set(n.buses.index.values) - set(n0_buses))
        new_components = {'links': new_links,
                          'generators': new_generators,
                          'loads': new_loads,
                          'stores': new_stores,
                          'buses': bus_list}
    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components

def file_name_network(n, n_flags, inputs_dict):
    '''function that automatically creates a file name give a network'''
    # the netwrok name includes the agents and the indipended variables H2_d, CO2_c, MeOH_d
    # example: Biogas_CHeat_RE_H2_MeOH_SymN_CO2c200_H2d297_MeOHd68
    CO2_cost = inputs_dict['CO2 cost']

    # loads
    if 'H2 grid' in n.loads.index.values:
        H2_d = int(n.loads_t.p_set['H2 grid'].sum() // 1000)  # yearly production of H2 in GWh
    else:
        H2_d = 0

    if 'Methanol' in n.loads.index.values:
        MeOH_d = int(n.loads_t.p_set['Methanol'].sum() // 1000)  # yearly prodcution of MeOH in GWh
    else:
        MeOH_d = 0

    # CO2 tax
    CO2_c = int(CO2_cost)  # CO2 price in currency

    # year
    year= int(p.En_price_year) # energy price year

    # agents
    file_name = n_flags['SkiveBiogas'] * 'SB_' + n_flags['central_heat'] * 'CH_' + n_flags['renewables'] * 'RE_' + \
                n_flags['electrolyzer'] * 'H2_' + n_flags['meoh'] * 'meoh_' + n_flags['symbiosis_net'] * 'SN_' + \
                n_flags['DH'] * 'DH_' + 'CO2c' + str(CO2_c) + '_' + 'H2d' + str(H2_d) + \
                '_' + 'MeOHd' + str(MeOH_d) + '_' + str(year)

    return file_name

def build_PyPSA_network_H2d_bioCH4d_MeOHd_V1(tech_costs, inputs_dict, n_flags):
    '''this function uses bioCH4 demand, H2 demand, and MeOH demand as input to build the PyPSA network'''
    # OUTPUTS: 1) Pypsa network, 2) nested dictionary with componets allocations to the agents (links, generators, loads, stores)

    '''--------------CREATE PYPSA NETWORK------------------'''
    override_component_attrs = override_components_mlinks()
    network = pypsa.Network(override_component_attrs=override_component_attrs)
    network.set_snapshots(p.hours_in_period)

    # Add external grids (no capital costs)
    network, comp_external_grids = add_external_grids(network, inputs_dict, n_flags)

    # Add agents if selected
    network, comp_biogas = add_biogas(network, n_flags, inputs_dict, tech_costs)
    network, comp_renewables = add_renewables(network, n_flags, inputs_dict, tech_costs)
    network, comp_electrolysis = add_electrolysis(network, n_flags, inputs_dict, tech_costs)
    network, comp_meoh = add_meoh(network, n_flags, inputs_dict, tech_costs)
    network, comp_central_H = add_central_heat_MT(network, n_flags, inputs_dict, tech_costs)
    network, comp_symbiosis = add_symbiosis(network, n_flags, inputs_dict, tech_costs)
    network, comp_DH = add_DH(network, n_flags, inputs_dict, tech_costs)

    # add prices and remove loads if required
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    network_comp_allocation = {'external_grids': comp_external_grids,
                               'SkiveBiogas': comp_biogas,
                               'renewables': comp_renewables,
                               'electrolyzer': comp_electrolysis,
                               'meoh': comp_meoh,
                               'central_heat': comp_central_H,
                               'symbiosis_net': comp_symbiosis,
                               'DH': comp_DH}

    # save comp allocation within network
    network.network_comp_allocation = network_comp_allocation
    # -----------Print & Save Network--------------------
    file_name = file_name_network(network, n_flags, inputs_dict)

    if n_flags['print']:
        file_name_topology = p.print_folder_NOpt + file_name + '.svg'
        pypsatopo.NETWORK_NAME = file_name
        pypsatopo.generate(network, file_output=file_name_topology, negative_efficiency=False, carrier_color=True)

    # -----------Export Network-------------------
    if n_flags['export']:
        network.export_to_netcdf(p.print_folder_NOpt + file_name + '.nc')

    return network


# --- OPTIMIZATION-----

def optimal_network_only(n_opt):
    '''function that removes unused: buses, links, stores, generators, storage_units and loads,
     from the plot of the optimal network'''
    n = n_opt

    idx_gen_zero = n.generators.p_nom_opt[n.generators.p_nom_opt == 0].index
    idx_lnk_zero = n.links.p_nom_opt[n.links.p_nom_opt == 0].index
    idx_str_zero = n.stores.e_nom_opt[n.stores.e_nom_opt == 0].index
    idx_stg_zero = n.storage_units.p_nom_opt[n.storage_units.p_nom_opt == 0].index

    for g in idx_gen_zero:
        n.remove('Generator', g)
    for l in idx_lnk_zero:
        n.remove('Link', l)
    for s in idx_str_zero:
        n.remove('Store', s)
    for su in idx_stg_zero:
        n.remove('StorageUnit', su)

    bus_ok = set(n.links.bus0.values) | set(n.links.bus1.values) | set(n.links.bus2.values) | set(
        n.links.bus3.values) | set(n.links.bus4.values) | set(n.generators.bus.values) | set(n.stores.bus.values) | set(
        n.storage_units.bus.values) | set(n.loads.bus.values)
    bus_zero = list(set(n.buses.index.values) - bus_ok)

    if len(bus_zero):
        for b in bus_zero:
            n.remove('Bus', b)
    return n

def OLPF_network_gurobi(n, n_flags_opt, n_flags, inputs_dict):
    """Optimization of OLPF"""
    status = n.lopf(n.snapshots,
                    pyomo=False,
                    solver_name='gurobi',
                    solver_options={'BarHomogeneous': 1})

    if status[0] != "ok":
        print("The solver did not reach a solution (infeasible or unbounded)!")
        sys.exit(-1)  # exit unsuccessfully

    # File name if print or export are True
    file_name = file_name_network(n, n_flags, inputs_dict)

    if n_flags_opt['print']:
        n_plot = optimal_network_only(n)
        file_name_topology = p.print_folder_Opt + file_name + '_OPT' + '.svg'
        pypsatopo.NETWORK_NAME = file_name + '_OPT'
        pypsatopo.generate(n_plot, file_output=file_name_topology, negative_efficiency=False, carrier_color=True)
    if n_flags_opt['export']:
        n.export_to_netcdf(p.print_folder_Opt + file_name + '_OPT' + '.nc')
    return n


# ----RESULTS SINGLE ANALYSIS----

def shadow_prices_violinplot(n,inputs_dict, tech_costs):
    '''function that plats a box plot from marginal prices (shadow prices) from a list of buses'''

    # Example of inputs
    CO2_cost = inputs_dict['CO2 cost']
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)


    H2_d = 0
    meoh_d = 0
    if 'H2 grid' in n.loads.index:
        H2_d = int(n.loads_t.p_set['H2 grid'].sum() // 1000)  # GWH/y
    if 'Methanol' in n.loads.index:
        meoh_d = n.loads_t.p_set['Methanol'].sum()
        bioCH4_y_d = n.loads_t.p_set['bioCH4'].sum()
        CO2_MeOH_plant = 1 / n.links.efficiency['Methanol plant']  # bus0 = CO2, bus1 = Methanol
        bioCH4_CO2plant = n.links.efficiency['SkiveBiogas'] / n.links.efficiency2[
            'SkiveBiogas']  # bus0 = biomass, bus1= bioCH4, bus2=CO2
        fC_MeOH = round((meoh_d * CO2_MeOH_plant) * bioCH4_CO2plant / bioCH4_y_d, 2)

    data = []
    x_ticks_plot = []
    for b in n.buses_t.marginal_price.columns:
        if b == 'ElDK1 bus':
            continue
        if np.sum(n.buses_t.marginal_price[b]) != 0:
            data.append(n.buses_t.marginal_price[b])
            x_ticks_plot.append(b)

    # add El Grid prices
    data.append(inputs_dict['Elspotprices'].squeeze())
    x_ticks_plot.append('Elspotprices')
    data.append(en_market_prices['el_grid_price'].squeeze())
    x_ticks_plot.append('ElDK1 (w/ tarif & CO2tax)')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    ax.violinplot(data, showmeans=False, showmedians=True)
    ax.set_xticks(range(1, len(x_ticks_plot) + 1), x_ticks_plot, rotation=90)
    ax.set_title('shadow prices in €/MWh or €/t (CO2)' + '\n' 'variability during year')
    ax.grid()

    # place a text box in upper left in axes coords
    textstr = '\n'.join((
        r'CO2 tax (€/t)=%.0f' % (CO2_cost),
        r'H2 prod (GWh/y)=%.0f' % (H2_d),
        r'fC MeOH (frac. CO2_bg)=%.0f' % (fC_MeOH)))
    ax.text(0.05, 0.85, textstr, transform=ax.transAxes, fontsize=10,
            va='center')

    plt.tight_layout()
    plt.show()

    folder = p.print_folder_Opt
    fig.savefig(folder + 'shd_prices_violin.png')

    return


def get_capital_cost(n_opt):
    '''function to retrive annualized capital cost for the '''
    # loads do not have capital or marginal costs
    # generatars: marginal + capital cost
    # links: marginal + capital costs
    # stores: marginal (only production) + capital costs
    cc_stores = n_opt.stores.capital_cost * n_opt.stores.e_nom_opt
    cc_generators = n_opt.generators.capital_cost * n_opt.generators.p_nom_opt
    cc_links = n_opt.links.capital_cost * n_opt.links.p_nom_opt

    return cc_stores, cc_generators, cc_links


def get_marginal_cost(n_opt):
    # calculate the marginal cost for every store: note mc is applied only to power generated
    mc_store = []
    # stores with constant marginal costs
    df_marginal_cost_s = n_opt.stores.marginal_cost * n_opt.stores_t.p
    mc_store = df_marginal_cost_s.sum()
    # generators with variable marginal cost
    df_marginal_cost_s2 = n_opt.stores_t.marginal_cost * n_opt.stores_t.p[
        n_opt.stores_t.marginal_cost.columns.values]
    mc_store[
        n_opt.stores_t.marginal_cost.columns.values] = df_marginal_cost_s2.sum()

    mc_gen = []
    # generators with constant marginal costs
    df_marginal_cost_g = n_opt.generators.marginal_cost * n_opt.generators_t.p
    mc_gen = df_marginal_cost_g.sum()
    # generators with variable marginal cost
    df_marginal_cost_g2 = n_opt.generators_t.marginal_cost * n_opt.generators_t.p[
        n_opt.generators_t.marginal_cost.columns.values]
    mc_gen[
        n_opt.generators_t.marginal_cost.columns.values] = df_marginal_cost_g2.sum()

    mc_link = []
    # links with constant marginal cost
    df_marginal_cost_l = n_opt.links.marginal_cost * n_opt.links_t.p0
    mc_link = df_marginal_cost_l.sum()
    # links with variable marginal cost
    df_marginal_cost_l2 = n_opt.links_t.marginal_cost * n_opt.links_t.p0[
        n_opt.links_t.marginal_cost.columns.values]
    mc_link[n_opt.links_t.marginal_cost.columns.values] = df_marginal_cost_l2.sum()

    return mc_store, mc_gen, mc_link


def get_system_cost(n_opt):
    'function that retunr total capital, marginal and system cost'
    # loads do not have capital or marginal costs
    # generatars: marginal + capital cost
    # links: marginal + capital costs
    # stores: marginal (only production) + capital costs

    # total capital cost
    cc_stores, cc_generators, cc_links = get_capital_cost(n_opt)
    tot_cc_stores = cc_stores.sum()
    tot_cc_generators = cc_generators.sum()
    tot_cc_links = cc_links.sum()
    tot_cc = [tot_cc_stores, tot_cc_generators, tot_cc_links]

    # Total marginal cost
    mc_store, mc_gen, mc_link = get_marginal_cost(n_opt)
    tot_mc_stores = mc_store.sum()
    tot_mc_generators = mc_gen.sum()
    tot_mc_links = mc_link.sum()
    tot_mc = [tot_mc_stores, tot_mc_generators, tot_mc_links]

    # total system cost
    tot_sc = np.sum(tot_cc) + np.sum(tot_mc)
    return tot_cc, tot_mc, tot_sc


def get_total_marginal_capital_cost_agents(n_opt,network_comp_allocation, plot_flag):
    ''' function that return 2 dicitonaries with total capital and marginal costs per agent
    it screens all the agents'''
    cc_stores, cc_generators, cc_links = get_capital_cost(n_opt)
    mc_stores, mc_generators, mc_links = get_marginal_cost(n_opt)

    agent_list_cost = []
    cc_tot_agent = {}
    mc_tot_agent = {}


    for key in network_comp_allocation:
        agent_list_cost.append(key)
        agent_links_n_opt = list(set(network_comp_allocation[key]['links']).intersection(set(n_opt.links.index)))
        agent_generators_n_opt = list(
            set(network_comp_allocation[key]['generators']).intersection(set(n_opt.generators.index)))
        agent_stores_n_opt = list(set(network_comp_allocation[key]['stores']).intersection(set(n_opt.stores.index)))

        cc_tot_agent[key] = cc_links[agent_links_n_opt].sum() + cc_generators[agent_generators_n_opt].sum() + cc_stores[
            agent_stores_n_opt].sum()
        mc_tot_agent[key] = mc_links[agent_links_n_opt].sum() + mc_generators[agent_generators_n_opt].sum() + mc_stores[
            agent_stores_n_opt].sum()

    if plot_flag:
        # creates plots vecotrs
        cc_plot = []
        mc_plot = []
        #totc_plot = []
        for a in agent_list_cost:
            cc_plot.append(cc_tot_agent[a])
            mc_plot.append(mc_tot_agent[a])
            #totc_plot.append(mc_tot_agent[a]+ cc_tot_agent[a])

        fig, ax = plt.subplots()

        ax.bar(agent_list_cost, cc_plot)
        ax.bar(agent_list_cost, mc_plot)
        ax.set_xlabel('agents')
        ax.set_ylabel('€/y')
        ax.legend(['fixed costs', 'operational costs'])
        ax.set_xticks(range(0, len(agent_list_cost)), agent_list_cost, rotation=60)
        ax.set_title('Total  system cost by agent')
        plt.tight_layout()
        plt.show()

        folder = p.print_folder_Opt
        fig.savefig(folder + 'cc_cm_agents.png')

    return cc_tot_agent, mc_tot_agent

def plot_duration_curve(ax,df_input,col_val):
    '''plto duration curve from dataframe (df) with index being a DateTimeIndex
     col_val (str) indicate the name of the column with the value that must be plotted
     OUTPUTS:df_1_sorted
      for duration curve plt: x = df_1_sorted['duration'] and y =df_1_sorted[col_val]'''

    df_1=df_input.copy()
    df_1['interval'] = 1 # time resolution of the index
    df_1_sorted = df_1.sort_values(by=[col_val], ascending=False)
    df_1_sorted['duration'] = df_1_sorted['interval'].cumsum()
    out = ax.plot(df_1_sorted['duration'], df_1_sorted[col_val])

    return out

def plot_El_Heat_prices(n_opt, inputs_dict, tech_costs):
    '''function that plots El and Heat prices in external grids and GLS'''

    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)
    el_grid_price_tariff, el_grid_sell_price_tariff = build_electricity_grid_price_w_tariff(inputs_dict['Elspotprices'])

    legend1=['DK1 price + tariff + CO2 tax','DK1 price + tariff','DK1 spotprice','GLS El price', 'GLS El price for H2']
    legend2=['DK NG price + CO2 tax','DK NG price','GLS Heat MT price','GLS Heat DH price']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

    ax1.plot(p.hours_in_period, en_market_prices['el_grid_price'])#, label='DK1 price + tariff + CO2 tax')
    ax1.plot(p.hours_in_period, el_grid_price_tariff)#, label='DK1 price + tariff')
    ax1.plot(p.hours_in_period, inputs_dict['Elspotprices'])#, label='DK1 spotprice')
    ax1.plot(p.hours_in_period, n_opt.buses_t.marginal_price['El2 bus'])#, label='GLS El price')
    ax1.plot(p.hours_in_period, n_opt.buses_t.marginal_price['El3 bus'])#, label='GLS El price')
    ax1.set_ylabel('€/MWh')
    ax1.grid(True)
    ax1.legend(legend1)
    ax1.set_title('El prices time series')

    plot_duration_curve(ax2,pd.DataFrame(en_market_prices['el_grid_price']),'SpotPrice EUR')
    plot_duration_curve(ax2,el_grid_price_tariff,'SpotPrice EUR')
    plot_duration_curve(ax2,inputs_dict['Elspotprices'],'SpotPrice EUR')
    plot_duration_curve(ax2,pd.DataFrame(n_opt.buses_t.marginal_price['El2 bus']),'El2 bus')
    plot_duration_curve(ax2,pd.DataFrame(n_opt.buses_t.marginal_price['El3 bus']),'El3 bus')
    ax2.set_ylabel('€/MWh')
    ax2.set_xlabel('h/y')
    ax2.legend(legend1)
    ax2.grid(True)
    ax2.set_title('El prices duration curve')

    ax3.plot(p.hours_in_period, en_market_prices['NG_grid_price'])#, label='DK NG price + CO2 tax')
    ax3.plot(p.hours_in_period, inputs_dict['NG_price_year'])#, label='DK NG price')
    ax3.plot(p.hours_in_period, n_opt.buses_t.marginal_price['Heat MT'])#, label='GLS Heat MT price')
    ax3.plot(p.hours_in_period, n_opt.buses_t.marginal_price['Heat DH'])#, label='GLS Heat DH price')
    ax3.set_ylabel('€/MWh')
    ax3.grid(True)
    ax3.legend(legend2)
    ax3.set_title('Heat prices time series')


    plot_duration_curve(ax4,pd.DataFrame(en_market_prices['NG_grid_price']),'Neutral gas price EUR/MWh')
    plot_duration_curve(ax4,inputs_dict['NG_price_year'],'Neutral gas price EUR/MWh')
    plot_duration_curve(ax4,pd.DataFrame(n_opt.buses_t.marginal_price['Heat MT']),'Heat MT')
    plot_duration_curve(ax4,pd.DataFrame(n_opt.buses_t.marginal_price['Heat DH']),'Heat DH')
    ax4.set_ylabel('€/MWh')
    ax4.set_xlabel('h/y')
    ax4.legend(legend2)
    ax4.grid(True)
    ax4.set_title('Heat prices durnation curves')

    folder = p.print_folder_Opt
    fig.savefig(folder + 'el_heat_prices.png')

    return

def plot_bus_list_shadow_prices(n_opt, bus_list, legend, start_date, end_date):
    '''function that plots shadow prices for the buses involved in the production of H2 and MeOH'''
    ''' period of time defined by the user '''

    # date format : '2022-01-01T00:00'

    time_ok = pd.date_range(start_date + 'Z', end_date + 'Z', freq='H')

    fig, (ax1, ax2)= plt.subplots(2, 1)
    for b in bus_list:
        df_data= n_opt.buses_t.marginal_price.copy()
        df_plot = df_data.loc[time_ok, :]
        # df_plot = df_data.loc[(df_data.index >= pd.Timestamp(start_date)) & (df_data.index <= pd.Timestamp(end_date))]
        ax1.plot(df_plot[b])
        plot_duration_curve(ax2, df_plot, b)

    ax1.set_ylabel('€/MWh or €/(t/h)')
    ax1.grid(True)
    ax1.legend(legend)
    ax1.set_title('time series')

    ax2.set_ylabel('€/MWh or €/(t/h)')
    ax2.set_xlabel('h/y')
    ax2.grid(True)
    ax2.legend(legend)
    ax2.set_title('duration curves')

    #folder = p.print_folder_Opt
    #fig.savefig(folder + 't_int_shadow_prices.png')

    return

def print_opt_components_table(n_opt,network_comp_allocation, flag_png ):

    "function that creates and saves as png a DF with all the components in the optimal nework, including their capacities and annualized capital costs"

    df_opt_componets= pd.DataFrame()

    agent_list_cost = []

    for key in network_comp_allocation:
        df_agent = pd.DataFrame(data=0, index=[],
                                columns=['Fixed cost (€/y)', 'capacity', 'component', 'reference inlet', 'unit', 'agent'])

        agent_list_cost.append(key)
        agent_generators_n_opt = list(
            set(network_comp_allocation[key]['generators']).intersection(set(n_opt.generators.index)))
        agent_links_n_opt = list(set(network_comp_allocation[key]['links']).intersection(set(n_opt.links.index)))
        agent_stores_n_opt = list(set(network_comp_allocation[key]['stores']).intersection(set(n_opt.stores.index)))

        for g in agent_generators_n_opt:
            df_agent.at[g, 'Fixed cost (€/y)'] = n_opt.generators.p_nom_opt[g] * n_opt.generators.capital_cost[g]
            df_agent.at[g, 'capacity'] = n_opt.generators.p_nom_opt[g]
            df_agent.at[g, 'reference inlet'] = n_opt.generators.bus[g]
            df_agent.at[g, 'unit'] = n_opt.buses.unit[n_opt.generators.bus[g]]
            df_agent.at[g, 'agent'] = key
            df_agent.at[g, 'component'] = 'generator'

        for l in agent_links_n_opt:
            df_agent.at[l, 'Fixed cost (€/y)'] = n_opt.links.capital_cost[l] * n_opt.links.p_nom_opt[l]
            df_agent.at[l, 'capacity'] = n_opt.links.p_nom_opt[l]
            df_agent.at[l, 'reference inlet'] = n_opt.links.bus0[l]
            df_agent.at[l, 'unit'] = n_opt.buses.unit[n_opt.links.bus0[l]]
            df_agent.at[l, 'agent'] = key
            df_agent.at[l, 'component'] = 'link'


        for s in agent_stores_n_opt:
            df_agent.at[s, 'Fixed cost (€/y)'] = n_opt.stores.capital_cost[s] * n_opt.stores.e_nom_opt[s]
            df_agent.at[s, 'capacity'] = n_opt.stores.e_nom_opt[s]
            df_agent.at[s,'reference inlet'] = n_opt.stores.bus[s]
            df_agent.at[s, 'unit'] = n_opt.buses.unit[n_opt.stores.bus[s]]
            df_agent.at[s, 'agent'] = key
            df_agent.at[s, 'component'] = 'store'

        df_opt_componets = pd.concat([df_opt_componets,df_agent])

    "save to png"
    if flag_png:
        folder = p.print_folder_Opt
        dfi.export(df_opt_componets, folder+'components_n_opt.png')

    return df_opt_componets

def plot_heat_map_single_comp(df_time_serie):

    "plot heat map for any time serie based on normalized value for week of the yeat and hour in a week"
    # input example : df_time_serie = pd.DataFrame(network_opt.stores_t.e['H2 HP']) - must be a DF!
    col_name = str(df_time_serie.columns.values.squeeze())
    df_2= df_time_serie.index.isocalendar()
    df_data=pd.concat([df_time_serie, df_2], axis= 1)
    df_data['hour of week'] =  ((df_data['day'] * 24 + 24) - (24 - df_data.index.hour))

    new_df = df_data.copy()
    new_df = new_df.drop('day', axis=1)
    new_df = new_df.drop('year', axis=1)
    new_df = new_df.set_index('week')
    new_df = new_df.pivot_table(index='week', columns="hour of week", values=col_name)

    fig = sns.heatmap(new_df, cmap='YlGn', vmin=0, cbar=True).set(title=col_name)


def heat_map_CF(network_opt, key_comp_dict):
    heat_map_comp_list = {
        'generators': list(set(key_comp_dict['generators']).intersection(set(network_opt.generators.index))),
        'links': list(set(key_comp_dict['links']).intersection(set(network_opt.links.index))),
        'stores': list(set(key_comp_dict['stores']).intersection(set(network_opt.stores.index)))}

    # build DF with all time series for heat map
    df_cf_comp_ts = pd.DataFrame()
    for g in heat_map_comp_list['generators']:
        df_cf_comp_ts[g] = network_opt.generators_t.p[g] / network_opt.generators.p_nom_opt[g]
    for l in heat_map_comp_list['links']:
        df_cf_comp_ts[l] = network_opt.links_t.p0[l] / network_opt.links.p_nom_opt[l]
    for s in heat_map_comp_list['stores']:
        df_cf_comp_ts[s] = network_opt.stores_t.e[s] / network_opt.stores.e_nom_opt[s]

    n_rows = math.ceil(len(df_cf_comp_ts.columns) ** 0.5)  # squared subplot
    position = range(1, len(df_cf_comp_ts.columns) + 1)

    fig = plt.figure()
    fig.suptitle(' Capacity factors - weekly patterns')

    for k in range(len(df_cf_comp_ts.columns)):
        ax = fig.add_subplot(n_rows, n_rows, position[k])
        df_time_serie = pd.DataFrame(df_cf_comp_ts.iloc[:, k])
        plot_heat_map_single_comp(df_time_serie)

    plt.subplots_adjust(hspace=0.7, wspace=0.5)
    fig.savefig(p.print_folder_Opt + 'heat_map_CF.png')

    return

