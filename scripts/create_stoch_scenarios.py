# Script that creates scenarios for stochastic optimization
# each scenario represents a different energy year
# energy years 2020, 2021, 2023, 2024
# with different CF for wind and solar, and TS for electricity (buy and sell) and NG
# extra can be set manually CO2 tax (different per scenario)

import scripts.config as c
import scripts.parameters as p
import pandas as pd
from scripts.preprocessing import prepare_all_inputs
from scripts.helpers import en_market_prices_w_CO2
from pathlib import Path

#----------------------------------------------------
#  define scenarios
#----------------------------------------------------
scenarios = {"2020": 0.25, "2021": 0.25, "2023": 0.25, "2024": 0.25}
CO2_cost_s = {"2020": 150, "2021": 150, "2023": 150, "2024": 150}
share_bio_NG_s = {"2020": 0, "2021": 0, "2023": 0, "2024": 0}


def set_input_paths(p, year):
    base_dir =Path(p.folder_model_inputs) / f"Inputs_{year}"

    p.El_price_input_file = base_dir / "Elspotprices_input.csv"
    p.CF_wind_input_file = base_dir / "CF_wind.csv"
    p.CF_solar_input_file = base_dir / "CF_solar.csv"
    p.NG_price_year_input_file = base_dir / "NG_price_year_input.csv"
    # add more if necessary

    return p


def create_inputs_per_scenario(s, n_flags_OK, tech_costs, CO2_cost_s, share_bio_NG_s ):
    # builds inputs per scenario reading the input data from different folders.

    # returns pd series

    # sets CO2 tax and bio_NG_share
    c.CO2_cost = CO2_cost_s[s]
    c.share_bio_NG = share_bio_NG_s[s]

    # change input data folder
    set_input_paths(p, str(s))

    # create inputs dict for each scenario
    inputs_dict = prepare_all_inputs(n_flags_OK=n_flags_OK,
                                     demand_H2=c.demand_H2,
                                     demand_meoh=c.demand_meoh,
                                     demand_CH4=c.demand_CH4,
                                     CO2_cost=c.CO2_cost,
                                     el_DK1_sale_el_RFNBO=c.el_DK1_sale_el_RFNBO,
                                     tech_costs=tech_costs,
                                     preprocess_flag=c.preprocess_flag)

    # --- get energy prices from external markets
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs, c.n_options)
    #en_market_prices = {k: v.reindex(n.snapshots).ffill() for k, v in en_market_prices.items()}

    # for each scenario set the values of the dynamic properties of components along the dimension "scenario"
    # RE CF time series
    # with safety check for nan in input TS (can happen depending on data source)
    CF_wind = inputs_dict["CF_wind"].astype(float).interpolate(method='linear').loc[:,'CF wind']
    CF_solar = inputs_dict["CF_solar"].astype(float).interpolate(method='linear').loc[:,'CF solar']
    el_price = en_market_prices["el_grid_price"].astype(float).interpolate(method='linear')
    el_grid_sell_price = en_market_prices["el_grid_sell_price"].astype(float).interpolate(method='linear')
    NG_price = en_market_prices["NG_grid_price"].astype(float).interpolate(method='linear')


    return CF_wind, CF_solar, el_price, el_grid_sell_price, NG_price


def create_scenarios(n, scenarios, CO2_cost_s, share_bio_NG_s, n_flags_OK, tech_costs):

    # IMPORTANT: ACCESS NETWORK TO INDEDIFY COMPONENTS BEFORE ADDING SCENARIOS!
    # DO NOT ACCESS NETOWRK AFTER CREATION OF SCENARIOS USING THE SAME API FOR A SINGLE NETWORK

    # identify components #TODO change the filtering -> based on carrier and bus0 and bus1
    solar_mask = n.generators.index.str.contains("solar", regex=True)
    solar_gens = n.generators.index[solar_mask]

    wind_mask = n.generators.index.str.contains("wind", regex=True)
    wind_gens = n.generators.index[wind_mask]

    dk1_buy = n.links.index.str.contains(r"DK1_to_", regex=True)
    dk1_buy_links = n.links.index[dk1_buy]

    dk1_sell = n.links.index.str.contains(r"_to_DK1", regex=True)
    dk1_sell_links = n.links.index[dk1_sell]

    dk1_NG = n.links.index.str.contains(r"NG boiler", regex=True)
    dk1_NG_links = n.links.index[dk1_NG]

    co2_liq = n.links.index.str.contains(r"CO2 Liq seq", regex=True)
    co2_liq_links = n.links.index[co2_liq]

    biochar = n.links.index.str.contains(r"biochar sequestration", regex=True)
    biochar_links = n.links.index[biochar]

    # set the scenarios with probability and names
    n.set_scenarios(scenarios)

    for s in n.scenarios:

        # create inputs per scenario
        CF_wind, CF_solar, el_price, el_grid_sell_price, NG_price = create_inputs_per_scenario(s, n_flags_OK, tech_costs, CO2_cost_s,
                                                                           share_bio_NG_s)

        # set input data to scenarios
        # Solar CFs
        for g in solar_gens:
            n.generators_t.p_max_pu.loc[:, (s, g)] = CF_solar.reindex(n.snapshots)

        # Wind CFs
        for g in wind_gens:
            n.generators_t.p_max_pu.loc[:, (s, g)] = CF_wind.reindex(n.snapshots)

        # DK1 electricity buy price
        for lk in dk1_buy_links:
            n.links_t.marginal_cost.loc[:, (s, lk)] = el_price.reindex(n.snapshots)

        # DK1 electricity sell price
        for lk in dk1_sell_links:
            n.links_t.marginal_cost.loc[:, (s, lk)] = el_grid_sell_price.reindex(n.snapshots)

        # DK1 NG price price
        for lk in dk1_NG_links:
            n.links_t.marginal_cost.loc[:, (s, lk)] = NG_price.reindex(n.snapshots)

        # CO2 Liq credits
        co2_credits = -1 * c.n_options.at['CO2 Liq credits', 'enable'] * pd.Series(float(c.CO2_cost),
                                                                                   index=n.snapshots)
        for lk in co2_liq_links:
            n.links_t.marginal_cost.loc[:, (s, lk)] = co2_credits

        # biochar credits
        co2_credits = -1 * c.n_options.at['biochar credits', 'enable'] * pd.Series(float(c.CO2_cost),
                                                                                   index=n.snapshots)
        for lk in biochar_links:
            n.links_t.marginal_cost.loc[:, (s, lk)] = co2_credits

    return

#####



