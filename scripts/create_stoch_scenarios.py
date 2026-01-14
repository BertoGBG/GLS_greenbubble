# Script that creates scenarios for stochastic optimization
# each scenario represents a different energy year
# energy years 2020, 2021, 2023, 2024
# with different CF for wind and solar, and TS for electricity (buy and sell) and NG
# extra can be set manually CO2 tax (different per scenario)

import scripts.config as c
import scripts.parameters as p
import pandas as pd
from scripts.preprocessing import prepare_all_inputs
from scripts.helpers import en_market_prices_w_CO2, add_el_grid_import_RFNBOs
from pathlib import Path

#----------------------------------------------------
#  define scenarios
#----------------------------------------------------
scenarios = {"2020": 0.25, "2021": 0.25, "2023": 0.25, "2024": 0.25}
CO2_cost_s = {"2020": 100, "2021": 100, "2023": 100, "2024": 100}
H2_price_s = {"2020": 120, "2021": 120, "2023": 120, "2024": 120}
MeOH_price_s = {"2020": 120, "2021": 120, "2023": 120, "2024": 120}


def set_input_paths(p, year):
    base_dir =Path(p.folder_model_inputs) / f"Inputs_{year}"

    p.El_price_input_file = base_dir / "Elspotprices_input.csv"
    p.CF_wind_input_file = base_dir / "CF_wind.csv"
    p.CF_solar_input_file = base_dir / "CF_solar.csv"
    p.NG_price_year_input_file = base_dir / "NG_price_year_input.csv"
    # add more if necessary

    return p


def create_inputs_per_scenario(n, s, n_flags_OK, tech_costs, CO2_cost_s):
    # builds inputs per scenario reading the input data from different folders.
    # returns pd series

    # sets CO2 tax
    c.CO2_cost = CO2_cost_s[s]

    # set H2 and Methanol prices
    c.price_H2 = H2_price_s[s],
    c.price_meoh = MeOH_price_s[s]

    # change input data folder
    set_input_paths(p, str(s))

    # create inputs dict for each scenario
    inputs_dict = prepare_all_inputs(n_flags_OK=n_flags_OK,
                                     targets_dict=c.targets_dict,
                                     CO2_cost=c.CO2_cost,
                                     max_RE_to_grid=c.max_RE_to_grid,
                                     preprocess_flag=c.preprocess_flag)

    # --- get energy prices from external markets
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs, c.n_options)
    p_max_pu_rfnbos = add_el_grid_import_RFNBOs(inputs_dict, c.rfnbos_dict)

    # for each scenario set the values of the dynamic properties of components along the dimension "scenario"
    # RE CF time series
    # with safety check for nan in input TS (can happen depending on data source)
    CF_wind = inputs_dict["CF_wind"].astype(float).interpolate(method='linear').loc[:,'CF wind']
    CF_solar = inputs_dict["CF_solar"].astype(float).interpolate(method='linear').loc[:,'CF solar']
    el_price = en_market_prices["el_grid_price"].astype(float).interpolate(method='linear')
    el_grid_sell_price = en_market_prices["el_grid_sell_price"].astype(float).interpolate(method='linear')
    NG_price = en_market_prices["NG_grid_price"].astype(float).interpolate(method='linear')
    p_bioCH4 = en_market_prices['bioCH4_grid_sell_price'].astype(float).interpolate(method='linear')
    p_max_pu_rfnbos = p_max_pu_rfnbos.reindex(n.snapshots).astype(float)

    return CF_wind, CF_solar, el_price, el_grid_sell_price, NG_price, p_max_pu_rfnbos, p_bioCH4


def create_scenarios(n, scenarios, CO2_cost_s, n_flags_OK, tech_costs):

    # IMPORTANT: ACCESS NETWORK TO INDENTIFY COMPONENTS BEFORE ADDING SCENARIOS!
    # DO NOT ACCESS NETWORK AFTER CREATION OF SCENARIOS USING THE SAME API FOR A SINGLE NETWORK

    # identify components #TODO (improvement) change the filtering based on carrier and bus0 and bus1
    solar_mask = n.generators.index.str.contains("solar", regex=True)
    solar_gens = n.generators.index[solar_mask]

    wind_mask = n.generators.index.str.contains("wind", regex=True)
    wind_gens = n.generators.index[wind_mask]

    dk1_buy = n.links.index.str.contains(r"DK1_to_", regex=True)
    dk1_buy_links = n.links.index[dk1_buy]

    rfnbos_link = f"DK1_to_El_H2"

    dk1_sell = n.links.index.str.contains(r"_to_DK1", regex=True)
    dk1_sell_links = n.links.index[dk1_sell]

    dk1_NG = n.links.index.str.contains(r"NG boiler", regex=True)
    dk1_NG_links = n.links.index[dk1_NG]

    bioCH4_sell = n.links.index.str.contains(r"bioCH4 delivery", regex=True)
    bioCH4_sell_links = n.links.index[bioCH4_sell]

    co2_liq = n.links.index.str.contains(r"CO2 Liq seq", regex=True)
    co2_liq_links = n.links.index[co2_liq]

    biochar = n.links.index.str.contains(r"biochar sequestration", regex=True)
    biochar_links = n.links.index[biochar]

    # set the scenarios with probability and names
    n.set_scenarios(scenarios)

    for s in n.scenarios:

        # create inputs per scenario
        CF_wind, CF_solar, el_price, el_grid_sell_price, NG_price, p_max_pu_rfnbos, p_bioCH4 = create_inputs_per_scenario(n, s, n_flags_OK, tech_costs, CO2_cost_s)

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

        # p_ma_pu constraint for RFNBOs
        n.links_t.p_max_pu.loc[:, (s, rfnbos_link)] = p_max_pu_rfnbos.reindex(n.snapshots)

        # DK1 electricity sell price
        for lk in dk1_sell_links:
            n.links_t.marginal_cost.loc[:, (s, lk)] = el_grid_sell_price.reindex(n.snapshots)

        # DK1 NG purchase price
        for lk in dk1_NG_links:
            n.links_t.marginal_cost.loc[:, (s, lk)] = NG_price.reindex(n.snapshots)

        # DK1 bioCH4 sell price
        for lk in bioCH4_sell_links:
            n.links_t.marginal_cost.loc[:, (s, lk)] = p_bioCH4.reindex(n.snapshots)

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



