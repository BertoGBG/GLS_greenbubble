import scripts.config as c
import scripts.parameters as p
import pandas as pd
from scripts.preprocessing import prepare_all_inputs
from scripts.helpers import en_market_prices_w_CO2, add_el_grid_import_RFNBOs
from pathlib import Path
import re
import math

# Example
#scenarios = {"2020": 0.25, "2021": 0.25, "2023": 0.25, "2024": 0.25}
#CO2_cost_s = {"2020": 100, "2021": 100, "2023": 100, "2024": 100}

scenarios = c.stochastic['scenarios']
CO2_cost_s = c.stochastic['CO2_cost_s']
CO2_cost_ref_year_s = c.stochastic['CO2_cost_ref_year_s']

def scenarios_check(scenarios: dict, CO2_cost_s: dict, CO2_cost_ref_year_s: dict, a: str = 'norm'):
    # function that checks if probability of scenarios are summing to 1
    # a == 'norm': normalize probability to sum 1

    prob_tot = list(scenarios.values())
    total_prob = sum(prob_tot)

    if not math.isclose(total_prob, 1.0):
        if a == 'norm':
            new_probs = [p/sum for p in prob_tot]
            checked_scenarios = dict(zip(scenarios.keys(), new_probs))
            print ('WARNING: sum of scenarios probability not 1. Probability were normalized')
        else:
            print('sum of scenarios probability' , sum)
            raise ValueError('WARNING: sum of scenarios probability not 1')

    else:
        checked_scenarios = scenarios

    # check CO2 cost per scenario
    if len(scenarios) != len(CO2_cost_s):
        raise ValueError("Length of scenarios and CO2 cost per scenario not equal")

    # check CO2 cost per scenario
    if len(scenarios) != len(CO2_cost_ref_year_s):
        raise ValueError("Length of scenarios and CO2 cost ref per scenario not equal")

    return checked_scenarios, CO2_cost_s, CO2_cost_ref_year_s


def set_input_paths(p, year, prefix="Inputs"):
    """
    Replace the last path component with f"{prefix}_{year}".
    Works even if folder_data has a trailing slash.
    """
    year = str(year)

    # normalize (remove trailing slash)
    cur = str(p.folder_data).rstrip("/")

    parent = str(Path(cur).parent)        # e.g. "data" or "data/California"
    p.folder_data = str(Path(parent) / f"{prefix}_{year}")

    # update derived file paths
    base_dir = Path(p.folder_data)
    p.El_price_input_file = base_dir / "Elspotprices_input.csv"
    p.CF_wind_input_file = base_dir / "CF_wind.csv"
    p.CF_solar_input_file = base_dir / "CF_solar.csv"
    p.NG_price_year_input_file = base_dir / "NG_price_year_input.csv"
    print('el file path', p.El_price_input_file)
    print('base dir', base_dir)
    return p


def create_inputs_per_scenario(n, s, tech_costs, CO2_cost_s, CO2_cost_ref_year_s):
    CO2_cost = CO2_cost_s[s]
    CO2_cost_ref_year = CO2_cost_ref_year_s[s]
    set_input_paths(p, str(s))

    inputs_dict = prepare_all_inputs(
        targets_dict=c.targets_dict,
        CO2_cost=CO2_cost,
        CO2_cost_ref_year = CO2_cost_ref_year,
        max_RE_to_grid=c.max_RE_to_grid,
        preprocess_flag=c.preprocess_flag,
    )

    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs, c.n_options)
    p_max_pu_rfnbos = add_el_grid_import_RFNBOs(inputs_dict, c.rfnbos_dict)

    CF_wind = inputs_dict["CF_wind"].astype(float).interpolate("linear").loc[:, "CF wind"]
    CF_solar = inputs_dict["CF_solar"].astype(float).interpolate("linear").loc[:, "CF solar"]
    el_price = en_market_prices["el_grid_price"].astype(float).interpolate("linear")
    el_grid_sell_price = en_market_prices["el_grid_sell_price"].astype(float).interpolate("linear")
    NG_price = en_market_prices["NG_grid_price"].astype(float).interpolate("linear")
    p_bioCH4 = en_market_prices["bioCH4_grid_sell_price"].astype(float).interpolate("linear")
    p_max_pu_rfnbos = p_max_pu_rfnbos.reindex(n.snapshots).astype(float)

    return CF_wind, CF_solar, el_price, el_grid_sell_price, NG_price, p_max_pu_rfnbos, p_bioCH4


def create_scenarios(n, scenarios, CO2_cost_s, CO2_cost_ref_year_s, n_flags_OK, tech_costs):
    # Guard: do not allow scenario expansion twice
    if isinstance(n.buses.index, pd.MultiIndex) and "scenario" in n.buses.index.names:
        raise RuntimeError(
            "Network already has scenarios set (buses index has 'scenario' level). "
            "Do not call n.set_scenarios twice."
        )

    checked_scenarios, CO2_cost_s, CO2_cost_ref_year_s = scenarios_check(scenarios=scenarios, CO2_cost_s=CO2_cost_s , CO2_cost_ref_year_s = CO2_cost_ref_year_s, a = 'stop')

    # Identify components BEFORE adding scenarios
    solar_gens = n.generators.index[n.generators.index.str.contains("solar", regex=True)]
    wind_gens  = n.generators.index[n.generators.index.str.contains("wind", regex=True)]

    dk1_buy_links  = n.links.index[n.links.index.str.contains(r"DK1_to_", regex=True)]
    dk1_sell_links = n.links.index[n.links.index.str.contains(r"_to_DK1", regex=True)]
    dk1_NG_links   = n.links.index[n.links.index.str.contains(r"NG boiler", regex=True)]
    co2_liq_links  = n.links.index[n.links.index.str.contains(r"CO2 Liq seq", regex=True)]
    biochar_links  = n.links.index[n.links.index.str.contains(r"biochar sequestration", regex=True)]

    # Identify sales links
    sale_links_by_product = {}

    if "is_product_sale" in n.links.columns:
        mask = n.links["is_product_sale"].fillna(False).astype(bool)
        sale = n.links.loc[mask]

        if not sale.empty:
            # carrier might be missing for some links; drop those
            sale = sale.dropna(subset=["carrier"])
            sale_links_by_product = sale.groupby("carrier").apply(lambda df: list(df.index)).to_dict()

    # find RFNBOS links
    rfnbos_links = list(n.links.index[n.links.carrier.eq("rfnbos_grid_import")])

    # Set scenarios (broadcast component tables)
    n.set_scenarios(scenarios)

    for s in n.scenarios:
        CF_wind, CF_solar, el_price, el_grid_sell_price, NG_price, p_max_pu_rfnbos, p_bioCH4 = \
            create_inputs_per_scenario(n, s, tech_costs, CO2_cost_s, CO2_cost_ref_year_s)

        # Solar / wind CFs
        for g in solar_gens:
            n.generators_t.p_max_pu.loc[:, (s, g)] = CF_solar.reindex(n.snapshots)
        for g in wind_gens:
            n.generators_t.p_max_pu.loc[:, (s, g)] = CF_wind.reindex(n.snapshots)

        # purchasing links - prices
        for lk in dk1_buy_links:
            n.links_t.marginal_cost.loc[:, (s, lk)] = el_price.reindex(n.snapshots)
        for lk in dk1_sell_links:
            n.links_t.marginal_cost.loc[:, (s, lk)] = el_grid_sell_price.reindex(n.snapshots)
        for lk in dk1_NG_links:
            n.links_t.marginal_cost.loc[:, (s, lk)] = NG_price.reindex(n.snapshots)

        # selling links - prices
        for lk in sale_links_by_product.get("bioCH4", []):
            n.links_t.marginal_cost.loc[:, (s, lk)] = p_bioCH4.reindex(n.snapshots)

        # RFNBO constraint
        for lk in rfnbos_links:
            n.links_t.p_max_pu.loc[:, (s, lk)] = p_max_pu_rfnbos.reindex(n.snapshots)

        # Credits
        co2_credits = -1 * c.n_options.at["CO2 Liq credits", "enable"] * pd.Series(float(CO2_cost_s[s]), index=n.snapshots)
        for lk in co2_liq_links:
            n.links_t.marginal_cost.loc[:, (s, lk)] = co2_credits

        biochar_credits = -1 * c.n_options.at["biochar credits", "enable"] * pd.Series(float(CO2_cost_s[s]), index=n.snapshots)
        for lk in biochar_links:
            n.links_t.marginal_cost.loc[:, (s, lk)] = biochar_credits


