import pandas as pd
import parameters as p
from scripts.preprocessing import en_market_prices_w_CO2



# -----CONSTRAINTS on GRID ELECTRICITY RFNBOs---------------
def p_max_pu_EU_renewable_el(Elspotprices, CO2_emiss_El):
    """ function that enables power from the grid tp be used for H2 production according to EU rules:
    1) price below limit, 2) emissionintensity below limit"""

    idx_renw_el_p = Elspotprices[Elspotprices.values <= p.rfnbos_dict['price_threshold']].index
    idx_renw_el_em = CO2_emiss_El[CO2_emiss_El.values <= p.rfnbos_dict['emission_threshold']].index
    p_max_pu_renew_el_price = pd.DataFrame(data=0, index=p.hours_in_period, columns=['p_max_pu el price'])
    p_max_pu_renew_em = pd.DataFrame(data=0, index=p.hours_in_period, columns=['p_max_pu emiss limit'])
    p_max_pu_renew_el_price.loc[idx_renw_el_p, 'p_max_pu el price'] = 1
    p_max_pu_renew_em.loc[idx_renw_el_em, 'p_max_pu emiss limit'] = 1

    return p_max_pu_renew_el_price, p_max_pu_renew_em


def add_link_El_grid_to_H2(n, inputs_dict, tech_costs):
    """ sets condition for use of electricity form the grid - depending on the year_EU and the legislation
    it is limiting the use of electricity form the grid after 2030 withouth installaiton of additional renewables"""

    Elspotprices = inputs_dict['Elspotprices']
    CO2_emiss_El = inputs_dict['CO2_emiss_El']
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    # Grid to H2 availability
    p_max_pu_renew_el_price, p_max_pu_renew_em = p_max_pu_EU_renewable_el(Elspotprices, CO2_emiss_El)

    # Link for use fo electricity from the grid to produce H2
    p_max_pu_grid_to_h2 = p.ref_df.copy()
    if p.rfnbos_dict['limit'] == 'None':
        p_max_pu_grid_to_h2.iloc[:, 0] = 1
    elif p.rfnbos_dict['limit'] == 'emissions':
        p_max_pu_grid_to_h2.iloc[:, 0] = p_max_pu_renew_em.iloc[:, 0]
    elif p.rfnbos_dict['limit'] == 'price':
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
