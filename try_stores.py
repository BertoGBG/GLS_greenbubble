import pypsa
import pandas
from
n= pypsa.Network()

n.add('Bus','el1')
n.add('Store', 'store1', bus='el1', e_nom_max = 10000)

n.add(
    "Link",
    'Dig biomass',
    bus0='Dig biomass market',
    bus1='Dig biomass',
    p_nom_extendable=True,
    p_min_pu=1,
    p_max_pu=1,
    marginal_cost=10,
    efficiency=1,
)

n.add(
    "Store",
    'Dig biomass',
    bus='Dig biomass market',
    e_nom_min=-float("inf"),
    e_nom_max=0,
    e_nom_extendable=True,
    e_min_pu=1.0,
    e_max_pu=0.0,
)