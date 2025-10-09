import numpy as np
import pypsa

from scripts import parameters as p
from scripts.preprocessing import en_market_prices_w_CO2
from scripts.grid_constraints import add_link_El_grid_to_H2
from scripts.config import (n_options,
                            n_config)

# ------- BUILD PYPSA NETWORK HANDLING FUNCTIONS-------------
def network_dependencies(n_flags):
    """Check if all required dependencies are satisfied when building the network based on n_flags dictionary in main,
    modifies n_flag dict """
    n_flags_OK = n_flags.copy()

    # SkiveBiogas : NO dependencies
    n_flags_OK['biogas'] = n_flags['biogas']

    # renewables : NO Dependencies
    n_flags_OK['renewables'] = n_flags['renewables']

    # H2 production Dependencies
    n_flags_OK['electrolysis'] = n_flags['electrolysis']

    # MeOH production Dependencies
    if n_flags['meoh'] and n_flags['electrolysis'] and n_flags['biogas'] and n_flags[
        'symbiosis']:
        n_flags_OK['meoh'] = True
    else:
        n_flags_OK['meoh'] = False

    # Methanation production Dependencies
    if n_flags['methanation'] and n_flags['electrolysis'] and n_flags['biogas'] and n_flags['symbiosis']:
        n_flags_OK['methanation'] = True
    else:
        n_flags_OK['methanation'] = False

    # Symbiosis net : NO Dependencies (but layout depends on the other n_flags_OK)
    n_flags_OK['symbiosis'] = n_flags['symbiosis']

    # Central heating Dependencies
    if n_flags['central_heat'] and n_flags['symbiosis']:
        n_flags_OK['central_heat'] = True
    else:
        n_flags_OK['central_heat'] = False

    return n_flags_OK


def override_components_mlinks():
    """function required by PyPSA for overwriting link component to multiple connecitons (multilink)
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


def add_requirements_buses(n, bus_dict):
    # function that checks if the required buses for a specific technology are present in the network and adds them if necessary
    # Example of Required buses:
    # bus_dict={'bus_list' : ['El3 bus', 'H2_distribution', 'H2 HP', 'H2 storage', 'Heat amb', 'H2 comp heat' ],
    #          'carrier_list' : ['AC', 'H2', 'H2', 'H2', 'Heat', 'Heat'],
    #          'unit_list' : ['MW', 'MW', 'MW', 'MW', 'MW', 'MW']}

    bus_list = bus_dict['bus_list']
    carrier_list = bus_dict['carrier_list']
    unit_list = bus_dict['unit_list']

    add_buses = list(set(bus_list) - set(n.buses.index.values))
    idx_add = [bus_list.index(i) for i in add_buses]
    if add_buses:
        n.madd('Bus', add_buses, carrier=[carrier_list[i] for i in idx_add], unit=[unit_list[i] for i in idx_add])

    return n


def get_network_status(n):
    # take a status of the network before adding components
    n0_dict = {
        'links': n.links.index.values,
        'generators': n.generators.index.values,
        'loads': n.loads.index.values,
        'stores': n.stores.index.values,
        'buses': n.buses.index.values}

    return n0_dict


def tech_to_add(tech, n0_dict):
    # functions that compared n_config and network status to decide what technolgies should be installed as initial cpacities or expansion capacities
    # Inputs:
    # techs : list  e.g.     tech = ['CO2 compressor', 'Biogas']
    # n0_dict = get_network_status(n)

    cap = [n_config.at[t,'initial capacity'] for t in tech]  # existing initial capacity for each tech
    exp = [n_config.at[t, 'expansion'] for t in tech]   # capacity expansion for each tech

    cap_missing = ['EXI_' + t for t in tech
               if 'EXI_' + t not in {x for k in ('links', 'generators', 'stores') for x in n0_dict.get(k, [])}]
    exp_missing = [t for t in tech
               if t not in {x for k in ('links', 'generators', 'stores') for x in n0_dict.get(k, [])}]
    cap_to_add = [t for t, c, m in zip(tech, cap, cap_missing) if m and (c is not None) and (c > 0)] # Initial capacities to be aded
    exp_to_add = [t for t, c, m in zip(tech, exp, exp_missing) if m and (c is not None) and (c > 0)] # capacity expansion to be added

    return cap_to_add, exp_to_add


def log_new_components(n, n0_dict):
    # take a status of the network after adding a technology and log the new components added
    # log new components
    new_components = {'links': list(set(n.links.index.values) - set(n0_dict['links'])),
                      'generators': list(set(n.generators.index.values) - set(n0_dict['generators'])),
                      'loads': list(set(n.loads.index.values) - set(n0_dict['loads'])),
                      'stores': list(set(n.stores.index.values) - set(n0_dict['stores'])),
                      'buses': list(set(n.buses.index.values) - set(n0_dict['buses']))}
    return new_components


def network_comp_allocation_add_buses_interface(network, network_comp_allocation):
    """function that creates the dict entry for buses for each agent and interface buses for that agent """

    # correct bus list per agent
    for key in network_comp_allocation:
        # find all buses included in aeach agent
        network_comp_allocation[key]['buses'] = []  # reset buses
        bus_list_lk = []
        bus_list_s = []
        bus_list_g = []
        for lk in network_comp_allocation[key]['links']:
            b_lk = [network.links.bus0[lk], network.links.bus1[lk], network.links.bus2[lk],
                    network.links.bus3[lk],
                    network.links.bus4[lk], network.links.bus5[lk],
                    network.links.bus6[lk]]  # list of buses connected to the link
            bus_list_lk.extend(b_lk)

        for s in network_comp_allocation[key]['stores']:
            b_s = [network.stores.bus[s]]
            bus_list_s.extend(b_s)

        for g in network_comp_allocation[key]['generators']:
            b_g = [network.generators.bus[g]]
            bus_list_g.extend(b_g)

        bus_list = list(set(bus_list_lk + bus_list_s + bus_list_g))

        if '' in bus_list:
            bus_list.remove('')

        network_comp_allocation[key]['buses'] = bus_list

    for key in network_comp_allocation:
        # identify interface buses
        network_comp_allocation[key]['interface_buses'] = []  # reset
        other_agents = list(set([key for key in network_comp_allocation]).difference(set([key])))
        other_buses = []
        [other_buses.extend(network_comp_allocation[i]['buses']) for i in other_agents]
        set1 = set(network_comp_allocation[key]['buses'])
        set2 = set(other_buses)
        network_comp_allocation[key]['interface_buses'] = list(set1.intersection(set2))

    return network_comp_allocation


# ------- BUILD PYPSA NETWORK AUXILIARY FUNCTIONS-------------
#  -------COMMON FUNCTIONS
def add_local_heat_connections(n, heat_bus_list, plant_name, n_flags, tech_costs):
    """function that creates local heat buses for each plant.
    heat leaving the plant can be rejected to the ambient for free.
    heat required by the plant can be supplied by symbiosys net ar added heating technologies"""

    new_buses = []

    for i in range(len(heat_bus_list)):
        b = heat_bus_list[i]  # bus in symbiosis net

        # add local bus --> can be used to install local boilers
        bus_name = b + '_' + plant_name
        n.add('Bus', bus_name, carrier='Heat', unit='MW')
        new_buses.append(bus_name)

        # for heat rejection add connection to Heat amb (cooling included in plant cost)
        link_name = b + '_' + plant_name + '_amb'
        n.add('Link',
              link_name,
              bus0=bus_name,
              bus1='Heat amb',
              efficiency=1,
              p_nom_extendable=True)

        # if symbiosis net is available, enable connection with heat grids and add cost (bidirectional)
        if n_flags['symbiosis']:
            if b not in n.buses.index.values:
                n.add('Bus', b, carrier='Heat', unit='MW')

            link_name = b + '_' + plant_name

            n.add('Link', link_name,
                  bus0=b,
                  bus1=bus_name,
                  efficiency=1,
                  p_min_pu=-1,
                  marginal_cost = 0,
                  p_nom_extendable=True,
                  capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * n_config.at['DH heat exchanger','cost factor'])

    return n, new_buses


def add_local_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs):
    """function that adds El connections for a plant
    one connection to the DK1 grid.
    one connection to the El2 bus if symbiosys net is active"""

    # ------ Create Local El bus
    n.add('Bus', local_EL_bus, carrier='AC', unit='MW')

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
              'electricity grid connection', 'fixed'] * n_config.at['grid connection', 'cost factor'],
          p_nom_extendable=True)

    # internal el connection
    if n_flags['symbiosis']:
        if 'El2 bus' not in n.buses.index.values:
            n.add('Bus', 'El2 bus', carrier='AC', unit='MW')

        link_name2 = 'El2_to_' + local_EL_bus
        n.add("Link",
              link_name2,
              bus0="El2 bus",
              bus1=local_EL_bus,  # 'El_biogas',
              efficiency=1,
              p_nom_extendable=True)
    return n


def add_local_boilers(n, local_EL_bus, local_heat_bus, name, heat_efficiency_plant,tech_costs, en_market_prices, capacity, expansion, capital_cost):
    """function that add a local El boiler and NG boiler for a reference plants requiring heating but not connected to the sybiosys net.
    both boilers need connections to local buses"""

    # inputs:
    # - n: pypsa network
    # - local_El_bus : str, El bus for local plant built using add_local_El_bus
    # - local_heat_bus : str, name of the local heat bus added via add_local_heat_connections
    # - name: str, name of the technolgy and plant
    # - heat_efficiency_plant: str, in the link representing of the reference plant, it is the efficiency to the local_heat_bus . e.g. "efficiency3"
    # - en_market_prices: generated using
    # - existing or additional NG boiler
    # - capacity: float , capacity of the reference plant from n_config
    # - expansion: bool, capacity expansion for the reference plant in n_config
    # - capital_cost: float, capital cost for the reference plant in n_config


    capacity_boiler = capacity * np.abs(n.links.at[name, heat_efficiency_plant]) #
    p_nom_max_boiler = n_config.at[name, 'max capacity'] * np.abs(n.links.at[name, 'efficiency3']) #

    n.add("Link",
          name + "_NG boiler",
          bus0="NG",
          bus1=local_heat_bus,
          efficiency=tech_costs.at['gas boiler steam', 'efficiency'],
          p_nom_extendable=expansion,
          p_nom=capacity_boiler / tech_costs.at['gas boiler steam', 'efficiency'] * 1.005,  # lock the capacity to the main plant,
          p_nom_max=p_nom_max_boiler / tech_costs.at['gas boiler steam', 'efficiency'] * 1.005,  # lock the max capacity to the main,
          capital_cost = tech_costs.at['gas boiler steam', 'fixed'] * n_config.at['NG boiler', 'cost factor'] * int(capital_cost > 0),
          marginal_cost=en_market_prices['NG_grid_price'] + tech_costs.at['gas boiler steam', 'VOM'],
          )

    # additional El boiler
    n.add('Link',
           name + '_El boiler',
          bus0=local_EL_bus,
          bus1=local_heat_bus,
          efficiency=tech_costs.at['electric boiler steam', 'efficiency'],
          p_nom_extendable=expansion,
          p_nom=capacity_boiler / tech_costs.at['electric boiler steam', 'efficiency'] * 1.005,# lock the boiler capacity to the main plant,
          p_nom_max=p_nom_max_boiler / tech_costs.at['electric boiler steam', 'efficiency'] * 1.005,# lock the max boiler capacity to the main,
          capital_cost=tech_costs.at['electric boiler steam', 'fixed'] * n_config.at['El boiler', 'cost factor'] * int(capital_cost > 0),
          marginal_cost=tech_costs.at['electric boiler steam', 'VOM'],
          )
    return n


def add_external_grids(network, inputs_dict, n_flags):
    """function building the external grids and loads according to n_flgas dict,
    this function DOES NOT allocate capital or marginal costs to any component"""

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

    # ----------NG source in local distriubtion------
    network.add("Generator",
                "NG grid",
                bus="NG",
                p_nom_extendable=True)

    # --------------District heating-------------------
    if n_options.at['DH','enable']:
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

    # new components
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


def mass_energy_balance_drying(initial_moisture: float = p.moisture_moist_biomass,
                               final_moisture: float = p.moisture_pellets, heat_drying: float = 1,
                               el_drying: float = 0.025):
    """function that calculates the water removed and the head demand from a biomass drying process, given the initial and final moisture
    inputs: - initial moisture (kg_H2O/kg_tot)
            - final moisture (kg_H2O/kg_tot)
            - heat for drying including, heat recovery and losses (MWh/tH2O)
            - el for drying including fans
    outputs:
            - moisture eveaporated : # t H2O remove / t DM
            - heat-input : # MW_th/ tDM
            - electricity-input : # MW_e/ tDM

    """
    water_removed = (initial_moisture / (1 - initial_moisture) - final_moisture / (
            1 - final_moisture))  # t H2O remove / t DM
    heat_input = water_removed * heat_drying  # MW_th/ tDM
    el_input = water_removed * el_drying  # MW_e/ tDM

    drying = {'moisture removed': water_removed,
              'heat-input': heat_input,
              'electricity-input': el_input}

    return drying

# ------- SUPPORT FUNCTIONS

def add_biomass_drying(n, tech_costs, n_flags, final_moisture: float = p.moisture_pellets, initial_moisture : float = p.moisture_moist_biomass):
    """
    Function that adds a biomass belt dryer and auxiliary processes: dewatering (for Digestate fibers) and pelletization (for digestate fibers or wood chips)
    :param n:
    :param final_moisture:
    :param tech_costs:
    :param pelletization: boolean if include or not pelletization
    :param dewatering
    :param initial moisture:
    :return: network with new link
    """

    # Allocation (Player that can build this function) and Dependencies
    allocation = n_flags['central_heat']
    dependencies = n_flags['symbiosis']

    if allocation and dependencies:

        # take a status of the network before adding components
        n0_dict = get_network_status(n)


        def add_biomass_belt_dryer_cap_exp(n, prefix, capital_cost, capacity, expansion):
            # Required buses
            bus_dict = {'bus_list': ['moist biomass', 'Heat MT', 'El2 bus', 'pellets'],
                        'carrier_list': ['moist biomass', 'Heat', 'AC', 'pellets'],
                        'unit_list': ['t/h DM', 'MW', 'MW', 'MW']
                        }

            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            # calculate energy and mass balance of drying process
            heat_drying = tech_costs.at['biomass belt dryer', 'heat-input']  # MW/tH2O removed
            el_drying = tech_costs.at['biomass belt dryer', 'electricity-input']  # MW/tH2O removed

            # return energy data in MW/tDM input (from input data based on  (X/t_H2O) removed)
            dryer_dict = mass_energy_balance_drying(initial_moisture=initial_moisture, final_moisture=final_moisture,
                                                    heat_drying=heat_drying, el_drying=el_drying)

            n.add('Link',
                  prefix + 'biomass belt dryer',
                  bus0='moist biomass',
                  bus1='pellets',
                  bus2='Heat MT',
                  bus3='El2 bus',
                  efficiency=p.lhv_dict['pellets'] / (1 - p.moisture_pellets),  # MWhpellets / tDM
                  efficiency2=- 1 * dryer_dict['heat-input'],  # MWh/tDM
                  efficiency3=- 1 * dryer_dict['electricity-input'] ,  # MWh/tDM
                  p_nom_extendable=expansion,
                  p_nom = capacity,
                  p_nom_max = n_config.at['biomass belt dryer', 'max capacity'],
                  capital_cost=capital_cost,
                  )

            if 'pellets store' not in n.stores.index.values:
                n.add('Store',
                      'pellets store',
                      bus = 'pellets',
                      e_nom_extendable = True,
                      e_nom_max=float('inf'),
                      e_cyclic=True
                      )

        # --------------------------------------------------------------------------
        techs = ['biomass belt dryer']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        t = techs[0]
        if t in cap_to_add:
            capacity = n_config.at['biomass belt dryer', 'initial capacity']
            add_biomass_belt_dryer_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['biomass belt dryer', 'fixed'] *  n_config.at['biomass belt dryer', 'cost factor']
            add_biomass_belt_dryer_cap_exp(n=n, prefix = '', capital_cost=capital_cost, capacity=0, expansion=True)


    return n

def add_CO2_liquefaction(n, n_flags, inputs_dict, tech_costs):
    # Function that adds CO2 liquefaction and storage

    # Allocation (Player that can build this function) and Dependencies
    allocation = n_flags['storage'],
    dependencies = [n_flags['symbiosis'],n_flags['biogas']]

    if allocation and all(dependencies):

        # take the status of the network
        n0_dict = get_network_status(n)

        # input data
        en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

        # ------- add EL connections------------
        local_EL_bus = 'El_CO2_liq'
        if 'El_CO2_liq' not in n.buses.index.values:
            n = add_local_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

        #------- Check techs to add ------------
        techs = ['CO2 Liq', 'CO2 Liq storage']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        # -----------CO2 Storage and liquefaction--------------------
        def add_CO2_Liq_aux(n):
            # Required buses
            bus_dict = {
                'bus_list': ['CO2_distribution', 'CO2 Liq storage',],
                'carrier_list': ['CO2', 'CO2', ],
                'unit_list': ['t/h', 't/h', ]
            }

            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            n.add('Link',
                  'CO2 liq return',
                  bus0='CO2 Liq storage',
                  bus1='CO2_distribution',
                  efficiency=1,
                  p_nom_extendable=True)

            return n

        def add_CO2_Liq_cap_exp(n, prefix, capital_cost, capacity, expansion):
            # Required buses
            bus_dict = {
                'bus_list': ['CO2_distribution',  'CO2 Liq storage'],
                'carrier_list': [ 'CO2', 'CO2'],
                'unit_list': ['t/h', 't/h']
            }

            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            n.add('Link',
                  prefix + 'CO2 Liq',
                  bus0='CO2_distribution',
                  bus1='CO2 Liq storage',
                  bus2=local_EL_bus,
                  efficiency=1,
                  efficiency2=-1 * tech_costs.at['CO2 liquefaction', 'electricity-input'],
                  capital_cost=capital_cost,
                  p_nom_extendable=expansion,
                  p_nom = capacity,
                  p_nom_max = n_config.at['CO2 Liq', 'max capacity'])

            return n

        def add_CO2_Liq_storage_cap_exp(n, prefix, capital_cost, capacity, expansion):
            # Required buses
            bus_dict = {
                'bus_list': ['CO2 Liq storage'],
                'carrier_list': ['CO2'],
                'unit_list': ['t/h']
            }

            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            n.add("Store",
                  prefix + 'CO2 Liq storage',
                  bus="CO2 Liq storage",
                  e_nom_extendable=expansion,
                  e_nom = capacity,
                  e_nom_max=n_config.at['CO2 Liq storage', 'max capacity'],
                  capital_cost=capital_cost,
                  marginal_cost=tech_costs.at['CO2 storage tank', 'VOM'],
                  e_cyclic=True)
            return n

        t = 'CO2 Liq'
        # add auxiliary components for both initial capacity and capacity expansion
        # NOTE, CO2 Liq storage cannot exist without liquefaction system
        if t in cap_to_add + exp_to_add:
            add_CO2_Liq_aux(n)

        # CO2 liquefaction ADD initial capacity and capacity expansion
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_CO2_Liq_cap_exp(n = n, prefix = 'EXI_', capital_cost = 0, capacity = capacity, expansion = False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['CO2 liquefaction', 'fixed'] * n_config.at[t,'cost factor']
            add_CO2_Liq_cap_exp(n = n, prefix = '' , capital_cost = capital_cost, capacity = 0, expansion = True)

        # CO2 Liq storage ADD initial capacity and capacity expansion
        t = 'CO2 Liq storage'
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_CO2_Liq_storage_cap_exp(n=n, prefix = 'EXI_', capital_cost=0, capacity=capacity, expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['CO2 storage tank', 'fixed'] * n_config.at[t,'cost factor']
            add_CO2_Liq_storage_cap_exp(n=n, prefix = '', capital_cost=capital_cost, capacity=0, expansion=True)

    return n


def add_CO2_compressor_HP(n, n_flags, inputs_dict, tech_costs):
    # Function that adds CO2 compression, storage as liquid store and CO2 cylinders
    # CAN BE USED BY DIFFERENT PLAYERS
    # Allocation (Player that can build this function) and Dependencies
    dependencies = [n_flags['biogas'] , n_flags['symbiosis']]

    if all(dependencies):

        # take the status of the network
        n0_dict = get_network_status(n)

        # input data
        en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

        # ------- add EL connections------------
        local_EL_bus = 'El_CO2_compressor'
        if 'El_CO2_compressor' not in n.buses.index.values:
            n = add_local_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

        #------- Check techs to add ------------
        techs = ['CO2 compressor', 'CO2 HP storage']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        # -----------ADD CO2 compressor -----------------------
        def add_CO2_compressor_aux(n):
            bus_dict = {
                'bus_list': ['Heat amb', 'Heat DH', 'Heat LT', 'CO2 comp heat LT', 'CO2 comp heat DH' ],
                'carrier_list': ['Heat', 'Heat', 'Heat', 'Heat', 'Heat'],
                'unit_list': ['MW', 'MW', 'MW', 'MW', 'MW']
            }
            n = add_requirements_buses(n, bus_dict)

            n.add('Link',
                  'CO2 comp heat rejection DH' ,
                  bus0='CO2 comp heat DH',
                  bus1='Heat amb',
                  efficiency=1,
                  p_nom_extendable=True)

            n.add('Link',
                  'CO2 comp heat rejection LT' ,
                  bus0='CO2 comp heat LT',
                  bus1='Heat amb',
                  efficiency=1,
                  p_nom_extendable=True)

            n.add('Link',
                  'CO2 comp heat integration LT' ,
                  bus0='CO2 comp heat LT',
                  bus1='Heat LT',
                  efficiency=1,
                  capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * n_config.at['DH heat exchanger','cost factor'],
                  p_nom_extendable=True)

            n.add('Link',
                  'CO2 comp heat integration DH',
                  bus0='CO2 comp heat DH',
                  bus1='Heat DH',
                  efficiency=1,
                  capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * n_config.at['DH heat exchanger','cost factor'],
                  p_nom_extendable=True)

            if n_flags['symbiosis']:
                if 'Heat LT' not in n.buses.index.values:
                    n.add('Bus', 'Heat LT', carrier='Heat', unit='MW')
            return (n)

        def add_CO2_compressor_cap_exp (n, prefix, capital_cost, capacity, expansion):
            bus_dict = {
                'bus_list': ['CO2 pure HP', 'CO2_distribution', 'CO2 comp heat LT', 'CO2 comp heat DH', ],
                'carrier_list': ['CO2', 'CO2', 'Heat', 'Heat'],
                'unit_list': ['t/h', 't/h', 'MW', 'MW', ]
            }
            n = add_requirements_buses(n, bus_dict)

            n.add("Link",
                  prefix + 'CO2 compressor',
                  bus0="CO2_distribution",
                  bus1="CO2 pure HP",
                  bus2=local_EL_bus,
                  bus3='CO2 comp heat DH',
                  bus4='CO2 comp heat LT',
                  efficiency=1,
                  efficiency2=-1 * tech_costs.at['CO2 industrial compressor', 'electricity-input'],
                  efficiency3=tech_costs.at['CO2 industrial compressor', 'heat output DH'],
                  efficiency4=tech_costs.at['CO2 industrial compressor', 'heat output LT'],
                  p_nom_extendable = expansion, #
                  p_nom = capacity,
                  p_nom_max = n_config.at['CO2 compressor', 'max capacity'],
                  capital_cost=capital_cost)
            return n

        t = "CO2 compressor"
        # add auxiliary components for both initial capacity and capacity expansion
        if  any( t in s for s in cap_to_add + exp_to_add) :
            add_CO2_compressor_aux(n)

        # Add initial capacity and expansion CO2 compressor
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_CO2_compressor_cap_exp(n = n, prefix = 'EXI_', capital_cost = 0, capacity = capacity, expansion = False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['CO2 industrial compressor', "fixed"] * n_config.at['CO2 compressor','cost factor']
            add_CO2_compressor_cap_exp(n = n, prefix = '' , capital_cost = capital_cost, capacity = 0, expansion = True)

        # -----------CO2 HP storage cylinders ---------------
        def add_CO2_storage_HP_aux (n):
            bus_dict = {
                'bus_list': ['CO2 pure HP', 'CO2 storage', 'CO2_distribution'],
                'carrier_list': ['CO2', 'CO2', 'CO2'],
                'unit_list': ['t/h', 't/h', 'CO2']
            }
            n = add_requirements_buses(n, bus_dict)

            n.add('Link',
                  'CO2 storage send',
                  bus0='CO2 pure HP',
                  bus1='CO2 storage',
                  efficiency=1,
                  p_nom_extendable=True)

            n.add('Link',
                  'CO2 extra compression',  #
                  bus0='CO2 storage',
                  bus1='CO2 pure HP',
                  bus2=local_EL_bus,
                  bus3='CO2 comp heat DH',
                  bus4='CO2 comp heat LT',
                  efficiency=1,
                  efficiency2=-1 * tech_costs.at['CO2 industrial compressor', 'extra electricity-input'],
                  efficiency3=tech_costs.at['CO2 industrial compressor', 'extra heat output DH'],
                  efficiency4=tech_costs.at['CO2 industrial compressor', 'extra heat output LT'],
                  p_nom_extendable=True)

            n.add('Link',
                  'CO2 HP return',
                  bus0='CO2 storage',
                  bus1='CO2_distribution',
                  efficiency=1,
                  p_nom_extendable=True)
            return n

        def add_CO2_storage_HP_cap_exp(n, prefix, capital_cost, capacity, expansion):
            bus_dict = {
                'bus_list': ['CO2 storage'],
                'carrier_list': ['CO2'],
                'unit_list': ['t/h']
            }
            n = add_requirements_buses(n, bus_dict)

            n.add("Store",
                  prefix + 'CO2 HP storage',
                  bus="CO2 storage",
                  e_nom_extendable=expansion,
                  e_nom = capacity,
                  capital_cost=capital_cost,
                  e_nom_max= n_config.at['CO2 HP storage','max capacity'],
                  e_cyclic=True)
            return n

        t = "CO2 HP storage"
        # add auxiliary components for both initial capacity and capacity expansion
        if t in cap_to_add + exp_to_add:
            add_CO2_storage_HP_aux(n = n)

        # ADD initial capacity and capacity expansion
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_CO2_storage_HP_cap_exp(n = n, prefix = 'EXI_', capital_cost = 0, capacity = capacity, expansion = False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['CO2 storage cylinders', 'fixed'] * n_config.at[t,'cost factor']
            add_CO2_storage_HP_cap_exp(n = n, prefix = '' , capital_cost = capital_cost, capacity = 0, expansion = True)

    return n


def add_H2_compressor(n, n_flags, inputs_dict, tech_costs):
    # CAN BE USED BY DIFFERENT PLANTS and Dependencies
    dependencies = [n_flags['electrolysis'] , n_flags['symbiosis']]

    if all(dependencies):
        # function that adds H2 store on the H2 HP bus
        # pressure 80 bars
        # input data
        en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

        # take the status of the network
        n0_dict = get_network_status(n)

        # ------- add EL connections------------
        local_EL_bus = 'El_H2_compressor'
        if 'El_H2_compressor' not in n.buses.index.values:
            n = add_local_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

        # ---- Check what to add ------
        techs = ['H2 compressor']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        # -----------adding functions  ----------------------
        def add_H2_comp_aux(n):
            # Required buses
            bus_dict = {'bus_list': ['Heat LT', 'Heat DH', 'Heat amb', 'H2 comp heat DH', 'H2 comp heat LT'],
                        'carrier_list': ['Heat', 'Heat', 'Heat', 'Heat', 'Heat'],
                        'unit_list': ['MW', 'MW', 'MW', 'MW', 'MW']
                        }

            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            n.add('Link',
                  'H2 comp heat rejection DH',
                  bus0='H2 comp heat DH',
                  bus1='Heat amb',
                  efficiency=1,
                  p_nom_extendable=True)

            n.add('Link',
                  'H2 comp heat rejection LT',
                  bus0='H2 comp heat LT',
                  bus1='Heat amb',
                  efficiency=1,
                  p_nom_extendable=True)

            if n_flags['symbiosis']:
                if 'Heat LT' not in n.buses.index.values:
                    n.add('Bus', 'Heat LT', carrier='Heat', unit='MW')

                n.add('Link',
                      'H2 comp heat integration LT',
                      bus0='H2 comp heat LT',
                      bus1='Heat LT',
                      efficiency=1,
                      capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * n_config.at['DH heat exchanger','cost factor'],
                      p_nom_extendable=True)

                n.add('Link',
                      'H2 comp heat integration DH',
                      bus0='H2 comp heat DH',
                      bus1='Heat DH',
                      efficiency=1,
                      capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * n_config.at['DH heat exchanger','cost factor'],
                      p_nom_extendable=True)
            return

        def add_H2_comp_cap_exp(n, prefix, capital_cost, capacity, expansion):
            # Required buses
            bus_dict = {'bus_list': ['H2_distribution', 'H2 HP', 'H2 comp heat DH', 'H2 comp heat LT'],
                        'carrier_list': ['H2', 'H2', 'Heat', 'Heat'],
                        'unit_list': ['MW', 'MW','MW', 'MW']
                        }

            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            n.add("Link",
                  prefix + 'H2 compressor',
                  bus0="H2_distribution",
                  bus1="H2 HP",
                  bus2=local_EL_bus,
                  bus3='H2 comp heat DH',
                  bus4='H2 comp heat LT',
                  efficiency=1,
                  efficiency2=-1 * tech_costs.at['hydrogen storage compressor', 'electricity-input'],
                  efficiency3=tech_costs.at['hydrogen storage compressor', 'heat output DH'],
                  efficiency4=tech_costs.at['hydrogen storage compressor', 'heat output LT'],
                  p_nom_extendable = expansion,
                  p_nom = capacity,
                  p_nom_max = n_config.at['H2 compressor','max capacity'],
                  capital_cost=capital_cost,
                  marginal_cost=tech_costs.at['hydrogen storage compressor', 'VOM'])
            return n

        # -----------H2 compressor  ---------------
        t = 'H2 compressor'
        # Add auxiliary systems
        if t in cap_to_add + exp_to_add:
            add_H2_comp_aux(n)

        # H2 compressor  initial capacity and capacity expansion
        if t in cap_to_add:
            capacity = n_config.at[t, 'initial capacity']
            add_H2_comp_cap_exp(n = n, prefix = 'EXI_', capital_cost = 0, capacity = capacity, expansion = False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['hydrogen storage compressor', 'fixed'] * n_config.at['H2 compressor','cost factor']
            add_H2_comp_cap_exp(n = n, prefix = '' , capital_cost = capital_cost, capacity = 0, expansion = True)

    return n


def add_H2_storage(n, n_flags, inputs_dict, tech_costs):

    # Allocation (Player that can build this function) and Dependencies
    allocation = n_flags['storage'],
    dependencies = [n_flags['symbiosis'],  n_flags['electrolysis']]

    if allocation and all(dependencies):
        # function that adds H2 store on the H2 HP bus
        # pressure 80 bars
        # input data
        en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

        # take the status of the network
        n0_dict = get_network_status(n)

        # ------- add EL connections------------
        local_EL_bus = 'El_H2_compressor'
        if 'El_H2_compressor' not in n.buses.index.values:
            n = add_local_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

        # ---- Check what to add ------
        techs = ['H2 compressor', 'H2 storage']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        # -----------adding functions  ----------------------
        def add_H2_storage_cap_exp(n, prefix, capital_cost, capacity, expansion):
            # H2 compressed local HP Storage
            # Required buses
            bus_dict = {
                'bus_list': ['H2 HP', 'H2 storage', 'H2 comp heat DH', 'H2 comp heat LT'],
                'carrier_list': ['H2', 'H2', 'Heat', 'Heat'],
                'unit_list': ['MW', 'MW', 'MW', 'MW']
                }

            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            n.add('Link',
                  prefix + 'H2 storage send',
                  bus0='H2 HP',
                  bus1='H2 storage',
                  bus2=local_EL_bus,
                  bus3='H2 comp heat DH',
                  bus4='H2 comp heat LT',
                  efficiency=1,
                  efficiency2=-1 * tech_costs.at['hydrogen storage compressor', 'extra electricity-input'],
                  efficiency3=tech_costs.at['hydrogen storage compressor', 'extra heat output DH'] ,
                  efficiency4=tech_costs.at['hydrogen storage compressor', 'extra heat output LT'],
                  p_nom_extendable=True)

            n.add('Link',
                  prefix + 'H2 storage return',
                  bus0='H2 storage',
                  bus1='H2 HP',
                  efficiency=1,
                  p_nom_extendable=True)

            n.add("Store",
                  prefix + 'H2 storage',
                  bus="H2 storage",
                  e_nom_extendable=expansion,
                  e_nom = capacity,
                  capital_cost=capital_cost,
                  marginal_cost=tech_costs.at['hydrogen storage tank type 1', 'VOM'],
                  e_nom_max= n_config.at['H2 storage', 'max capacity'],
                  e_cyclic=True)

        # -----------H2 storage  ---------------
        # H2 storage vessels  initial capacity and capacity expansion
        t = 'H2 storage'
        if t in cap_to_add:
            capacity = n_config.at[t, 'initial capacity']
            add_H2_storage_cap_exp(n = n, prefix = 'EXI_', capital_cost = 0, capacity = capacity, expansion = False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['hydrogen storage tank type 1', 'fixed'] * n_config.at[t,'cost factor']
            add_H2_storage_cap_exp(n = n, prefix = '' , capital_cost = capital_cost, capacity = 0, expansion = True)

    return n


def add_battery(n, n_flags, inputs_dict, tech_costs):

    # Allocation (Player can ca build this function) and Dependencies
    allocation = n_flags['storage'],
    dependencies = [n_flags['symbiosis'] , n_flags['renewables']]

    if allocation and all(dependencies):

        # take a status of the network before adding components
        n0_dict = get_network_status(n)

        # battery is installed with priority on the El bus with renewables
        if n_flags['renewables']:
            el_bus = 'El3 bus'
        else:
            el_bus = 'El2 bus'

        def add_battery_cap_exp(n, prefix: str, capital_cost: list, capacity: list, expansion: list, el_bus=el_bus):
            # inputs are give as a list with order, ['battery', 'inverter']

            bus_dict = {'bus_list': ['battery', el_bus],
                        'carrier_list': ['battery', 'AC'],
                        'unit_list': ['MW', 'MW']}
            n = add_requirements_buses(n, bus_dict)

            # Add battery as storage. Note time resolution = 1h, hence battery max C-rate (ch  & dch) is 1
            n.add("Store",
                  prefix + "battery",
                  bus="battery",
                  e_cyclic=True,
                  e_nom_extendable=expansion[0],
                  e_nom=capacity[0],
                  e_nom_max=n_config.at['battery', 'max capacity'],
                  capital_cost=capital_cost[
                      0])  # tech_costs.at["battery storage", 'fixed'] * n_config.at['battery','cost factor'])  #

            n.add("Link",
                  prefix + "battery charger",
                  bus0=el_bus,
                  bus1="battery",
                  efficiency=tech_costs.at["battery inverter", 'efficiency'],
                  p_nom=capacity[1],
                  p_nom_extendable=expansion[1],
                  p_nom_max=n_config.at['battery inverter', 'max capacity'],
                  capital_cost=capital_cost[1])

            # assumption: charging and discharing power are exaclty the same.
            n.add("Link",
                  prefix + "battery discharger",
                  bus0="battery",
                  bus1=el_bus,
                  efficiency=tech_costs.at["battery inverter", 'efficiency'],
                  p_nom_extendable=True)

        techs = ['battery']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        t = 'battery'
        if t in cap_to_add:
            capacity = [n_config.at['battery', 'initial capacity'], n_config.at['battery inverter', 'initial capacity']]
            add_battery_cap_exp(n=n, prefix='EXI_', capital_cost=[0, 0], capacity=capacity, expansion=[False, False])

        if t in exp_to_add:
            capital_cost = [tech_costs.at['battery storage', 'fixed'] * n_config.at[t, 'cost factor'],
                            tech_costs.at['battery inverter', 'fixed'] * n_config.at['battery inverter', 'cost factor']]
            add_battery_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=[0, 0], expansion=[True, True])
    return


def add_thermal_storage(n, n_flags, inputs_dict, tech_costs):
    # Allocation (Player can ca build this function) and Dependencies
    allocation = n_flags['storage'],
    dependencies = n_flags['symbiosis']

    if allocation and dependencies:
        # take a status of the network before adding components
        n0_dict = get_network_status(n)

        # check what to add
        techs = ['TES DH' , 'TES concrete']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        #  ---- Thermal energy storage in Water tanks
        def add_TES_storage_DH_cap_exp(n, prefix, capital_cost, capacity, expansion):
            bus_dict = {'bus_list': ['Heat DH storage', 'Heat DH'],
                        'carrier_list': ['Heat', 'Heat'],
                        'unit_list': ['MW', 'MW']}
            n = add_requirements_buses(n, bus_dict)

            # water tank on Heat DH
            n.add('Store',
                  prefix + 'Water tank DH storage',
                  bus='Heat DH storage',
                  e_nom_extendable=expansion,
                  e_nom = capacity,
                  e_nom_min=n_config.at['TES DH', 'min capacity'],
                  e_nom_max=n_config.at['TES DH', 'max capacity'],
                  standing_loss=n_config.at['TES DH', 'standing loss'],
                  e_cyclic=True,
                  capital_cost=capital_cost) #

            n.add("Link",
                  prefix + "Heat DH storage charger",
                  bus0="Heat DH",
                  bus1="Heat DH storage",
                  p_nom_extendable=expansion,
                  p_nom = capacity * n_config.at['TES DH','ramp limit up'],
                  p_nom_max = n_config.at['TES DH', 'max capacity'] * n_config.at['TES DH','ramp limit up'],
                  capital_cost= tech_costs.at['DH heat exchanger', "fixed"] * n_config.at['DH heat exchanger','cost factor'] * int(capital_cost > 0)
                  )

            n.add("Link",
                  prefix + "Heat storage discharger",
                  bus0="Heat DH storage",
                  bus1="Heat DH",
                  p_nom_extendable=expansion,
                  p_nom = capacity * n_config.at['TES DH','ramp limit down'],
                  p_nom_max = n_config.at['TES DH', 'max capacity'] * n_config.at['TES DH','ramp limit down'])

        t = 'TES DH'
        if t in cap_to_add:
            capacity = n_config.at['TES DH', 'initial capacity']
            add_TES_storage_DH_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['central water tank storage', 'fixed'] * n_config.at['TES DH', 'cost factor']
            add_TES_storage_DH_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True)

        # ----- Concrete Heat storage on Heat MT
        def add_TES_storage_concrete_cap_exp(n, prefix, capital_cost, capacity, expansion):

            bus_dict = {'bus_list': ['Heat MT storage', 'Heat MT'],
                        'carrier_list': ['Heat', 'Heat'],
                        'unit_list': ['MW', 'MW']}
            n = add_requirements_buses(n, bus_dict)

            n.add('Store',
                  prefix + 'Concrete Heat MT storage',
                  bus='Heat MT storage',
                  e_nom_extendable=expansion,
                  e_nom=capacity,
                  e_nom_min=n_config.at['TES concrete','min capacity'],
                  e_nom_max=n_config.at['TES concrete', 'max capacity'],
                  standing_loss=n_config.at['TES concrete','standing loss'],
                  e_cyclic=True,
                  capital_cost=capital_cost)

            n.add("Link",
                  prefix + "Heat MT storage charger",
                  bus0="Heat MT",
                  bus1="Heat MT storage",
                  p_nom_extendable=expansion,
                  p_nom=capacity * n_config.at['TES concrete', 'ramp limit up'],
                  p_nom_max=n_config.at['TES concrete', 'max capacity'],
                  capital_cost= tech_costs.at['Concrete-charger', 'fixed'] * n_config.at['TES concrete', 'cost factor'] * int(capital_cost > 0)
                  )

            # p/en <= max

            n.add("Link",
                  prefix + "Heat MT storage discharger",
                  bus0="Heat MT storage",
                  bus1="Heat MT",
                  p_nom_extendable=expansion,
                  p_nom=capacity * n_config.at['TES concrete', 'ramp limit down'],
                  p_nom_max=n_config.at['TES DH', 'max capacity'],
                  capital_cost= tech_costs.at['Concrete-discharger', 'fixed'] * n_config.at['TES concrete', 'cost factor'] * int(capital_cost > 0)
                  )

        t = 'TES concrete'
        if t in cap_to_add:
            capacity = n_config.at['TES concrete', 'initial capacity']
            add_TES_storage_concrete_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['Concrete-store', "fixed"] * n_config.at['TES concrete', 'cost factor']
            add_TES_storage_concrete_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True)

    return


def add_heat_pump(n, n_flags, inputs_dict, tech_costs):
    """function that adds heat pump between LT and DH buses """
    allocation = n_flags['symbiosis']
    dependencies = n_flags['symbiosis']
    if allocation and dependencies:

        def add_heat_pump_cap_exp(n, prefix, capital_cost, capacity, expansion):

            # add required buses if not in the network
            bus_dict = {'bus_list': ['Heat DH', 'Heat LT', 'Heat DH'],
                        'carrier_list': ['Heat', 'Heat', 'Heat', ],
                        'unit_list': ['MW', 'MW', 'MW']}
            n = add_requirements_buses(n, bus_dict)

            # ------- add EL connections------------
            local_EL_bus = 'El_heat_pump'
            en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)
            n = add_local_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

            # Heat pump for increasing LT heat temperature to DH temperature
            n.add('Link',
                  prefix + 'heat pump',
                  bus0=local_EL_bus,
                  bus1='Heat DH',
                  bus2='Heat LT',
                  efficiency=tech_costs.at['industrial heat pump medium temperature', 'efficiency'],
                  efficiency2=-(tech_costs.at['industrial heat pump medium temperature', 'efficiency'] - 1),
                  capital_cost=capital_cost,
                  marginal_cost=tech_costs.at['industrial heat pump medium temperature', 'VOM'],
                  p_nom_extendable=expansion,
                  p_nom_max=n_config.at['heat pump', 'max capacity'],
                  p_nom=capacity)

        # take a status of the network before adding components
        n0_dict = get_network_status(n)

        # check what to add
        techs = ['heat pump', 'heat pump']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)
        # Add heat pump
        t = 'heat pump'
        if t in cap_to_add:
            capacity = n_config.at['heat pump', 'initial capacity']
            add_heat_pump_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['industrial heat pump medium temperature', 'fixed'] * n_config.at['heat pump', 'cost factor']
            add_heat_pump_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True)


    return

# ------- BUILD PYPSA NETWORK MAIN FUNCTIONS-------------
def add_demands(n, n_flags, inputs_dict):
    # take a status of the network before adding components
    n0_dict = get_network_status(n)

    # import demands TS
    bioCH4_demand = inputs_dict['bioCH4_demand']
    H2_input_demand = inputs_dict['H2_input_demand']
    Methanol_input_demand = inputs_dict['Methanol_input_demand']

    # ------- CH4 ----------
    if n_flags['biogas'] or n_flags['methanation']:
        bus_dict = {'bus_list': ['bioCH4'],
                    'carrier_list': ['gas'],
                    'unit_list': ['MW']}
        # add required buses if not in the network
        n = add_requirements_buses(n, bus_dict)

        n.add("Load",
              "bioCH4",
              bus="bioCH4",
              p_set=bioCH4_demand.iloc[:, 0])

        # infinite store capacity for CH4 grid (NOTE: production over time is controlled by Dig biomass generator)
        n.add("Store",
              "bioCH4 delivery",
              bus="bioCH4",
              e_nom_extendable=True,
              e_cyclic=True)

    # ------- H2 ------------
    if n_flags['electrolysis']:
        bus_dict = {'bus_list': ['H2', 'H2 delivery'],
                    'carrier_list': ['H2', 'H2'],
                    'unit_list': ['MW', 'MW']}
        # add required buses if not in the network
        n = add_requirements_buses(n, bus_dict)

        # ------------H2 Grid for selling H2 (flexible delivery) -------
        n.add("Load",
              "H2 grid",
              bus="H2 delivery",
              p_set=H2_input_demand.iloc[:, 0])

        # monodirectional link for production H2 for the grid
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

    # ------- Methanol ------
    if n_flags['meoh']: # or n_flahs['eSMR meoh']
        bus_dict = {
            'bus_list': ['Methanol'],
            'carrier_list': ['Methanol'],
            'unit_list': ['MW']}
        # add required buses if not in the network
        n = add_requirements_buses(n, bus_dict)

        n.add('Store',
              'Methanol prod',
              bus='Methanol',
              e_nom_extendable=True,
              e_nom_min=0,
              e_nom_max=float("inf"),
              e_cyclic=True,
              )

        # ----------MeOH flexible delivery storage-------
        n.add("Load",
              "Methanol",
              bus="Methanol",
              p_set=Methanol_input_demand.iloc[:, 0])

    # log new components
    new_components = log_new_components(n, n0_dict)

    return n, new_components

# PLAYERS
def add_biogas(n, n_flags, inputs_dict, tech_costs):
    """function that add the biogas plant to the network and all the dependecies if not preset in the network yet"""

    GL_eff = inputs_dict['GL_eff']
    GL_inputs = inputs_dict['GL_inputs']
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    # take a status of the network before adding components
    n0_dict = get_network_status(n)


    if n_flags['biogas']:

        # ------- add EL connections------------
        local_EL_bus = 'El_biogas'
        n = add_local_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

        # -----add local heat connections
        plant_name = 'biogas'
        heat_bus_list = ["Heat MT", "Heat LT"]
        n, new_heat_buses = add_local_heat_connections(n, heat_bus_list, plant_name, n_flags, tech_costs)

        # ------- adding functions ------------

        def add_biogas_load_aux(n):
            bus_dict = {'bus_list': ['Dig biomass market', 'Dig biomass', 'Digestate', 'biogas', 'bioCH4'],
                        'carrier_list': ['Dig biomass', 'Dig biomass', 'Digestate', 'gas', 'gas'],
                        'unit_list': ['t/h', 't/h DM', 't/h DM', 'MW', 'MW']}
            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            # ------- Digestible biomass -------
            n.add(
                "Link",
                'Dig biomass',
                bus0='Dig biomass market',
                bus1='Dig biomass',
                p_nom_extendable=True,
                p_min_pu = 1,
                p_max_pu = 1,
                marginal_cost=n_options.at['Dig biomass' , 'price'] / GL_eff.loc["bioCH4", "SkiveBiogas"],
                efficiency=1,
                )


            n.add(
                "Store",
                'Dig biomass',
                bus='Dig biomass market',
                e_nom_min= - n_options.at['Dig biomass','max capacity'],
                e_nom_max=0,
                e_nom_extendable=True,
                e_min_pu=1.0,
                e_max_pu=0.0,
                )


            # ---- DM digestate  store
            n.add("Store",
                  "Digestate",
                  bus="Digestate",
                  e_nom_extendable=True,
                  e_nom_max=float("inf"),
                  e_cyclic=False)
            return n

        def add_biogas_exp_cap(n, prefix, capital_cost, capacity, expansion):
            bus_dict = {'bus_list': ['Dig biomass', 'Digestate', 'biogas'],
                        'carrier_list': ['Dig biomass', 'Digestate', 'gas', 'gas'],
                        'unit_list': ['t/h', 't/h DM', 'MW', 'MW', 'MW']}
            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            name = prefix + 'biogas'
            n.add("Link",
                  name = name,
                  bus0="Dig biomass",
                  bus1="biogas",
                  bus2=new_heat_buses[1],  # "Heat LT",
                  bus3=local_EL_bus,  # 'El_biogas',
                  bus4='Digestate',
                  efficiency=GL_eff.loc["bioCH4", "SkiveBiogas"],
                  efficiency2=GL_eff.loc["Heat LT", "SkiveBiogas"],
                  efficiency3=GL_eff.loc["El2 bus", "SkiveBiogas"] * 0.5,
                  efficiency4=GL_eff.loc["DM digestate", "SkiveBiogas"],
                  p_nom_extendable = expansion,
                  p_nom = capacity ,
                  p_nom_max = n_config.at['biogas', 'max capacity'],
                  capital_cost = capital_cost )
            return

        def add_biogas_storage_exp_cap(n, prefix, capital_cost, capacity, expansion):
            bus_dict = {'bus_list': ['biogas'],
                        'carrier_list': ['gas', 'gas'],
                        'unit_list': ['MW']}
            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            n.add('Store',
                  name = prefix + 'biogas store',
                  bus='biogas',
                  e_nom_extendable=expansion,
                  e_nom = capacity,
                  capital_cost=capital_cost,
                  e_nom_max=n_config.at['biogas storage','max capacity'],
                  e_cyclic=True)
            return

        def add_biogas_upgrading_aux (n):

            bus_dict = {'bus_list': ['CO2 sep', 'CO2 pure atm'],
                        'carrier_list': ['CO2', 'CO2'],
                        'unit_list': ['t/h', 't/h']}

            n = add_requirements_buses(n, bus_dict)

            # -----------infinite Store of biogenic CO2 (venting to ATM)
            n.add("Store",
                  "CO2 biogenic out",
                  bus="CO2 pure atm",
                  e_nom_extendable=True,
                  e_nom_min=0,
                  e_nom_max=float("inf"),
                  e_cyclic=False,
                  marginal_cost=0,
                  )

            n.add("Link",
                  "CO2 sep to atm",
                  bus0="CO2 sep",
                  bus1="CO2 pure atm",
                  efficiency=1,
                  p_nom_extendable=True)

            return

        def add_biogas_upgrading_exp_cap(n, prefix, capital_cost, capacity, expansion):
            bus_dict = {'bus_list': ['NG', 'CO2 sep', 'biogas', 'bioCH4'],
                        'carrier_list': ['gas', 'CO2', 'gas', 'gas'],
                        'unit_list': ['MW', 't/h', 'MW', 'MW']}
            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            n.add("Link",
                  name =  prefix + 'biogas upgrading',
                  bus0="biogas",
                  bus1="bioCH4",
                  bus2="CO2 sep",
                  bus3=new_heat_buses[0],  # "Heat MT",
                  bus4=local_EL_bus,
                  efficiency=1,
                  efficiency2=GL_eff.loc["CO2 pure", "SkiveBiogas"] / GL_eff.loc["bioCH4", "SkiveBiogas"],
                  efficiency3=GL_eff.loc["Heat MT", "SkiveBiogas"] / GL_eff.loc["bioCH4", "SkiveBiogas"],
                  efficiency4=GL_eff.loc["El2 bus", "SkiveBiogas"] * 0.5 / GL_eff.loc["bioCH4", "SkiveBiogas"],
                  p_nom_extendable=expansion,
                  p_nom = capacity,
                  p_nom_max = n_config.at['biogas upgrading', 'max capacity'],
                  capital_cost= capital_cost,
                  marginal_cost=tech_costs.at['biogas upgrading', 'VOM'])

            # existing or additional NG boiler
            capacity_boiler = np.abs(capacity * n.links.at[prefix + 'biogas upgrading', 'efficiency3'] / tech_costs.at['gas boiler steam', 'efficiency']) * 1.01 # lock the capacity to the biogas upgrading,
            p_nom_max_boiler = n_config.at['biogas upgrading', 'max capacity'] * np.abs(n.links.at[prefix + 'biogas upgrading', 'efficiency3']) / \
                        tech_costs.at['gas boiler steam', 'efficiency'] * 1.01  # lock the max capacity to the biogas upgrading

            n.add("Link",
                  name = prefix + 'biogas upgrading' + '_' + "NG boiler",
                  bus0="NG",
                  bus1=new_heat_buses[0],
                  efficiency=tech_costs.at['gas boiler steam', 'efficiency'],
                  p_nom_extendable=expansion,
                  p_nom = capacity_boiler,
                  p_nom_max = p_nom_max_boiler,
                  capital_cost= tech_costs.at['gas boiler steam', 'fixed'] * n_config.at['NG boiler','cost factor'] * int(capital_cost > 0),
                  marginal_cost=en_market_prices['NG_grid_price'] + tech_costs.at['gas boiler steam', 'VOM'],
            )

            # enables NG boiler to supply heat to the symbiosis network
            if n_flags['symbiosis'] and capacity:
                n.links.p_min_pu.at[prefix + 'biogas upgrading' + '_' + "NG boiler"] = -1

        def add_dewatering_cap_exp(n, prefix, capital_cost, capacity, expansion):

            # Required buses
            bus_dict = {'bus_list': ['Digestate', 'moist biomass', 'El2 bus'],
                        'carrier_list': ['Digestate', 'moist biomass', 'AC'],
                        'unit_list': ['t/h DM', 't/h DM', 'MW']
                        }

            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            n.add('Link',
                  name =  prefix + 'dewatering',
                  bus0='Digestate',
                  bus1='moist biomass',
                  bus2=local_EL_bus,
                  efficiency=tech_costs.at['centrifugal dewatering', 'DM separation'],  # tDM out /in
                  efficiency2=-1 * tech_costs.at['centrifugal dewatering', 'electricity-input'],  # MWel / tDM in
                  p_nom_extendable=expansion,
                  p_nom = capacity,
                  p_nom_max = n_config.at['dewatering', 'max capacity'],
                  capital_cost=capital_cost)

        # ------- Check techs to add ------------
        techs = ['biogas','biogas storage', 'biogas upgrading', 'dewatering']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        # add biogas plant
        t = 'biogas'
        if t in cap_to_add + exp_to_add:
            add_biogas_load_aux(n)

        if t in cap_to_add:
            capacity = n_config.at[t, 'initial capacity'] / GL_eff.loc["bioCH4", "SkiveBiogas"]
            add_biogas_exp_cap(n= n, prefix = 'EXI_', capital_cost = 0, capacity = capacity, expansion= False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['biogas', 'fixed'] / GL_eff.loc["bioCH4", "SkiveBiogas"] * n_config.at[t,'cost factor']
            add_biogas_exp_cap(n=n, prefix = '', capital_cost=capital_cost, capacity = 0, expansion=True)

        # Add biogas storage
        t = 'biogas storage'
        if t in cap_to_add:
            capacity = n_config.at[t, 'initial capacity']
            add_biogas_storage_exp_cap(n= n, prefix = 'EXI_', capital_cost = 0, capacity= capacity, expansion= False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['biogas storage', 'fixed'] * n_config.at[t,'cost factor']
            add_biogas_storage_exp_cap(n=n, prefix = '', capital_cost=capital_cost, capacity=0, expansion=True)

        # Biogas upgrading
        t = 'biogas upgrading'
        if t in cap_to_add + exp_to_add:
            add_biogas_upgrading_aux(n)

        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_biogas_upgrading_exp_cap(n= n, prefix = 'EXI_', capital_cost = 0, capacity= capacity, expansion= False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['biogas upgrading', 'fixed'] * n_config.at[t,'cost factor']
            add_biogas_upgrading_exp_cap(n=n, prefix = '', capital_cost=capital_cost, capacity=0, expansion=True)

        # dewatering of digestate fibers
        t = 'dewatering'
        if t in cap_to_add:
            capacity = n_config.at['dewatering', 'initial capacity']
            add_dewatering_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['centrifugal dewatering', "fixed"] * n_config.at['dewatering', 'cost factor']
            add_dewatering_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True)

        # log new components
        new_components = log_new_components(n, n0_dict)

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

    # take a status of the network before adding components
    n0_dict = get_network_status(n)


    if n_flags['renewables']:

        def add_grid_connection_cap_exp(n, name, capital_cost, capacity, expansion):
            # add link to sell power to the external El grid
            bus_dict = {'bus_list': ['El3 bus', 'ElDK1 bus'],
                        'carrier_list': ['AC', 'AC'],
                        'unit_list': ['MW',  'MW']}
            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            n.add("Link",
                  name, # "El3_to_DK1",
                  bus0="El3 bus",
                  bus1="ElDK1 bus",
                  efficiency=1,
                  marginal_cost=en_market_prices['el_grid_sell_price'],
                  p_nom_extendable=expansion,
                  p_nom = capacity,
                  p_nom_max = n_config.at['grid connection', 'max capacity'],
                  capital_cost= capital_cost)

        def add_onwind_cap_exp(n, prefix, capital_cost, capacity, expansion):
            bus_dict = {'bus_list': ['El3 bus', 'ElDK1 bus'],
                        'carrier_list': ['AC',  'AC'],
                        'unit_list': ['MW', 'MW']}
            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            # Add onshore wind generators
            n.add("Generator",
                  prefix + "onshorewind",
                  bus="El3 bus",
                  p_nom_max=n_config.at['onwind','max capacity'],
                  p_nom_extendable=expansion,
                  p_nom = capacity,
                  capital_cost=capital_cost,
                  marginal_cost=tech_costs.at['onwind', 'VOM'],
                  p_max_pu=CF_wind['CF wind'])

        def add_salar_cap_exp(n, prefix, capital_cost, capacity, expansion):
            bus_dict = {'bus_list': ['El3 bus', 'ElDK1 bus'],
                        'carrier_list': ['AC', 'AC'],
                        'unit_list': ['MW', 'MW']}
            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            # add PV utility generators
            n.add("Generator",
                  prefix + "solar",
                  bus="El3 bus",
                  p_nom_max=n_config.at['solar','max capacity'],
                  p_nom_extendable=expansion,
                  p_nom = capacity,
                  capital_cost=capital_cost,
                  marginal_cost=tech_costs.at['solar', 'VOM'],
                  p_max_pu=CF_solar['CF solar'])

        techs = ['onwind', 'solar', 'grid connection']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        # Add initial capacity and expansion onshore wind
        t = 'onwind'
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_onwind_cap_exp(n= n, prefix = 'EXI_', capital_cost = 0, capacity= capacity, expansion= False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['onwind', 'fixed'] * n_config.at[t,'cost factor']
            add_onwind_cap_exp(n=n, prefix = '', capital_cost=capital_cost, capacity=0, expansion=True)

        # Add initial capacity and expansion solar
        t = 'solar'
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_salar_cap_exp(n= n, prefix = 'EXI_', capital_cost = 0, capacity= capacity, expansion= False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['solar', 'fixed'] * n_config.at[t,'cost factor']
            add_salar_cap_exp(n=n, prefix = '', capital_cost=capital_cost, capacity=0, expansion=True)

        # Add initial capacity and expansion grid connection
        t = 'grid connection'
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_grid_connection_cap_exp(n= n, name = 'EXI_'+ 'El3_to_DK1', capital_cost = 0, capacity= capacity, expansion= False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['electricity grid connection', 'fixed'] * n_config.at[t,'cost factor']
            add_grid_connection_cap_exp(n=n, name = 'El3_to_DK1', capital_cost=capital_cost, capacity=0, expansion=True)


        # log new components
        new_components = log_new_components(n, n0_dict)

    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components


def add_electrolysis(n, n_flags, inputs_dict, tech_costs):
    GL_eff = inputs_dict['GL_eff']
    H2_input_demand = inputs_dict['H2_input_demand']

    # take a status of the network before adding components
    n0_dict = get_network_status(n)

    if n_flags['electrolysis']:


        # ---------- Add RFNBOs constraint for use of electricity form the grid without additional RE----
        n = add_link_El_grid_to_H2(n, inputs_dict, tech_costs)

        # -----add local heat connections
        plant_name = 'electrolysis'
        heat_bus_list = ['Heat MT', "Heat DH", "Heat LT"]
        n, new_heat_buses = add_local_heat_connections(n, heat_bus_list, plant_name, n_flags, tech_costs)

        # -----------Electrolyzer------------------
        # cost_electrolysis dependent on scale (grid ot MeOH only)
        if H2_input_demand.iloc[:, 0].sum() > 0:
            electrolysis_cost = tech_costs.at['electrolysis', 'fixed'] * n_config.at['electrolysis','cost factor']
        else:
            electrolysis_cost = tech_costs.at['electrolysis small', 'fixed'] * n_config.at['electrolysis','cost factor']

        def add_H2_cap_exp(n, prefix , capital_cost , capacity , expansion ):
            bus_dict = {'bus_list': ['El3 bus', 'H2'],
                        'carrier_list': ['AC', 'H2'],
                        'unit_list': ['MW', 'MW']}
            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            n.add("Link",
                  prefix + 'electrolysis', # "electrolysis",
                  bus0="El3 bus",
                  bus1="H2",
                  bus2=new_heat_buses[2],
                  efficiency=GL_eff.at['H2', 'GreenHyScale'],
                  efficiency2=GL_eff.at['Heat LT', 'GreenHyScale'],
                  capital_cost=capital_cost, # electrolysis_cost,
                  marginal_cost=0,
                  p_nom_extendable=expansion,
                  p_nom = capacity,
                  p_nom_max= n_config.at['electrolysis', 'max capacity'],
                  p_min_pu=n_config.at['electrolysis', 'min load'],
                  ramp_limit_up=n_config.at['electrolysis', 'ramp limit up'],
                  ramp_limit_down=n_config.at['electrolysis', 'ramp limit down'],)

        # ------- Check techs to add ------------
        techs = ['electrolysis']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        # add auxiliary components for both initial capacity and capacity expansion
        t = 'electrolysis'
        # Add initial capacity and expansion CO2 compressor
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_H2_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False)

        if t in exp_to_add:
            capital_cost = electrolysis_cost * n_config.at['CO2 compressor','cost factor']
            add_H2_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True)

        # log new components
        new_components = log_new_components(n, n0_dict)

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

    # take a status of the network before adding components
    n0_dict = get_network_status(n)

    # ------- ADD METHANOL DEMAND
    if n_flags['meoh']:

        def add_meoh_load_aux(n):
            # ------- add EL connections------------
            local_EL_bus = 'El_meoh'
            n = add_local_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

            # add local heat connections
            heat_bus_list = ['Heat MT', "Heat DH"]
            n, new_heat_buses = add_local_heat_connections(n, heat_bus_list, plant_name='meoh', n_flags=n_flags, tech_costs=tech_costs)

            return n, new_heat_buses, local_EL_bus

        def add_meoh_cap_exp(n, prefix , capital_cost , capacity , expansion, new_heat_buses, local_EL_bus):
            # required buses
            bus_dict = {
                'bus_list': ['El3 bus', 'H2 HP', 'CO2 pure HP', 'Methanol'],
                'carrier_list': ['AC', 'H2', 'CO2 pure', 'Methanol', 'Heat'],
                'unit_list': [ 'MW', 'MW', 't/h', 'MW', 'MW']}
            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            n.add("Link",
                  prefix + 'meoh',
                  bus0="CO2 pure HP",
                  bus1="Methanol",
                  bus2="H2 HP",
                  bus3= local_EL_bus,
                  bus4=new_heat_buses[0],
                  bus5=new_heat_buses[1],
                  efficiency=GL_eff.loc["Methanol", "Methanol plant"],
                  efficiency2=GL_eff.loc["H2", "Methanol plant"],
                  efficiency3=GL_eff.loc["El2 bus", "Methanol plant"],
                  efficiency4=GL_eff.at['Heat MT', 'Methanol plant'],
                  efficiency5=GL_eff.at['Heat DH', 'Methanol plant'],
                  p_nom_extendable=expansion,
                  p_nom = capacity,
                  p_nom_max = n_config.at['meoh', 'max capacity'],
                  capital_cost=capital_cost,
                  p_min_pu=n_config.at['meoh', 'min load'],
                  ramp_limit_up=n_config.at['meoh', 'ramp limit up'],
                  ramp_limit_down=n_config.at['meoh', 'ramp limit down'],)

            if not n_flags['central_heat']:
                # add local NG and El boiler to produce MT heat
                add_local_boilers(n = n, local_EL_bus = 'El3 bus', local_heat_bus = new_heat_buses[0], name = prefix + 'meoh', heat_efficiency_plant = 'efficiency4', tech_costs = tech_costs,
                                  en_market_prices = en_market_prices, capacity =  capacity, expansion = expansion, capital_cost =capital_cost)

        # ------- Check techs to add ------------
        techs = ['meoh']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        # add auxiliary components for both initial capacity and capacity expansion
        t = 'meoh'
        if any(t in s for s in cap_to_add + exp_to_add):
            n, new_heat_buses, local_EL_bus = add_meoh_load_aux(n)

        # Add initial capacity and expansion CO2 compressor
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_meoh_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False, new_heat_buses= new_heat_buses, local_EL_bus = local_EL_bus)

        if t in exp_to_add:
            capital_cost = tech_costs.at['methanolisation', "fixed"] * n_config.at['CO2 compressor','cost factor']
            add_meoh_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True, new_heat_buses= new_heat_buses, local_EL_bus= local_EL_bus)

        # ------ADD CO2 compressor ( and storage Liquid and Cylinders) and add H2 compressor and Storage (Steel Vessel)------
        n = add_CO2_compressor_HP(n, n_flags, inputs_dict, tech_costs)
        n = add_H2_compressor(n, n_flags, inputs_dict, tech_costs)

        # log new components
        new_components = log_new_components(n, n0_dict)

    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components


def add_methanation(n, n_flags, inputs_dict, tech_costs):
    ''' function installing required methanation facilities: biomethanation and cathalitic methanation
    H2 system can be supplied with own electolyzer but does not have a CO2 source
    To enable CO2 trade is NEEDED the symbiosis net and the source (Biogas)'''

    # take a status of the network before adding components
    n0_dict = get_network_status(n)
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)


    if n_flags['methanation']:

        # ------- add EL connections------------
        local_EL_bus = 'methanation'
        n = add_local_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

        # add local heat connections
        heat_bus_list = ['Heat MT']
        n, new_heat_buses = add_local_heat_connections(n, heat_bus_list, plant_name='methanation', n_flags=n_flags,
                                                       tech_costs=tech_costs)

        # ----------BIO-METHANATION PLANT (biogas + H2)---------
        def add_biomethanation_biogas_cap_exp(n, prefix, capital_cost, capacity, expansion):
            # Required buses
            bus_dict = {
                'bus_list': ['H2_distribution','biogas', 'bioCH4'],
                'carrier_list': ['H2', 'gas', 'gas'],
                'unit_list': ['MW', 'MW', 'MW']}

            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            n.add("Link",
                  prefix + 'biomethanation biogas',
                  bus0="H2_distribution",
                  bus1="bioCH4",
                  bus2="biogas",
                  bus3=local_EL_bus,
                  efficiency=tech_costs.at['biomethanation', "Methane Output"],
                  efficiency2 = - tech_costs.at['biomethanation', "Biogas Input"],
                  efficiency3 = - tech_costs.at['biomethanation', "electricity input"],
                  p_nom=capacity,
                  p_nom_extendable = expansion,
                  p_nom_max = n_config.at['biomethanation biogas','max capacity'],
                  p_min_pu=n_config.at['biomethanation biogas', 'min load'],
                  capital_cost = capital_cost,
                  ramp_limit_up=n_config.at['biomethanation biogas', 'ramp limit up'],
                  ramp_limit_down=n_config.at['biomethanation biogas', 'ramp limit down'],
                  marginal_cost=tech_costs.at['biomethanation', "VOM"]) #

        # ----------BIO-METHANATION PLANT (CO2 + H2)---------

        def add_biomethanation_CO2_cap_exp(n, prefix, capital_cost, capacity, expansion):
            # Required buses
            bus_dict = {
                'bus_list': ['H2_distribution','CO2_distribution', 'bioCH4'],
                'carrier_list': ['H2', 'CO2', 'gas'],
                'unit_list': ['MW', 'MW', 'MW']}

            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            # capital cost refer to biomaethantion_biogas and are scaled by the volumetric flow of input gas to the reactor
            v_ch4_v_co2 = (tech_costs.at['biomethanation', "Biogas Input"] / p.lhv_dict['CH4'] / p.density_CH4_1atm) / \
                          tech_costs.at['biomethanation', "CO2 Input"] / p.density_CO2_1atm
            v_h2 = 1 / p.lhv_dict['H2'] * 1e3 / p.density_H2_1atm  # m3/h/MW_h2
            v_co2 = tech_costs.at['biomethanation', "CO2 Input"] / p.density_CO2_1atm * 1e3  # m3/h/MW_h2
            v_ch4 = v_co2 * v_ch4_v_co2  # m3/h/MW_h2
            input_vol_flow_onlyco2_biogas_biometh = (v_h2 + v_co2) / (
                    v_h2 + v_co2 + v_ch4)  # ratio of vol flow for CO2 only vs biogas (for 1MW H2)

            n.add("Link",
                  prefix + 'biomethanation CO2',
                  bus0="H2_distribution",
                  bus1="bioCH4",
                  bus2="CO2_distribution",
                  bus3= local_EL_bus,
                  efficiency=tech_costs.at['biomethanation', "Methane Output"] - tech_costs.at[
                      'biomethanation', "Biogas Input"],  # only generated Methane not input biogas
                  efficiency2=- tech_costs.at['biomethanation', "CO2 Input"],
                  efficiency3=- tech_costs.at['biomethanation', "electricity input"],
                  p_nom_extendable=expansion,
                  p_nom = capacity,
                  p_nom_max = n_config.at['biomethanation CO2', 'max capacity' ],
                  p_min_pu=n_config.at['biomethanation CO2', 'min load'],
                  ramp_limit_up=n_config.at['biomethanation CO2', 'ramp limit up'],
                  ramp_limit_down=n_config.at['biomethanation CO2', 'ramp limit down'],
                  capital_cost=capital_cost * input_vol_flow_onlyco2_biogas_biometh,
                  marginal_cost=tech_costs.at['biomethanation', "VOM"] * input_vol_flow_onlyco2_biogas_biometh)
            return

        # --------- CATALYTIC METHANATION PLANT (biogas + H2)

        def add_cat_methanation_biogas_cap_exp(n, prefix, capital_cost, capacity, expansion):
            # Required buses
            bus_dict = {
                'bus_list': ['H2_distribution','biogas', 'bioCH4'],
                'carrier_list': ['H2', 'gas', 'gas'],
                'unit_list': ['MW', 'MW', 'MW']}

            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            n.add("Link",
                  prefix + 'cat methanation biogas',
                  bus0="H2_distribution",
                  bus1="bioCH4",
                  bus2="biogas",
                  bus3= local_EL_bus,
                  bus4=new_heat_buses[0],
                  efficiency=tech_costs.at['biogas plus hydrogen', "Methane Output"],
                  efficiency2= -1 * tech_costs.at['biogas plus hydrogen', "Biogas Input"],
                  efficiency3= -1 * tech_costs.at['biogas plus hydrogen', "electricity input"],
                  efficiency4= tech_costs.at['biogas plus hydrogen', "heat output"],
                  p_nom= capacity,
                  p_nom_extendable=expansion,
                  p_nom_max=n_config.at['cat methanation biogas', 'max capacity'],
                  p_min_pu=n_config.at['cat methanation biogas', 'min load'],
                  ramp_limit_up=n_config.at['cat methanation biogas', 'ramp limit up'],
                  ramp_limit_down=n_config.at['cat methanation biogas', 'ramp limit down'],
                  capital_cost= capital_cost ,# cost per MWh_H2 input
                  marginal_cost=tech_costs.at['biogas plus hydrogen', "VOM"] )# - en_market_prices['NG_grid_price'] * tech_costs.at['biogas plus hydrogen', "Methane Output"] )

        # --------- CATALYTIC METHANATION PLANT (CO2 + H2)
        def add_cat_methanation_CO2_cap_exp(n, prefix, capital_cost, capacity, expansion):

            # Required buses
            bus_dict = {
                'bus_list': ['H2_distribution', 'CO2_distribution', 'bioCH4'],
                'carrier_list': ['H2', 'CO2', 'gas'],
                'unit_list': ['MW', 'MW', 'MW']}

            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict)

            # capital costs are scaled by the volumetric flow of input gas  to the reactor
            v_ch4_v_co2 = (tech_costs.at['biogas plus hydrogen', "Biogas Input"] / p.lhv_dict[
                'CH4'] / p.density_CH4_1atm) / \
                          tech_costs.at['biogas plus hydrogen', "CO2 Input"] / p.density_CO2_1atm
            v_h2 = 1 / p.lhv_dict['H2'] * 1e3 / p.density_H2_1atm  # m3/h/MW_h2
            v_co2 = tech_costs.at['biogas plus hydrogen', "CO2 Input"] / p.density_CO2_1atm * 1e3  # m3/h/MW_h2
            v_ch4 = v_co2 * v_ch4_v_co2  # m3/h/MW_h2
            input_vol_flow_onlyco2_biogas_cat_meth = (v_h2 + v_co2) / (
                    v_h2 + v_co2 + v_ch4)  # ratio of vol flow for CO2 only vs biogas (for 1MW H2)

            n.add("Link",
                  prefix + 'cat methanation CO2',
                  bus0="H2_distribution",
                  bus1="bioCH4",
                  bus2="CO2_distribution",
                  bus3= local_EL_bus,
                  bus4=new_heat_buses[0],
                  efficiency = tech_costs.at['biogas plus hydrogen', "Methane Output"] - tech_costs.at[
                      'biogas plus hydrogen', "Biogas Input"],
                  efficiency2= -1 * tech_costs.at['biogas plus hydrogen', "CO2 Input"],  # tCO2/MWh_H2
                  efficiency3 = -1 * tech_costs.at['biogas plus hydrogen', "electricity input"],
                  efficiency4 = tech_costs.at['biogas plus hydrogen', "heat output"],
                  p_nom_extendable=expansion,
                  p_nom = capacity,
                  p_min_pu=n_config.at['cat methanation CO2', 'min load'],
                  ramp_limit_up=n_config.at['cat methanation CO2', 'ramp limit up'],
                  ramp_limit_down=n_config.at['cat methanation CO2', 'ramp limit down'],
                  capital_cost = capital_cost * input_vol_flow_onlyco2_biogas_cat_meth,#
                  marginal_cost=tech_costs.at['biogas plus hydrogen', "VOM"] * input_vol_flow_onlyco2_biogas_cat_meth) # - en_market_prices['NG_grid_price'] * tech_costs.at['biogas plus hydrogen', "Methane Output"] - tech_costs.at[ 'biogas plus hydrogen', "Biogas Input"] )

        # ------- Check techs to add ------------
        techs = ['biomethanation biogas', 'biomethanation CO2', 'cat methanation biogas' ,'cat methanation CO2']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        # biological methanation
        t = 'biomethanation biogas'
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_biomethanation_biogas_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['biomethanation', "fixed"] * n_config.at[t,'cost factor']
            add_biomethanation_biogas_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True)

        t = 'biomethanation CO2'
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_biomethanation_CO2_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['biomethanation', "fixed"] * n_config.at[t,'cost factor']
            add_biomethanation_CO2_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True)

        # Catalytic methanation
        t = 'cat methanation biogas'
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_cat_methanation_biogas_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['biogas plus hydrogen', "fixed"] * n_config.at[t,'cost factor']
            add_cat_methanation_biogas_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True)

        t = 'cat methanation CO2'
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_cat_methanation_CO2_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['biogas plus hydrogen', "fixed"] * n_config.at[t,'cost factor']
            add_cat_methanation_CO2_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True)

        # ----------- ADD CO2 compression ----------
        n = add_CO2_compressor_HP(n, n_flags, inputs_dict, tech_costs)

        # log new components
        new_components = log_new_components(n, n0_dict)

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

    # take a status of the network before adding components
    n0_dict = get_network_status(n)

    bus_dict = {'bus_list': ['pellets', 'ElDK1 bus', 'NG'],
                'carrier_list': ['pellets', 'AC', 'NG'],
                'unit_list': ['MW', 'MW', 'MW']}

    if n_flags['central_heat']:
        # add required buses if not in the network
        n = add_requirements_buses(n, bus_dict)

        # ------- add EL connections------------
        local_EL_bus = 'El_C_heat'
        n = add_local_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

        # ------- add Heat MT bus ------
        if n_flags['symbiosis']:
            if 'Heat MT' not in n.buses.index.values:
                n.add('Bus', 'Heat MT', carrier='Heat', unit='MW')

        # ---- add generator for straw (market)
        if n_options.at['pellets market','enable']:
            bus_dict = {'bus_list': ['pellets', 'pellets market'],
                        'carrier_list': ['pellets', 'pellets'],
                        'unit_list': ['MW', 'MW']}
            n = add_requirements_buses(n, bus_dict)

            n.add(
                "Link",
                'pellets market',
                bus0='pellets market',
                bus1='pellets',
                p_nom_extendable=True,
                efficiency = 1,
                marginal_cost=n_options.at['pellets market','price']
                )

            n.add(
                "Store",
                'pellets market',
                bus='pellets market',
                e_nom_min= -n_options.at['pellets market','max capacity'],
                e_nom_max=0,
                e_nom_extendable=True,
                e_min_pu=1.0,
                e_max_pu=0.0,
                e_cyclic=False,
                )

        # ---- add generator for chip biomass (market)
        if n_options.at['moist biomass market','enable']:
            bus_dict = {'bus_list': ['moist biomass', 'moist biomass market'],
                        'carrier_list': ['moist biomass', ' moist biomass'],
                        'unit_list': ['MW', 'MW']}
            n = add_requirements_buses(n, bus_dict)

            n.add(
                "Link",
                'moist biomass market',
                bus0='moist biomass market',
                bus1='moist biomass',
                p_nom_extendable=True,
                efficiency = 1,
                marginal_cost=n_options.at['moist biomass market','price']
            )

            n.add(
                "Store",
                'moist biomass market',
                bus='moist biomass market',
                e_nom_min= - n_options.at['moist biomass market','max capacity'],
                e_nom_max=0,
                e_nom_extendable=True,
                e_min_pu=1.0,
                e_max_pu=0.0,
                e_cyclic = False,
            )

        # -----add biomass drying

        add_biomass_drying(n, tech_costs, n_flags)

        # ---------add Biochar Pyrolysis---------
        def add_pyrolysis_aux(n):
            bus_dict = {'bus_list': ['biochar', 'biochar sequestration'],
                        'carrier_list': ['CO2','CO2'],
                        'unit_list': ['t/h', 't/h']}
            n = add_requirements_buses(n, bus_dict)

            n.add('Link',
                  'biochar sequestration',
                  bus0= 'biochar',
                  bus1 = 'biochar sequestration',
                  efficiency = 1,
                  marginal_cost =  - n_options.at['biochar credits', 'enable'] * CO2_cost,
                  )

            n.add('Store',
                  'biochar sequestred',
                  bus="biochar sequestration",
                  e_nom_extendable=True,
                  e_nom_min=0,
                  e_nom_max=float("inf"),
                  e_cyclic=False)

            return


        def add_pyrolysis_cap_exp(n, prefix, capital_cost, capacity, expansion):
            bus_dict = {'bus_list': ['pellets', 'biochar'],
                        'carrier_list': ['pellets','CO2'],
                        'unit_list': ['MW','t/h']}
            n = add_requirements_buses(n, bus_dict)


            n.add("Link",
                  prefix + "pyrolysis",
                  bus0='pellets',
                  bus1='Heat MT',
                  bus2=local_EL_bus,
                  bus3='biochar',  # as CO2 stored
                  efficiency = tech_costs.at['biochar pyrolysis', 'heat output'] / tech_costs.at[
                      'biochar pyrolysis', 'biomass input'],
                  efficiency2 = -1 * tech_costs.at['biochar pyrolysis', 'electricity input'] / tech_costs.at[
                      'biochar pyrolysis', 'biomass input'],
                  efficiency3 =  1 / tech_costs.at['biochar pyrolysis', 'biomass input'],
                  marginal_cost=tech_costs.at[
                                    'biomass HOP', 'VOM'] / tech_costs.at[
                                    'biochar pyrolysis', 'biomass input'],
                  p_nom_extendable=expansion,
                  p_nom = capacity,
                  p_nom_max=n_config.at['pyrolysis', 'max capacity'],
                  capital_cost= capital_cost)
            return

        techs = ['pyrolysis']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        # add aux
        add_pyrolysis_aux(n)

        t = 'pyrolysis'
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_pyrolysis_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['biochar pyrolysis', "fixed"] / tech_costs.at['biochar pyrolysis', 'biomass input'] * n_config.at[t, 'cost factor']
            add_pyrolysis_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True)

        # ------ BIOMASS BOILER (pellets)-------
        def add_C_biomass_boiler_cap_exp(n, prefix, capital_cost, capacity, expansion):
            bus_dict = {'bus_list': ['pellets', 'Heat MT'],
                        'carrier_list': ['pellets','Heat'],
                        'unit_list': ['MW','MW']}
            n = add_requirements_buses(n, bus_dict)

            n.add("Link",
                  prefix + 'pellets boiler',
                  bus0='pellets',
                  bus1="Heat MT",
                  efficiency=tech_costs.at['biomass HOP', 'efficiency'] * p.lhv_dict['pellets'],
                  marginal_cost=(tech_costs.at['biomass HOP', 'VOM']),
                  p_nom_extendable=expansion,
                  p_nom = capacity,
                  p_nom_max = n_config.at['biomass boiler', 'max capacity'],
                  capital_cost=capital_cost )
            return

        techs = ['biomass boiler']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        t = 'biomass boiler'
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_C_biomass_boiler_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['biomass HOP', 'fixed'] * n_config.at[t, 'cost factor']
            add_C_biomass_boiler_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True)

        # ------ NG BOILER -------
        def add_C_NG_boiler_cap_exp(n, prefix, capital_cost, capacity, expansion):
            # TODO add dependecies
            bus_dict = {'bus_list': ['NG', 'Heat MT'],
                        'carrier_list': ['gas','Heat'],
                        'unit_list': ['MW','MW']}
            n = add_requirements_buses(n, bus_dict)

            n.add("Link",
                  prefix + "NG boiler",
                  bus0="NG",
                  bus1="Heat MT",
                  efficiency=tech_costs.at['central gas boiler', 'efficiency'],
                  p_nom_extendable=expansion,
                  p_nom = capacity,
                  p_nom_max = n_config.at['NG boiler', 'max capacity'],
                  capital_cost=capital_cost ,
                  marginal_cost=en_market_prices['NG_grid_price'] +
                                tech_costs.at['gas boiler steam', 'VOM'])

        techs = ['NG boiler']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        t = 'NG boiler'
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_C_NG_boiler_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['central gas boiler', 'fixed'] * n_config.at[t, 'cost factor']
            add_C_NG_boiler_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True)


        # ------- El boiler -------
        def add_C_El_boiler_cap_exp(n, prefix, capital_cost, capacity, expansion):
            bus_dict = {'bus_list': [local_EL_bus, 'Heat MT'],
                        'carrier_list': ['gas', 'Heat'],
                        'unit_list': ['MW', 'MW']}
            n = add_requirements_buses(n, bus_dict)

            n.add('Link',
                  prefix + 'El boiler',
                  bus0=local_EL_bus,
                  bus1='Heat MT',
                  efficiency=tech_costs.at['electric boiler steam', 'efficiency'],
                  marginal_cost=tech_costs.at['electric boiler steam', 'VOM'],
                  p_nom_extendable=expansion,
                  p_nom = capacity,
                  p_nom_max = n_config.at['El boiler', 'max capacity'],
                  capital_cost=capital_cost )

        techs = ['El boiler']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        t = 'El boiler'
        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            add_C_El_boiler_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['electric boiler steam', 'fixed'] * n_config.at[t, 'cost factor']
            add_C_El_boiler_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True)


        # log new components
        new_components = log_new_components(n, n0_dict)

    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components


def add_storage(n, n_flags, inputs_dict, tech_costs):

    # take a status of the network before adding components
    n0_dict = get_network_status(n)

    if n_flags['storage']:
        add_battery(n, n_flags, inputs_dict, tech_costs)
        add_thermal_storage(n, n_flags, inputs_dict, tech_costs)
        add_CO2_liquefaction(n, n_flags, inputs_dict, tech_costs)
        add_H2_storage(n, n_flags, inputs_dict, tech_costs)

        # log new components
        new_components = log_new_components(n, n0_dict)

    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components


def add_symbiosis(n, n_flags, inputs_dict, tech_costs):
    '''this function builds the symbiosis net with: Buses, Links, Storages
     The services includes: RE, Heat MT, H2, CO2, connection to DH'''

    GL_inputs = inputs_dict['GL_inputs']

    # take a status of the network before adding components
    n0_dict = get_network_status(n)


    if n_flags['symbiosis']:
        # add required buses if not in the network

        # Link for trading of RE in the park----------------
        if n_flags['renewables']:
            bus_dict = {'bus_list': ['El3 bus', 'El2 bus'],
                        'carrier_list': ['AC', 'AC', ],
                        'unit_list': ['MW', 'MW']}
            n = add_requirements_buses(n, bus_dict)

            n.add("Link",
                  "El3_to_El2",
                  bus0="El3 bus",
                  bus1="El2 bus",
                  efficiency=1,
                  capital_cost = tech_costs.at['electricity grid connection', 'fixed'] * n_config.at['grid connection', 'cost factor'] * n_options.at['symbiosis El transformer','enable'],
                  p_nom_extendable=True)

        # Link for sale of DH
        if n_options.at['DH', 'enable']:
            bus_dict = {'bus_list': ['DH grid', 'Heat DH'],
                        'carrier_list': ['Heat', 'Heat', ],
                        'unit_list': ['MW', 'MW']}
            n = add_requirements_buses(n, bus_dict)

            n.add('Link',
                    'DH GL_to_DH grid',
                    bus0='Heat DH',
                    bus1='DH grid',
                    efficiency=1,
                    p_nom_extendable=True,
                    marginal_cost= - n_options.at['DH', 'price'])

        # ------- Trading of  H2 (35 bars)---------------
        if n_flags['electrolysis']:
            bus_dict = {'bus_list': ['H2', 'H2_distribution'],
                        'carrier_list': ['H2', 'H2', ],
                        'unit_list': ['MW', 'MW']}
            n = add_requirements_buses(n, bus_dict)
            n.add("Link",
                  "H2 pipe",
                  bus0="H2",
                  bus1="H2_distribution",
                  efficiency=1,
                  p_nom_extendable=True,
                  capital_cost=tech_costs.at[
                                   'H2 pipe', "fixed"] * tech_costs.at['H2 pipe', 'distance'] * n_config.at['H2 pipe', 'cost factor'])

        # -------- Trading of CO2 (LP)-----
        if n_flags['biogas'] and (n_config.at['biogas upgrading','expansion'] or n_config.at['biogas upgrading','initial capacity']) :
            bus_dict = {'bus_list': ['CO2 sep', 'CO2_distribution'],
                        'carrier_list': ['CO2 pure', 'CO2 pure', ],
                        'unit_list': ['MW', 'MW']}
            n = add_requirements_buses(n, bus_dict)
            n.add("Link",
                  "CO2_pipe",
                  bus0="CO2 sep",
                  bus1="CO2_distribution",
                  efficiency=1,
                  p_nom_extendable=True,
                  capital_cost=tech_costs.at['CO2 gas pipe', "fixed"] *  tech_costs.at['CO2 gas pipe', 'distance'] * n_config.at['CO2 pipe', 'cost factor'])

        # -------- HEAT NETWORKS---------------
        # MT Heat to ambient (additional heat exchanger)
        n.add("Link",
              "Heat_MT_to_amb",
              bus0="Heat MT",
              bus1='Heat amb',
              efficiency=1,
              p_nom_extendable=True,
              capital_cost= 0) # It is assumed that each plant has the capacity to fully reject heat (only shared heat is transferred).

        # DH heat to ambient
        n.add("Link",
              "Heat_DH_to_amb",
              bus0="Heat DH",
              bus1='Heat amb',
              efficiency=1,
              p_nom_extendable=True,
              capital_cost=0) # It is assumed that each plant has the capacity to fully reject heat (only shared heat is transferred).

        # LT heat to ambient
        n.add("Link",
              "Heat_LT_to_amb",
              bus0="Heat LT",
              bus1='Heat amb',
              efficiency=1,
              p_nom_extendable=True,
              capital_cost=0) # It is assumed that each plant has the capacity to fully reject heat (only shared heat is transferred).

        # HEAT INTEGRATION (heat cascade) - HEX
        n.add("Link",
              "Heat_MT_to_DH",
              bus0="Heat MT",
              bus1='Heat DH',
              efficiency=1,
              p_nom_extendable=True,
              capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * n_config.at['DH heat exchanger', 'cost factor'])

        n.add("Link",
              "Heat_MT_to_LT",
              bus0="Heat MT",
              bus1='Heat LT',
              efficiency=1,
              p_nom_extendable=True,
              capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * n_config.at['DH heat exchanger', 'cost factor'])

        n.add("Link",
              "Heat_DH_to_LT",
              bus0="Heat DH",
              bus1='Heat LT',
              efficiency=1,
              p_nom_extendable=True,
              capital_cost=tech_costs.at['DH heat exchanger', "fixed"] *  n_config.at['DH heat exchanger', 'cost factor'])

        # add heat pump
        add_heat_pump(n, n_flags, inputs_dict, tech_costs)
        # TODO add mixing as an option to reduce heat pump capacity

        # log new components
        new_components = log_new_components(n, n0_dict)

    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components

# BUILD THE NETWORK
def build_network(tech_costs, inputs_dict, n_flags):
    """this function uses bioCH4 demand, H2 demand, and MeOH demand as input to build the PyPSA network"""
    # OUTPUTS: 1) Pypsa network, 2) nested dictionary with componets allocations to the agents

    '''--------------CREATE PYPSA NETWORK------------------'''
    override_component_attrs = override_components_mlinks()
    network = pypsa.Network(override_component_attrs=override_component_attrs)
    network.set_snapshots(p.hours_in_period)

    # Add external grids (no capital or marginal costs) and demands
    network, comp_external_grids = add_external_grids(network, inputs_dict, n_flags)
    network, comp_demands = add_demands(network, n_flags, inputs_dict)

    # Add agents if selected
    network, comp_biogas = add_biogas(network, n_flags, inputs_dict, tech_costs)
    network, comp_renewables = add_renewables(network, n_flags, inputs_dict, tech_costs)
    network, comp_electrolysis = add_electrolysis(network, n_flags, inputs_dict, tech_costs)
    network, comp_meoh = add_meoh(network, n_flags, inputs_dict, tech_costs)
    network, comp_central_H = add_central_heat_MT(network, n_flags, inputs_dict, tech_costs)
    network, comp_symbiosis = add_symbiosis(network, n_flags, inputs_dict, tech_costs)
    network, comp_methanation = add_methanation(network, n_flags, inputs_dict, tech_costs)
    network, comp_storage = add_storage(network, n_flags, inputs_dict, tech_costs)

    network_comp_allocation = {'external_grids': comp_external_grids,
                               'SkiveBiogas': comp_biogas,
                               'renewables': comp_renewables,
                               'electrolysis': comp_electrolysis,
                               'meoh': comp_meoh,
                               'methanation': comp_methanation,
                               'central_heat': comp_central_H,
                               'symbiosis': comp_symbiosis,
                               'storage': comp_storage,
                               'demands' : comp_demands,
                               }

    # add buses per agent and interface buses per agent
    network_comp_allocation = network_comp_allocation_add_buses_interface(network, network_comp_allocation)

    # save comp allocation within network
    network.network_comp_allocation = network_comp_allocation

    return network
