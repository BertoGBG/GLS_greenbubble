import pandas as pd
import numpy as np
import pypsa
import pypsatopo
import parameters as p
import math
from scripts.preprocessing import en_market_prices_w_CO2
from scripts.grid_constraints import add_link_El_grid_to_H2

# ------- BUILD PYPSA NETWORK-------------

def network_dependencies(n_flags):
    """Check if all required dependencies are satisfied when building the network based on n_flags dictionary in main,
    modifies n_flag dict """
    n_flags_OK = n_flags.copy()

    # SkiveBiogas : NO dependencies
    n_flags_OK['SkiveBiogas'] = n_flags['SkiveBiogas']

    # renewables : NO Dependencies
    n_flags_OK['renewables'] = n_flags['renewables']

    # H2 production Dependencies
    n_flags_OK['electrolyzer'] = n_flags['electrolyzer']

    # MeOH production Dependencies
    if n_flags['meoh'] and n_flags['electrolyzer'] and n_flags['renewables'] and n_flags['SkiveBiogas'] and n_flags[
        'symbiosis_net']:
        n_flags_OK['meoh'] = True
    else:
        n_flags_OK['meoh'] = False

    # Methanation production Dependencies
    if n_flags['methanation'] and n_flags['electrolyzer'] and n_flags['renewables'] and n_flags['SkiveBiogas'] and n_flags[
        'symbiosis_net']:
        n_flags_OK['methanation'] = True
    else:
        n_flags_OK['methanation'] = False

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
    'links' : n.links.index.values,
    'generators' : n.generators.index.values,
    'loads' : n.loads.index.values,
    'stores' : n.stores.index.values,
    'buses' : n.buses.index.values }

    return n0_dict

def log_new_components(n, n0_dict):
    # take a status of the network after adding a technology and log the new components added
    # log new components
    new_components = {'links': list(set(n.links.index.values) - set(n0_dict['links'])),
                      'generators': list(set(n.generators.index.values) - set(n0_dict['generators'])),
                      'loads': list(set(n.loads.index.values) - set(n0_dict['loads'])),
                      'stores': list(set(n.stores.index.values) - set(n0_dict['stores'])),
                      'buses':  list(set(n.buses.index.values) - set(n0_dict['buses']))}
    return new_components

def add_local_heat_connections(n, heat_bus_list, GL_eff, plant_name, n_flags, tech_costs):
    """function that creates local heat buses for each plant.
    heat leaving the plant can be rejected to the ambient for free.
    heat required by the plant can be supplied by symbiosys net ar added heating technologies"""

    new_buses = ['', '', '']

    for i in range(len(heat_bus_list)):
        b = heat_bus_list[i]  # symbiosys net bus
        if not math.isnan(GL_eff.loc[b, plant_name]):
            sign_eff = np.sign(
                GL_eff.loc[b, plant_name])  # negative is consumed by  the agent, positive is produced by the agent

            # add local bus (input)
            bus_name = b + '_' + plant_name
            new_buses[i] = bus_name

            n.add('Bus', bus_name, carrier='Heat', unit='MW')

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
                    n.add('Bus', b, carrier='Heat', unit='MW')
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
                           'electricity grid connection', 'fixed'] * p.currency_multiplier,
          p_nom_extendable=True)

    # internal el connection
    if n_flags['symbiosis_net']:
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


def add_local_boilers(n, local_EL_bus, local_heat_bus, plant_name, tech_costs, en_market_prices):
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
    """function that add the biogas plant to the network and all the dependecies if not preset in the network yet"""

    bioCH4_demand = inputs_dict['bioCH4_demand']
    GL_eff = inputs_dict['GL_eff']
    GL_inputs = inputs_dict['GL_inputs']
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    # take a status of the network before adding components
    n0_dict = get_network_status(n)

    bus_dict={'bus_list' : ['Dig_biomass', 'Digest DM', 'ElDK1 bus', 'biogas', 'bioCH4'],
    'carrier_list' : ['Dig_biomass', 'Digest DM', 'AC', 'gas', 'gas'],
    'unit_list' : ['MW', 't/h', 'MW', 'MW', 'MW']}


    if n_flags['SkiveBiogas']:
        # add required buses if not in the network
        n = add_requirements_buses(n, bus_dict)

        # ----------External BioMethane Load---------------------
        n.add("Load",
              "bioCH4",
              bus="bioCH4",
              p_set=bioCH4_demand['bioCH4 demand MWh'])

        # ------- Digestible biomass generator -------
        n.add("Generator",
              "Dig_biomass",
              bus="Dig_biomass",
              p_nom_extendable=True)

        # ------- add EL connections------------
        local_EL_bus = 'El_biogas'
        n = add_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

        # -----add local heat connections
        plant_name = 'SkiveBiogas'
        heat_bus_list = ["Heat MT", "Heat LT", "Heat DH"]
        n, new_heat_buses = add_local_heat_connections(n, heat_bus_list, GL_eff, plant_name, n_flags, tech_costs)

        # -----Biogas plant --------------
        # NOTE 1: OPERATES STEADY STATE DUE TO CONSTANT DEMAND
        # NOTE 2: REFERENCE in the study is that standard operation of the Biogas plant has a cost = 0
        # Hence there is an opportunity for revenue by NOT using the NG boiler and Grid electricity.
        # In the calculation the plant is allocated this "Revenue" as marginal cost (every hour).

        NG_opportunity_revenue = -(en_market_prices['NG_grid_price'] * np.abs(
            GL_eff.loc["Heat MT", "SkiveBiogas"]) / tech_costs.at[
                                       'gas boiler steam', 'efficiency'])  # €/(t_biomass)

        EL_opportunity_revenue = -(en_market_prices['el_grid_price'] * np.abs(
            GL_eff.loc["El2 bus", "SkiveBiogas"] * 0.5))  # €/(t_biomass))

        n.add("Link",
              "SkiveBiogas",
              bus0="Dig_biomass",
              bus1="biogas",
              bus2=new_heat_buses[2],  # "Heat LT",
              bus3=local_EL_bus,  # 'El_biogas',
              bus4='Digest DM',
              efficiency=GL_eff.loc["bioCH4", "SkiveBiogas"],
              efficiency2=GL_eff.loc["Heat LT", "SkiveBiogas"],
              efficiency3=GL_eff.loc["El2 bus", "SkiveBiogas"]* 0.5 ,
              efficiency4=GL_eff.loc["DM digestate", "SkiveBiogas"],
              p_nom=np.abs(GL_inputs.loc["Biomass", 'SkiveBiogas']),
              marginal_cost=(p.Dig_biomass_price + NG_opportunity_revenue + EL_opportunity_revenue) * p.currency_multiplier,
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

        # Add biogas storage
        n = add_biogas_store(n, n_flags, inputs_dict, tech_costs)

        # log new components
        new_components = log_new_components(n, n0_dict)

    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components


def add_biogas_upgrading(n, n_flags, inputs_dict, tech_costs, brown_field):

    GL_eff = inputs_dict['GL_eff']
    GL_inputs = inputs_dict['GL_inputs']
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    # take a status of the network before adding components
    n0_dict = get_network_status(n)

    bus_dict={'bus_list' : ['ElDK1 bus', 'biogas', 'bioCH4','NG', 'CO2 sep', 'CO2 pure atm'],
    'carrier_list' : ['AC', 'gas', 'gas', 'gas', 'CO2 pure', 'CO2 pure'],
    'unit_list' : ['MW', 'MW', 'MW', 'MW', 't/h', 't/h']}


    if n_flags['SkiveBiogas']:
        # add required buses if not in the network
        n = add_requirements_buses(n, bus_dict)

        # log new components
        new_components = log_new_components(n, n0_dict)

        # ------- add EL connections------------
        local_EL_bus = 'El_biogas_upgrading'
        n = add_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

        # -----add local heat connections
        new_heat_buses = ['Heat MT' + '_' + 'SkiveBiogas']

        #------ brownfield
        if brown_field:

            # ----- add biogas upgrading
            n.add("Link",
                  "biogas upgrading",
                  bus0="biogas",
                  bus1="bioCH4",
                  bus2="CO2 sep",
                  bus3=new_heat_buses[0],  # "Heat MT",
                  bus4=local_EL_bus,
                  efficiency=1,
                  efficiency2=GL_eff.loc["CO2 pure", "SkiveBiogas"] / GL_eff.loc["bioCH4", "SkiveBiogas"],
                  efficiency3=GL_eff.loc["Heat MT", "SkiveBiogas"] / GL_eff.loc["bioCH4", "SkiveBiogas"],
                  efficiency4=GL_eff.loc["El2 bus", "SkiveBiogas"] * 0.5 / GL_eff.loc["bioCH4", "SkiveBiogas"],
                  p_nom = np.abs(GL_inputs.loc["bioCH4", 'SkiveBiogas']),
                  p_nom_extendable=False,
                  # capital_cost=tech_costs.at['biogas upgrading', 'fixed'] * p.currency_multiplier,  # EUR/MWhCH4
                  marginal_cost=tech_costs.at['biogas upgrading', 'VOM'] * p.currency_multiplier)

            # existing NG boiler
            n.add("Link",
                  "NG boiler upgrading",
                  bus0="NG",
                  bus1="Heat MT",
                  efficiency=tech_costs.at['central gas boiler', 'efficiency'],
                  p_nom_extendable=False,
                  p_nom=np.abs(
                      GL_inputs.loc['Heat MT', 'SkiveBiogas'] / tech_costs.at['gas boiler steam', 'efficiency']),
                  # capital_cost=tech_costs.at['central gas boiler', 'fixed'] * p.currency_multiplier,
                  marginal_cost=en_market_prices['NG_grid_price'] +
                                tech_costs.at['gas boiler steam', 'VOM'] * p.currency_multiplier)

            # enables existing NG boiler to supply heat to the symbiosys network
            if n_flags['symbiosis_net']:
                n.links.p_min_pu.at[new_heat_buses[0]] = -1

        else:
            # ----- add biogas upgrading
            n.add("Link",
                  "biogas upgrading",
                  bus0="biogas",
                  bus1="bioCH4",
                  bus2="CO2 sep",
                  bus3=new_heat_buses[0],  # "Heat MT",
                  bus4=local_EL_bus,
                  efficiency=1,
                  efficiency2=GL_eff.loc["CO2 pure", "SkiveBiogas"] / GL_eff.loc["bioCH4", "SkiveBiogas"],
                  efficiency3=GL_eff.loc["Heat MT", "SkiveBiogas"] / GL_eff.loc["bioCH4", "SkiveBiogas"],
                  efficiency4=GL_eff.loc["El2 bus", "SkiveBiogas"] * 0.5 / GL_eff.loc["bioCH4", "SkiveBiogas"],
                  p_nom_extendable=True,
                  capital_cost=tech_costs.at['biogas upgrading', 'fixed'] * p.currency_multiplier,  # EUR/MWhCH4
                  marginal_cost=tech_costs.at['biogas upgrading', 'VOM'] * p.currency_multiplier)

            if not n_flags['symbiosis_net']:
                # additional NG boiler
                n.add("Link",
                      "NG boiler upgrading",
                      bus0="NG",
                      bus1="Heat MT",
                      efficiency=tech_costs.at['central gas boiler', 'efficiency'],
                      p_nom_extendable=True,
                      capital_cost=tech_costs.at['central gas boiler', 'fixed'] * p.currency_multiplier,
                      marginal_cost=en_market_prices['NG_grid_price'] +
                                    tech_costs.at['gas boiler steam', 'VOM'] * p.currency_multiplier)

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

    bus_dict ={    'bus_list' : ['El3 bus', 'ElDK1 bus'],
    'carrier_list' : ['AC', 'AC', 'AC'],
    'unit_list' : ['MW', 'MW', 'MW']}


    if n_flags['renewables']:
        # add required buses if not in the network
        n = add_requirements_buses(n, bus_dict)

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
              p_max_pu=CF_wind['CF wind'])

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
              p_max_pu=CF_solar['CF solar'])

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

    # Grid to H2 availability
    # p_max_pu_renew_el_price, p_max_pu_renew_em = p_max_pu_EU_renewable_el(Elspotprices, CO2_emiss_El)

    bus_dict = {    'bus_list' : ['El3 bus', 'H2', 'H2 delivery', 'Heat amb'],
    'carrier_list' : ['AC', 'H2', 'H2', 'Heat'],
    'unit_list' : ['MW', 'MW', 'MW', 'MW']}


    if n_flags['electrolyzer']:
        # add required buses if not in the network
        n = add_requirements_buses(n, bus_dict)

        # ---------- conditions for use of electricity form the grid without additional RE----
        n = add_link_El_grid_to_H2(n, inputs_dict, tech_costs)

        # -----add local heat connections
        plant_name = 'GreenHyScale'
        heat_bus_list = ['Heat MT', "Heat DH", "Heat LT"]
        n, new_heat_buses = add_local_heat_connections(n, heat_bus_list, GL_eff, plant_name, n_flags, tech_costs)

        # -----------Electrolyzer------------------
        # cost_electrolysis dependent on scale (grid ot MeOH only)
        if H2_input_demand.iloc[:, 0].sum() > 0:
            electrolysis_cost = tech_costs.at['electrolysis', 'fixed'] * p.electrolysis_dict['cost_factor'] * p.currency_multiplier
        else:
            electrolysis_cost = tech_costs.at['electrolysis small', 'fixed'] * p.currency_multiplier

        n.add("Link",
              "Electrolyzer",
              bus0="El3 bus",
              bus1="H2",
              bus2=new_heat_buses[2],
              efficiency=GL_eff.at['H2', 'GreenHyScale'],
              efficiency2=GL_eff.at['Heat LT', 'GreenHyScale'],
              capital_cost=electrolysis_cost,
              marginal_cost=0,
              p_nom_extendable=True,
              p_min_pu=p.electrolysis_dict['p_min_pu'],
              ramp_limit_up=p.electrolysis_dict['ramp_limit_up'],
              ramp_limit_down=p.electrolysis_dict['ramp_limit_down'])

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
    Methanol_input_demand = inputs_dict['Methanol_input_demand']
    H2_input_demand = inputs_dict['H2_input_demand']

    # take a status of the network before adding components
    n0_dict = get_network_status(n)

    # required buses
    bus_dict={  'bus_list' : ['ElDK1 bus', 'El3 bus', 'H2 HP', 'H2_distribution', 'CO2 pure HP', 'Methanol', 'Heat amb'],
                'carrier_list' : ['AC', 'AC', 'H2', 'H2', 'CO2 pure', 'Methanol', 'Heat'],
                'unit_list' : ['MW', 'MW', 'MW', 'MW', 't/h', 'MW', 'MW']}

    #------- ADD METHANOL DEMAND
    if n_flags['meoh']:
        # add required buses if not in the network
        n = add_requirements_buses(n, bus_dict)

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
        # local_EL_bus='El_meoh'
        # n = add_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

        # --------add H2 grid connection if available -----
        # if n_flags['electrolyzer']:
        if H2_input_demand.iloc[:, 0].sum() > 0:  # external H2 demand
            n.add("Link",
                  "H2grid_to_meoh",
                  bus0="H2 delivery",
                  bus1='H2_distribution',
                  efficiency=1,
                  p_nom_extendable=True,
                  capital_cost=tech_costs.at[
                                   'H2_pipeline_gas', "fixed"] * p.dist_H2_pipe * p.currency_multiplier)


        # ----------METHANOL PLANT---------
        # add local heat connections
        plant_name = 'Methanol plant'
        heat_bus_list = ['Heat MT', "Heat DH", "Heat LT"]
        n, new_heat_buses = add_local_heat_connections(n, heat_bus_list, GL_eff, plant_name, n_flags, tech_costs)

        if not n_flags['central_heat']:
            # add_local_boilers(n, local_EL_bus, new_heat_buses[0], plant_name, tech_costs, en_market_prices)
            add_local_boilers(n, 'El3 bus', new_heat_buses[0], plant_name, tech_costs, en_market_prices)

        n.add("Link",
              "Methanol plant",
              bus0="CO2 pure HP",
              bus1="Methanol",
              bus2="H2 HP",
              bus3='El3 bus',  # local_EL_bus,
              bus4=new_heat_buses[0],
              bus5=new_heat_buses[1],
              efficiency=GL_eff.loc["Methanol", "Methanol plant"],
              efficiency2=GL_eff.loc["H2", "Methanol plant"],
              efficiency3=GL_eff.loc["El2 bus", "Methanol plant"],
              efficiency4=GL_eff.at['Heat MT', 'Methanol plant'],
              efficiency5=GL_eff.at['Heat DH', 'Methanol plant'],
              p_nom_extendable=True,
              capital_cost=tech_costs.at['methanolisation', "fixed"] * p.meoh_dict['cost_factor'] * p.currency_multiplier,
              p_min_pu=p.meoh_dict['p_min_pu'],
              ramp_limit_up=p.meoh_dict['ramp_limit_up'],
              ramp_limit_down=p.meoh_dict['ramp_limit_down'])

        # ------ADD CO2 compressor ( and storage Liquid and Cylinders) and add H2 compressor and Storage (Steel Vassel)------
        n = add_CO2_store(n, n_flags, inputs_dict, tech_costs)
        n = add_H2_store(n, n_flags, inputs_dict, tech_costs)

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

    GL_eff = inputs_dict['GL_eff']
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    # calc input demand methanation
    Methananation_input_demand = inputs_dict['Methanation_input_demand']

    # take a status of the network before adding components
    n0_dict = get_network_status(n)

    # Required buses
    bus_dict = {'bus_list' : ['ElDK1 bus', 'El3 bus', 'H2_distribution', 'CO2_distribution', 'biogas', 'bioCH4', 'Heat MT', 'methanation'],
    'carrier_list' : ['AC', 'AC', 'H2', 'CO2 pure', 'gas', 'gas', 'Heat', 'gas'],
    'unit_list' : ['MW', 'MW', 'MW', 't/h', 'MW', 'MW', 'MW', 'MW']}


    if n_flags['methanation']:
        # add required buses if not in the network
        n = add_requirements_buses(n, bus_dict)

        # ------- add Store for production rewarded with NG price ------------
        n.add('Store',
              'methanation',
              bus='methanation',
              e_nom_extendable=True,
              e_nom_min=0,
              e_nom_max=float("inf"),
              e_cyclic=True,
              marginal_cost= 0,# -en_market_prices['NG_grid_price'],
              capital_cost=0)

        # ----------MeOH deliver infinite storage-------
        n.add("Load",
              "methanation",
              bus="methanation",
              p_set=Methananation_input_demand.iloc[:, 0])

        # ----------BIO-METHANATION PLANT (biogas + H2)---------
        if p.biometh_dict['active']:
            n.add("Link",
                  "biomethanation_biogas",
                  bus0="H2_distribution",
                  bus1="methanation",
                  bus2="biogas",
                  bus3='El3 bus',  # local_EL_bus,
                  efficiency=tech_costs.at['biomethanation', "Methane Output"],
                  efficiency2=tech_costs.at['biomethanation', "Biogas Input"] ,  # tCO2/MWh_H2 * MWCH4/tCO2 = MW_biogas/MW_H2
                  efficiency3=tech_costs.at['biomethanation', "electricity input"],
                  p_nom_extendable=True,
                  capital_cost=tech_costs.at['biomethanation', "fixed"] * p.currency_multiplier,
                  marginal_cost =  tech_costs.at['biomethanation', "VOM"] * p.currency_multiplier)

            # ----------BIO-METHANATION PLANT (CO2 + H2)---------
            # capital costs are scaled by the volumetric flow of input gas  to the reactor
            v_ch4_v_co2 = (tech_costs.at['biomethanation', "Biogas Input"] / p.lhv_ch4 / p.density_CH4_1atm) / tech_costs.at['biomethanation', "CO2 Input"] / p.density_CO2_1atm
            v_h2 =  1/p.lhv_h2 * 1e3 / p.density_H2_1atm # m3/h/MW_h2
            v_co2 = tech_costs.at['biomethanation', "CO2 Input"] / p.density_CO2_1atm * 1e3 # m3/h/MW_h2
            v_ch4 = v_co2 * v_ch4_v_co2 # m3/h/MW_h2
            input_vol_flow_onlyco2_biogas_biometh = (v_h2 + v_co2)/(v_h2 + v_co2 + v_ch4) # ratio of vol flow for CO2 only vs biogas (for 1MW H2)

            #print('input_vol_flow_onlyco2_biogas_bio')
            #print(input_vol_flow_onlyco2_biogas)

            n.add("Link",
                  "biomethanation_co2",
                  bus0="H2_distribution",
                  bus1="methanation",
                  bus2="CO2_distribution",
                  bus3='El3 bus',  # local_EL_bus,
                  efficiency=tech_costs.at['biomethanation', "Methane Output"] - tech_costs.at['biomethanation', "Biogas Input"], # only generated Methane not input biogas
                  efficiency2=tech_costs.at['biomethanation', "CO2 Input"],
                  efficiency3=tech_costs.at['biomethanation', "electricity input"],
                  p_nom_extendable=True,
                  capital_cost=tech_costs.at['biomethanation', "fixed"] * input_vol_flow_onlyco2_biogas_biometh * p.currency_multiplier ,
                  marginal_cost =  tech_costs.at['biomethanation', "VOM"] * input_vol_flow_onlyco2_biogas_biometh * p.currency_multiplier)


        # --------- CATALYTIC METHANATION PLANT (biogas + H2)
        if p.catmeth_dict['active']:
            n.add("Link",
                  "cat_methanation_biogas",
                  bus0="H2_distribution",
                  bus1="methanation",
                  bus2="biogas",
                  bus3='El3 bus',  # local_EL_bus,
                  bus4='Heat MT',
                  efficiency=tech_costs.at['biogas plus hydrogen', "Methane Output"],
                  efficiency2=tech_costs.at['biogas plus hydrogen', "Biogas Input"],
                  efficiency3=tech_costs.at['biogas plus hydrogen', "electricity input"],
                  efficiency4=tech_costs.at['biogas plus hydrogen', "heat output"],
                  p_nom_extendable=True,
                  p_min_pu = p.catmeth_dict['p_min_pu'],
                  ramp_limit_up=p.catmeth_dict['ramp_limit_up'],
                  ramp_limit_down=p.catmeth_dict['ramp_limit_down'],
                  capital_cost=tech_costs.at['biogas plus hydrogen', "fixed"] * p.ramp_limit_down['cost_factor'] * p.currency_multiplier, # cost per MWh_H2 input
                  marginal_cost = tech_costs.at['biogas plus hydrogen', "VOM"]  * p.currency_multiplier)

            # --------- CATALYTIC METHANATION PLANT (CO2 + H2)
            # capital costs are scaled by the volumetric flow of input gas  to the reactor
            v_ch4_v_co2 = (tech_costs.at['biogas plus hydrogen', "Biogas Input"] / p.lhv_ch4 / p.density_CH4_1atm) / tech_costs.at['biogas plus hydrogen', "CO2 Input"] / p.density_CO2_1atm
            v_h2 =  1/p.lhv_h2 * 1e3 / p.density_H2_1atm # m3/h/MW_h2
            v_co2 = tech_costs.at['biogas plus hydrogen', "CO2 Input"] / p.density_CO2_1atm * 1e3 # m3/h/MW_h2
            v_ch4 = v_co2 * v_ch4_v_co2 # m3/h/MW_h2
            input_vol_flow_onlyco2_biogas_cat_meth = (v_h2 + v_co2)/(v_h2 + v_co2 + v_ch4) # ratio of vol flow for CO2 only vs biogas (for 1MW H2)

            #print('input_vol_flow_onlyco2_biogas_cat')
            #print(input_vol_flow_onlyco2_biogas)

            n.add("Link",
                  "cat_methanation_co2",
                  bus0="H2_distribution",
                  bus1="methanation",
                  bus2="CO2_distribution",
                  bus3='El3 bus',  # local_EL_bus,
                  bus4='Heat MT',
                  efficiency=tech_costs.at['biogas plus hydrogen', "Methane Output"] - tech_costs.at['biogas plus hydrogen', "Biogas Input"],
                  efficiency2=tech_costs.at['biogas plus hydrogen', "CO2 Input"], # tCO2/MWh_H2
                  efficiency3=tech_costs.at['biogas plus hydrogen', "electricity input"],
                  efficiency4=tech_costs.at['biogas plus hydrogen', "heat output"],
                  p_nom_extendable=True,
                  p_min_pu=p.p_min_pu_cat_meth,
                  ramp_limit_up=p.ramp_limit_up_cat_meth,
                  ramp_limit_down=p.ramp_limit_down_cat_meth,
                  capital_cost=tech_costs.at['biogas plus hydrogen', "fixed"]  * input_vol_flow_onlyco2_biogas_cat_meth * p.currency_multiplier,
                  marginal_cost = tech_costs.at['biogas plus hydrogen', "VOM"] * input_vol_flow_onlyco2_biogas_cat_meth * p.currency_multiplier)

        # ----------- Add link between methanation bus and bioCH4 bus to satisfy external demand
        n.add('Link',
              'methanation_to_NG',
              bus0='methanation',
              bus1='bioCH4',
              efficiency=1,
              p_nom_extendable=True)

        # ----------- ADD CO2 (Liquid and Cylinders) and H2 Storage (Steel Vassel)----------
        n = add_CO2_store(n, n_flags, inputs_dict, tech_costs)
        n = add_H2_store(n, n_flags, inputs_dict, tech_costs)

        # log new components
        new_components = log_new_components(n, n0_dict)

    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components


def add_biogas_store(n, n_flags, inputs_dict, tech_costs):
    # function that adds biogas storage (baloon)

    # input data
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    # Required buses
    bus_dict={'bus_list' : ['biogas'],
              'carrier_list' : ['gas'],
              'unit_list' : ['MW']
    }

    # add required buses if not in the network
    n = add_requirements_buses(n, bus_dict)

    # add store
    n.add('Store',
          'biogas storage',
          bus='biogas',
          e_nom_extendable =  True,
          capital_cost=tech_costs.at[
                           'biogas storage', 'fixed'] * p.currency_multiplier,
          e_nom_max=p.e_nom_max_biogas_storage,
          e_cyclic=True)

    return n


def add_CO2_store(n, n_flags, inputs_dict, tech_costs):
    # Function that adds CO2 storage as liquid store and CO2 cylinders

    # input data
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    # Required buses
    bus_dict={'bus_list' : ['El3 bus', 'CO2_distribution', 'CO2 pure HP', 'CO2 storage', 'Heat amb', 'CO2 comp heat', 'CO2 liq storage', 'CO2 liq heat LT'],
              'carrier_list' : ['AC', 'CO2', 'CO2', 'CO2', 'Heat', 'CO2', 'CO2', 'Heat'],
              'unit_list' : ['MW', 't/h', 't/h', 't/h', 'MW', 't/h', 't/h', 'MW']
    }

    # add required buses if not in the network
    n = add_requirements_buses(n, bus_dict)

    # ------- add EL connections------------
    local_EL_bus = 'El_CO2_compressor'
    if 'El_CO2_compressor' not in n.buses.index.values:
        n = add_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

    # -----------CO2 compressor -----------------------
    # check if CO2 compressor already in network
    if "CO2 compressor" not in n.links.index.values:

        n.add("Link",
              "CO2 compressor",
              bus0="CO2_distribution",
              bus1="CO2 pure HP",
              bus2= local_EL_bus,
              bus3='CO2 comp heat',
              efficiency=1,
              efficiency2=-1 * p.el_comp_CO2,
              efficiency3=1 * p.heat_comp_CO2,
              p_nom_extendable=True,
              capital_cost=tech_costs.at['CO2_industrial_compressor', "fixed"] * p.currency_multiplier)

        n.add('Link',
              'CO2 comp heat rejection',
              bus0='CO2 comp heat',
              bus1='Heat amb',
              efficiency=1,
              p_nom_extendable=True)

        if n_flags['symbiosis_net']:
            if 'Heat LT' not in n.buses.index.values:
                n.add('Bus', 'Heat LT', carrier='Heat', unit='MW')
            n.add('Link',
                  'CO2 comp heat integration',
                  bus0='CO2 comp heat',
                  bus1='Heat LT',
                  efficiency=1,
                  capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * p.currency_multiplier,
                  p_nom_extendable=True)

# -----------CO2 HP storage cylinders ---------------
    if 'CO2 pure HP' not in n.stores.index.values:

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
    # check if CO2 liquid storage already in the network
    if 'CO2 Liq' not in n.stores.index.values:

        n.add('Link',
              'CO2 liq send',
              bus0='CO2_distribution',
              bus1='CO2 liq storage',
              bus2= local_EL_bus,
              bus3='CO2 liq heat LT',
              efficiency=1,
              efficiency2=-1 * p.El_CO2_liq,
              efficiency3=p.Heat_CO2_liq_DH,
              capital_cost=tech_costs.at['CO2 liquefaction', 'fixed'] * p.currency_multiplier,
              p_nom_extendable=True)

        n.add('Link',
              'CO2 liq return',
              bus0='CO2 liq storage',
              bus1='CO2_distribution',
              capital_cost=p.CO2_evap_annualized_cost * p.currency_multiplier,
              efficiency=1,
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

    return n


def add_H2_store(n, n_flags, inputs_dict, tech_costs):
    # function that adds H2 store on the H2 HP bus
    # pressure 80 bars

    # input data
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    # Required buses
    bus_dict={'bus_list' : ['El3 bus', 'H2_distribution', 'H2 HP', 'H2 storage', 'Heat amb', 'H2 comp heat' ],
              'carrier_list' : ['AC', 'H2', 'H2', 'H2', 'Heat', 'Heat'],
              'unit_list' : ['MW', 'MW', 'MW', 'MW', 'MW', 'MW']
    }


    # add required buses if not in the network
    n = add_requirements_buses(n, bus_dict)


    # ------- add EL connections------------
    local_EL_bus = 'El_H2_compressor'
    if 'El_H2_compressor' not in n.buses.index.values:
        n = add_el_conections(n, local_EL_bus, en_market_prices, n_flags, tech_costs)

    # -----------H2 compressor ----------------------
    # Check if H2 compressor already in the network
    if 'H2 compressor' not in n.links.index.values:

        n.add("Link",
              "H2 compressor",
              bus0="H2_distribution",
              bus1="H2 HP",
              bus2=local_EL_bus,
              bus3='H2 comp heat',
              efficiency=1,
              efficiency2=-1 * p.el_comp_H2,
              efficiency3=1 * p.heat_comp_H2,
              p_nom_extendable=True,
              capital_cost=tech_costs.at['hydrogen storage compressor', 'fixed'] * p.currency_multiplier,
              marginal_cost=tech_costs.at['hydrogen storage compressor', 'VOM'] * p.currency_multiplier)

        n.add('Link',
              'H2 comp heat rejection',
              bus0='H2 comp heat',
              bus1='Heat amb',
              efficiency=1,
              p_nom_extendable=True)

        if n_flags['symbiosis_net']:
            if 'Heat LT' not in n.buses.index.values:
                n.add('Bus', 'Heat LT', carrier='Heat', unit='MW')

            n.add('Link',
                  'H2 comp heat integration',
                  bus0='H2 comp heat',
                  bus1='Heat LT',
                  efficiency=1,
                  capital_cost=tech_costs.at['DH heat exchanger', "fixed"] * p.currency_multiplier,
                  p_nom_extendable=True)

    # -----------H2 HP storage cylinders ---------------
    # check if H2 storage already in the network
    if 'H2 HP' not in n.stores.index.values:

        # H2 compressed local HP Storage
        n.add('Link',
              'H2 storage send',
              bus0='H2 HP',
              bus1='H2 storage',
              bus2= local_EL_bus,
              efficiency=1,
              efficiency2=-1 * p.El_H2_storage_add,
              p_nom_extendable=True)

        n.add('Link',
              'H2 storage return',
              bus0='H2 storage',
              bus1='H2 HP',
              efficiency=1,
              p_nom_extendable=True)

        n.add("Store",
              "H2 HP",
              bus="H2 storage",
              e_nom_extendable=True,
              capital_cost=tech_costs.at['hydrogen storage tank type 1', 'fixed'] * p.currency_multiplier,
              marginal_cost=tech_costs.at['hydrogen storage tank type 1', 'VOM'] * p.currency_multiplier,
              e_nom_max=p.e_nom_max_H2_HP,
              e_cyclic=True)

    return n


def add_central_heat_MT(n, n_flags, inputs_dict, tech_costs):
    '''this function adds expansion capacity for heating technology'''

    GL_eff = inputs_dict['GL_eff']
    GL_inputs = inputs_dict['GL_inputs']
    CO2_cost = inputs_dict['CO2 cost']
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    # take a status of the network before adding components
    n0_dict = get_network_status(n)

    bus_dict = { 'bus_list' : ['Straw Pellets', 'Digest DM', 'ElDK1 bus', 'NG', 'biochar', 'biochar storage'],
    'carrier_list' : ['Straw Pellets', 'Digest DM', 'AC', 'NG', 'CO2 pure', 'CO2 pure'],
    'unit_list' : ['t/h', 't/h', 'MW', 'MW', 't/h', 't/h']}


    if n_flags['central_heat']:
        # add required buses if not in the network
        n = add_requirements_buses(n, bus_dict)

        # ------- add EL connections------------
        local_EL_bus = 'El_C_heat'
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
        # NOTE: electricity in Skyclean is moslty for pelletization of digestate fibers,
        # hence it is balanced (produced for free) by this link when pellets are purchased
        n.add("Link",
              "Straw to Skyclean",
              bus0="Straw Pellets",
              bus1="Digest DM",
              bus2=local_EL_bus,
              efficiency=p.lhv_straw_pellets / p.lhv_dig_pellets,
              efficiency2=-GL_eff.at['El2 bus', 'SkyClean'] * p.lhv_straw_pellets / p.lhv_dig_pellets,
              p_nom_extendable=True)

        if n_flags['bioChar']:
            biochar_cost = -CO2_cost
        else:
            biochar_cost = 0

        n.add("Link",
              "SkyClean",
              bus0='Digest DM',
              bus1='Heat MT',
              bus2=local_EL_bus,
              bus3='biochar',
              efficiency=GL_eff.at['Heat MT', 'SkyClean'],
              efficiency2=GL_eff.at['El2 bus', 'SkyClean'],
              efficiency3=-GL_eff.at['CO2e bus', 'SkyClean'],  # NOTE: negative sign for CO2e in the input file
              marginal_cost=(tech_costs.at[
                  'biomass HOP', 'VOM']) * p.currency_multiplier,
              p_nom_extendable=True,
              p_nom_max=p.p_nom_max_skyclean / p.lhv_dig_pellets,
              capital_cost=tech_costs.at['biochar pyrolysis', "fixed"] * p.currency_multiplier)  #

        n.add('Link',
              'biochar credits',
              bus0='biochar',
              bus1='biochar storage',
              efficiency=1,
              marginal_cost=biochar_cost * p.currency_multiplier,  # REWARD FOR NEGATIVE EMISSIONS
              p_nom_extendable=True)

        n.add('Store',
              'biochar storage',
              bus="biochar storage",
              e_nom_extendable=True,
              e_nom_min=0,
              e_nom_max=float("inf"),  # Total emission limit
              e_cyclic=False)

        # ------ BIOMASS BOILER (no biochar)-------
        n.add("Link",
              "Pellets boiler",
              bus0='Digest DM',
              bus1="Heat MT",
              bus2=local_EL_bus,
              efficiency=tech_costs.at['biomass HOP', 'efficiency'] * p.lhv_dig_pellets,
              efficiency2=GL_eff.at['El2 bus', 'SkyClean'],
              marginal_cost=(tech_costs.at[
                  'biomass HOP', 'VOM']) * p.currency_multiplier,
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

        # log new components
        new_components = log_new_components(n, n0_dict)

    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components


def add_symbiosis(n, n_flags, inputs_dict, tech_costs):
    '''this function builds the simbiosys net with: Buses, Links, Storeges
     The services includes: RE, Heat MT, H2, CO2, connection to DH'''

    GL_inputs = inputs_dict['GL_inputs']

    # take a status of the network before adding components
    n0_dict = get_network_status(n)

    bus_dict = {'bus_list' : ['El2 bus', 'Heat MT', 'Heat LT', 'Heat DH', 'battery', 'Heat DH storage', 'Heat MT storage', 'H2', 'H2_distribution', 'CO2 sep', 'CO2_distribution', 'Heat DH'],
                'carrier_list' : ['AC', 'Heat', 'Heat', 'Heat', 'battery', 'Heat', 'Heat', 'H2', 'H2','CO2 pure', 'CO2 pure', 'Heat'],
                'unit_list' : ['MW', 'MW', 'MW', 'MW', 'MW', 'MW', 'MW', 'MW', 'MW', 't/h', 't/h','MW']}


    if n_flags['symbiosis_net']:
        # add required buses if not in the network
        n = add_requirements_buses(n, bus_dict)

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
              "H2_pipe",
              bus0="H2",
              bus1="H2_distribution",
              efficiency=1,
              p_nom_extendable=True,
              capital_cost=tech_costs.at[
                               'H2_pipeline_gas', "fixed"] * p.dist_H2_pipe * p.currency_multiplier)

        # -------- Trading of CO2 (LP)-----
        n.add("Link",
              "CO2_pipe",
              bus0="CO2 sep",
              bus1="CO2_distribution",
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
                  efficiency=1,
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
        if p.TES_conc_dict['active']:
            n.add('Store',
                  'Concrete Heat MT storage',
                  bus='Heat MT storage',
                  e_nom_extendable=True,
                  e_nom_min=0,
                  e_nom_max=p.e_nom_max_Heat_MT_storage,
                  e_cyclic=True,
                  standing_loss = p.TES_conc_dict['standing_loss'],
                  capital_cost=tech_costs.at['Concrete-store', 'fixed'] * p.TES_conc_dict['cost_factor'] * p.currency_multiplier)

        n.add("Link",
              "Heat MT storage charger",
              bus0="Heat MT",
              bus1="Heat MT storage",
              p_nom_extendable=True,
              capital_cost=tech_costs.at['Concrete-charger', 'fixed'] * p.currency_multiplier)

        n.add("Link",
              "Heat MT storage discharger",
              bus0="Heat MT storage",
              bus1="Heat MT",
              p_nom_extendable=True,
              capital_cost=tech_costs.at['Concrete-discharger', 'fixed'] * p.currency_multiplier)

        # log new components
        new_components = log_new_components(n, n0_dict)

    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components


def add_DH(n, n_flags, inputs_dict, tech_costs):
    """function that adds DH infrastruture in the park and grid outside"""
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs)

    # take a status of the network before adding components
    n0_dict = get_network_status(n)

    bus_dict = {'bus_list' : ['ElDK1 bus', 'Heat DH', 'DH grid', 'DH GL'],
    'carrier_list' : ['AC', 'Heat', 'Heat', 'Heat', ],
    'unit_list' : ['MW', 'MW', 'MW', 'MW']}


    # options for DH if selected
    if n_flags['DH']:
        # add required buses if not in the network
        n = add_requirements_buses(n, bus_dict)

        # ------- add EL connections------------
        local_EL_bus = 'El_DH'
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

        # log new components
        new_components = log_new_components(n, n0_dict)


    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components


def file_name_network(n, n_flags, inputs_dict):
    """function that automatically creates a file name give a network"""
    # the netwrok name includes: the agents included,  the demands variables H2_d, MeOH_d, CO2 cost, bioChar credits
    # and max fraction of electricity sold externally
    # example: Biogas_CHeat_RE_H2_MeOH_SymN_CO2c200_H2d297_MeOHd68
    CO2_cost = inputs_dict['CO2 cost']

    # loads
    if 'H2 grid' in n.loads.index.values:
        H2_d = int(n.loads_t.p_set['H2 grid'].sum() // 1000)  # yearly production of H2 in GWh
    else:
        H2_d = 0

    if 'Methanol' in n.loads.index.values:
        MeOH_d = int(n.loads_t.p_set['Methanol'].sum() // 1000)  # yearly production of MeOH in GWh
    else:
        MeOH_d = 0

    if 'methanation' in n.loads.index.values:
        Methanation_d = int(n.loads_t.p_set['methanation'].sum() // 1000)  # yearly production of MeOH in GWh
    else:
        Methanation_d = 0

    # CO2 tax
    CO2_c = int(CO2_cost)  # CO2 price in currency

    # year
    year = int(p.En_price_year)  # energy price year

    # max El to DK1
    el_DK1_sale_el_RFNBO = inputs_dict['el_DK1_sale_el_RFNBO']

    # agents
    file_name = n_flags['SkiveBiogas'] * 'SB_' + n_flags['central_heat'] * 'CH_' + n_flags['renewables'] * 'RE_' + \
                n_flags['electrolyzer'] * 'H2_' + n_flags['meoh'] * 'meoh_' + n_flags['methanation'] * 'Meth_' + str(Methanation_d) + n_flags['symbiosis_net'] * 'SN_' + \
                n_flags['DH'] * 'DH_' + 'CO2c' + str(CO2_c) + '_' + 'H2d' + str(H2_d) + \
                '_' + 'MeOHd' + str(MeOH_d) + '_' + str(year) + n_flags[
                    'bioChar'] * '_bCh' + '_' + 'El2DK1' + '_' + str(el_DK1_sale_el_RFNBO)

    return file_name


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


def build_network(tech_costs, inputs_dict, n_flags):
    """this function uses bioCH4 demand, H2 demand, and MeOH demand as input to build the PyPSA network"""
    # OUTPUTS: 1) Pypsa network, 2) nested dictionary with componets allocations to the agents

    '''--------------CREATE PYPSA NETWORK------------------'''
    override_component_attrs = override_components_mlinks()
    network = pypsa.Network(override_component_attrs=override_component_attrs)
    network.set_snapshots(p.hours_in_period)

    # Add external grids (no capital costs)
    network, comp_external_grids = add_external_grids(network, inputs_dict, n_flags)

    # Add agents if selected
    network, comp_biogas = add_biogas(network, n_flags, inputs_dict, tech_costs)
    network, comp_biogas_upgrading = add_biogas_upgrading(network, n_flags, inputs_dict, tech_costs, brown_field=True)
    network, comp_renewables = add_renewables(network, n_flags, inputs_dict, tech_costs)
    network, comp_electrolysis = add_electrolysis(network, n_flags, inputs_dict, tech_costs)
    network, comp_meoh = add_meoh(network, n_flags, inputs_dict, tech_costs)
    network, comp_central_H = add_central_heat_MT(network, n_flags, inputs_dict, tech_costs)
    network, comp_symbiosis = add_symbiosis(network, n_flags, inputs_dict, tech_costs)
    network, comp_DH = add_DH(network, n_flags, inputs_dict, tech_costs)
    network, comp_methanation = add_methanation(network, n_flags, inputs_dict, tech_costs)


    network_comp_allocation = {'external_grids': comp_external_grids,
                               'SkiveBiogas': comp_biogas,
                               'Biogas_upgrading' : comp_biogas_upgrading,
                               'renewables': comp_renewables,
                               'electrolyzer': comp_electrolysis,
                               'meoh': comp_meoh,
                               'methanation' : comp_methanation,
                               'central_heat': comp_central_H,
                               'symbiosis_net': comp_symbiosis,
                               'DH': comp_DH}

    # add buses per agent and interface buses per agent
    network_comp_allocation = network_comp_allocation_add_buses_interface(network, network_comp_allocation)

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
