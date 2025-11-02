import numpy as np
import pypsa
import pandas as pd
from toolz.functoolz import return_none

from scripts import parameters as p
from scripts.helpers import en_market_prices_w_CO2
from scripts.grid_constraints import add_link_El_grid_to_H2
from scripts.config import (n_options,
                            n_config,
                            rfnbos_dict,
                            run_name)
from scripts.technology_inputs import symbiosis_n
import CoolProp.CoolProp as CP
from pypsa.optimization.constraints import define_total_supply_constraints

# ------- BUILD PYPSA NETWORK HANDLING FUNCTIONS-------------
def network_dependencies(n_flags, ):
    """Check if all required dependencies are satisfied when building the network based on n_flags dictionary in main,
    modifies n_flag dict """
    n_flags_OK = n_flags.copy()

    # SkiveBiogas : NO dependencies
    n_flags_OK['biogas'] = n_flags['biogas']

    # renewables : NO Dependencies
    n_flags_OK['renewables'] = n_flags['renewables']

    # H2 production Dependencies
    cond1 = n_flags['electrolysis'] and rfnbos_dict['limit'] != ('emissions' or 'price')
    cond2 = n_flags['electrolysis'] and n_flags['renewables']

    if cond1 or cond2 :
        n_flags_OK['electrolysis'] = True
    else:
        n_flags_OK['electrolysis'] = False

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


def add_requirements_buses(n, bus_dict, symbiosis_n=None):
    """
    Ensure carriers exist, then add any missing buses with the specified attributes.
    If symbiosis_n is provided, assign each bus a 'properties' value equal to the
    index of symbiosis_n where it appears in the 'buses' column.

    Special case:
      - If bus name ends with 'CO2 HP storage' or 'H2 HP storage',
        match using only that suffix, regardless of any prefix (e.g. 'PlantA H2 HP storage').
    """

    valid_entries = [i for i, b in enumerate(bus_dict["bus_list"]) if pd.notna(b)]
    bus_list = [bus_dict["bus_list"][i] for i in valid_entries]
    carrier_list = [bus_dict["carrier_list"][i] for i in valid_entries]
    unit_list = [bus_dict.get("unit_list", [""] * len(bus_dict["bus_list"]))[i] for i in valid_entries]

    # Ensure carriers exist
    needed_carriers = {c for c in carrier_list if c}
    missing_carriers = [c for c in needed_carriers if c not in n.carriers.index]
    for c in missing_carriers:
        n.add("Carrier", c)

    # Add missing buses
    to_add = [b for b in bus_list if b not in n.buses.index]
    if to_add:
        idx = [bus_list.index(b) for b in to_add]
        n.add(
            "Bus",
            to_add,
            carrier=[carrier_list[i] for i in idx],
            unit=[unit_list[i] for i in idx],
        )

    # Assign properties from symbiosis_n
    if symbiosis_n is not None:
        # Ensure 'properties' column exists
        if "properties" not in n.buses.columns:
            n.buses["properties"] = None

        # Precompute mapping: bus_name -> symbiosis index
        bus_to_property = {}
        for prop_name, row in symbiosis_n.iterrows():
            buses = row.get("buses", [])
            if isinstance(buses, list):
                for b in buses:
                    if b not in bus_to_property:
                        bus_to_property[b] = prop_name
                    else:
                        print(f"⚠️ Warning: Bus '{b}' appears in multiple symbiosis_n rows: "
                              f"{bus_to_property[b]} and {prop_name}")

        # Helper: function to resolve special suffix mapping
        def resolve_property_name(bus_name):
            if bus_name.endswith("CO2 HP storage"):
                return bus_to_property.get("CO2 HP storage")
            elif bus_name.endswith("H2 HP storage"):
                return bus_to_property.get("H2 HP storage")
            return bus_to_property.get(bus_name)

        # Assign properties
        for b in bus_list:
            if b in n.buses.index:
                prop_value = resolve_property_name(b)
                if prop_value:
                    current = n.buses.at[b, "properties"]
                    if pd.isna(current) or current is None:
                        n.buses.at[b, "properties"] = prop_value

    return n


def get_network_status(n):
    """Return a snapshot of current component names in the network."""
    def safe_index(table):
        return list(table.index) if table is not None else []

    return {
        'links': safe_index(n.links),
        'generators': safe_index(n.generators),
        'loads': safe_index(n.loads),
        'stores': safe_index(n.stores),
        'buses': safe_index(n.buses),
    }


def tech_to_add(techs, n0_dict):
    # functions that compared n_config and network status to decide what technolgies should be installed as initial cpacities or expansion capacities
    # Inputs:
    # techs : list  e.g.     tech = ['CO2 compressor', 'Biogas']
    # n0_dict = get_network_status(n)

    cap = [n_config.at[t,'initial capacity'] for t in techs]  # existing initial capacity for each tech
    exp = [n_config.at[t, 'expansion'] for t in techs]   # capacity expansion for each tech

    cap_missing = ['EXI_' + t for t in techs
               if 'EXI_' + t not in {x for k in ('links', 'generators', 'stores') for x in n0_dict.get(k, [])}]
    exp_missing = [t for t in techs
               if t not in {x for k in ('links', 'generators', 'stores') for x in n0_dict.get(k, [])}]
    cap_to_add = [t for t, c, m in zip(techs, cap, cap_missing) if m and (c is not None) and (c > 0)] # Initial capacities to be added
    exp_to_add = [t for t, c, m in zip(techs, exp, exp_missing) if m and (c is not None) and (c > 0)] # capacity expansion to be added

    return cap_to_add, exp_to_add


def log_new_components(n, n0_dict):
    """
    Compare the network before/after adding components and log new items.
    Safe for uninitialized component tables (which are None in PyPSA 1.0).
    """
    new_components = {}
    for comp in ["links", "generators", "loads", "stores", "buses"]:
        before = set(n0_dict.get(comp, []))

        table = getattr(n, comp, None)
        if table is not None:
            after = set(table.index)
        else:
            after = set()

        new_components[comp] = list(after - before)
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
                    network.links.bus4[lk],
                    network.links.bus5[lk],
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
def add_local_heat_connections(n, heat_bus_dict, plant_name, n_flags, tech_costs, n_config=None):
    """
    Create plant-local heat buses, a rejection link to 'Heat amb',
    and (optionally) a bidirectional connection to the symbiosis heat grid.

    heat_bus_dict = {'Heat MT' : -1,
                    'Heat DH' : -1,
                    'Heat LT' : -1}

    # heat_bus_list can be only ['Heat MT', 'Heat DH', 'Heat LT']
    # symbiosis_dir =  -1  the plant is receiving from the symbiosis network
    # symbiosis_dir =  1  the plant is supplying heat to the symbiosis network

    PyPSA 1.0 notes:
      - Ensure carriers exist before adding buses.
    """

    # --- Ensure required carrier(s) exist ---
    if "Heat" not in n.carriers.index:
        n.add("Carrier", "Heat")

    new_buses = []

    for b in heat_bus_dict.keys():
        # 1) Local bus at the plant (for local boilers etc.)
        local_bus = f"{b}_{plant_name}"

        # direction of the heat flow with respect to the main plant
        symbiosis_dir = heat_bus_dict[b]

        if local_bus not in n.buses.index:
            n.add("Bus", local_bus, carrier="Heat", unit="MW")
        new_buses.append(local_bus)

        # if n_flags['symbiosis'] == True
        if n_flags.get("symbiosis", False):
            # add het bus if not existing
            if b not in n.buses.index:
                n.add("Bus", b, carrier="Heat", unit="MW")

            if int(symbiosis_dir>0):
                # 2) Heat rejection to symbiosis net (on heat bus)
                sym_link = f"{b}_{plant_name}_to_symb"
                if sym_link not in n.links.index:
                    n.add(
                        "Link",
                        sym_link,
                        bus0=local_bus,  # plant-local side
                        bus1=b,  # symbiosis side
                        efficiency=1.0,
                        p_min_pu=0,
                        p_nom_extendable=True,
                        marginal_cost=5e-6,
                        capital_cost=tech_costs.at["DH heat exchanger", "fixed"]
                                     * n_config.at["DH heat exchanger", "cost factor"],
                    )

            elif int(symbiosis_dir<0):
                # 1) Heat supplied by the symbiosis network (from heat bus)
                sym_link = f"{b}_{plant_name}_from_symb"
                if sym_link not in n.links.index:
                    n.add(
                        "Link",
                        sym_link,
                        bus0=b,                 # symbiosis side
                        bus1=local_bus,         # plant-local side
                        efficiency=1.0,
                        p_min_pu=0,
                        p_nom_extendable=True,
                        marginal_cost=5e-6,
                        capital_cost=tech_costs.at["DH heat exchanger", "fixed"]
                                     * n_config.at["DH heat exchanger", "cost factor"],
                    )

        # if n_flags['symbiosis'] == False
        else:
            # --- Ensure the ambient heat sink bus exists ---
            if "Heat amb" not in n.buses.index:
                n.add("Bus", "Heat amb", carrier="Heat", unit="MW")  #

            # 1) Heat rejection to ambient (one-way)
            amb_link = f"{b}_{plant_name}_amb"
            if amb_link not in n.links.index:
                n.add(
                    "Link",
                    amb_link,
                    bus0=local_bus,
                    bus1="Heat amb",
                    efficiency=1.0,
                    p_nom_extendable=True,
                    marginal_cost=0.0,
                )

    return n, new_buses

def add_local_el_connections(n, local_EL_bus, inputs_dict, n_flags, tech_costs, n_config, n_options):
    """Add electricity connections for a plant:
       - connection to the DK1 grid
       - optional connection to El2 bus (symbiosis network)
    """

    # --- get energy prices from external markets
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs, n_options)
    en_market_prices = {k: v.reindex(n.snapshots).ffill() for k, v in en_market_prices.items()}

    # --- Ensure carrier exists ---
    if "AC" not in n.carriers.index:
        n.add("Carrier", "AC")

    # --- Local electricity bus ---
    if local_EL_bus not in n.buses.index:
        n.add("Bus", local_EL_bus, carrier="AC", unit="MW")

    # --- Grid connection link (DK1 → local bus) ---
    link_name1 = f"DK1_to_{local_EL_bus}"
    if link_name1 not in n.links.index:
        cap_cost = tech_costs.at["electricity grid connection", "fixed"]
        if n_config is not None:
            cap_cost *= n_config.at["grid connection", "cost factor"]

        n.add(
            "Link",
            link_name1,
            bus0="ElDK1 bus",
            bus1=local_EL_bus,
            efficiency=1.0,
            capital_cost=float(cap_cost),
            p_nom_extendable=True,
        )

    # --- Assign time-dependent marginal cost  ---
    if "el_grid_price" in en_market_prices:
        mc_series = en_market_prices["el_grid_price"]
        n.links_t.marginal_cost[link_name1] = mc_series

    # --- Optional internal connection to symbiosis network ---
    if n_flags['renewables'] and n_flags['symbiosis']:
        el_bus_symbiosis = 'El3 bus'

        if el_bus_symbiosis not in n.buses.index:
            n.add("Bus", el_bus_symbiosis, carrier="AC", unit="MW")

        link_name2 = f"{el_bus_symbiosis}_to_{local_EL_bus}"
        if link_name2 not in n.links.index:
            n.add(
                "Link",
                link_name2,
                bus0=el_bus_symbiosis,
                bus1=local_EL_bus,
                efficiency=1.0,
                p_nom_extendable=True,
            )

    return n


def add_local_boilers(n, local_EL_bus, local_heat_bus, name,
                      heat_efficiency_plant, tech_costs,
                      inputs_dict, capacity, expansion,
                      capital_cost, n_config, n_options):
    """
    Add local NG and electric boilers for a reference plant requiring heating
    but not connected to the symbiosis network.
    """

    # --- get energy prices from external markets
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs, n_options)
    en_market_prices = {k: v.reindex(n.snapshots).ffill() for k, v in en_market_prices.items()}

    # --- Derived parameters ---
    η_ng = tech_costs.at['gas boiler steam', 'efficiency']
    η_el = tech_costs.at['electric boiler steam', 'efficiency']
    VOM_ng = tech_costs.at['gas boiler steam', 'VOM']
    VOM_el = tech_costs.at['electric boiler steam', 'VOM']

    # --- Reference plant efficiencies and capacities ---
    η_ref = abs(n.links.at[name, heat_efficiency_plant])
    η_ref3 = abs(n.links.at[name, 'efficiency3'])

    capacity_boiler = capacity * η_ref
    p_nom_max_boiler = n_config.at[name, 'max capacity'] * η_ref3

    # --- Natural gas boiler ---
    n.add("Link",
          f"{name}_NG boiler",
          bus0="NG",
          bus1=local_heat_bus,
          efficiency=η_ng,
          p_nom_extendable=expansion,
          p_nom=capacity_boiler / η_ng * 1.005,
          p_nom_max=p_nom_max_boiler / η_ng * 1.005,
          capital_cost=tech_costs.at['gas boiler steam', 'fixed']
                       * n_config.at['NG boiler', 'cost factor']
                       * int(capital_cost > 0),
          marginal_cost = VOM_ng,
          )

    # Add marginal cost time series
    mc = en_market_prices['NG_grid_price'] + VOM_ng
    n.links_t.marginal_cost[f"{name}_NG boiler"] = mc

    # --- Electric boiler ---
    n.add("Link",
          f"{name}_El boiler",
          bus0=local_EL_bus,
          bus1=local_heat_bus,
          efficiency=η_el,
          p_nom_extendable=expansion,
          p_nom=capacity_boiler / η_el * 1.005,
          p_nom_max=p_nom_max_boiler / η_el * 1.005,
          capital_cost=tech_costs.at['electric boiler steam', 'fixed']
                       * n_config.at['El boiler', 'cost factor']
                       * int(capital_cost > 0),
          marginal_cost=VOM_el,
          )

    return n

def add_external_grids(network, inputs_dict, n_options):
    """
    Build external grids and loads according to configuration flags.
    No capital or marginal costs are assigned here.
    """

    # --- store current network state ---
    n0_dict = get_network_status(network)

    # --- electricity demand & supply (DK1) ---
    bus_dict = {
        "bus_list": ["ElDK1 bus"],
        "carrier_list": ["AC"],
        "unit_list": [ "MW"],
    }
    network = add_requirements_buses(network, bus_dict, symbiosis_n)

    el = (
        inputs_dict["El_demand_DK1"].iloc[:, 0].astype(float)
        .reindex(network.snapshots)
        .ffill()
    )
    network.add("Load", "Grid Load", bus="ElDK1 bus")
    network.loads_t.p_set["Grid Load"] = el

    network.add("Generator",
                "Grid gen",
                carrier="AC",
                bus="ElDK1 bus",
                p_nom_extendable=True)

    # --- ambient heat sink store ---
    if "Heat amb" not in network.stores.index:
        bus_dict = {
            "bus_list": ["Heat amb"],
            "carrier_list": ["Heat"],
            "unit_list": ["MW"],
        }
        network = add_requirements_buses(network, bus_dict, symbiosis_n)

        network.add("Store",
                    "Heat amb",
                    bus="Heat amb",
                    e_nom_extendable=True,
                    e_nom_max=float("inf"),
                    e_cyclic=False)

    # --- natural gas grid generator ---
    if "NG grid" not in network.generators.index:
        bus_dict = {
            "bus_list": ["NG"],
            "carrier_list": ["gas"],
            "unit_list": ["MW"],
        }
        network = add_requirements_buses(network, bus_dict, symbiosis_n)

        ng = el.copy()
        ng[:] = inputs_dict['NG_demand_DK'].values
        ng = ng.astype(float).reindex(network.snapshots).ffill()

        network.add("Load", "NG Grid Load", bus="NG")
        network.loads_t.p_set["NG Grid Load"] = ng

        network.add("Generator",
                    "NG grid",
                    bus="NG",
                    carrier="gas",
                    p_nom_extendable=True)


    # --- optional district heating grid ---
    if n_options.at["DH", "enable"]:
        bus_dict = {
            "bus_list": ["DH grid"],
            "carrier_list": ["Heat"],
            "unit_list": ["MW"],
        }
        network = add_requirements_buses(network, bus_dict,symbiosis_n)

        dh = (
            inputs_dict["DH_external_demand"]["DH demand MWh"]
            .astype(float)
            .interpolate("linear")
            .reindex(network.snapshots)
            .ffill()
        )

        if "DH load" not in network.loads.index:
            network.add("Load", "DH load", bus="DH grid")
            network.loads_t.p_set["DH load"] = dh * n_options.at['DH','dh_load_multiplier']

        if "DH gen" not in network.generators.index:
            network.add("Generator",
                        "DH gen",
                        bus="DH grid",
                        carrier="Heat",
                        p_nom_extendable=True)

    # --- record newly added components ---
    new_components = log_new_components(network, n0_dict)
    return network, new_components

def mass_energy_balance_drying(initial_moisture: float = symbiosis_n.at['chips','moisture'],
                               final_moisture: float = symbiosis_n.at['pellets','moisture'], heat_drying: float = 1,
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
def set_plant_connection(n, buses, tech, inputs_dict, n_flags, tech_costs):
    # ----------------------------------------------------------------------
    # creates buses for a specific plant or technolgy.
    # Input:
    #   - buses : df (index = local bus names, cols = technology/plant name, data=bus names in network )
    # ----------------------------------------------------------------------
    # check that buses exists
    bus_dict = {
        "bus_list": [b for b in buses.loc[:, tech].iloc[1:]],
        "carrier_list": [c for c in buses.loc[:, 'carrier'].iloc[1:]],
        "unit_list": [u for u in buses.loc[:, 'unit'].iloc[1:]]
    }
    n = add_requirements_buses(n, bus_dict, symbiosis_n)

    # Add local El ---------------------------------------
    local_EL_bus = buses.at['local EL bus', tech]
    n = add_local_el_connections(n, local_EL_bus, inputs_dict, n_flags, tech_costs, n_config, n_options)

    return n, buses

def add_biomass_drying(
    n,
    tech_costs,
    n_flags,
    n_config,
    final_moisture=None,
    initial_moisture=None,
    local_EL_bus="El_C_heat"
):
    """
    Add a biomass belt dryer and auxiliary processes:
    - dewatering (for digestate fibers)
    - pelletization (for digestate fibers or wood chips)
    """

    # --- moisture defaults ---
    if final_moisture is None:
        final_moisture = symbiosis_n.at['pellets','moisture']
    if initial_moisture is None:
        initial_moisture = symbiosis_n.at['chips','moisture']

    # --- allocation logic ---
    allocation = n_flags.get("central_heat", False)
    dependencies = n_flags.get("symbiosis", False)

    if not (allocation and dependencies):
        return n  # nothing to add

    # --- store current network state ---
    n0_dict = get_network_status(n)

    def add_biomass_belt_dryer_cap_exp(n, prefix, capital_cost, capacity, expansion):
        # required buses
        bus_dict = {
            "bus_list": ["moist biomass", "Heat MT", "pellets"],
            "carrier_list": ["moist biomass", "Heat", "AC", "pellets"],
            "unit_list": ["t/h DM", "MW", "MW", "MW"],
        }

        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        # energy and mass balance of drying process
        heat_drying = tech_costs.at["biomass belt dryer", "heat-input"]  # MW/tH2O
        el_drying = tech_costs.at["biomass belt dryer", "electricity-input"]  # MW/tH2O

        dryer_dict = mass_energy_balance_drying(
            initial_moisture=initial_moisture,
            final_moisture=final_moisture,
            heat_drying=heat_drying,
            el_drying=el_drying,
        )

        n.add(
            "Link",
            prefix + "biomass belt dryer",
            bus0="moist biomass",
            bus1="pellets",
            bus2="Heat MT",
            bus3=local_EL_bus,
            efficiency=symbiosis_n.at['pellets','LHV'] / (1 - symbiosis_n.at['pellets','moisture']),
            efficiency2=-dryer_dict["heat-input"],
            efficiency3=-dryer_dict["electricity-input"],
            p_nom_extendable=expansion,
            p_nom=capacity,
            p_nom_max=n_config.at["biomass belt dryer", "max capacity"],
            capital_cost=capital_cost,
        )

        if "pellets store" not in n.stores.index:
            n.add(
                "Store",
                "pellets store",
                bus="pellets",
                e_nom_extendable=True,
                e_nom_max=float("inf"),
                capital_cost = 5e-6,
                e_cyclic=True,
            )

        return n

    # --- expansion logic ---
    techs = ["biomass belt dryer"]
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)
    t = techs[0]

    if t in cap_to_add:
        capacity = n_config.at["biomass belt dryer", "initial capacity"]
        n = add_biomass_belt_dryer_cap_exp(n, prefix="EXI_", capital_cost=0, capacity=capacity, expansion=False)

    if t in exp_to_add:
        capital_cost = tech_costs.at["biomass belt dryer", "fixed"] * n_config.at["biomass belt dryer", "cost factor"]
        n = add_biomass_belt_dryer_cap_exp(n, prefix="", capital_cost=capital_cost, capacity=0, expansion=True)

    return n

def add_CO2_liquefaction(n, n_flags, inputs_dict, tech_costs, n_config, n_options):
    """
    Add CO₂ liquefaction and storage modules.
    Includes optional sequestration and CO₂ credits.
    """

    # --- activation conditions ---
    allocation = n_flags.get("storage", False)
    dependencies = [n_flags.get("symbiosis", False), n_flags.get("biogas", False)]

    if not (allocation and all(dependencies)):
        return n

    # --- snapshot network state ---
    n0_dict = get_network_status(n)

    # === Helper functions ===
    def add_CO2_liquid_sequestration(n, inputs_dict, n_options, co2_liq_bus):
        if n_options.at['CO2 Liq credits','enable']:
            bus_seq = 'CO2 Liq sequestration'
            bus_dict = {
                "bus_list": [bus_seq, co2_liq_bus],
                "carrier_list": ["CO2"]*2,
                "unit_list": ["t/h"]*2,
            }

            n = add_requirements_buses(n, bus_dict, symbiosis_n)

            # CO2 credits for sequestration of from liquefied CO2
            co2_credits = pd.Series(float(inputs_dict["CO2 cost"]), index=n.snapshots)
            n.add('Link',
                  'CO2 Liq seq',
                  bus0=co2_liq_bus,
                  bus1=bus_seq,
                  efficiency=0.9, # n_options.at['CO2 Liq credits','efficiency'],
                  p_nom_extendable=True,
                  marginal_cost= -1 *  co2_credits,
                  )

            n.add("Store",
                  'CO2 Liq sequestration',
                  bus=bus_seq,
                  e_nom_extendable=True,
                  e_cyclic=False)
        else:
            return n
        return n

    def add_CO2_Liq_storage_cap_exp(n, prefix, capital_cost, capacity, expansion):
        # --- add local buses ---
        co2_bus = "CO2 distribution"
        bust_st = "CO2 Liq storage"

        bus_dict = {
            "bus_list": [co2_bus, bust_st],
            "carrier_list": ["CO2", "CO2"],
            "unit_list": ["t/h", "t/h"],
        }

        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        # --- add local electricity connection ---
        local_EL_bus = "El_CO2_liq"
        n = add_local_el_connections(n, local_EL_bus, inputs_dict, n_flags, tech_costs, n_config, n_options)

        n.add("Store",
              prefix + "CO2 Liq storage",
              bus= bust_st,
              e_nom_extendable=expansion,
              e_nom=capacity,
              e_nom_max=n_config.at["CO2 Liq storage", "max capacity"],
              capital_cost=capital_cost,
              marginal_cost=tech_costs.at["CO2 storage tank", "VOM"],
              e_cyclic=True)

        n.add("Link",
              prefix + "CO2 liquefaction return",
              bus0=bust_st,
              bus1=co2_bus,
              efficiency=1,
              marginal_cost=5e-6,
              p_nom_extendable=expansion,
              p_nom = capacity * (n.snapshots[1].hour -  n.snapshots[0].hour), # ramp limit up and down set to 1
              p_nom_max=n_config.at["CO2 Liq storage", "max capacity"] * (n.snapshots[1].hour -  n.snapshots[0].hour))

        n.add("Link",
              prefix + "CO2 liquefaction",
              bus0=co2_bus,
              bus1=bust_st,
              bus2=local_EL_bus,
              efficiency=1,
              efficiency2= -1 * tech_costs.at["CO2 liquefaction small", "electricity-input"],
              capital_cost=int(capital_cost>0) * tech_costs.at["CO2 liquefaction", "fixed"] * 10e3, #TODO source error: (place holder for  new DEA input)
              p_nom_extendable=expansion,
              marginal_cost=5e-6,
              p_nom=capacity * (n.snapshots[1].hour -  n.snapshots[0].hour), # ramp limit up and down set to 1
              p_nom_max=n_config.at["CO2 Liq storage", "max capacity"] * (n.snapshots[1].hour -  n.snapshots[0].hour))

        #add_CO2_liquid_sequestration(n, inputs_dict, n_options, bust_st )

        return n

    # --- determine capacity additions ---
    techs = ["CO2 Liq storage"]
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

    # === Main additions ===
    t = "CO2 Liq storage"
    if t in cap_to_add:
        capacity = n_config.at[t, "initial capacity"]
        n = add_CO2_Liq_storage_cap_exp(n, prefix="EXI_", capital_cost=0, capacity=capacity, expansion=False)

    if t in exp_to_add:
        capital_cost = tech_costs.at["CO2 storage tank small", "fixed"] * n_config.at[t, "cost factor"]
        n = add_CO2_Liq_storage_cap_exp(n, prefix="", capital_cost=capital_cost, capacity=0, expansion=True)

    return n


def add_compressor_and_storage(n, n_flags, tech_costs, n_config, comp_dict):
    """
    Add general compression and gas storage (cylinders/vessels/baloon) systems.
    Includes heat integration to LT/DH heat networks and auxiliary electric buses.



    return n and comp_dict (updated)
    """

    #comp_dict = {'plant' : plant_name, # ----> '' for centralized H2 compressor
    #            'local EL bus': local_EL_bus,
    #            'Heat DH bus' :local_heat_buses [0],
    #            'Heat LT bus' :local_heat_buses [1],
    #            'in bus' : 'H2 production', # look up in n_symbiosis
    #            'out bus' : 'H2 to MeOH',  # look up in n_symbiosis
    #            'storage bus' : 'H2 storage', # look up in n_symbiosis
    #            'compressor capacity' :   0, # from plant call: compressor initial capacity
    #            'storage capacity' : 0, # from plant call:  storage initial capacity
    #            'compressor expansion' :   0, # from plant call: compressor expansion
    #            'storage expansion' : 0, #H2 from plant call: storage expansion
    # }


    # --- Snapshot network state ---
    n0_dict = get_network_status(n)

    # ==========================================================
    # 1. CHECK AND ADD BUSES
    # ==========================================================
    def en_balance_comp_storage(n, comp_dict):
        # INPUTS:
        # comp_dict = {'plant' : 'cat methanation biogas', # ----> '' for centralized H2 compressor
        #           'local EL bus': methanation_buses.at['local EL bus', 'cat methanation biogas'],
        #           'Heat DH bus' :methanation_buses.at['Heat DH', 'cat methanation biogas'],
        #           'Heat LT bus' :methanation_buses.at['Heat LT', 'cat methanation biogas'],
        #           'IN bus' : methanation_buses.at['CO2 in bus', 'methanation'],
        #           'OUT bus' : methanation_buses.at['CO2 in bus', 'cat methanation biogas'],
        #           'ST bus' : '',
        # }
        from scripts.technology_inputs import symbiosis_n, compressor_calculation

        # Match inputs buses to streams in symbiosis_n
        comp_streams = {}
        if 'IN bus' in comp_dict and comp_dict['IN bus'] in n.buses.index:
            comp_streams['IN stream'] = n.buses.loc[comp_dict['IN bus'], 'properties']

        if 'OUT bus' in comp_dict and comp_dict['OUT bus'] in n.buses.index:
            comp_streams['OUT stream'] = n.buses.loc[comp_dict['OUT bus'], 'properties']

        if 'ST bus' in comp_dict and comp_dict['ST bus'] in n.buses.index:
            comp_streams['ST stream'] = n.buses.loc[comp_dict['ST bus'], 'properties']
            fluid = symbiosis_n.at[n.buses.loc[comp_dict['ST bus'], 'properties'], 'fluid']
            if 'CO2' in fluid:
                comp_streams['ST OUT stream'] = f'{fluid} from HP storage'

        # ------- calculate compressor energy demand --------
        compressor_data = compressor_calculation(comp_streams, symbiosis_n)

        return compressor_data
    # ==========================================================
    # 2. COMPRESSION LINK
    # ==========================================================

    def add_compressor_cap_exp(n, prefix, capital_cost, marginal_cost, capacity, expansion, comp_dict, compressor_data):
        # this function identifies in which of the three modes the compressor + HP storage can be build depending on the process

        # check if main compressor is requested
        if compressor_data.at["electricity-input", 'main compression']:

            n.add("Link",
                  prefix + f"{plant_name} {fluid} compressor",
                  bus0=comp_dict['IN bus'],
                  bus1=comp_dict['OUT bus'],
                  bus2=comp_dict['local EL bus'],
                  bus3=comp_dict['Heat DH bus'],
                  bus4=comp_dict['Heat LT bus'],
                  efficiency=1,
                  efficiency2=-compressor_data.at["electricity-input", 'main compression'],
                  efficiency3=compressor_data.at['heat-output DH', 'main compression'],
                  efficiency4=compressor_data.at["heat-output LT", 'main compression'],
                  p_nom_extendable=expansion,
                  p_nom=capacity,
                  p_nom_max=n_config.at[f"{fluid} compressor", "max capacity"],
                  capital_cost=capital_cost,
                  marginal_cost=marginal_cost)


        return n

    # ==========================================================
    # 2. HIGH-PRESSURE VESSELS / CYLINDERS STORAGE
    # ==========================================================
    def add_HP_storage_aux(n, comp_dict, compressor_data):

        # check if main compressor is requested
        if compressor_data.at["electricity-input", 'storage compression']:

            # --- Charging (compression to storage) ---
            capex_recomp = 0.001 * tech_costs.at["hydrogen storage compressor", "fixed"] * n_config.at["H2 compressor", "cost factor"]
            n.add("Link",
                  f"{plant_name} {fluid} storage send comp",
                  bus0=comp_dict['OUT bus'],
                  bus1=comp_dict['ST bus'],
                  bus2=comp_dict['local EL bus'],
                  bus3=comp_dict['Heat DH bus'],
                  bus4=comp_dict['Heat LT bus'],
                  efficiency=1,
                  efficiency2=-compressor_data.at["electricity-input", 'storage compression'],
                  efficiency3=compressor_data.at['heat-output DH', 'storage compression'],
                  efficiency4=compressor_data.at["heat-output LT", 'storage compression'],
                  p_nom_extendable=True,
                  capital_cost = capex_recomp
                  )

            # --- Discharging (from storage to HP network) ---
            n.add("Link",
                  f"{plant_name} {fluid} storage return",
                  bus0=comp_dict['ST bus'],
                  bus1=comp_dict['OUT bus'],
                  efficiency=1,
                  p_nom_extendable=True,
                  marginal_cost=5e-6)

        return n

    def add_HP_storage_cap_exp(n, prefix, capital_cost, capacity, expansion, comp_dict):

        n.add("Store",
              prefix + f"{plant_name} {fluid} HP storage",
              bus=comp_dict['ST bus'],
              e_nom_extendable=expansion,
              e_nom=capacity,
              e_nom_max=n_config.at[f"{fluid} HP storage", "max capacity"],
              capital_cost=capital_cost,
              e_cyclic=True)

        return n

    # ==========================================================
    # 4. CAPITAL AND MARGINAL COSTS FOR DIFFERENT FLUIDS
    # ==========================================================
    def get_cc_mc_compressor(fluid):
        # gets correct capital and marginal cost for the compressor depending on the fluid

        if fluid == 'Hydrogen' or 'H2':
            capital_cost = tech_costs.at["hydrogen storage compressor", "fixed"] * n_config.at['H2 HP storage', "cost factor"]
            marginal_cost = tech_costs.at["hydrogen storage compressor", "VOM"] * n_config.at['H2 HP storage', "cost factor"]

        elif fluid == 'CarbonDioxide' or 'CO2':
            capital_cost = tech_costs.at["CO2 industrial compressor", "fixed"] * n_config.at[
                "CO2 compressor", "cost factor"]
            marginal_cost = tech_costs.at["CO2 industrial compressor", "VOM"] * n_config.at[
                "CO2 compressor", "cost factor"]

        elif fluid == 'Methane' or 'CH4':
            capital_cost = tech_costs.at['CH4 (g) fill compressor station', 'fixed'] * n_config.at[
                "CH4 compressor", "cost factor"]
            marginal_cost = tech_costs.at['CH4 (g) fill compressor station', "VOM"] * n_config.at[
                "CH4 compressor", "cost factor"]
        else:
            # Unknown fluid type → skip safely
            print(f"⚠️ Skipping {fluid} compressor: unsupported fluid '{fluid}'.")
            return None, None

        return capital_cost, marginal_cost

    def get_cc_mc_hp_storage(fluid):
        """
        Return capital_cost and marginal_cost for HP storage, depending on the fluid.
        Returns (None, None) if the fluid is not recognized (e.g. CH4).
        """

        if fluid in ("Hydrogen", "H2"):
            capital_cost = (
                    tech_costs.at["hydrogen storage tank type 1", "fixed"]
                    * n_config.at["H2 HP storage", "cost factor"]
            )
            marginal_cost = (
                    tech_costs.at["hydrogen storage tank type 1", "VOM"]
                    * n_config.at["H2 HP storage", "cost factor"]
            )

        elif fluid in ("CarbonDioxide", "CO2"):
            capital_cost = (
                    tech_costs.at["CO2 storage cylinders", "fixed"]
                    * n_config.at["CO2 HP storage", "cost factor"]
            )
            marginal_cost = (
                    tech_costs.at["CO2 storage cylinders", "VOM"]
                    * n_config.at["CO2 HP storage", "cost factor"]
            )

        else:
            # Unknown fluid type → skip safely
            print(f"⚠️ Skipping {fluid} HP storage: unsupported fluid '{fluid}'.")
            return None, None

        return capital_cost, marginal_cost

    # ==========================================================
    # 4. Build components
    # ==========================================================
    # add fluid that is compressed top components name
    fluid = symbiosis_n.at[n.buses.at[comp_dict['IN bus'],'properties'], 'fluid'] # note must be compatible with symbiosis_n and n_congif #TODO make it more general

    # --- Centralized {fluid} compressor and HP storage
    if not comp_dict['plant']:
        techs = [f"{fluid} compressor", f"{fluid} HP storage"]

        # check if tech exists already in the model (versus n_config.yaml settings)
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        # sanity check : assign initial capacities for central plants only if they exist in n_config
        capacity = [
            n_config.at[f"{fluid} compressor", 'capacity']
            if (f"{fluid} compressor" in n_config.index and 'capacity' in n_config.columns)
            else 0,
            n_config.at[f"{fluid} HP storage", 'capacity']
            if (f"{fluid} storage" in n_config.index and 'capacity' in n_config.columns)
            else 0
        ]

    # --- Plant-specific components
    else:
        plant_name = comp_dict['plant']
        techs = [f"{plant_name} {fluid} compressor", f"{plant_name} {fluid} HP storage"]

        # check if any initial capacity is requested from the plant_function
        capacity = [comp_dict['compressor capacity'], comp_dict['storage capacity']]

        # check if expansion is requested from the plant_function and if it is allowed by n_config
        if fluid == 'biogas': # HP storage for biogas not available
            n_config.at[f"{fluid} HP storage", 'expansion'] = False

        expansion = [comp_dict['compressor expansion'] * n_config.at[f"{fluid} compressor", 'expansion'],
                     comp_dict['storage expansion'] * n_config.at[f"{fluid} HP storage", 'expansion']]

        # check if the components are already present in the network (sanity check)
        cap_missing = ['EXI_' + t for t in techs if 'EXI_' + t not in {x for k in ('links', 'stores') for x in n0_dict.get(k, [])}]
        exp_missing = [t for t in techs if t not in {x for k in ('links', 'stores') for x in n0_dict.get(k, [])}]

        # Add t if, it has a capacity to be installed from the plant call and if it is missing in the network
        cap_to_add = [t for t, b, m in zip(techs, capacity, cap_missing) if b and m] # Initial capacities to be added

        # Add t if, the capacity of the plant needs to be expanded and if it is allowed to expand the compont, and if it is missing in the network
        exp_to_add = [t for t, c, m in zip(techs, expansion, exp_missing) if
                      m and (c is not None) and (c > 0)]  # capacity expansion to be added

    # --- add compressor ---------------
    t = techs[0]

    # calculate compressor and storage energy balance
    if t in cap_to_add or exp_to_add:
        compressor_data = en_balance_comp_storage(n, comp_dict)

    # add compressor
    capital_cost, marginal_cost = get_cc_mc_compressor(fluid)

    if t in cap_to_add:
        n = add_compressor_cap_exp(n = n, prefix=f"EXI_", capital_cost=0, marginal_cost = marginal_cost, capacity=capacity[0], expansion=False, comp_dict = comp_dict, compressor_data = compressor_data)

    if t in exp_to_add:
        n = add_compressor_cap_exp(n = n, prefix="", capital_cost=capital_cost, marginal_cost=marginal_cost, capacity=0, expansion=True, comp_dict = comp_dict, compressor_data=compressor_data)

    # --- add HP Storage (only H2 and CO2) ---
    if n_flags["storage"]:
        t = techs[1]
        capital_cost, marginal_cost = get_cc_mc_hp_storage(fluid)

        if t in cap_to_add or t in exp_to_add:
            n = add_HP_storage_aux(n, comp_dict, compressor_data)

        if t in cap_to_add:
            n = add_HP_storage_cap_exp(
                n, prefix="EXI_", capital_cost=0, capacity=capacity[1], expansion=False, comp_dict = comp_dict)

        if t in exp_to_add:
            n = add_HP_storage_cap_exp(n, prefix="", capital_cost=capital_cost, capacity=0, expansion=True, comp_dict = comp_dict)

    return n


def add_CO2_compressor_HP_storage(n, n_flags, tech_costs, n_config, CO2_comp_dict):
    """
    Add CO₂ compression and high-pressure storage (cylinders) systems.
    Includes heat integration to LT/DH heat networks and auxiliary electric buses.

    CO2_comp_dict = {'plant' : plant_name,  ----> '' for centralized CO2 compressor
             'local EL bus': local_EL_bus,
             'Heat DH bus' :local_heat_buses [0],
             'Heat LT bus' :local_heat_buses [1],
              'CO2 LP bus' : 'CO2 distribution'
              'CO2 HP bus' : local_CO2_HP,
              'CO2 HP storage bus' : local_CO2_HP_storage
              'CO2 comp capacity' :   # CO2 compressor initial capacity
              'CO2 HP storage capacity' : CO2 HP storage initial capacity
              'CO2 comp expansion' : bool
              'CO2 HP storage expansion' : bool

    CO2_comp_values : float

    return n and CO2_comp_dict (updated)
    """

    # --- Dependencies ---
    dependencies = [n_flags.get("biogas", False)]
    if not all(dependencies):
        return n

    # --- Snapshot network state ---
    n0_dict = get_network_status(n)

    # ==========================================================
    # 1. COMPRESSION LINK
    # ==========================================================

    def add_CO2_compressor_aux(n):
        # add / check for required buses
        bus_dict = {
            "bus_list": [CO2_comp_dict['CO2 LP bus'],
                         CO2_comp_dict['CO2 HP bus'],
                         CO2_comp_dict['local Heat DH bus'],
                         CO2_comp_dict['local Heat LT bus'],
                         CO2_comp_dict['local EL bus']],
            "carrier_list": [ "CO2", "CO2", 'Heat', 'Heat', 'AC' ] ,
            "unit_list": ["t/h", "t/h", 'MW', 'MW', 'MW'],

        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        return n

    def add_CO2_compressor_cap_exp(n, prefix, capital_cost, capacity, expansion):

        n.add("Link",
              prefix + f"{plant_name}CO2 compressor",
              bus0=CO2_comp_dict['CO2 LP bus'],
              bus1=CO2_comp_dict['CO2 HP bus'],
              bus2=CO2_comp_dict['local EL bus'],
              bus3=CO2_comp_dict['local Heat DH bus'],
              bus4=CO2_comp_dict['local Heat LT bus'],
              efficiency=1,
              efficiency2=-tech_costs.at["CO2 industrial compressor", "electricity-input"],
              efficiency3=tech_costs.at["CO2 industrial compressor", "heat output DH"],
              efficiency4=tech_costs.at["CO2 industrial compressor", "heat output LT"],
              p_nom_extendable=expansion,
              p_nom=capacity,
              p_nom_max=n_config.at["CO2 compressor", "max capacity"],
              capital_cost=capital_cost)

        return n

    # ==========================================================
    # 3. HIGH-PRESSURE STORAGE
    # ==========================================================
    def add_CO2_storage_HP_aux(n):
        # --- create local CO2 HP storage bus

        bus_dict = {
            "bus_list": [CO2_comp_dict['CO2 HP storage bus']],
            "carrier_list": ["CO2"] ,
            "unit_list": ["t/h"] ,
            }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        n.add("Link",
              f"{plant_name}CO2 storage send",
              bus0=CO2_comp_dict['CO2 HP bus'],
              bus1=CO2_comp_dict['CO2 HP storage bus'],
              efficiency=1,
              p_nom_extendable=True,
              )

        capex_recomp = 0.0001 * tech_costs.at["CO2 industrial compressor", "fixed"] * n_config.at["CO2 compressor", "cost factor"]
        n.add("Link",
              f"{plant_name}CO2 return extra comp",
              bus0=CO2_comp_dict['CO2 HP storage bus'],
              bus1=CO2_comp_dict['CO2 HP bus'],
              bus2=CO2_comp_dict['local EL bus'],
              bus3=CO2_comp_dict['local Heat DH bus'],
              bus4=CO2_comp_dict['local Heat LT bus'],
              efficiency=1,
              efficiency2 = -tech_costs.at["CO2 industrial compressor", "extra electricity-input"],
              efficiency3 = tech_costs.at["CO2 industrial compressor", "extra heat output DH"],
              efficiency4 = tech_costs.at["CO2 industrial compressor", "extra heat output LT"],
              p_nom_extendable=True,
              capital_cost= capex_recomp,
              )

        return n

    def add_CO2_storage_HP_cap_exp(n, prefix, capital_cost, capacity, expansion):
        n.add("Store",
              prefix + f"{plant_name}CO2 HP storage",
              bus=CO2_comp_dict['CO2 HP storage bus'],
              e_nom_extendable=expansion,
              e_nom=capacity,
              e_nom_max=n_config.at["CO2 HP storage", "max capacity"],
              capital_cost=capital_cost,
              e_cyclic=True,
              )
        return n

    # ==========================================================
    # 4. Build components
    # ==========================================================


    # --- Centralized CO2 compressor and CO2 HP storage
    if not CO2_comp_dict['plant name']:
        techs = ["CO2 compressor", "CO2 HP storage"]

        # check if tech exists already in the model (versus n_config.yaml settings)
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)
        capacity =[n_config.at["CO2 compressor", 'capacity'], n_config.at['CO2 HP storage', 'capacity']]

    # --- Plant-specific components
    else:
        plant_name = CO2_comp_dict['plant name'] + ' '
        techs = [f"{plant_name}CO2 compressor", f"{plant_name}CO2 HP storage"]

        # check if tech exists already in the model (versus CO2_comp_dict and n_config)
        # allows expansion only if the main plant is expanding (CO2_comp_dict), and it is allowed to expand capacity of the single components (n_config)
        capacity = [CO2_comp_dict['CO2 comp capacity'], CO2_comp_dict['CO2 HP storage capacity']]
        expansion = [CO2_comp_dict['CO2 comp expansion'] * n_config.at['CO2 compressor', 'expansion'], CO2_comp_dict['CO2 HP storage expansion'] * n_config.at['CO2 HP storage', 'expansion']]

        cap_to_add =  [a for a, b in zip(techs, [int(c > 0) for c in capacity]) if b]
        exp_to_add =  [a for a, b in zip(techs, expansion) if b]

    # --- add CO2 compressor
    t = techs[0]
    if t in cap_to_add or t in exp_to_add:
        n = add_CO2_compressor_aux(n)

    if t in cap_to_add:
        n = add_CO2_compressor_cap_exp(n, prefix=f"EXI_", capital_cost=0, capacity=capacity[0], expansion=False)

    if t in exp_to_add:
        capital_cost = tech_costs.at["CO2 industrial compressor", "fixed"] * n_config.at["CO2 compressor", "cost factor"]
        n = add_CO2_compressor_cap_exp(n, prefix="", capital_cost=capital_cost, capacity=0, expansion=True)

    # --- add CO2 HP Storage ---
    if n_flags['storage']:
        t = techs[1]
        if t in cap_to_add or t in exp_to_add:
            # add aux components
            CO2_comp_dict['CO2 HP storage bus'] = f"{plant_name}CO2 HP storage"
            n = add_CO2_storage_HP_aux(n)

        if t in cap_to_add:
            n = add_CO2_storage_HP_cap_exp(n, prefix="EXI_", capital_cost=0, capacity=capacity[1], expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at["CO2 storage cylinders", "fixed"] * n_config.at['CO2 HP storage', "cost factor"]
            n = add_CO2_storage_HP_cap_exp(n, prefix="", capital_cost=capital_cost, capacity=0, expansion=True)

    return n, CO2_comp_dict


def add_H2_compressor_HP_storage(n, n_flags, tech_costs, n_config, H2_comp_dict):
    """
    Add CO₂ compression and high-pressure storage (cylinders) systems.
    Includes heat integration to LT/DH heat networks and auxiliary electric buses.

    H2_comp_dict = {'plant' : plant_name,  ----> '' for centralized H2 compressor
             'local EL bus': local_EL_bus,
             'Heat DH bus' :local_heat_buses [0],
             'Heat LT bus' :local_heat_buses [1],
              'H2 LP bus' : 'CO2 distribution'
              'H2 HP bus' : local_H2_HP,
              'H2 storage bus' : local_CO2_HP_storage
              'H2 comp capacity' :   float # H2 compressor initial capacity
              'H2 storage capacity' : float #H2  storage initial capacity

    H2_comp_dict.values : float

    return n and H2_comp_dict (updated)
    """

    # --- Dependencies ---
    dependencies = [n_flags.get("electrolysis", False)]
    if not all(dependencies):
        return n

    # --- Snapshot network state ---
    n0_dict = get_network_status(n)

    # ==========================================================
    # 1. COMPRESSION LINK
    # ==========================================================

    def add_H2_compressor_aux(n):
        # add / check for required buses
        bus_dict = {
            "bus_list": [H2_comp_dict['H2 LP bus'],
                         H2_comp_dict['H2 HP bus'],
                         H2_comp_dict['local Heat DH bus'],
                         H2_comp_dict['local Heat LT bus'],
                         H2_comp_dict['local EL bus']],
            "carrier_list": ["H2", "H2", 'Heat', 'Heat', 'AC'],
            "unit_list": ["MW", "MW", 'MW', 'MW', 'MW'],
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        return n

    def add_H2_compressor_cap_exp(n, prefix, capital_cost, capacity, expansion):

        n.add("Link",
              prefix + f"{plant_name}H2 compressor",
              bus0=H2_comp_dict['H2 LP bus'],
              bus1=H2_comp_dict['H2 HP bus'],
              bus2=H2_comp_dict['local EL bus'],
              bus3=H2_comp_dict['local Heat DH bus'],
              bus4=H2_comp_dict['local Heat LT bus'],
              efficiency=1,
              efficiency2=-tech_costs.at["hydrogen storage compressor", "electricity-input"],
              efficiency3=tech_costs.at["hydrogen storage compressor", "heat output DH"],
              efficiency4=tech_costs.at["hydrogen storage compressor", "heat output LT"],
              p_nom_extendable=expansion,
              p_nom=capacity,
              p_nom_max=n_config.at["H2 compressor", "max capacity"],
              capital_cost=capital_cost,
              marginal_cost=tech_costs.at["hydrogen storage compressor", "VOM"])

        return n

    # ==========================================================
    # 3. HIGH-PRESSURE VESSELS STORAGE
    # ==========================================================
    def add_H2_storage_aux(n):
        # --- create local CO2 HP storage bus
        bus_dict = {
            "bus_list": [H2_comp_dict['H2 storage bus']],
            "carrier_list": ["H2"],
            "unit_list": ["MW"],
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        # --- Discharging (from storage to HP network) ---
        n.add("Link",
              f"{plant_name}H2 return",
              bus0=H2_comp_dict['H2 storage bus'],
              bus1=H2_comp_dict['H2 HP bus'],
              efficiency=1,
              p_nom_extendable=True,
              marginal_cost=5e-6)

        # --- Charging (compression to storage) ---
        capex_recomp = 0.001 * tech_costs.at["hydrogen storage compressor", "fixed"] * n_config.at["H2 compressor", "cost factor"]
        n.add("Link",
              f"{plant_name}H2 storage send extra comp",
              bus0=H2_comp_dict['H2 HP bus'],
              bus1=H2_comp_dict['H2 storage bus'],
              bus2=H2_comp_dict['local EL bus'],
              bus3=H2_comp_dict['local Heat DH bus'],
              bus4=H2_comp_dict['local Heat LT bus'],
              efficiency=1,
              efficiency2=-tech_costs.at["hydrogen storage compressor", "extra electricity-input"],
              efficiency3=tech_costs.at["hydrogen storage compressor", "extra heat output DH"],
              efficiency4=tech_costs.at["hydrogen storage compressor", "extra heat output LT"],
              p_nom_extendable=True,
              capital_cost = capex_recomp
              )

        return n

    def add_H2_storage_cap_exp(n, prefix, capital_cost, capacity, expansion):
        n.add("Store",
              prefix + f"{plant_name}H2 storage",
              bus=H2_comp_dict['H2 storage bus'],
              e_nom_extendable=expansion,
              e_nom=capacity,
              e_nom_max=n_config.at["H2 storage", "max capacity"],
              capital_cost=capital_cost,
              e_cyclic=True)

        return n

    # ==========================================================
    # 4. Build components
    # ==========================================================

    # --- Centralized CO2 compressor and CO2 HP storage
    if not H2_comp_dict['plant name']:
        techs = ["H2 compressor", "H2 storage"]

        # check if tech exists already in the model (versus n_config.yaml settings)
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)
        capacity = [n_config.at["H2 compressor", 'capacity'], n_config.at['H2 storage', 'capacity']]

    # --- Plant-specific components
    else:
        plant_name = H2_comp_dict['plant name'] + ' '
        techs = [f"{plant_name}H2 compressor", f"{plant_name}H2 storage"]

        # check if components should be added (vs CO2_comp_dict)
        capacity = [H2_comp_dict['H2 comp capacity'], H2_comp_dict['H2 storage capacity']]
        expansion = [H2_comp_dict['H2 comp expansion'] * n_config.at['H2 compressor', 'expansion'], H2_comp_dict['H2 storage expansion'] * n_config.at['H2 storage', 'expansion']]


        cap_to_add = [a for a, b in zip(techs, [int(c > 0) for c in capacity]) if b]
        exp_to_add = [a for a, b in zip(techs, expansion) if b]

    # --- add H2 compressor
    t = techs[0]
    if t in cap_to_add or t in exp_to_add:
        n = add_H2_compressor_aux(n)

    if t in cap_to_add:
        n = add_H2_compressor_cap_exp(n, prefix=f"EXI_", capital_cost=0, capacity=capacity[0], expansion=False)

    if t in exp_to_add:
        capital_cost = tech_costs.at["hydrogen storage compressor", "fixed"] * n_config.at[
            "H2 compressor", "cost factor"]
        n = add_H2_compressor_cap_exp(n, prefix="", capital_cost=capital_cost, capacity=0, expansion=True)

    # --- add H2 Storage ---
    if n_flags['storage']:
        t = techs[1]
        if t in cap_to_add or t in exp_to_add:
            # add aux components
            H2_comp_dict['H2 storage bus'] = f"{plant_name} H2 storage"
            n = add_H2_storage_aux(n)

        if t in cap_to_add:
            n = add_H2_storage_cap_exp(n, prefix="EXI_", capital_cost=0, capacity=capacity[1], expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at["hydrogen storage tank type 1", "fixed"] * n_config.at['H2 storage', "cost factor"]
            n = add_H2_storage_cap_exp(n, prefix="", capital_cost=capital_cost, capacity=0, expansion=True)

    return n, H2_comp_dict


def add_battery(n, n_flags, inputs_dict, tech_costs, n_config):
    """
    Add a battery storage system connected to the main or renewable electricity bus.

    Includes inverter and optional capacity expansion.
    """

    # --- Allocation & Dependencies ---
    allocation = n_flags.get("storage", False)
    dependencies = [n_flags.get("renewables", False)]
    if not (allocation and any(dependencies)): # just one between symbiosis and renewable is necessary
        return n

    # --- Snapshot network state ---
    n0_dict = get_network_status(n)


    # ==========================================================
    # 1. ADD BATTERY (STORE + CHARGER/DISCHARGER)
    # ==========================================================
    def add_battery_cap_exp(n, prefix, capital_cost, capacity, expansion):
        """
        Add a battery system with inverter (AC/DC coupling).
        """
        st_bus = "battery"
        local_EL_bus = 'El3 bus'

        # Ensure required buses exist
        bus_dict = {
            "bus_list": [st_bus, local_EL_bus],
            "carrier_list": ["battery", 'AC'],
            "unit_list": ["MW", 'MW']
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        # Add electricity connection
        #local_EL_bus = 'El_battery'
        #n = add_local_el_connections(n, local_EL_bus, inputs_dict, n_flags, tech_costs, n_config, n_options)

        # --- Storage unit ---
        n.add("Store",
              prefix + "battery",
              bus=st_bus,
              e_cyclic=True,
              e_nom_extendable=expansion,
              e_nom=capacity,
              e_nom_max=n_config.at["battery", "max capacity"],
              capital_cost=capital_cost,
              marginal_cost= 5e-6)

        # --- Charging link (AC → DC) ---
        n.add("Link",
              prefix + "battery charger",
              bus0=local_EL_bus,
              bus1=st_bus,
              efficiency=tech_costs.at["battery inverter", "efficiency"],
              p_nom=capacity * n_config.at["battery", "ramp limit up"],
              p_nom_extendable=expansion,
              capital_cost=(tech_costs.at["battery inverter", "fixed"]
                            * n_config.at["battery", "cost factor"]
                            * int(capital_cost > 0)),
              marginal_cost = 5e-6)

        # --- Discharging link (DC → AC) ---
        n.add("Link",
              prefix + "battery discharger",
              bus0=st_bus,
              bus1=local_EL_bus,
              efficiency=tech_costs.at["battery inverter", "efficiency"],
              p_nom=capacity * n_config.at["battery", "ramp limit down"],
              p_nom_extendable=expansion
              )  # inverter cost only on charger side

        return n

    # ==========================================================
    # 2. BUILD COMPONENTS
    # ==========================================================

    techs = ["battery"]
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

    t = "battery"
    if t in cap_to_add:
        capacity = n_config.at[t, "initial capacity"]
        n = add_battery_cap_exp(n=n, prefix="EXI_", capital_cost=0, capacity=capacity, expansion=False)

    if t in exp_to_add:
        capital_cost = tech_costs.at["battery storage", "fixed"] * n_config.at[t, "cost factor"]
        n = add_battery_cap_exp(n=n, prefix="", capital_cost=capital_cost, capacity=0, expansion=True)

    return n


def add_thermal_storage(n, n_flags, inputs_dict, tech_costs, n_config):
    """
    Add thermal energy storage systems:
      - District heating water tank (TES DH)
      - Medium-temperature concrete storage (TES concrete)
    """

    # --- Allocation and dependencies ---
    allocation = n_flags.get("storage", False)
    dependencies = [n_flags.get("symbiosis", False)]

    if not (allocation and all(dependencies)):
        return n

    # --- Snapshot network state ---
    n0_dict = get_network_status(n)

    # --- Determine which techs to add ---
    techs = ["TES DH", "TES concrete"]
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

    # ==========================================================
    # 1. DISTRICT HEATING WATER TANK (TES DH)
    # ==========================================================
    def add_TES_storage_DH_cap_exp(n, prefix, capital_cost, capacity, expansion):
        """
        Add district heating water tank (low-temperature storage).
        """
        heat_bus = "Heat DH"
        bus_dict = {
            "bus_list": ["Heat DH storage", heat_bus],
            "carrier_list": ["Heat", "Heat"],
            "unit_list": ["MW", "MW"],

        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        # --- Storage tank ---
        n.add("Store",
              prefix + "TES DH storage",
              bus="Heat DH storage",
              e_nom_extendable=expansion,
              e_nom=capacity,
              e_nom_min=n_config.at["TES DH", "min capacity"],
              e_nom_max=n_config.at["TES DH", "max capacity"],
              standing_loss=n_config.at["TES DH", "standing loss"],
              e_cyclic=True,
              capital_cost=capital_cost)

        # --- Charging  ---
        n.add("Link",
              prefix + "TES DH charger",
              bus0=heat_bus,
              bus1="Heat DH storage",
              efficiency=1,
              p_nom_extendable=expansion,
              p_nom=capacity * n_config.at["TES DH", "ramp limit up"],
              capital_cost=(tech_costs.at["DH heat exchanger", "fixed"]
                            * n_config.at["DH heat exchanger", "cost factor"]
                            * int(capital_cost > 0)),
              marginal_cost =5e-6)

        # --- Discharging (heat out of tank) ---
        n.add("Link",
              prefix + "TES DH discharger",
              bus0="Heat DH storage",
              bus1=heat_bus,
              efficiency=1,
              p_nom_extendable=expansion,
              p_nom=capacity * n_config.at["TES DH", "ramp limit down"],
              )

        return n

    # ==========================================================
    # 2. MEDIUM-TEMPERATURE CONCRETE STORAGE (TES CONCRETE)
    # ==========================================================
    def add_TES_storage_concrete_cap_exp(n, prefix, capital_cost, capacity, expansion):
        """
        Add medium-temperature concrete storage (e.g. 120–400°C).
        """
        # Add electricity connection
        local_EL_bus = 'El_TES_concrete'
        n = add_local_el_connections(n, local_EL_bus, inputs_dict, n_flags, tech_costs, n_config, n_options)

        # add heat
        heat_bus = "Heat MT"
        TES_bus = "Heat MT storage"
        bus_dict = {
            "bus_list": [TES_bus, heat_bus],
            "carrier_list": ["Heat", "Heat"],
            "unit_list": ["MW", "MW"],

        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        # --- Concrete storage block ---
        n.add("Store",
              prefix + "TES concrete storage",
              bus=TES_bus,
              e_nom_extendable=expansion,
              e_nom=capacity,
              e_nom_min=n_config.at["TES concrete", "min capacity"],
              e_nom_max=n_config.at["TES concrete", "max capacity"],
              standing_loss=n_config.at["TES concrete", "standing loss"],
              e_cyclic=True,
              capital_cost=capital_cost)

        # --- Charging ---
        n.add("Link",
              prefix + "TES concrete charger",
              bus0=local_EL_bus,
              bus1=TES_bus,
              efficiency=1,
              p_nom_extendable=expansion,
              p_nom=capacity * n_config.at["TES concrete", "ramp limit up"],
              capital_cost=tech_costs.at["Concrete-charger", "fixed"]
                            * n_config.at["TES concrete", "cost factor"]
                            * int(capital_cost > 0),
              marginal_cost = 5e-6,
              )

        # --- Discharging ---
        n.add("Link",
              prefix + "TES concrete discharger",
              bus0=TES_bus,
              bus1=heat_bus,
              efficiency=1,
              p_nom_extendable=expansion,
              p_nom=capacity * n_config.at["TES concrete", "ramp limit down"],
              capital_cost=(tech_costs.at["Concrete-discharger", "fixed"]
                            * n_config.at["TES concrete", "cost factor"]
                            * int(capital_cost > 0)),
              )

        return n

    # ==========================================================
    # 3. BUILD COMPONENTS
    # ==========================================================
    # --- TES DH ---
    t = "TES DH"
    if t in cap_to_add:
        capacity = n_config.at[t, "initial capacity"]
        n = add_TES_storage_DH_cap_exp(n, prefix="EXI_", capital_cost=0, capacity=capacity, expansion=False)
    if t in exp_to_add:
        capital_cost = tech_costs.at["central water tank storage", "fixed"] * n_config.at[t, "cost factor"]
        n = add_TES_storage_DH_cap_exp(n, prefix="", capital_cost=capital_cost, capacity=0, expansion=True)

    # --- TES CONCRETE ---
    t = "TES concrete"
    if t in cap_to_add:
        capacity = n_config.at[t, "initial capacity"]
        n = add_TES_storage_concrete_cap_exp(n, prefix="EXI_", capital_cost=0, capacity=capacity, expansion=False)
    if t in exp_to_add:
        capital_cost = tech_costs.at["Concrete-store", "fixed"] * n_config.at[t, "cost factor"]
        n = add_TES_storage_concrete_cap_exp(n, prefix="", capital_cost=capital_cost, capacity=0, expansion=True)

    return n


def add_heat_pump(n, n_flags, inputs_dict, tech_costs):
    """Add an industrial heat pump connecting LT and DH heat networks."""

    # Allocation (who can build it) and dependencies
    allocation = n_flags['symbiosis']
    dependencies = n_flags['symbiosis']

    if allocation and dependencies:

        def add_heat_pump_cap_exp(n, prefix, capital_cost, capacity, expansion):
            # Ensure buses exist
            bus_dict = {
                'bus_list': ['Heat DH', 'Heat LT'],
                'carrier_list': ['Heat', 'Heat'],
                'unit_list': ['MW', 'MW'],

            }
            n = add_requirements_buses(n, bus_dict, symbiosis_n)

            # Add electricity connection
            local_EL_bus = 'El_heat_pump'

            n = add_local_el_connections(n, local_EL_bus, inputs_dict, n_flags, tech_costs, n_config, n_options)

            # Add the heat pump link
            COP = tech_costs.at['industrial heat pump medium temperature', 'efficiency']

            n.add(
                'Link',
                prefix + 'heat pump',
                bus0=local_EL_bus,     # electricity input
                bus1='Heat DH',        # useful heat output
                bus2='Heat LT',        # low-temperature heat source
                efficiency=COP,        # output (DH)
                efficiency2=-(COP - 1),# input (LT), negative because it’s consumed
                capital_cost=capital_cost,
                marginal_cost=tech_costs.at['industrial heat pump medium temperature', 'VOM'],
                p_nom_extendable=expansion,
                p_nom=capacity,
                p_nom_max=n_config.at['heat pump', 'max capacity'],
            )
            return n

        # Snapshot of the current network (for tech_to_add)
        n0_dict = get_network_status(n)

        # Determine whether to add existing or expandable capacity
        techs = ['heat pump']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        t = 'heat pump'

        if t in cap_to_add:
            capacity = n_config.at[t, 'initial capacity']
            n = add_heat_pump_cap_exp(
                n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False
            )

        if t in exp_to_add:
            capital_cost = (
                tech_costs.at['industrial heat pump medium temperature', 'fixed']
                * n_config.at[t, 'cost factor']
            )
            n = add_heat_pump_cap_exp(
                n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True
            )

    return n

# ------- BUILD PYPSA NETWORK MAIN FUNCTIONS-------------
def add_demands(n, n_flags, inputs_dict):
    """Add exogenous energy demands (bioCH4, H2, Methanol) and corresponding delivery/storage links."""

    # Take a snapshot of network state
    n0_dict = get_network_status(n)

    # ---- Helper to process demand series safely ----
    def clean_demand_series(df, network):
        s = df.iloc[:, 0].astype(float)
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        return s.reindex(network.snapshots).fillna(0.0)

    # ---- Import and align demand time series ----
    s_ch4 = clean_demand_series(inputs_dict['bioCH4_demand'], n)
    s_h2  = clean_demand_series(inputs_dict['H2_input_demand'], n)
    s_meoh = clean_demand_series(inputs_dict['Methanol_input_demand'], n)

    # ==============================================================
    # 1. BIOCH4
    # ==============================================================
    if n_flags.get('biogas') or n_flags.get('methanation'):

        bus_dict = {
            'bus_list': ['bioCH4'],
            'carrier_list': ['gas'],
            'unit_list': ['MW'],

        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        # Load representing CH4 demand
        n.add("Load", "bioCH4", bus="bioCH4", carrier="gas")
        n.loads_t.p_set["bioCH4"] = s_ch4

        # Infinite CH4 delivery store
        n.add("Store",
              "bioCH4 delivery",
              bus="bioCH4",
              e_nom_extendable=True,
              e_cyclic=True)

    # ==============================================================
    # 2. HYDROGEN
    # ==============================================================
    if n_flags.get('electrolysis'):

        bus_dict = {
            'bus_list': ['H2', 'H2 delivery'],
            'carrier_list': ['H2', 'H2'],
            'unit_list': ['MW', 'MW'],
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        # H2 demand (grid)
        n.add("Load", "H2 grid", bus="H2 delivery")
        n.loads_t.p_set["H2 grid"] = s_h2

        # Link from production (H2) to delivery (H2 delivery)
        if "H2_to_delivery" not in n.links.index:
            n.add("Link",
                  "H2_to_delivery",
                  bus0="H2",
                  bus1="H2 delivery",
                  efficiency=1.0,
                  p_nom_extendable=True)

        # Infinite delivery storage
        if "H2 delivery" not in n.stores.index:
            n.add("Store",
                  "H2 delivery",
                  bus="H2 delivery",
                  e_nom_extendable=True,
                  e_cyclic=True)

    # ==============================================================
    # 3. METHANOL
    # ==============================================================
    if n_flags.get('meoh'):

        bus_dict = {
            'bus_list': ['Methanol'],
            'carrier_list': ['Methanol'],
            'unit_list': ['MW'],
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        # Methanol production storage (infinite)
        n.add("Store",
              "Methanol prod",
              bus="Methanol",
              e_nom_extendable=True,
              e_nom_max=float("inf"),
              e_cyclic=True)

        # Methanol demand
        n.add("Load", "Methanol", bus="Methanol")
        n.loads_t.p_set["Methanol"] = s_meoh

    # ==============================================================
    # 4. Log newly added components
    # ==============================================================
    new_components = log_new_components(n, n0_dict)

    return n, new_components


# PLAYERS
def add_biogas(n, n_flags, inputs_dict, tech_costs):
    """function that add the biogas plant to the network and all the dependecies if not preset in the network yet"""

    GL_eff = inputs_dict['GL_eff']
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs, n_options)
    en_market_prices = {k: v.reindex(n.snapshots).ffill() for k, v in en_market_prices.items()}

    # take a status of the network before adding components
    n0_dict = get_network_status(n)

    if n_flags['biogas']:

        # ------- add EL connections------------
        local_EL_bus = 'El_biogas'
        n = add_local_el_connections(n, local_EL_bus, inputs_dict, n_flags, tech_costs, n_config, n_options)

        # -----add local heat connections
        plant_name = 'biogas'
        heat_bus_dict = {'Heat MT': -1,
                         'Heat LT': 1}
        n, new_heat_buses = add_local_heat_connections(n, heat_bus_dict, plant_name=plant_name, n_flags=n_flags,
                                                       tech_costs=tech_costs, n_config=n_config)

        # ------- adding functions ------------

        def add_biogas_aux(n):
            bus_dict = {'bus_list': ['Dig biomass', 'Digestate', 'biogas', 'bioCH4'],
                        'carrier_list': ['Dig biomass', 'Digestate', 'gas', 'gas'],
                        'unit_list': ['t/h DM', 't/h DM', 'MW', 'MW'],
                        }
            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict, symbiosis_n)

            # ------- Digestible biomass -------
            n.add(
                "Generator",
                "Dig biomass market",
                bus='Dig biomass',
                carrier='Dig biomass',
                p_nom_extendable=True,
                marginal_cost=n_options.at['Dig biomass' , 'price'] / GL_eff.loc["bioCH4", "SkiveBiogas"],
            )
            n.generators.loc["Dig biomass market", "e_sum_max"] = n_options.at['Dig biomass','max capacity']

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
                        'carrier_list': ['Dig biomass', 'Digestate', 'gas'],
                        'unit_list': ['t/h', 't/h DM', 'MW'],
                        }

            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict, symbiosis_n)

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
            return n

        def add_biogas_storage_exp_cap(n, prefix, capital_cost, capacity, expansion):
            bus_dict = {'bus_list': ['biogas'],
                        'carrier_list': ['gas'],
                        'unit_list': ['MW'],
                        }
            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict, symbiosis_n)

            n.add('Store',
                  name = prefix + 'biogas store',
                  bus='biogas',
                  e_nom_extendable=expansion,
                  e_nom = capacity,
                  capital_cost=capital_cost,
                  e_nom_max=n_config.at['biogas storage','max capacity'],
                  e_cyclic=True)
            return n

        def add_biogas_upgrading_aux (n):

            bus_dict = {'bus_list': ['CO2 sep', 'CO2 pure atm'],
                        'carrier_list': ['CO2', 'CO2'],
                        'unit_list': ['t/h', 't/h'],
                        }

            n = add_requirements_buses(n, bus_dict, symbiosis_n)

            # -----------infinite Store of biogenic CO2 (venting to ATM)
            n.add("Store",
                  "CO2 biogenic out",
                  bus="CO2 pure atm",
                  e_nom_extendable=True,
                  e_nom_max=float("inf"),
                  e_cyclic=False,
                  )

            n.add("Link",
                  "CO2 sep to atm",
                  bus0="CO2 sep",
                  bus1="CO2 pure atm",
                  efficiency=1,
                  p_nom_extendable=True)

            return n

        def add_biogas_upgrading_exp_cap(n, prefix, capital_cost, capacity, expansion):
            bus_dict = {'bus_list': ['NG', 'CO2 sep', 'biogas', 'bioCH4'],
                        'carrier_list': ['gas', 'CO2', 'gas', 'gas'],
                        'unit_list': ['MW', 't/h', 'MW', 'MW'],
                        }
            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict, symbiosis_n)

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

            # enables  NG boiler to supply heat to the symbiosis network
            if n_flags['symbiosis'] and capacity:
                p_min_pu_val = -1
            else:
                p_min_pu_val = 0

            name_lk = prefix + 'biogas upgrading' + '_' + "NG boiler"
            n.add("Link",
                  name = name_lk,
                  bus0="NG",
                  bus1=new_heat_buses[0],
                  efficiency=tech_costs.at['gas boiler steam', 'efficiency'],
                  p_nom_extendable=expansion,
                  p_nom = capacity_boiler,
                  p_nom_max = p_nom_max_boiler,
                  p_min_pu = p_min_pu_val,
                  capital_cost= tech_costs.at['gas boiler steam', 'fixed'] * n_config.at['NG boiler','cost factor'] * int(capital_cost > 0),
                  )
            n.links_t.marginal_cost.loc[:, name_lk] = en_market_prices["NG_grid_price"]

            return n

        def add_dewatering_cap_exp(n, prefix, capital_cost, capacity, expansion):

            # Required buses
            bus_dict = {'bus_list': ['Digestate', 'moist biomass'],
                        'carrier_list': ['Digestate', 'moist biomass'],
                        'unit_list': ['t/h DM', 't/h DM'],
                        }

            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict, symbiosis_n)

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
            return n

        # ------- Check techs to add ------------
        techs = ['biogas','biogas storage', 'biogas upgrading', 'dewatering']
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

        # add biogas plant
        t = 'biogas'
        if t in cap_to_add + exp_to_add:
            n = add_biogas_aux(n)

        if t in cap_to_add:
            capacity = n_config.at[t, 'initial capacity'] / GL_eff.loc["bioCH4", "SkiveBiogas"]
            n = add_biogas_exp_cap(n= n, prefix = 'EXI_', capital_cost = 0, capacity = capacity, expansion= False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['biogas', 'fixed'] / GL_eff.loc["bioCH4", "SkiveBiogas"] * n_config.at[t,'cost factor']
            n = add_biogas_exp_cap(n=n, prefix = '', capital_cost=capital_cost, capacity = 0, expansion=True)

        # Add biogas storage
        t = 'biogas storage'
        if t in cap_to_add:
            capacity = n_config.at[t, 'initial capacity']
            n = add_biogas_storage_exp_cap(n= n, prefix = 'EXI_', capital_cost = 0, capacity= capacity, expansion= False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['biogas storage', 'fixed'] * n_config.at[t,'cost factor']
            n = add_biogas_storage_exp_cap(n=n, prefix = '', capital_cost=capital_cost, capacity=0, expansion=True)

        # Biogas upgrading
        t = 'biogas upgrading'
        if t in cap_to_add + exp_to_add:
            n = add_biogas_upgrading_aux(n)

        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            n = add_biogas_upgrading_exp_cap(n= n, prefix = 'EXI_', capital_cost = 0, capacity= capacity, expansion= False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['biogas upgrading', 'fixed'] * n_config.at[t,'cost factor']
            n = add_biogas_upgrading_exp_cap(n=n, prefix = '', capital_cost=capital_cost, capacity=0, expansion=True)

        # dewatering of digestate fibers
        t = 'dewatering'
        if t in cap_to_add:
            capacity = n_config.at['dewatering', 'initial capacity']
            n = add_dewatering_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at['centrifugal dewatering', "fixed"] * n_config.at['dewatering', 'cost factor']
            n = add_dewatering_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True)

        # log new components
        new_components = log_new_components(n, n0_dict)

    else:
        keylist = ['links', 'generators', 'loads', 'stores', 'buses']
        new_components = {key: [] for key in keylist}

    return n, new_components

def add_renewables(n, n_flags, inputs_dict, tech_costs):
    """Add renewable generation (wind and PV) and grid connection to the network."""

    # Retrieve time series and reindex to match network snapshots
    CF_wind = inputs_dict["CF_wind"].reindex(n.snapshots).astype(float)
    CF_solar = inputs_dict["CF_solar"].reindex(n.snapshots).astype(float)

    # Market prices (incl. CO2 adjustment)
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs, n_options)
    en_market_prices = {k: v.reindex(n.snapshots).ffill() for k, v in en_market_prices.items()}

    # Snapshot of network before adding new components
    n0_dict = get_network_status(n)

    if not n_flags.get('renewables', False):
        empty = {k: [] for k in ['links', 'generators', 'loads', 'stores', 'buses']}
        return n, empty

    # ----------------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------------

    def add_grid_connection_cap_exp(n, name, capital_cost, capacity, expansion):
        bus_dict = {'bus_list': ['El3 bus', 'ElDK1 bus'],
                    'carrier_list': ['AC', 'AC'],
                    'unit_list': ['MW', 'MW']}
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        n.add("Link",
              name=name,
              bus0="El3 bus",
              bus1="ElDK1 bus",
              efficiency=1,
              p_nom_extendable=expansion,
              p_nom=capacity,
              p_nom_max=n_config.at['grid connection', 'max capacity'],
              capital_cost=capital_cost)
        n.links_t.marginal_cost[name] = en_market_prices["el_grid_price"]
        return n

    def add_onwind_cap_exp(n, prefix, capital_cost, capacity, expansion):
        n = add_requirements_buses(n, {
            'bus_list': ['El3 bus'],
            'carrier_list': ['AC'],
            'unit_list': ['MW']
        }, symbiosis_n)
        n.add("Carrier", "wind")

        name = f"{prefix}onshorewind"
        n.add("Generator",
              name=name,
              bus="El3 bus",
              carrier="wind",
              p_nom_max=n_config.at['onwind', 'max capacity'],
              p_nom_extendable=expansion,
              p_nom=capacity,
              capital_cost=capital_cost,
              marginal_cost=tech_costs.at['onwind', 'VOM'],
              p_max_pu=CF_wind["CF wind"])
        n.generators_t.p_max_pu[name] = CF_wind
        return n

    def add_solar_cap_exp(n, prefix, capital_cost, capacity, expansion):
        n = add_requirements_buses(n, {
            'bus_list': ['El3 bus'],
            'carrier_list': ['AC'],
            'unit_list': ['MW']
        }, symbiosis_n)
        n.add("Carrier", "solar")

        name = f"{prefix}solar"
        n.add("Generator",
              name=name,
              bus="El3 bus",
              carrier="solar",
              p_nom_max=n_config.at['solar', 'max capacity'],
              p_nom_extendable=expansion,
              p_nom=capacity,
              capital_cost=capital_cost,
              marginal_cost=tech_costs.at['solar', 'VOM'])
        n.generators_t.p_max_pu[name] = CF_solar
        return n

    # ----------------------------------------------------------------------
    # Add technologies
    # ----------------------------------------------------------------------

    techs = ['onwind', 'solar', 'grid connection']
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

    # Onshore wind
    if 'onwind' in cap_to_add:
        cap = n_config.at['onwind', 'initial capacity']
        n = add_onwind_cap_exp(n, 'EXI_', 0, cap, False)
    if 'onwind' in exp_to_add:
        cost = tech_costs.at['onwind', 'fixed'] * n_config.at['onwind', 'cost factor']
        n = add_onwind_cap_exp(n, '', cost, 0, True)

    # Solar PV
    if 'solar' in cap_to_add:
        cap = n_config.at['solar', 'initial capacity']
        n = add_solar_cap_exp(n, 'EXI_', 0, cap, False)
    if 'solar' in exp_to_add:
        cost = tech_costs.at['solar', 'fixed'] * n_config.at['solar', 'cost factor']
        n = add_solar_cap_exp(n, '', cost, 0, True)

    # Grid connection
    if 'grid connection' in cap_to_add:
        cap = n_config.at['grid connection', 'initial capacity']
        n = add_grid_connection_cap_exp(n, 'EXI_El3_to_DK1', 0, cap, False)
    if 'grid connection' in exp_to_add:
        cost = tech_costs.at['electricity grid connection', 'fixed'] * n_config.at['grid connection', 'cost factor']
        n = add_grid_connection_cap_exp(n, 'El3_to_DK1', cost, 0, True)


    # ----------------------------------------------------------------------
    new_components = log_new_components(n, n0_dict)
    return n, new_components

def add_electrolysis(n, n_flags, inputs_dict, tech_costs):
    """Add electrolysis system (H2 production) to the network."""

    GL_eff = inputs_dict['GL_eff']
    H2_input_demand = inputs_dict['H2_input_demand']
    n0_dict = get_network_status(n)

    if not n_flags.get('electrolysis', False):
        empty = {k: [] for k in ['links', 'generators', 'loads', 'stores', 'buses']}
        return n, empty

    # ---------- Add RFNBOs constraint: grid ↔ local electricity
    n = add_requirements_buses(n, {
        'bus_list': ['ElDK1 bus', 'El3 bus'],
        'carrier_list': ['AC', 'AC'],
        'unit_list': ['MW', 'MW']
    }, symbiosis_n)

    n = add_link_El_grid_to_H2(n, inputs_dict, tech_costs)

    # ---------- Add local heat connections
    plant_name = 'electrolysis'
    heat_bus_dict = {'Heat LT': 1}
    n, new_heat_buses = add_local_heat_connections(n, heat_bus_dict, plant_name=plant_name, n_flags=n_flags,
                                                   tech_costs=tech_costs, n_config=n_config)

    # ---------- Choose CAPEX depending on H2 demand
    if H2_input_demand.iloc[:, 0].sum() > 0:
        electrolysis_cost = tech_costs.at['electrolysis', 'fixed'] * n_config.at['electrolysis', 'cost factor']
    else:
        electrolysis_cost = tech_costs.at['electrolysis small', 'fixed'] * n_config.at['electrolysis', 'cost factor']

    # ---------- Electrolyzer component builder
    def add_H2_cap_exp(n, prefix, capital_cost, capacity, expansion):
        n = add_requirements_buses(n, {
            'bus_list': ['El3 bus', 'H2'],
            'carrier_list': ['AC', 'H2'],
            'unit_list': ['MW', 'MW'],
        }, symbiosis_n)

        name = f"{prefix}electrolysis"

        n.add("Link",
              name=name,
              bus0="El3 bus",
              bus1="H2",
              bus2=new_heat_buses[0],  # Heat LT
              efficiency=GL_eff.at['H2', 'GreenHyScale'],
              efficiency2=GL_eff.at['Heat LT', 'GreenHyScale'],
              capital_cost=capital_cost,
              p_nom_extendable=expansion,
              p_nom=capacity,
              p_nom_max=n_config.at['electrolysis', 'max capacity'],
              p_min_pu0= n_config.at['electrolysis', 'min load'],
              ramp_limit_up0 = n_config.at['electrolysis', 'ramp limit up'],
              ramp_limit_down0 = n_config.at['electrolysis', 'ramp limit down']
              )

        return n

    # ---------- Determine what to add
    techs = ['electrolysis']
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)
    t = 'electrolysis'

    if t in cap_to_add:
        cap = n_config.at[t, 'initial capacity']
        n = add_H2_cap_exp(n, 'EXI_', 0, cap, False)
    if t in exp_to_add:
        n = add_H2_cap_exp(n, '', electrolysis_cost, 0, True)

    new_components = log_new_components(n, n0_dict)
    return n, new_components

def add_meoh(n, n_flags, inputs_dict, tech_costs):
    """
    Add methanol synthesis system and required auxiliary units.
    Methanol system can include its own electrolyzer but requires CO2 source
    (from biogas/symbiosis network) for operation.
    """

    # sanity check
    if not n_flags.get("meoh", False):
        empty = {k: [] for k in ["links", "generators", "loads", "stores", "buses"]}
        return n, empty

    GL_eff = inputs_dict['GL_eff']

    n0_dict = get_network_status(n)

    # ----------------------------------------------------------------------
    # Methanol synthesis reactor
    # ----------------------------------------------------------------------
    def add_methanolisation_cap_exp(n, prefix, capital_cost, capacity, expansion, meoh_buses):
        # update methanation_buses
        meoh_buses.at['H2 in bus', 'methanolisation'] = 'H2 to methanolisation'
        meoh_buses.at['product bus', 'methanolisation'] = meoh_buses.at['product bus', 'meoh']
        meoh_buses.at['CO2 in bus', 'methanolisation'] = 'CO2 to methanolisation'
        meoh_buses.at['local EL bus', 'methanolisation'] = meoh_buses.at['local EL bus', 'meoh']
        meoh_buses.at['CO2 storage bus', 'methanolisation'] = meoh_buses.at['CO2 storage bus', 'meoh']
        meoh_buses.at['H2 storage bus', 'methanolisation'] = meoh_buses.at['H2 storage bus', 'meoh']

        # check that the buses are actually existing
        n, meoh_buses = set_plant_connection(n, buses = meoh_buses , tech ='methanolisation', inputs_dict =inputs_dict, n_flags =n_flags, tech_costs=tech_costs)

        # add Heat MT bus
        meth_heat_directions = {'Heat MT': -1,
                                'Heat DH': 1,  # for compressor
                                'Heat LT': 1}  # for compressor
        n, new_heat_buses = add_local_heat_connections(n, meth_heat_directions, 'methanolisation', n_flags,
                                                       tech_costs, n_config)

        # update methanation_buses (could be different among plants)
        meoh_buses.loc['Heat MT', 'meoh'] = new_heat_buses[0]
        meoh_buses.loc['Heat DH', 'meoh'] = new_heat_buses[1]
        meoh_buses.loc['Heat LT', 'meoh'] = new_heat_buses[2]
        meoh_buses.loc['Heat MT', 'methanolisation'] = new_heat_buses[0]
        meoh_buses.loc['Heat DH', 'methanolisation'] = new_heat_buses[1]
        meoh_buses.loc['Heat LT', 'methanolisation'] = new_heat_buses[2]

        # ----------------------------------------------------------------------
        # Add Methanolisation plant
        # ----------------------------------------------------------------------
        name = f"{prefix}methanolisation"
        n.add(
            "Link",
            name=name,
            bus0=meoh_buses.at['CO2 in bus', 'methanolisation'],#meoh_comp_dict['CO2 HP bus'],
            bus1=meoh_buses.at['product bus', 'methanolisation'],
            bus2=meoh_buses.at['H2 in bus', 'methanolisation'],
            bus3=meoh_buses.at['local EL bus', 'methanolisation'],
            bus4=meoh_buses.at['Heat MT', 'methanolisation'],
            bus5=meoh_buses.at['Heat DH', 'methanolisation'],
            efficiency=GL_eff.loc["Methanol", "Methanol plant"],
            efficiency2=GL_eff.loc["H2", "Methanol plant"],
            efficiency3=GL_eff.loc["El2 bus", "Methanol plant"],
            efficiency4=GL_eff.at["Heat MT", "Methanol plant"],
            efficiency5=GL_eff.at["Heat DH", "Methanol plant"],
            p_nom_extendable=expansion,
            p_nom=capacity,
            p_nom_max=n_config.at["methanolisation", "max capacity"],
            capital_cost=capital_cost,
            p_min_pu0=n_config.at["methanolisation", "min load"],
            ramp_limit_up0 = n_config.at['methanolisation', 'ramp limit up'],
            ramp_limit_down0 = n_config.at['methanolisation', 'ramp limit down'],
            )


        # Call compressor for CO2 w/ storage
        methanolisation_compCO2 = {'plant' : 'methanolisation',
                   'local EL bus': meoh_buses.at['local EL bus', 'methanolisation'],
                   'Heat DH bus' :meoh_buses.at['Heat DH', 'methanolisation'],
                   'Heat LT bus' :meoh_buses.at['Heat LT', 'methanolisation'],
                   'IN bus' : meoh_buses.at['CO2 in bus', 'meoh'],
                   'OUT bus' : meoh_buses.at['CO2 in bus', 'methanolisation'],
                   'ST bus' : meoh_buses.at['CO2 storage bus', 'methanolisation'],
                   'compressor capacity' :   capacity,
                   'storage capacity' : 0,
                   'compressor expansion' :   expansion,
                   'storage expansion' : expansion,
        }
        n = add_compressor_and_storage(n, n_flags, tech_costs, n_config, methanolisation_compCO2)

        # Call storage for H2
        methanolisation_compH2 = {'plant' : 'methanolisation',
                   'local EL bus': meoh_buses.at['local EL bus', 'methanolisation'],
                   'Heat DH bus' :meoh_buses.at['Heat DH', 'methanolisation'],
                   'Heat LT bus' :meoh_buses.at['Heat LT', 'methanolisation'],
                   'IN bus' : meoh_buses.at['H2 in bus', 'meoh'],
                   'OUT bus' : meoh_buses.at['H2 in bus', 'methanolisation'],
                   'ST bus' : meoh_buses.at['H2 storage bus', 'methanolisation'],
                   'compressor capacity' :  capacity * GL_eff.loc["H2", "Methanol plant"],
                   'storage capacity' : 0,
                   'compressor expansion' :  expansion,
                   'storage expansion' : expansion,
        }

        n = add_compressor_and_storage(n, n_flags, tech_costs, n_config, methanolisation_compH2)

        # ----------------------------------------------------------------------
        # Add local boilers (El, and NG) if central heat not available
        # ----------------------------------------------------------------------
        if not n_flags.get("central_heat", False):
            add_local_boilers(
                n=n,
                local_EL_bus=meoh_buses.at['local EL bus', 'methanolisation'],
                local_heat_bus=meoh_buses.at['Heat MT', 'methanolisation'],
                name=name,
                heat_efficiency_plant="efficiency4",
                tech_costs=tech_costs,
                inputs_dict=inputs_dict,
                capacity=capacity,
                expansion=expansion,
                capital_cost=capital_cost,
                n_config=n_config,
                n_options = n_options
                )

        return n, meoh_buses

    # ----------------------------------------------------------------------
    # Add plant depending on tech status
    # ----------------------------------------------------------------------

    # check what technologies to add
    techs = ["methanolisation"]
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

    if cap_to_add or exp_to_add:
        # initialize meoh_buses with existing bus
        idx = ['local EL bus', 'CO2 in bus', 'H2 in bus', 'product bus']
        carriers= ['AC',"CO2", "H2", "Methanol"]
        units = ['MW', "t/h", "MW", "MW"]
        buse_names = ['EL_meoh', 'CO2 distribution', 'H2 distribution', 'Methanol']
        meoh_buses = pd.DataFrame(index =idx, columns=['meoh'] + techs, data = ''  )
        meoh_buses.loc[:,'meoh'] = buse_names
        meoh_buses.loc[:,'carrier'] = carriers
        meoh_buses.loc[:,'unit'] = units

        # add H2 and CO2 storage bus for all methanol production techs if allowed
        if n_config.at['H2 HP storage', 'expansion']:
            meoh_buses.at['H2 storage bus', 'meoh'] = 'meoh H2 HP storage'
            meoh_buses.at['H2 storage bus', 'carrier'] = 'H2'
            meoh_buses.at['H2 storage bus', 'unit'] = 'MW'
        else:
            meoh_buses.at['H2 storage bus', 'meoh'] = ''
            meoh_buses.at['H2 storage bus', 'carrier'] = ''
            meoh_buses.at['H2 storage bus', 'unit'] = ''

        if n_config.at['CO2 HP storage', 'expansion']:
            meoh_buses.at['CO2 storage bus', 'meoh'] = 'meoh CO2 HP storage'
            meoh_buses.at['CO2 storage bus', 'carrier'] = 'CO2'
            meoh_buses.at['CO2 storage bus', 'unit'] = 't/h'
        else:
            meoh_buses.at['CO2 storage bus', 'meoh'] = ''
            meoh_buses.at['CO2 storage bus', 'carrier'] = ''
            meoh_buses.at['CO2 storage bus', 'unit'] = ''

        n, meoh_buses = set_plant_connection(n, buses = meoh_buses , tech ='meoh', inputs_dict =inputs_dict, n_flags =n_flags, tech_costs=tech_costs)

    else:
        empty = {k: [] for k in ['links', 'generators', 'loads', 'stores', 'buses']}
        return n, empty

    # ----------------------------------------------------------------------
    # Add technologies
    # ----------------------------------------------------------------------

    t = 'methanolisation'

    if t in cap_to_add:
        cap = n_config.at[t, "initial capacity"]
        n, meoh_buses = add_methanolisation_cap_exp(n, "EXI_", 0, cap, False, meoh_buses)

    if t in exp_to_add:
        cost = tech_costs.at["methanolisation", "fixed"] * n_config.at["methanolisation", "cost factor"]
        n, meoh_buses = add_methanolisation_cap_exp(n, "", cost, 0, True, meoh_buses)

    new_components = log_new_components(n, n0_dict)

    return n, new_components


def add_methanation(n, n_flags, inputs_dict, tech_costs):
    """
    Add methanation facilities (biological and catalytic) to the network.
    Methanation can use biogas or CO2 as carbon source and requires H2.
    """

    n0_dict = get_network_status(n)

    if not n_flags.get("methanation", False):
        empty = {k: [] for k in ["links", "generators", "loads", "stores", "buses"]}
        return n, empty

    # ----------------------------------------------------------------------
    # BIOLOGICAL METHANATION (biogas)
    # ----------------------------------------------------------------------
    def add_biomethanation_biogas_cap_exp(n, prefix, capital_cost, capacity, expansion, methanation_buses):

        # update methanation_buses
        methanation_buses.at['H2 in bus', 'biomethanation biogas'] = methanation_buses.at['H2 in bus', 'methanation']
        methanation_buses.at['product bus', 'biomethanation biogas'] = methanation_buses.at['product bus', 'methanation']
        methanation_buses.at['biogas in bus', 'biomethanation biogas'] = methanation_buses.at['biogas in bus', 'methanation']
        methanation_buses.at['local EL bus', 'biomethanation biogas'] = methanation_buses.at['local EL bus', 'methanation']
        methanation_buses.at['H2 storage bus', 'biomethanation biogas'] = methanation_buses.at['H2 storage bus', 'methanation']


        # check that the buses are actually existing
        n, methanation_buses = set_plant_connection(n, buses = methanation_buses , tech ='biomethanation biogas', inputs_dict =inputs_dict, n_flags =n_flags, tech_costs=tech_costs)

        # add Heat  bus
        meth_heat_directions = {'Heat DH': 1,  # for compressor
                                'Heat LT': 1}  # for compressor

        n, new_heat_buses = add_local_heat_connections(n, meth_heat_directions, 'methanation', n_flags,
                                                       tech_costs, n_config)
        # update methanation_buses
        methanation_buses.at['Heat DH', 'methanation'] = new_heat_buses[0]
        methanation_buses.at['Heat DH', 'biomethanation biogas'] = new_heat_buses[0]
        methanation_buses.at['Heat LT', 'methanation'] = new_heat_buses[1]
        methanation_buses.at['Heat LT', 'biomethanation biogas'] = new_heat_buses[1]

        name = f"{prefix}biomethanation biogas"
        n.add(
            "Link",
            name,
            bus0=methanation_buses.at['H2 in bus', 'biomethanation biogas'],
            bus1=methanation_buses.at['product bus', 'biomethanation biogas'],
            bus2=methanation_buses.at['biogas in bus', 'biomethanation biogas'],
            bus3=methanation_buses.at['local EL bus', 'biomethanation biogas'],
            efficiency=tech_costs.at["biomethanation", "Methane Output"],
            efficiency2=-tech_costs.at["biomethanation", "Biogas Input"],
            efficiency3=-tech_costs.at["biomethanation", "electricity input"],
            p_nom=capacity,
            p_nom_extendable=expansion,
            p_nom_max=n_config.at["biomethanation biogas", "max capacity"],
            p_min_pu0=n_config.at["biomethanation biogas", "min load"],
            capital_cost=capital_cost,
            marginal_cost=tech_costs.at["biomethanation", "VOM"],
            ramp_limit_up0 = n_config.at['biomethanation biogas', 'ramp limit up'],
            ramp_limit_down0 = n_config.at['biomethanation biogas', 'ramp limit down'],
            )


        # Call storage for H2
        bio_meth_comp_H2 = {'plant' : 'methanation', # ----> '' for centralized compressor
                   'local EL bus': methanation_buses.at['local EL bus', 'biomethanation biogas'],
                   'Heat DH bus' :methanation_buses.at['Heat DH', 'biomethanation biogas'],
                   'Heat LT bus' :methanation_buses.at['Heat LT', 'biomethanation biogas'],
                   'IN bus' : methanation_buses.at['H2 in bus', 'methanation'],
                   'OUT bus' : methanation_buses.at['H2 in bus', 'methanation'],
                   'ST bus' : methanation_buses.at['H2 storage bus', 'methanation'],
                   'compressor capacity' : 0,
                   'storage capacity' : 0,
                   'compressor expansion' :  expansion,
                   'storage expansion' : expansion,
                            }

        n = add_compressor_and_storage(n, n_flags, tech_costs, n_config, bio_meth_comp_H2)


        return n, methanation_buses

    # ----------------------------------------------------------------------
    # BIOLOGICAL METHANATION (CO2)
    # ----------------------------------------------------------------------
    def add_biomethanation_CO2_cap_exp(n, prefix, capital_cost, capacity, expansion, methanation_buses):

        # update methanation_buses
        methanation_buses.at['H2 in bus', 'biomethanation CO2'] = methanation_buses.at['H2 in bus', 'methanation']
        methanation_buses.at['product bus', 'biomethanation CO2'] = methanation_buses.at['product bus', 'methanation']
        methanation_buses.at['CO2 in bus', 'biomethanation CO2'] = methanation_buses.at['CO2 in bus', 'methanation']
        methanation_buses.at['local EL bus', 'biomethanation CO2'] = methanation_buses.at['local EL bus', 'methanation']
        methanation_buses.at['CO2 storage bus', 'biomethanation CO2'] = methanation_buses.at['CO2 storage bus', 'methanation']
        methanation_buses.at['H2 storage bus', 'biomethanation CO2'] = methanation_buses.at['H2 storage bus', 'methanation']

        # check that the buses are actually existing
        n, methanation_buses = set_plant_connection(n, buses = methanation_buses , tech ='biomethanation CO2', inputs_dict =inputs_dict, n_flags =n_flags, tech_costs=tech_costs)

        # add Heat  bus
        meth_heat_directions = {'Heat DH': 1,  # for compressor
                                'Heat LT': 1}  # for compressor

        n, new_heat_buses = add_local_heat_connections(n, meth_heat_directions, 'methanation', n_flags,
                                                       tech_costs, n_config)
        # update methanation_buses
        methanation_buses.at['Heat DH', 'methanation'] = new_heat_buses[0]
        methanation_buses.at['Heat DH', 'biomethanation CO2'] = new_heat_buses[0]
        methanation_buses.at['Heat LT', 'methanation'] = new_heat_buses[1]
        methanation_buses.at['Heat LT', 'biomethanation CO2'] = new_heat_buses[1]

        fl1= n.buses.loc[methanation_buses.at['product bus', 'biomethanation CO2'],'properties']
        fl2 = n.buses.loc[methanation_buses.at['CO2 in bus', 'biomethanation CO2'], 'properties']
        fl3 = n.buses.loc[methanation_buses.at['H2 in bus', 'biomethanation CO2'], 'properties']

        lhv_biomethane = symbiosis_n.at[fl1,'LHV']
        lhv_h2 = symbiosis_n.at[fl3,'LHV']
        density_biomethane = CP.PropsSI("D", "T", symbiosis_n.at[fl1,'T'] + 273, "P", symbiosis_n.at[fl1,'P'] *1e5, symbiosis_n.at['bioCH4','fluid'])
        density_co2 = CP.PropsSI("D", "T", symbiosis_n.at[fl2,'T'] + 273, "P", symbiosis_n.at[fl2,'P'] *1e5, symbiosis_n.at[fl2,'fluid'])
        density_h2 = CP.PropsSI("D", "T", symbiosis_n.at[fl3,'T'] + 273, "P", symbiosis_n.at[fl3,'P'] *1e5, symbiosis_n.at[fl2,'fluid'])

        v_ch4_v_co2 = (tech_costs.at["biomethanation", "Biogas Input"] / lhv_biomethane / density_biomethane) / (tech_costs.at["biomethanation", "CO2 Input"] / density_co2)
        v_h2 = 1 / lhv_h2 * 1e3 / density_h2
        v_co2 = tech_costs.at["biomethanation", "CO2 Input"] / density_co2 * 1e3
        v_ch4 = v_co2 * v_ch4_v_co2
        vol_ratio = (v_h2 + v_co2) / (v_h2 + v_co2 + v_ch4)

        name = f"{prefix}biomethanation CO2"

        n.add(
            "Link",
            name,
            bus0=methanation_buses.at['H2 in bus', 'biomethanation CO2'],
            bus1=methanation_buses.at['product bus', 'biomethanation CO2'],
            bus2=methanation_buses.at['CO2 in bus', 'biomethanation CO2'],
            bus3=methanation_buses.at['local EL bus', 'biomethanation CO2'],
            efficiency=tech_costs.at["biomethanation", "Methane Output"]
            - tech_costs.at["biomethanation", "Biogas Input"],
            efficiency2=-tech_costs.at["biomethanation", "CO2 Input"],
            efficiency3=-tech_costs.at["biomethanation", "electricity input"],
            p_nom_extendable=expansion,
            p_nom=capacity,
            p_nom_max=n_config.at["biomethanation CO2", "max capacity"],
            p_min_pu0=n_config.at["biomethanation CO2", "min load"],
            capital_cost=capital_cost * vol_ratio,
            marginal_cost=tech_costs.at["biomethanation", "VOM"] * vol_ratio,
            ramp_limit_up0 = n_config.at['biomethanation CO2', 'ramp limit up'],
            ramp_limit_down0 = n_config.at['biomethanation CO2', 'ramp limit down'],
            )

        # Call compressor for CO2 w/ storage
        bio_meth_compCO2 = {'plant' : 'biomethanation CO2', # ----> '' for centralized compressor
                   'local EL bus': methanation_buses.at['local EL bus', 'biomethanation CO2'],
                   'Heat DH bus' :methanation_buses.at['Heat DH', 'biomethanation CO2'],
                   'Heat LT bus' :methanation_buses.at['Heat LT', 'biomethanation CO2'],
                   'IN bus' : methanation_buses.at['CO2 in bus', 'methanation'],
                   'OUT bus' : methanation_buses.at['CO2 in bus', 'biomethanation CO2'],
                   'ST bus' : methanation_buses.at['CO2 storage bus', 'methanation'],
                   'compressor capacity' :  capacity * tech_costs.at["biomethanation", "CO2 Input"] ,
                   'storage capacity' : 0,
                   'compressor expansion' :   expansion,
                   'storage expansion' : expansion,
        }

        n = add_compressor_and_storage(n, n_flags, tech_costs, n_config, bio_meth_compCO2)

        # Call storage for H2
        bio_meth_comp_H2 = {'plant' : 'methanation', # ----> '' for centralized compressor
                   'local EL bus': methanation_buses.at['local EL bus', 'biomethanation CO2'],
                   'Heat DH bus' :methanation_buses.at['Heat DH', 'biomethanation CO2'],
                   'Heat LT bus' :methanation_buses.at['Heat LT', 'biomethanation CO2'],
                   'IN bus' : methanation_buses.at['H2 in bus', 'methanation'],
                   'OUT bus' : methanation_buses.at['H2 in bus', 'methanation'],
                   'ST bus' : methanation_buses.at['H2 storage bus', 'methanation'],
                   'compressor capacity' :  0 ,
                   'storage capacity' : 0,
                   'compressor expansion' :  expansion,
                   'storage expansion' : expansion,
        }
        n = add_compressor_and_storage(n, n_flags, tech_costs, n_config, bio_meth_comp_H2)

        return n, methanation_buses

    # ----------------------------------------------------------------------
    # CATALYTIC METHANATION (biogas)
    # ----------------------------------------------------------------------
    def add_cat_methanation_biogas_cap_exp(n, prefix, capital_cost, capacity, expansion, methanation_buses):

        # update methanation_buses
        methanation_buses.at['H2 in bus', 'cat methanation biogas'] = methanation_buses.at['H2 in bus', 'methanation']
        methanation_buses.at['product bus', 'cat methanation biogas'] = methanation_buses.at['product bus', 'methanation']
        methanation_buses.at['biogas in bus', 'cat methanation biogas'] = 'biogas to cat methanation'
        methanation_buses.at['local EL bus', 'cat methanation biogas'] = methanation_buses.at['local EL bus', 'methanation']
        methanation_buses.at['H2 storage bus', 'cat methanation biogas'] = methanation_buses.at['H2 storage bus', 'methanation']

        # check that the buses are actually existing
        n, methanation_buses = set_plant_connection(n, buses = methanation_buses , tech ='cat methanation biogas', inputs_dict =inputs_dict, n_flags =n_flags, tech_costs=tech_costs)

        # add Heat MT bus
        meth_heat_directions = {'Heat MT': 1,
                                'Heat DH': 1, # for compressor
                                'Heat LT': 1} # for compressor
        n, new_heat_buses = add_local_heat_connections(n, meth_heat_directions, 'methanation', n_flags,
                                                       tech_costs, n_config)
        # update methanation_buses
        methanation_buses.at['Heat MT', 'methanation']= new_heat_buses[0]
        methanation_buses.at['Heat MT', 'cat methanation biogas'] = new_heat_buses[0]
        methanation_buses.at['Heat DH', 'methanation']= new_heat_buses[1]
        methanation_buses.at['Heat DH', 'cat methanation biogas'] = new_heat_buses[1]
        methanation_buses.at['Heat LT', 'methanation']= new_heat_buses[2]
        methanation_buses.at['Heat LT', 'cat methanation biogas'] = new_heat_buses[2]

        # add plant
        name = f"{prefix}cat methanation biogas"

        n.add(
            "Link",
            name,
            bus0=methanation_buses.at['H2 in bus', 'cat methanation biogas'],
            bus1=methanation_buses.at['product bus', 'cat methanation biogas'],
            bus2=methanation_buses.at['biogas in bus', 'cat methanation biogas'],
            bus3=methanation_buses.at['local EL bus', 'cat methanation biogas'],
            bus4=methanation_buses.at['Heat MT', 'cat methanation biogas'],
            efficiency=tech_costs.at["biogas plus hydrogen", "Methane Output"],
            efficiency2=-tech_costs.at["biogas plus hydrogen", "Biogas Input"],
            efficiency3=-tech_costs.at["biogas plus hydrogen", "electricity input"],
            efficiency4=tech_costs.at["biogas plus hydrogen", "heat output"],
            p_nom=capacity,
            p_nom_extendable=expansion,
            p_nom_max=n_config.at["cat methanation biogas", "max capacity"],
            p_min_pu0=n_config.at["cat methanation biogas", "min load"],
            capital_cost=capital_cost,
            marginal_cost=tech_costs.at["biogas plus hydrogen", "VOM"],
            ramp_limit_up0 = n_config.at['cat methanation biogas', 'ramp limit up'],
            ramp_limit_down0 = n_config.at['cat methanation biogas', 'ramp limit down']
            )

        # Call compressor for biogas
        cat_meth_comp_bg = {'plant' : 'cat methanation biogas', # ----> '' for centralized H2 compressor
                   'local EL bus': methanation_buses.at['local EL bus', 'cat methanation biogas'],
                   'Heat DH bus' :methanation_buses.at['Heat DH', 'cat methanation biogas'],
                   'Heat LT bus' :methanation_buses.at['Heat LT', 'cat methanation biogas'],
                   'IN bus' : methanation_buses.at['biogas in bus', 'methanation'],
                   'OUT bus' : methanation_buses.at['biogas in bus', 'cat methanation biogas'],
                   'storage bus' : '',
                   'compressor capacity' :   capacity * tech_costs.at["biogas plus hydrogen", "Biogas Input"] ,
                   'storage capacity' : 0,
                   'compressor expansion' :   expansion,
                   'storage expansion' : 0,
        }
        n = add_compressor_and_storage(n, n_flags, tech_costs, n_config, cat_meth_comp_bg)


        # Call storage for H2
        cat_meth_comp_H2 = {'plant' : 'methanation', # ----> '' for centralized compressor
                   'local EL bus': methanation_buses.at['local EL bus', 'cat methanation biogas'],
                   'Heat DH bus' :methanation_buses.at['Heat DH', 'cat methanation biogas'],
                   'Heat LT bus' :methanation_buses.at['Heat LT', 'cat methanation biogas'],
                   'IN bus' : methanation_buses.at['H2 in bus', 'methanation'],
                   'OUT bus' : methanation_buses.at['H2 in bus', 'methanation'],
                   'ST bus' : methanation_buses.at['H2 storage bus', 'methanation'],
                   'compressor capacity' :  0 ,
                   'storage capacity' : 0,
                   'compressor expansion' :  expansion,
                   'storage expansion' : expansion,
        }
        n = add_compressor_and_storage(n, n_flags, tech_costs, n_config, cat_meth_comp_H2)

        return n, methanation_buses

    # ----------------------------------------------------------------------
    # CATALYTIC METHANATION (CO2)
    # ----------------------------------------------------------------------
    def add_cat_methanation_CO2_cap_exp(n, prefix, capital_cost, capacity, expansion, methanation_buses):

        # update methanation_buses
        methanation_buses.at['H2 in bus', 'cat methanation CO2'] = methanation_buses.at['H2 in bus', 'methanation']
        methanation_buses.at['product bus', 'cat methanation CO2'] = methanation_buses.at['product bus', 'methanation']
        methanation_buses.at['CO2 in bus', 'cat methanation CO2'] = 'CO2 to cat methanation'
        methanation_buses.at['local EL bus', 'cat methanation CO2'] = methanation_buses.at['local EL bus', 'methanation']
        methanation_buses.at['CO2 storage bus', 'cat methanation CO2'] = methanation_buses.at['CO2 storage bus', 'methanation']
        methanation_buses.at['H2 storage bus', 'cat methanation CO2'] = methanation_buses.at['H2 storage bus', 'methanation']

        # check that the buses are actually existing
        n, methanation_buses = set_plant_connection(n, buses = methanation_buses , tech ='cat methanation CO2', inputs_dict =inputs_dict, n_flags =n_flags, tech_costs=tech_costs)

        # add Heat MT bus
        meth_heat_directions = {'Heat MT': 1,
                                'Heat DH': 1,  # for compressor
                                'Heat LT': 1}  # for compressor
        n, new_heat_buses = add_local_heat_connections(n, meth_heat_directions, 'methanation', n_flags,
                                                       tech_costs, n_config)

        # update methanation_buses (could be different among plants)
        methanation_buses.at['Heat MT', 'methanation'] = new_heat_buses[0]
        methanation_buses.at['Heat DH', 'methanation'] = new_heat_buses[1]
        methanation_buses.at['Heat LT', 'methanation'] = new_heat_buses[2]
        methanation_buses.at['Heat MT', 'cat methanation CO2'] = new_heat_buses[0]
        methanation_buses.at['Heat DH', 'cat methanation CO2'] = new_heat_buses[1]
        methanation_buses.at['Heat LT', 'cat methanation CO2'] = new_heat_buses[2]

        # adjust volume flows from biogas cat methanation
        fl1 = n.buses.loc["bioCH4", 'properties']
        fl2 = n.buses.loc['CO2 distribution', 'properties']
        fl3 = n.buses.loc['H2 distribution', 'properties']

        lhv_biomethane = symbiosis_n.at[fl1, 'LHV']
        lhv_h2 = symbiosis_n.at[fl3, 'LHV']
        density_biomethane = CP.PropsSI("D", "T", symbiosis_n.at[fl1, 'T'] + 273, "P", symbiosis_n.at[fl1, 'P'] * 1e5,
                                        symbiosis_n.at['bioCH4', 'fluid'])
        density_co2 = CP.PropsSI("D", "T", symbiosis_n.at[fl2, 'T'] + 273, "P", symbiosis_n.at[fl2, 'P'] * 1e5,
                                 symbiosis_n.at[fl2, 'fluid'])
        density_h2 = CP.PropsSI("D", "T", symbiosis_n.at[fl3, 'T'] + 273, "P", symbiosis_n.at[fl3, 'P'] * 1e5,
                                symbiosis_n.at[fl2, 'fluid'])

        v_ch4_v_co2 = (
            tech_costs.at["biogas plus hydrogen", "Biogas Input"] / lhv_biomethane / density_biomethane
        ) / (tech_costs.at["biogas plus hydrogen", "CO2 Input"] / density_co2)
        v_h2 = 1 / lhv_h2 * 1e3 / density_h2
        v_co2 = tech_costs.at["biogas plus hydrogen", "CO2 Input"] / density_co2 * 1e3
        v_ch4 = v_co2 * v_ch4_v_co2
        vol_ratio = (v_h2 + v_co2) / (v_h2 + v_co2 + v_ch4)

        # add plant
        name = f"{prefix}cat methanation CO2"

        n.add(
            "Link",
            name,
            bus0=methanation_buses.at['H2 in bus', 'cat methanation CO2'],
            bus1=methanation_buses.at['product bus', 'cat methanation CO2'],
            bus2=methanation_buses.at['CO2 in bus', 'cat methanation CO2'],
            bus3=methanation_buses.at['local EL bus', 'cat methanation CO2'],
            bus4=methanation_buses.at['Heat MT', 'cat methanation CO2'],
            efficiency=tech_costs.at["biogas plus hydrogen", "Methane Output"]
            - tech_costs.at["biogas plus hydrogen", "Biogas Input"],
            efficiency2=-tech_costs.at["biogas plus hydrogen", "CO2 Input"],
            efficiency3=-tech_costs.at["biogas plus hydrogen", "electricity input"],
            efficiency4=tech_costs.at["biogas plus hydrogen", "heat output"],
            p_nom_extendable=expansion,
            p_nom=capacity,
            p_min_pu0=n_config.at["cat methanation CO2", "min load"],
            capital_cost=capital_cost * vol_ratio,
            marginal_cost=tech_costs.at["biogas plus hydrogen", "VOM"] * vol_ratio,
            ramp_limit_up0 = n_config.at['cat methanation CO2', 'ramp limit up'],
            ramp_limit_down0 = n_config.at['cat methanation CO2', 'ramp limit down']
            )

        # Call compressor for CO2 w/ storage
        cat_meth_compCO2 = {'plant' : 'cat methanation CO2', # ----> '' for centralized compressor
                   'local EL bus': methanation_buses.at['local EL bus', 'cat methanation CO2'],
                   'Heat DH bus' :methanation_buses.at['Heat DH', 'cat methanation CO2'],
                   'Heat LT bus' :methanation_buses.at['Heat LT', 'cat methanation CO2'],
                   'IN bus' : methanation_buses.at['CO2 in bus', 'methanation'],
                   'OUT bus' : methanation_buses.at['CO2 in bus', 'cat methanation CO2'],
                   'ST bus' : methanation_buses.at['CO2 storage bus', 'methanation'],
                   'compressor capacity' :   capacity * tech_costs.at["biogas plus hydrogen", "CO2 Input"] ,
                   'storage capacity' : 0,
                   'compressor expansion' :   expansion,
                   'storage expansion' : expansion,
        }

        n = add_compressor_and_storage(n, n_flags, tech_costs, n_config, cat_meth_compCO2)

        # Call storage for H2
        cat_meth_comp_H2 = {'plant' : 'methanation', # ----> '' for centralized compressor
                   'local EL bus': methanation_buses.at['local EL bus', 'cat methanation CO2'],
                   'Heat DH bus' :methanation_buses.at['Heat DH', 'cat methanation CO2'],
                   'Heat LT bus' :methanation_buses.at['Heat LT', 'cat methanation CO2'],
                   'IN bus' : methanation_buses.at['H2 in bus', 'methanation'],
                   'OUT bus' : methanation_buses.at['H2 in bus', 'methanation'],
                   'ST bus' : methanation_buses.at['H2 storage bus', 'methanation'],
                   'compressor capacity' :  0 ,
                   'storage capacity' : 0,
                   'compressor expansion' :  expansion,
                   'storage expansion' : expansion,
        }
        n = add_compressor_and_storage(n, n_flags, tech_costs, n_config, cat_meth_comp_H2)

        return n, methanation_buses

    # ----------------------------------------------------------------------
    # Add technologies
    # ----------------------------------------------------------------------

    # check what technologies to add
    techs = [
        "biomethanation biogas",
        "biomethanation CO2",
        "cat methanation biogas",
        "cat methanation CO2",
    ]
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

    if cap_to_add or exp_to_add:
        # initialize methanation_buses with existing bus
        idx = ['local EL bus','CO2 in bus', 'H2 in bus', 'biogas in bus', 'product bus']
        carriers= ['AC',"CO2", "H2", "gas", "gas"]
        units = ['MW', "t/h", "MW", "MW", "MW"]
        buses_methanation = ['EL_methanation', 'CO2 distribution', 'H2 distribution', 'biogas', 'bioCH4']
        methanation_buses = pd.DataFrame(index =idx, columns=['methanation'] + techs, data = ''  )
        methanation_buses.loc[:,'methanation'] = buses_methanation
        methanation_buses.loc[:,'carrier'] = carriers
        methanation_buses.loc[:,'unit'] = units
        # add H2 and CO2 storage bus for all methanation techs if allowed

        if n_config.at['H2 HP storage', 'expansion']:
            methanation_buses.at['H2 storage bus', 'methanation'] = 'methanation H2 HP storage'
            methanation_buses.at['H2 storage bus', 'carrier'] = 'H2'
            methanation_buses.at['H2 storage bus', 'unit'] = 'MW'
        else:
            methanation_buses.at['H2 storage bus', 'methanation'] = ''
            methanation_buses.at['H2 storage bus', 'carrier'] = ''
            methanation_buses.at['H2 storage bus', 'unit'] = ''
        if n_config.at['CO2 HP storage', 'expansion']:
            methanation_buses.at['CO2 storage bus', 'methanation'] = 'methanation CO2 HP storage'
            methanation_buses.at['CO2 storage bus', 'carrier'] = 'CO2'
            methanation_buses.at['CO2 storage bus', 'unit'] = 't/h'
        else:
            methanation_buses.at['CO2 storage bus', 'methanation'] = ''
            methanation_buses.at['CO2 storage bus', 'carrier'] = ''
            methanation_buses.at['CO2 storage bus', 'unit'] = ''

        n, methanation_buses = set_plant_connection(n, buses = methanation_buses , tech ='methanation', inputs_dict =inputs_dict, n_flags =n_flags, tech_costs=tech_costs)

    else:
        empty = {k: [] for k in ['links', 'generators', 'loads', 'stores', 'buses']}
        return n, empty

    # add each technology with initial capacity or expansion
    for t, add_fn in [
        ("biomethanation biogas", add_biomethanation_biogas_cap_exp),
        ("biomethanation CO2", add_biomethanation_CO2_cap_exp),
        ("cat methanation biogas", add_cat_methanation_biogas_cap_exp),
        ("cat methanation CO2", add_cat_methanation_CO2_cap_exp),
    ]:

        if t in cap_to_add:
            cap = n_config.at[t, "initial capacity"]
            n, methanation_buses = add_fn(n, "EXI_", 0, cap, False, methanation_buses)
        if t in exp_to_add:
            cost = (tech_costs.at["biogas plus hydrogen", "fixed"]
                if "cat" in t
                else tech_costs.at["biomethanation", "fixed"]) * n_config.at[t, "cost factor"]
            n, methanation_buses = add_fn(n, "", cost, 0, True, methanation_buses)

    # add gas (CO2 and H2) storages and compressors

    #if n_flags['storage']:
    #    add_compressors_storage_CO2_H2_methanation(n)

    new_components = log_new_components(n, n0_dict)
    return n, new_components


def add_central_heat_MT(n, n_flags, inputs_dict, tech_costs):
    """Add central heating technologies (biomass, gas, electric, pyrolysis) to the Heat MT bus."""

    n0_dict = get_network_status(n)

    # --- get energy prices from external markets
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs, n_options)
    en_market_prices = {k: v.reindex(n.snapshots).ffill() for k, v in en_market_prices.items()}

    # Core fuel and grid buses
    bus_dict = {
        "bus_list": ["pellets", "ElDK1 bus", "NG"],
        "carrier_list": ["pellets", "AC", "NG"],
        "unit_list": ["MW", "MW", "MW"],
    }

    if not n_flags.get("central_heat", False):
        empty = {k: [] for k in ["links", "generators", "loads", "stores", "buses"]}
        return n, empty

    n = add_requirements_buses(n, bus_dict, symbiosis_n)

    # Local electricity hub
    local_EL_bus = "El_C_heat"
    n = add_local_el_connections(n, local_EL_bus, inputs_dict, n_flags, tech_costs, n_config, n_options)

    # Add Heat MT bus (if symbiosis network active)
    if n_flags.get("symbiosis", False):
        if "Heat MT" not in n.buses.index:
            n.add("Bus", "Heat MT", carrier="Heat", unit="MW")

    # ---------------------------------------------------------
    # Pellet market
    # ---------------------------------------------------------
    if n_options.at["pellets market", "enable"]:
        n = add_requirements_buses(
            n, {"bus_list": ["pellets"], "carrier_list": ["pellets"], "unit_list": ["MW"]}
        , symbiosis_n)

        n.add(
            "Generator",
            "pellets market",
            bus="pellets",
            carrier="pellets",
            p_nom_extendable=True,
            marginal_cost=n_options.at["pellets market", "price"],
        )
        n.generators.loc["pellets market", "e_sum_max"] = n_options.at[
            "pellets market", "max capacity"
        ]

    # ---------------------------------------------------------
    # Moist biomass market
    # ---------------------------------------------------------
    if n_options.at["moist biomass market", "enable"]:
        n = add_requirements_buses(
            n, {"bus_list": ["moist biomass"], "carrier_list": ["moist biomass"], "unit_list": ["MW"]}
        , symbiosis_n)
        n.add("Carrier", "moist biomass")
        n.add(
            "Generator",
            "moist biomass market",
            bus="moist biomass",
            carrier="moist biomass",
            p_nom_extendable=True,
            marginal_cost=n_options.at["moist biomass market", "price"],
        )
        n.generators.loc["moist biomass market", "e_sum_max"] = n_options.at[
            "moist biomass market", "max capacity"
        ]

    # ---------------------------------------------------------
    # Biomass drying
    # ---------------------------------------------------------
    n = add_biomass_drying(n, tech_costs, n_flags, n_config, local_EL_bus=local_EL_bus)

    # ---------------------------------------------------------
    # Biochar pyrolysis
    # ---------------------------------------------------------
    def add_pyrolysis_aux(n):
        bus_dict = {
            "bus_list": ["biochar", "biochar sequestration"],
            "carrier_list": ["CO2", "CO2"],
            "unit_list": ["t/h", "t/h"],
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        co2_credits = pd.Series(float(inputs_dict["CO2 cost"]), index=n.snapshots)
        n.add(
            "Link",
            "biochar sequestration",
            bus0="biochar",
            bus1="biochar sequestration",
            efficiency=1,
            p_nom_extendable = True,
            marginal_cost=-1 * n_options.at["biochar credits", "enable"] * co2_credits,
        )

        n.add(
            "Store",
            "biochar sequestred",
            bus="biochar sequestration",
            e_nom_extendable=True,
            e_nom_max=float("inf"),
            e_cyclic=False,
        )
        return n

    def add_pyrolysis_cap_exp(n, prefix, capital_cost, capacity, expansion):
        bus_dict = {
            "bus_list": ["pellets", "biochar"],
            "carrier_list": ["pellets", "CO2"],
            "unit_list": ["MW", "t/h"],
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        n.add(
            "Link",
            prefix + "pyrolysis",
            bus0="pellets",
            bus1="Heat MT",
            bus2=local_EL_bus,
            bus3="biochar",
            efficiency=tech_costs.at["biochar pyrolysis", "heat output"]
            / tech_costs.at["biochar pyrolysis", "biomass input"],
            efficiency2=-tech_costs.at["biochar pyrolysis", "electricity input"]
            / tech_costs.at["biochar pyrolysis", "biomass input"],
            efficiency3=1 / tech_costs.at["biochar pyrolysis", "biomass input"],
            marginal_cost=tech_costs.at["biomass HOP", "VOM"]
            / tech_costs.at["biochar pyrolysis", "biomass input"],
            p_nom_extendable=expansion,
            p_nom=capacity,
            p_nom_max=n_config.at["pyrolysis", "max capacity"],
            capital_cost=capital_cost,
        )
        return n

    techs = ["pyrolysis"]
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

    if "pyrolysis" in cap_to_add or "pyrolysis" in exp_to_add:
        n = add_pyrolysis_aux(n)
    if "pyrolysis" in cap_to_add:
        n = add_pyrolysis_cap_exp(n, "EXI_", 0, n_config.at["pyrolysis", "initial capacity"], False)
    if "pyrolysis" in exp_to_add:
        cost = (
            tech_costs.at["biochar pyrolysis", "fixed"]
            / tech_costs.at["biochar pyrolysis", "biomass input"]
            * n_config.at["pyrolysis", "cost factor"]
        )
        n = add_pyrolysis_cap_exp(n, "", cost, 0, True)

    # ---------------------------------------------------------
    # Biomass boiler (pellets)
    # ---------------------------------------------------------
    def add_C_biomass_boiler_cap_exp(n, prefix, capital_cost, capacity, expansion):
        n = add_requirements_buses(
            n,
            {"bus_list": ["pellets", "Heat MT"], "carrier_list": ["pellets", "Heat"], "unit_list": ["MW", "MW"]}, symbiosis_n)
        n.add(
            "Link",
            prefix + "pellets boiler",
            bus0="pellets",
            bus1="Heat MT",
            efficiency=tech_costs.at["biomass HOP", "efficiency"] * symbiosis_n.at[n.buses.loc['pellets', 'properties'], 'LHV'],
            marginal_cost=tech_costs.at["biomass HOP", "VOM"],
            p_nom_extendable=expansion,
            p_nom=capacity,
            p_nom_max=n_config.at["biomass boiler", "max capacity"],
            capital_cost=capital_cost,
        )
        return n

    techs = ["biomass boiler"]
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)
    if "biomass boiler" in cap_to_add:
        n = add_C_biomass_boiler_cap_exp(n, "EXI_", 0, n_config.at["biomass boiler", "initial capacity"], False)
    if "biomass boiler" in exp_to_add:
        cost = tech_costs.at["biomass HOP", "fixed"] * n_config.at["biomass boiler", "cost factor"]
        n = add_C_biomass_boiler_cap_exp(n, "", cost, 0, True)

    # ---------------------------------------------------------
    # NG boiler
    # ---------------------------------------------------------
    def add_C_NG_boiler_cap_exp(n, prefix, capital_cost, capacity, expansion):
        n = add_requirements_buses(
            n, {"bus_list": ["NG", "Heat MT"], "carrier_list": ["gas", "Heat"], "unit_list": ["MW", "MW"]}, symbiosis_n
        )
        n.add(
            "Link",
            prefix + "NG boiler",
            bus0="NG",
            bus1="Heat MT",
            efficiency=tech_costs.at["central gas boiler", "efficiency"],
            p_nom_extendable=expansion,
            p_nom=capacity,
            p_nom_max=n_config.at["NG boiler", "max capacity"],
            capital_cost=capital_cost,
            marginal_cost=en_market_prices["NG_grid_price"]
            + tech_costs.at["gas boiler steam", "VOM"],
        )
        return n

    techs = ["NG boiler"]
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)
    if "NG boiler" in cap_to_add:
        n = add_C_NG_boiler_cap_exp(n, "EXI_", 0, n_config.at["NG boiler", "initial capacity"], False)
    if "NG boiler" in exp_to_add:
        cost = tech_costs.at["central gas boiler", "fixed"] * n_config.at["NG boiler", "cost factor"]
        n = add_C_NG_boiler_cap_exp(n, "", cost, 0, True)

    # ---------------------------------------------------------
    # Electric boiler
    # ---------------------------------------------------------
    def add_C_El_boiler_cap_exp(n, prefix, capital_cost, capacity, expansion):
        n = add_requirements_buses(
            n, {"bus_list": [local_EL_bus, "Heat MT"], "carrier_list": ["AC", "Heat"], "unit_list": ["MW", "MW"]}, symbiosis_n
        )
        n.add(
            "Link",
            prefix + "El boiler",
            bus0=local_EL_bus,
            bus1="Heat MT",
            efficiency=tech_costs.at["electric boiler steam", "efficiency"],
            marginal_cost=tech_costs.at["electric boiler steam", "VOM"],
            p_nom_extendable=expansion,
            p_nom=capacity,
            p_nom_max=n_config.at["El boiler", "max capacity"],
            capital_cost=capital_cost,
        )
        return n

    techs = ["El boiler"]
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)
    if "El boiler" in cap_to_add:
        n = add_C_El_boiler_cap_exp(n, "EXI_", 0, n_config.at["El boiler", "initial capacity"], False)
    if "El boiler" in exp_to_add:
        cost = tech_costs.at["electric boiler steam", "fixed"] * n_config.at["El boiler", "cost factor"]
        n = add_C_El_boiler_cap_exp(n, "", cost, 0, True)

    # ---------------------------------------------------------
    # Log additions
    # ---------------------------------------------------------
    new_components = log_new_components(n, n0_dict)
    return n, new_components


def add_storage(n, n_flags, inputs_dict, tech_costs):
    """
    Add all storage-related technologies to the network:
      - Battery storage
      - Thermal storage
      - CO2 liquefaction and storage
    """

    # Take a snapshot of current network state
    n0_dict = get_network_status(n)

    if not n_flags.get("storage", False):
        empty = {k: [] for k in ["links", "generators", "loads", "stores", "buses"]}
        return n, empty

    # ---------------------------------------------------------
    # Add storage systems (ensure n is updated if functions return it)
    # ---------------------------------------------------------
    for add_func in [add_battery, add_thermal_storage, add_CO2_liquefaction ]:
        result = add_func(n, n_flags, inputs_dict, tech_costs, n_config, n_options) \
            if add_func.__name__ == "add_CO2_liquefaction" else \
            add_func(n, n_flags, inputs_dict, tech_costs, n_config)

        # handle return semantics flexibly
        if result is not None:
            n = result

    # ---------------------------------------------------------
    # Log additions
    # ---------------------------------------------------------
    new_components = log_new_components(n, n0_dict)
    return n, new_components


def add_symbiosis(n, n_flags, inputs_dict, tech_costs):
    """
    Build the industrial symbiosis network connecting:
      - Renewable electricity (El3, El2)
      - District heat (DH grid)
      - Hydrogen (H2 distribution)
      - CO2 (CO2 distribution)
      - Heat networks (MT, DH, LT, ambient)
    """

    n0_dict = get_network_status(n)

    if not n_flags.get("symbiosis", False):
        empty = {k: [] for k in ["links", "generators", "loads", "stores", "buses"]}
        return n , empty

    # ---------------------------------------------------------
    # District heating export
    # ---------------------------------------------------------
    if n_options.at["DH", "enable"]:
        bus_dict = {
            "bus_list": ["DH grid", "Heat DH"],
            "carrier_list": ["Heat", "Heat"],
            "unit_list": ["MW", "MW"],
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        n.add(
            "Link",
            "DH_GL_to_DH_grid",
            bus0="Heat DH",
            bus1="DH grid",
            efficiency=1,
            p_nom_extendable=True,
            marginal_cost=-n_options.at["DH", "price"],
        )

    # ---------------------------------------------------------
    # Hydrogen distribution
    # ---------------------------------------------------------
    if n_flags.get("electrolysis", False):
        bus_dict = {
            "bus_list": ["H2", "H2 distribution"],
            "carrier_list": ["H2", "H2"],
            "unit_list": ["MW", "MW"],
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        n.add(
            "Link",
            "H2_pipe",
            bus0="H2",
            bus1="H2 distribution",
            efficiency=1,
            p_nom_extendable=True,
            capital_cost=tech_costs.at["H2 pipe", "fixed"]
            * tech_costs.at["H2 pipe", "distance"]
            * n_config.at["H2 pipe", "cost factor"],
        )

    # ---------------------------------------------------------
    # CO2 distribution (low pressure)
    # ---------------------------------------------------------
    if n_flags.get("biogas", False) and (
            n_config.at["biogas upgrading", "expansion"] == True
            or n_config.at["biogas upgrading", "initial capacity"] > 0):

        bus_dict = {
            "bus_list": ["CO2 sep", "CO2 distribution"],
            "carrier_list": ["CO2 pure", "CO2 pure"],
            "unit_list": ["MW", "MW"],
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        n.add(
            "Link",
            "CO2_pipe",
            bus0="CO2 sep",
            bus1="CO2 distribution",
            efficiency=1,
            p_nom_extendable=True,
            capital_cost=tech_costs.at["CO2 gas pipe", "fixed"]
            * tech_costs.at["CO2 gas pipe", "distance"]
            * n_config.at["CO2 pipe", "cost factor"],
        )

    # ---------------------------------------------------------
    # Heat network (MT, DH, LT, ambient)
    # ---------------------------------------------------------
    bus_dict = {
        "bus_list": ["Heat MT", "Heat DH", "Heat LT", "Heat amb"],
        "carrier_list": ["Heat", "Heat", "Heat", "Heat"],
        "unit_list": ["MW", "MW", "MW", "MW"],
    }
    n = add_requirements_buses(n, bus_dict, symbiosis_n)

    # Heat rejection links (MT/DH/LT → ambient)
    for src in ["Heat MT", "Heat DH", "Heat LT"]:
        n.add(
            "Link",
            f"{src}_to_amb",
            bus0=src,
            bus1="Heat amb",
            efficiency=1,
            p_min_pu= 0.0,
            p_nom_extendable=True,
            capital_cost=5e-6,  # Assumes plants can reject heat freely
        )

    # Heat cascade links (MT → DH → LT)
    cascade_links = [
        ("Heat_MT_to_DH", "Heat MT", "Heat DH"),
        ("Heat_MT_to_LT", "Heat MT", "Heat LT"),
        ("Heat_DH_to_LT", "Heat DH", "Heat LT"),
    ]
    for name, b0, b1 in cascade_links:
        n.add(
            "Link",
            name,
            bus0=b0,
            bus1=b1,
            efficiency=1,
            p_nom_extendable=True,
            capital_cost=tech_costs.at["DH heat exchanger", "fixed"]
            * n_config.at["DH heat exchanger", "cost factor"],
        )

    # ---------------------------------------------------------
    # Add shared heat pump (LT → DH)
    # ---------------------------------------------------------
    n = add_heat_pump(n, n_flags, inputs_dict, tech_costs)

    # ---------------------------------------------------------
    # Log new components
    # ---------------------------------------------------------
    new_components = log_new_components(n, n0_dict)
    return n, new_components

# BUILD THE NETWORK
def build_network(tech_costs, inputs_dict, n_flags, n_options, p):
    """
    Build the full PyPSA network (Greenbubble) using all modular plant functions.

    Parameters
    ----------
    tech_costs : pd.DataFrame
        Table of technology costs (fixed, variable, efficiency, etc.)
    inputs_dict : dict
        Global inputs (efficiencies, market prices, CO2 cost, etc.)
    n_flags : dict
        Boolean flags controlling which subsystems are active
    n_options : pd.DataFrame
        Configuration options (market toggles, cost factors, etc.)
    p : module or object
        Parameter container with time series, constants, etc.

    Returns
    -------
    network : pypsa.Network
    """

    # ---------------------------------------------------------
    # 1. Initialize the network
    # ---------------------------------------------------------
    network = pypsa.Network()
    network.name = run_name
    network.set_snapshots(p.hours_in_period)

    # ---------------------------------------------------------
    # 2. Add basic layers (external + demand)
    # ---------------------------------------------------------
    network, comp_external_grids = add_external_grids(network, inputs_dict, n_options)
    network, comp_demands = add_demands(network, n_flags, inputs_dict)

    # ---------------------------------------------------------
    # 3. Add production plants and technologies
    # ---------------------------------------------------------
    network, comp_biogas = add_biogas(network, n_flags, inputs_dict, tech_costs)
    network, comp_renewables = add_renewables(network, n_flags, inputs_dict, tech_costs)
    network, comp_electrolysis = add_electrolysis(network, n_flags, inputs_dict, tech_costs)
    network, comp_meoh = add_meoh(network, n_flags, inputs_dict, tech_costs)
    network, comp_central_H = add_central_heat_MT(network, n_flags, inputs_dict, tech_costs)
    network, comp_symbiosis = add_symbiosis(network, n_flags, inputs_dict, tech_costs)
    network, comp_methanation = add_methanation(network, n_flags, inputs_dict, tech_costs)
    network, comp_storage = add_storage(network, n_flags, inputs_dict, tech_costs)

    # ---------------------------------------------------------
    # 4. Apply system-wide constraints
    # ---------------------------------------------------------
    define_total_supply_constraints(network, network.snapshots, component='Generator')

    # ---------------------------------------------------------
    # 5. Collect all component logs
    # ---------------------------------------------------------
    network_comp_allocation = {
        "external_grids": comp_external_grids,
        "demands": comp_demands,
        "biogas": comp_biogas,
        "renewables": comp_renewables,
        "electrolysis": comp_electrolysis,
        "meoh": comp_meoh,
        "methanation": comp_methanation,
        "central_heat": comp_central_H,
        "symbiosis": comp_symbiosis,
        "storage": comp_storage,
    }

    # Optionally: build per-agent interfaces and bus maps
    # network_comp_allocation = network_comp_allocation_add_buses_interface(network, network_comp_allocation)

    # Store allocation inside network object
    network.network_comp_allocation = network_comp_allocation

    # ---------------------------------------------------------
    # 6. Return full network
    # ---------------------------------------------------------
    # fix for some efficiencies not assigned becoming NaN instead than 1 # TODO remove when solved
    for col in [c for c in network.components["Link"].static.columns if c.startswith("efficiency")]:
        network.links[col] = network.links[col].fillna(1.0)

    return network

