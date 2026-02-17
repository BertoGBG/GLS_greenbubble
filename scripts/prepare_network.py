from sys import prefix

import numpy as np
import pypsa
import pandas as pd
import hashlib
import re
from scripts.parameters import loop_tol
from scripts.helpers import en_market_prices_w_CO2, add_el_grid_import_RFNBOs, ensure_bus, ensure_carrier
from scripts.config import (n_options,
                            n_config,
                            rfnbos_dict,
                            run_name,
                            targets_dict,
                            H2_profile_flag)
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

    # renewables : Needs Users (avoids unbounded sales)
    users = [n_flags.get("electrolysis", False), n_flags.get("biogas", False), n_flags.get("meoh", False),
                    n_flags.get("methanation", False)]

    if (n_flags.get("renewables", False) and n_flags.get("symbiosis", False) and any(users)):
        n_flags_OK['renewables'] = True
    else:
        n_flags_OK['renewables'] = False

    # H2 production Dependencies
    cond1 = n_flags['electrolysis'] and rfnbos_dict['limit'] == 'unlimited'
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

    print('n_flags_OK', n_flags_OK)

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
    if n.buses.index.has_duplicates:
        d = n.buses.index[n.buses.index.duplicated(keep=False)]
        raise ValueError(f"[add_requirements_buses] n.buses already has duplicates (examples): {sorted(set(d))[:30]}")

    valid_entries = [i for i, b in enumerate(bus_dict["bus_list"]) if pd.notna(b)]
    bus_list = [bus_dict["bus_list"][i] for i in valid_entries]
    bus_list = list(pd.Index(bus_list).dropna().unique()) # guard for bus list with duplicate names
    carrier_list = [bus_dict["carrier_list"][i] for i in valid_entries]
    unit_list = [bus_dict.get("unit_list", [""] * len(bus_dict["bus_list"]))[i] for i in valid_entries]

    # Ensure carriers exist
    needed_carriers = {c for c in carrier_list if c}
    #missing_carriers = [c for c in needed_carriers if c not in n.carriers.index]
    for c in needed_carriers:
        ensure_carrier(n, c)

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
    Safe for uninitialized component tables (which are None in PyPSA >1.0).
    """
    new_components = {}
    for comp in ["links", "generators", "loads", "stores", "buses", "storage_units"]:
        before = set(n0_dict.get(comp, []))

        table = getattr(n, comp, None)
        if table is not None:
            after = set(table.index)
        else:
            after = set()

        new_components[comp] = list(after - before)
    return new_components


# ------- Unique Links or Stores ----------------

def _is_timeseries_like(v) -> bool:
    # treat these as profiles you will set later -> ignore in signature
    return isinstance(v, (pd.Series, pd.DataFrame, np.ndarray, list, tuple, dict))


def _canon_scalar(v, nd=10):
    # canonicalize only scalar-ish values
    if v is None:
        return None
    if isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, (float, np.floating)):
        return round(float(v), nd)
    if isinstance(v, (np.integer,)):
        return int(v)
    # anything else (objects) -> ignore by returning a marker (or raise)
    return ("_obj_", type(v).__name__)


def _ensure_registry(n):
    if getattr(n, "meta", None) is None:
        n.meta = {}
    # main registry: maps signature-hash -> component name
    n.meta.setdefault("component_registry", {})
    # optional reverse lookup (purely informative, JSON-safe)
    n.meta.setdefault("component_signatures", {})
    return n.meta["component_registry"]


# ---------- signature field selection ----------

def _sorted_port_keys(keys, base):
    # e.g. bus0, bus1, p_min_pu0, ramp_limit_up0 ...
    def portnum(k):
        m = re.match(rf"^{re.escape(base)}(\d+)$", k)
        return int(m.group(1)) if m else 10**9
    return sorted([k for k in keys if re.match(rf"^{re.escape(base)}\d+$", k)], key=portnum)


def link_signature(kwargs: dict, nd=10):
    keys = set(kwargs.keys())

    # variable number of buses + efficiencies
    bus_keys = _sorted_port_keys(keys, "bus")
    eff_keys = ["efficiency"] + _sorted_port_keys(keys, "efficiency")  # efficiency, efficiency2...
    eff_keys = [k for k in eff_keys if k in keys]

    # include port-dependent constraints/costs if they are scalars
    port_patterns = [
        "p_min_pu", "p_max_pu",
        "ramp_limit_up", "ramp_limit_down",
        "ramp_limit_start_up", "ramp_limit_shut_down",
        "marginal_cost",
    ]
    port_keys = []
    for base in port_patterns:
        port_keys += [k for k in ([base] + _sorted_port_keys(keys, base)) if k in keys]

    # common scalar fields (only if present)
    common = [
        "p_nom_extendable", "p_nom_min", "p_nom_max",
        "capital_cost",
        "committable",
        "start_up_cost", "shut_down_cost",
        "min_up_time", "min_down_time",
    ]
    common = [k for k in common if k in keys]

    sig_items = []

    for k in common + bus_keys + eff_keys + port_keys:
        v = kwargs.get(k)
        if _is_timeseries_like(v):
            continue  # ignore profiles
        sig_items.append((k, _canon_scalar(v, nd=nd)))

    return ("Link", tuple(sig_items))


def store_signature(kwargs: dict, nd=10):
    keys = set(kwargs.keys())

    common = [
        "bus",
        "e_nom_extendable", "e_nom_min", "e_nom_max",
        "e_cyclic", "e_initial",
        "standing_loss",
        "capital_cost", "marginal_cost",
    ]
    common = [k for k in common if k in keys]

    sig_items = []
    for k in common:
        v = kwargs.get(k)
        if _is_timeseries_like(v):
            continue  # ignore profiles
        sig_items.append((k, _canon_scalar(v, nd=nd)))

    return ("Store", tuple(sig_items))


# ---------- generic "add if new" ----------


def add_component_if_new(n, component: str, name_prefix: str, kwargs: dict, nd=10):
    reg = _ensure_registry(n)

    if component == "Link":
        sig = link_signature(kwargs, nd=nd)
        table = n.links
    elif component == "Store":
        sig = store_signature(kwargs, nd=nd)
        table = n.stores
    else:
        raise ValueError("Only 'Link' and 'Store' supported.")

    # make JSON-safe registry key
    sig_hash = hashlib.sha1(repr(sig).encode("utf-8")).hexdigest()  # long, stable
    reg_key = f"{component}:{sig_hash}"                              # str key -> JSON ok

    if reg_key in reg:
        return n, reg[reg_key], False

    # deterministic name from same signature
    name = f"{name_prefix}_{sig_hash[:10]}"

    if name in table.index:
        i = 1
        while f"{name}_{i}" in table.index:
            i += 1
        name = f"{name}_{i}"

    n.add(component, name, **kwargs)
    reg[reg_key] = name
    return n, name, True


# convenience wrappers

def add_link_if_new(n, name_prefix: str, link_kwargs: dict, nd=10):
    """
    Add a Link only if an identical one (by signature) is not already present.
    - Keeps human-readable names
    - Uses a hidden hash-based registry to avoid duplicates
    """
    reg = _ensure_registry(n)

    # build signature (ignoring time-series-like parameters)
    sig = link_signature(link_kwargs, nd=nd)

    # make a JSON-safe registry key from the signature
    sig_hash = hashlib.sha1(repr(sig).encode("utf-8")).hexdigest()
    reg_key = f"Link:{sig_hash}"

    # --- If identical link already exists, reuse it ---
    if reg_key in reg:
        return n, reg[reg_key], False

    # --- Create clean, human-readable name ---
    base = name_prefix.strip()
    name = base

    # avoid collision with existing names
    if name in n.links.index:
        i = 2
        while f"{base}_{i}" in n.links.index:
            i += 1
        name = f"{base}_{i}"

    # actually add the link
    n.add("Link", name, **link_kwargs)

    # store in registry (JSON-safe)
    reg[reg_key] = name
    n.meta["component_signatures"][name] = sig_hash  # optional bookkeeping

    return n, name, True


def add_store_if_new(n, name_prefix: str, store_kwargs: dict, nd=10):
    reg = _ensure_registry(n)

    sig = store_signature(store_kwargs, nd=nd)
    sig_hash = hashlib.sha1(repr(sig).encode("utf-8")).hexdigest()
    reg_key = f"Store:{sig_hash}"

    if reg_key in reg:
        return n, reg[reg_key], False

    base = name_prefix.strip()
    name = base

    if name in n.stores.index:
        i = 2
        while f"{base}_{i}" in n.stores.index:
            i += 1
        name = f"{base}_{i}"

    n.add("Store", name, **store_kwargs)

    reg[reg_key] = name
    n.meta["component_signatures"][name] = sig_hash

    return n, name, True


# ------- BUILD PYPSA NETWORK AUXILIARY FUNCTIONS-------------

#  -------COMMON FUNCTIONS -----------
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
    ensure_carrier(n, "Heat")

    new_buses = []

    for b in heat_bus_dict.keys():
        # 1) Local bus at the plant (for local boilers etc.)
        local_bus = f"{b}_{plant_name}"

        # direction of the heat flow with respect to the main plant
        symbiosis_dir = heat_bus_dict[b]

        # ensure loca_bus
        ensure_bus(n, local_bus, carrier="Heat", unit="MW")
        new_buses.append(local_bus)

        if n_flags.get("symbiosis", False):
            ensure_carrier(n, "symbiosys net")

            # ensure bus
            if b not in n.buses.index:
                ensure_bus(n, b, carrier="Heat", unit="MW")

            if int(symbiosis_dir>0):
                # 2) Heat rejection to symbiosis net (on heat bus)
                sym_link = f"{b}_{plant_name}_to_symb"
                if sym_link not in n.links.index:
                    n.add(
                        "Link",
                        sym_link,
                        carrier = 'symbiosys net',
                        bus0=local_bus,  # plant-local side
                        bus1=b,  # symbiosis side
                        efficiency=tech_costs.at["DH heat exchanger", "efficiency"],
                        p_min_pu=0,
                        p_nom_extendable=True,
                        marginal_cost=loop_tol,
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
                        carrier = 'symbiosys net',
                        bus0=b,                 # symbiosis side
                        bus1=local_bus,         # plant-local side
                        efficiency=tech_costs.at["DH heat exchanger", "efficiency"],
                        p_min_pu=0,
                        p_nom_extendable=True,
                        marginal_cost=loop_tol,
                        capital_cost=tech_costs.at["DH heat exchanger", "fixed"]
                                     * n_config.at["DH heat exchanger", "cost factor"],
                    )

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
    ensure_carrier(n, "El")

    # --- Local electricity bus ---
    ensure_bus(n, local_EL_bus, carrier="El", unit="MW")

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
        el_bus_symbiosis = "El3" #"El3 bus"

        if el_bus_symbiosis not in n.buses.index:
            n.add("Bus", el_bus_symbiosis, carrier="El", unit="MW")

        link_name2 = f"{el_bus_symbiosis}_to_{local_EL_bus}"
        if link_name2 not in n.links.index:
            ensure_carrier(n, "symbiosys net")

            n.add(
                "Link",
                link_name2,
                carrier='symbiosys net',
                bus0=el_bus_symbiosis,
                bus1=local_EL_bus,
                efficiency=1.0,
                p_nom_extendable=True,
            )

    return n


def add_local_boilers(n, local_EL_bus, local_heat_bus, name,
                      heat_efficiency_plant, tech_costs,
                      inputs_dict, capacity, expansion, carrier,
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
    mc_NG = en_market_prices['NG_grid_price'] + VOM_ng

    # --- Reference plant efficiencies and capacities ---
    η_ref = abs(n.links.at[name, heat_efficiency_plant])
    η_ref3 = abs(n.links.at[name, 'efficiency3'])

    capacity_boiler = capacity * η_ref
    p_nom_max_boiler = n_config.at[name, 'max capacity'] * η_ref3

    # --- Natural gas boiler ---
    n.add("Link",
          f"{name}_NG boiler",
          carrier = carrier,
          bus0="NG",
          bus1=local_heat_bus,
          efficiency=η_ng,
          p_nom_extendable=expansion,
          p_nom=capacity_boiler / η_ng * 1.005,
          p_nom_max=p_nom_max_boiler / η_ng * 1.005,
          lifetime = tech_costs.at['gas boiler steam', 'lifetime'],
          capital_cost=tech_costs.at['gas boiler steam', 'fixed']
                       * n_config.at['NG boiler', 'cost factor']
                       * int(capital_cost > 0) + tech_costs.at['NG grid connection', 'fixed'],
          marginal_cost = mc_NG,
          )


    # --- Electric boiler ---
    n.add("Link",
          f"{name}_El boiler",
          carrier=carrier,
          bus0=local_EL_bus,
          bus1=local_heat_bus,
          efficiency=η_el,
          p_nom_extendable=expansion,
          p_nom=capacity_boiler / η_el * 1.005,
          p_nom_max=p_nom_max_boiler / η_el * 1.005,
          lifetime = tech_costs.at['electric boiler steam', 'lifetime'],
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
        "carrier_list": ["El"],
        "unit_list": [ "MW"],
    }
    network = add_requirements_buses(network, bus_dict, symbiosis_n)
    ensure_carrier(network, 'Grid')

    network.add("Generator",
                "Grid gen",
                carrier="Grid",
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
    # creates buses for a specific plant or technology.
    # Input:
    #   - buses : df (index = local bus names, cols = technology/plant name, data=bus names in network )
    # ----------------------------------------------------------------------
    # check that buses exists, if it does avoid installing '' buses
    def _is_blank(x):
        return (x is None) or (isinstance(x, float) and np.isnan(x)) or (isinstance(x, str) and x.strip() == "")

    # Build tuples and filter blanks
    rows = []
    for b, c, u in zip(
        buses.loc[:, tech].iloc[1:],
        buses.loc[:, "carrier"].iloc[1:],
        buses.loc[:, "unit"].iloc[1:],
    ):
        if _is_blank(b):
            continue
        rows.append((b, c, u))

    bus_dict = {
        "bus_list":   [r[0] for r in rows],
        "carrier_list":[r[1] for r in rows],
        "unit_list":  [r[2] for r in rows],
    }
    n = add_requirements_buses(n, bus_dict, symbiosis_n)

    # add El connections to El_meoh bus
    local_EL_bus = buses.at["local EL bus", tech]
    n = add_local_el_connections(n, local_EL_bus, inputs_dict, n_flags, tech_costs, n_config, n_options)

    return n, buses

def add_biomass_drying(
    n,
    tech_costs,
    n_flags,
    n_config,
    final_moisture=None,
    initial_moisture=None,
    local_EL_bus="El_central_heat"
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
            "carrier_list": ["moist biomass", "Heat", "El", "pellets"],
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
            lifetime = tech_costs.at['biomass belt dryer', 'lifetime'],
            capital_cost=capital_cost,
        )

        if "pellets store" not in n.stores.index:
            n.add(
                "Store",
                "pellets store",
                bus="pellets",
                e_nom_extendable=True,
                e_nom_max=float("inf"),
                capital_cost = loop_tol,
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
                "carrier_list": ["CO2 Liq"]*2,
                "unit_list": ["t/h"]*2,
            }

            n = add_requirements_buses(n, bus_dict, symbiosis_n)

            # CO2 credits for sequestration of from liquefied CO2
            co2_credits = pd.Series(float(inputs_dict["CO2 cost"]), index=n.snapshots)
            c = 'CO2 Liq'
            ensure_carrier(n, c)

            n.add('Link',
                  'CO2 Liq seq',
                  carrier=c,
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

    def add_CO2_Liq_storage_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier):
        # --- add local buses ---
        co2_bus = "CO2 distribution"
        bust_st = "CO2 Liq storage"

        bus_dict = {
            "bus_list": [co2_bus, bust_st],
            "carrier_list": ["CO2 Liq", "CO2"],
            "unit_list": ["t/h", "t/h"],
        }

        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        # --- add local electricity connection ---
        local_EL_bus = "El_CO2_liq"
        n = add_local_el_connections(n, local_EL_bus, inputs_dict, n_flags, tech_costs, n_config, n_options)

        n.add("Store",
              prefix + "CO2 Liq storage",
              carrier = carrier,
              bus= bust_st,
              e_nom_extendable=expansion,
              e_nom=capacity,
              e_nom_max=n_config.at["CO2 Liq storage", "max capacity"],
              capital_cost=capital_cost,
              marginal_cost=tech_costs.at["CO2 storage tank", "VOM"],
              lifetime = tech_costs.at["CO2 storage tank", "lifetime"],
              e_cyclic=True)

        n.add("Link",
              prefix + "CO2 liquefaction return",
              bus0=bust_st,
              bus1=co2_bus,
              efficiency=1,
              marginal_cost=loop_tol,
              p_nom_extendable=expansion,
              p_nom = capacity * (n.snapshots[1].hour -  n.snapshots[0].hour), # ramp limit up and down set to 1
              p_nom_max=n_config.at["CO2 Liq storage", "max capacity"] * (n.snapshots[1].hour -  n.snapshots[0].hour))

        n.add("Link",
              prefix + "CO2 liquefaction",
              carrier = carrier,
              bus0=co2_bus,
              bus1=bust_st,
              bus2=local_EL_bus,
              efficiency=1,
              efficiency2= -1 * tech_costs.at["CO2 liquefaction small", "electricity-input"],
              capital_cost=int(capital_cost>0) * tech_costs.at["CO2 liquefaction small", "fixed"],
              p_nom_extendable=expansion,
              marginal_cost=loop_tol,
              lifetime = tech_costs.at["CO2 liquefaction small", 'lifetime'],
              p_nom=capacity * (n.snapshots[1].hour -  n.snapshots[0].hour), # ramp limit up and down set to 1
              p_nom_max=n_config.at["CO2 Liq storage", "max capacity"] * (n.snapshots[1].hour -  n.snapshots[0].hour))

        add_CO2_liquid_sequestration(n, inputs_dict, n_options, bust_st )

        return n

    # --- determine capacity additions ---
    techs = ["CO2 Liq storage"]
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

    # === Main additions ===
    t = "CO2 Liq storage"
    ensure_carrier(n, name=t)

    if t in cap_to_add:
        capacity = n_config.at[t, "initial capacity"]
        n = add_CO2_Liq_storage_cap_exp(n, prefix="EXI_", capital_cost=0, capacity=capacity, expansion=False, carrier= t)

    if t in exp_to_add:
        capital_cost = tech_costs.at["CO2 storage tank small", "fixed"] * n_config.at[t, "cost factor"]
        n = add_CO2_Liq_storage_cap_exp(n, prefix="", capital_cost=capital_cost, capacity=0, expansion=True, carrier = t)

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

    def _is_blank(x):
        return (x is None) or (isinstance(x, float) and np.isnan(x)) or (isinstance(x, str) and x.strip() == "")

    def _has_bus(n, bus_name):
        return (not _is_blank(bus_name)) and (bus_name in n.buses.index)

    def _warn_skip(msg):
        print(f"⚠️ [compressor/storage] {msg}")

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

    def add_compressor_cap_exp(n, prefix, capital_cost, marginal_cost, lifetime, capacity, expansion, comp_dict, compressor_data):
        # Ensure all buses referenced by this link exist
        for k in ["IN bus", "OUT bus", "local EL bus", "Heat DH bus", "Heat LT bus"]:
            if not _has_bus(n, comp_dict.get(k)):
                _warn_skip(f"skip main compressor link because '{k}' bus is blank/missing")
                return n

        # Determine whether main compression is required
        main_el = None
        if compressor_data is not None and not compressor_data.empty:
            main_el = compressor_data.at["electricity-input", "main compression"]

        # Skip if compressor not needed / no data / zero work
        if main_el is None:
            return n
        try:
            main_el = float(main_el)
        except (TypeError, ValueError):
            return n
        if main_el <= 0.0:
            return n

        # Add compressor link
        link_kwargs = dict(
            bus0=comp_dict["IN bus"],
            bus1=comp_dict["OUT bus"],
            bus2=comp_dict["local EL bus"],
            bus3=comp_dict["Heat DH bus"],
            bus4=comp_dict["Heat LT bus"],
            carrier = 'compressors',
            efficiency=1.0,
            efficiency2=-main_el,
            efficiency3=float(compressor_data.at["heat-output DH", "main compression"]),
            efficiency4=float(compressor_data.at["heat-output LT", "main compression"]),
            p_nom_extendable=expansion,
            p_nom=capacity,
            p_nom_max=n_config.at[f"{fluid} compressor", "max capacity"],
            capital_cost=capital_cost,
            marginal_cost=marginal_cost,
            lifetime = lifetime,
        )

        name_prefix = prefix + f"{plant_name} {fluid} compressor"
        n, link_name, added = add_link_if_new(n, name_prefix, link_kwargs)
        if not added:
            print(f"{name_prefix} skipped because of an equal component present in the network")

        return n

    # ==========================================================
    # 2. HIGH-PRESSURE VESSELS / CYLINDERS STORAGE
    # ==========================================================
    def add_HP_storage_aux(n, comp_dict, compressor_data):

        # Need a valid storage bus to build storage send/return links
        if not _has_bus(n, comp_dict.get("ST bus")):
            _warn_skip(f"skip HP storage aux for plant='{comp_dict.get('plant')}' because ST bus is blank/missing")
            return n

        # check if storage compressor is requested
        if compressor_data.at["electricity-input", 'storage compression']:
            # --- Charging (compression to storage) ---
            capex_recomp = tech_costs.at["hydrogen storage compressor", "fixed"] * n_config.at["H2 compressor", "cost factor"]
            link_kwargs = dict(
                  bus0=comp_dict['OUT bus'],
                  bus1=comp_dict['ST bus'],
                  bus2=comp_dict['local EL bus'],
                  bus3=comp_dict['Heat DH bus'],
                  bus4=comp_dict['Heat LT bus'],
                  carrier = 'compressors',
                  efficiency=1,
                  efficiency2=-compressor_data.at["electricity-input", 'storage compression'],
                  efficiency3=compressor_data.at['heat-output DH', 'storage compression'],
                  efficiency4=compressor_data.at["heat-output LT", 'storage compression'],
                  p_nom_extendable=True,
                  capital_cost = capex_recomp
                  )

            name_prefix = f"{plant_name} {fluid} storage send comp"
            n, link_name, added = add_link_if_new(n, name_prefix, link_kwargs)
            if not added:
                print(f"{name_prefix} skipped because of an equal component present in the network")

            # --- Discharging (from storage to HP network) ---
            link_kwargs = dict(
                  bus0=comp_dict['ST bus'],
                  bus1=comp_dict['OUT bus'],
                  efficiency=1,
                  p_nom_extendable=True,
                  marginal_cost=loop_tol)

            name_prefix = f"{plant_name} {fluid} storage return"
            n, link_name, added = add_link_if_new(n, name_prefix, link_kwargs)
            if not added:
                print(f"{name_prefix} skipped because of an equal component present in the network")

        return n

    def add_HP_storage_cap_exp(n, prefix, capital_cost, lifetime, capacity, expansion, comp_dict):
        # Need a valid storage bus to build storage send/return links
        if not _has_bus(n, comp_dict.get("ST bus")):
            _warn_skip(f"skip HP storage aux for plant='{comp_dict.get('plant')}' because ST bus is blank/missing")
            return n

        link_kwargs = dict(
              bus=comp_dict['ST bus'],
              carrier= 'HP gas storage',
              e_nom_extendable=expansion,
              e_nom=capacity,
              e_nom_max=n_config.at[f"{fluid} HP storage", "max capacity"],
              capital_cost=capital_cost,
              lifetime=lifetime,
              e_cyclic=True)

        name_prefix = prefix + f"{plant_name} {fluid} HP storage"
        n, link_name, added = add_store_if_new(n, name_prefix, link_kwargs)
        if not added:
            print(f"{name_prefix} skipped because of an equal component present in the network")

        return n

    # ==========================================================
    # 4. CAPITAL AND MARGINAL COSTS FOR DIFFERENT FLUIDS
    # ==========================================================
    def get_cc_mc_compressor(fluid):
        # gets correct capital and marginal cost for the compressor depending on the fluid

        if fluid in ('Hydrogen', 'H2'):
            capital_cost = tech_costs.at["hydrogen storage compressor", "fixed"] * n_config.at['H2 HP storage', "cost factor"]
            marginal_cost = tech_costs.at["hydrogen storage compressor", "VOM"] * n_config.at['H2 HP storage', "cost factor"]
            lifetime =  tech_costs.at["hydrogen storage compressor", "lifetime"]

        elif fluid in ('CarbonDioxide', 'CO2'):
            capital_cost = tech_costs.at["CO2 industrial compressor", "fixed"] * n_config.at[
                "CO2 compressor", "cost factor"]
            marginal_cost = tech_costs.at["CO2 industrial compressor", "VOM"] * n_config.at[
                "CO2 compressor", "cost factor"]
            lifetime = tech_costs.at["CO2 industrial compressor", "lifetime"]

        elif fluid in ('Methane', 'CH4', 'biogas', 'Biogas'):
            capital_cost = tech_costs.at['CH4 (g) fill compressor station', 'fixed'] * n_config.at[
                "CH4 compressor", "cost factor"]
            marginal_cost = tech_costs.at['CH4 (g) fill compressor station', "VOM"] * n_config.at[
                "CH4 compressor", "cost factor"]
            lifetime = tech_costs.at['CH4 (g) fill compressor station', "lifetime"]

        else:
            # Unknown fluid type → skip safely
            print(f"⚠️ Skipping {fluid} compressor: unsupported fluid '{fluid}'.")
            return None, None, None

        return capital_cost, marginal_cost, lifetime

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
            lifetime =  tech_costs.at["hydrogen storage tank type 1", "lifetime"]


        elif fluid in ("CarbonDioxide", "CO2"):
            capital_cost = (
                    tech_costs.at["CO2 storage cylinders", "fixed"]
                    * n_config.at["CO2 HP storage", "cost factor"]
            )
            marginal_cost = (
                    tech_costs.at["CO2 storage cylinders", "VOM"]
                    * n_config.at["CO2 HP storage", "cost factor"]
            )
            lifetime =  tech_costs.at["CO2 storage cylinders", "lifetime"]

        else:
            # Unknown fluid type → skip safely
            print(f"⚠️ Skipping {fluid} HP storage: unsupported fluid '{fluid}'.")
            return None, None, None

        return capital_cost, marginal_cost, lifetime

    # ==========================================================
    # 4. Build components
    # ==========================================================
    # Required buses for ANY compressor/link to make sense
    req_keys = ["IN bus", "OUT bus", "local EL bus", "Heat DH bus", "Heat LT bus"]
    missing = [k for k in req_keys if not _has_bus(n, comp_dict.get(k))]
    if missing:
        _warn_skip(f"skip build for plant='{comp_dict.get('plant')}' missing/blank buses: {missing}")
        return n

    # add fluid that is compressed top components name
    fluid = symbiosis_n.at[n.buses.at[comp_dict['IN bus'],'properties'], 'fluid'] # note must be compatible with symbiosis_n and n_congif #TODO make it more general

    # dd carriers
    ensure_carrier(n, 'compressors')
    ensure_carrier(n,'HP gas storage')

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
    if (t in cap_to_add) or (t in exp_to_add):
        compressor_data = en_balance_comp_storage(n, comp_dict)

    # add compressor
    capital_cost, marginal_cost, lifetime = get_cc_mc_compressor(fluid)

    if t in cap_to_add:
        n = add_compressor_cap_exp(n = n, prefix=f"EXI_", capital_cost=0, marginal_cost = marginal_cost, lifetime =lifetime, capacity=capacity[0], expansion=False, comp_dict = comp_dict, compressor_data = compressor_data)

    if t in exp_to_add:
        n = add_compressor_cap_exp(n = n, prefix="", capital_cost=capital_cost, marginal_cost=marginal_cost, lifetime =lifetime, capacity=0, expansion=True, comp_dict = comp_dict, compressor_data = compressor_data)

    # --- add HP Storage (only H2 and CO2) ---
    if n_flags["storage"]:
        t = techs[1]
        capital_cost, marginal_cost, lifetime = get_cc_mc_hp_storage(fluid)

        if (t in cap_to_add) or (t in exp_to_add):
            n = add_HP_storage_aux(n, comp_dict, compressor_data)

        if t in cap_to_add:
            n = add_HP_storage_cap_exp(
                n, prefix="EXI_", capital_cost=0, lifetime= lifetime, capacity=capacity[1], expansion=False, comp_dict = comp_dict)

        if t in exp_to_add:
            n = add_HP_storage_cap_exp(n, prefix="", capital_cost=capital_cost, lifetime= lifetime, capacity=0, expansion=True, comp_dict = comp_dict)

    return n


def add_battery_old(n, n_flags, inputs_dict, tech_costs, n_config):
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
    def add_battery_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier):
        """
        Add a battery system with inverter (AC/DC coupling).
        """
        st_bus = "battery"
        local_EL_bus = 'El3' #'El3 bus'

        # Ensure required buses exist
        bus_dict = {
            "bus_list": [st_bus, local_EL_bus],
            "carrier_list": ["El", 'El'],
            "unit_list": ["MW", 'MW']
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        # Add electricity connection
        #local_EL_bus = 'El_battery'
        #n = add_local_el_connections(n, local_EL_bus, inputs_dict, n_flags, tech_costs, n_config, n_options)

        # --- Storage unit ---
        n.add("Store",
              prefix + "battery",
              carrier=carrier,
              bus=st_bus,
              e_cyclic=True,
              e_nom_extendable=expansion,
              e_nom=capacity,
              e_nom_max=n_config.at["battery", "max capacity"],
              lifetime = tech_costs.at["battery storage", 'lifetime'],
              capital_cost=capital_cost,
              marginal_cost= loop_tol)

        # --- Charging link (AC → DC) ---
        n.add("Link",
              prefix + "battery charger",
              carrier=carrier,
              bus0=local_EL_bus,
              bus1=st_bus,
              efficiency=tech_costs.at["battery inverter", "efficiency"],
              lifetime = tech_costs.at["battery inverter", "lifetime"],
              p_nom=capacity / n_config.at["battery", "max hours"],
              p_nom_extendable=expansion,
              capital_cost=(tech_costs.at["battery inverter", "fixed"]
                            * n_config.at["battery", "cost factor"]
                            * int(capital_cost > 0)),
              marginal_cost = loop_tol)

        # --- Discharging link (DC → AC) ---
        n.add("Link",
              prefix + "battery discharger",
              carrier = carrier,
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
        n = add_battery_cap_exp(n=n, prefix="EXI_", capital_cost=0, capacity=capacity, expansion=False,carrier = t)

    if t in exp_to_add:
        capital_cost = tech_costs.at["battery storage", "fixed"] * n_config.at[t, "cost factor"]
        n = add_battery_cap_exp(n=n, prefix="", capital_cost=capital_cost, capacity=0, expansion=True, carrier = t)

    return n


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
    def add_battery_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier):
        """
        Add a battery system with inverter (AC/DC coupling).
        """
        local_EL_bus = 'El3' #'El3 bus'

        # Ensure required buses exist
        bus_dict = {
            "bus_list": [local_EL_bus],
            "carrier_list": ['El'],
            "unit_list": ['MW']
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        # Add electricity connection

        # --- Storage unit ---
        n.add("StorageUnit",
              prefix + "battery",
              bus=local_EL_bus,
              carrier=carrier,
              max_hours=n_config.at["battery", "max hours"],
              efficiency_store=tech_costs.at["battery inverter", "efficiency"],
              efficiency_dispatch=tech_costs.at["battery inverter", "efficiency"],
              lifetime=tech_costs.at["battery storage", "lifetime"],
              p_nom_extendable=expansion,
              p_nom= capacity,
              p_nom_max= n_config.at["battery", "max capacity"],
              cyclic_state_of_charge=True,
              capital_cost=capital_cost)
        return n

    # ==========================================================
    # 2. BUILD COMPONENTS
    # ==========================================================

    techs = ["battery"]
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

    t = "battery"
    ensure_carrier(n, t)

    if t in cap_to_add:
        capacity = n_config.at[t, "initial capacity"]
        n = add_battery_cap_exp(n=n, prefix="EXI_", capital_cost=0, capacity=capacity, expansion=False,carrier = t)

    if t in exp_to_add:
        capital_cost = (tech_costs.at["battery storage", "fixed"]/n_config.at["battery", "max hours"] + tech_costs.at["battery inverter", "fixed"])  * n_config.at[t, "cost factor"]
        n = add_battery_cap_exp(n=n, prefix="", capital_cost=capital_cost, capacity=0, expansion=True, carrier = t)

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
    def add_TES_storage_DH_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier):
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
        n.add("StorageUnit",
              prefix + "TES DH storage",
              bus="Heat DH storage",
              carrier=carrier,
              max_hours=tech_costs.at["decentral water tank storage", "energy to power ratio"],
              efficiency_store=tech_costs.at["water tank charger", "efficiency"],
              efficiency_dispatch=tech_costs.at["water tank discharger", "efficiency"],
              standing_loss=n_config.at["TES DH", "standing loss"],
              p_nom_max=n_config.at["TES DH", "max capacity"],
              lifetime=tech_costs.at["decentral water tank storage", "lifetime"],
              p_nom_extendable=expansion,
              p_nom= capacity,
              cyclic_state_of_charge=True,
              capital_cost=capital_cost)

        return n


    # ==========================================================
    # 2. MEDIUM-TEMPERATURE CONCRETE STORAGE (TES CONCRETE)
    # ==========================================================
    def add_TES_storage_concrete_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier):
        """
        Add medium-temperature concrete storage (e.g. 120–400°C).
        """
        # Add electricity connection
        local_EL_bus = 'El_TES_concrete'
        n = add_local_el_connections(n, local_EL_bus, inputs_dict, n_flags, tech_costs, n_config, n_options)

        # add heat
        heat_bus = "Heat MT"

        bus_dict = {
            "bus_list": [heat_bus],
            "carrier_list": ["Heat"],
            "unit_list": ["MW"],

        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        # --- Concrete storage block ---
        n.add("StorageUnit",
              prefix + "TES concrete storage",
              bus=heat_bus,
              carrier = carrier,
              p_nom_extendable=expansion,
              p_nom=capacity,
              p_nom_max=n_config.at["TES concrete", "max capacity"],
              standing_loss=n_config.at["TES concrete", "standing loss"],
              max_hours = n_config.at["TES concrete", "max hours"],
              lifetime = tech_costs.at["Concrete-store", 'lifetime'],
              efficiency_store = n_config.at["TES concrete", "efficiency store"],
              efficiency_dispatch = n_config.at["TES concrete", "efficiency dispatch"],
              cyclic_state_of_charge=True,
              capital_cost=capital_cost)

        return n


    # ==========================================================
    # 3. BUILD COMPONENTS
    # ==========================================================
    # --- TES DH ---
    t = "TES DH"
    ensure_carrier(n, t)

    if t in cap_to_add:
        capacity = n_config.at[t, "initial capacity"]
        n = add_TES_storage_DH_cap_exp(n, prefix="EXI_", capital_cost=0, capacity=capacity, expansion=False, carrier = t)
    if t in exp_to_add:
        capital_cost = tech_costs.at["central water tank storage", "fixed"] * n_config.at[t, "cost factor"]
        n = add_TES_storage_DH_cap_exp(n, prefix="", capital_cost=capital_cost, capacity=0, expansion=True, carrier = t)

    # --- TES CONCRETE ---
    t = "TES concrete"
    ensure_carrier(n, t)

    if t in cap_to_add:
        capacity = n_config.at[t, "initial capacity"]
        n = add_TES_storage_concrete_cap_exp(n, prefix="EXI_", capital_cost=0, capacity=capacity, expansion=False, carrier = t)
    if t in exp_to_add:
        capital_cost = tech_costs.at["Concrete-store", "fixed"] * n_config.at[t, "cost factor"]
        n = add_TES_storage_concrete_cap_exp(n, prefix="", capital_cost=capital_cost, capacity=0, expansion=True, carrier = t)

    return n


def add_heat_pump(n, n_flags, inputs_dict, tech_costs):
    """Add an industrial heat pump connecting LT and DH heat networks."""

    # Allocation (who can build it) and dependencies
    allocation = n_flags['symbiosis']
    dependencies = n_flags['symbiosis']

    if allocation and dependencies:

        def add_heat_pump_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier):
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
                carrier = carrier,
                bus0=local_EL_bus,     # electricity input
                bus1='Heat DH',        # useful heat output
                bus2='Heat LT',        # low-temperature heat source
                efficiency=COP,        # output (DH)
                efficiency2=-(COP - 1),# input (LT), negative because it’s consumed
                capital_cost=capital_cost,
                marginal_cost=tech_costs.at['industrial heat pump medium temperature', 'VOM'],
                lifetime = tech_costs.at['industrial heat pump medium temperature', 'lifetime'],
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
        ensure_carrier(n,t)

        if t in cap_to_add:
            capacity = n_config.at[t, 'initial capacity']
            n = add_heat_pump_cap_exp(
                n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False, carrier =t)

        if t in exp_to_add:
            capital_cost = (
                tech_costs.at['industrial heat pump medium temperature', 'fixed']
                * n_config.at[t, 'cost factor']
            )
            n = add_heat_pump_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True, carrier = t)

    return n

# ------- BUILD PYPSA NETWORK MAIN FUNCTIONS-------------


def add_targets(n, plant, inputs_dict, tech_costs, n_options, targets_dict):
    """Add exogenous energy demands/ or selling links (bioCH4, H2, Methanol) and corresponding delivery/storage links

    INPUTS
    plant : str # usually it matched t in the adding plant section
    OUTPUTS
    product_bus : 'str' # name of the bus where the plant alloctes its product -> used to build multilinks in add_plant functions
    """

    # HELPERS
    def clean_series(df, network):
        # ---- Helper to process demand/price series safely (expects 1-col DataFrames) ----
        s = df.iloc[:, 0].astype(float)
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        return s.reindex(network.snapshots).fillna(0.0)

    def build_bus_list_demand_or_price(driver, plant):

        # driver : str  'price' or 'demand'
        # plant : str

        bus_list = None
        demand_ts = None
        price_ts = None
        e_product = None

        if driver == "price":

            # Import and align selling price time series
            p_H2 = clean_series(inputs_dict["price_H2"], n)
            p_meoh = clean_series(inputs_dict["price_meoh"], n)

            if "price_bioCH4" in inputs_dict:
                p_bioCH4 = clean_series(inputs_dict["price_bioCH4"], n)
            else:
                en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs, n_options)
                p_bioCH4 = en_market_prices["bioCH4_grid_sell_price"].reindex(n.snapshots).ffill()

            # ---- Import and align demand time series / max production limits ----

            weights = n.snapshot_weightings["objective"]

            if any(k in plant.lower() for k in ["biogas", "methanation"]):
                bus_list = [f"bioCH4 {plant}", "bioCH4 delivery"]
                demand_ts = clean_series(inputs_dict["bioCH4_demand"], n)
                e_product = float((demand_ts * weights).sum())
                price_ts = p_bioCH4

            if any(k in plant.lower() for k in ["electrolysis"]):
                bus_list = [f"H2 {plant}", "H2 delivery"]
                demand_ts = clean_series(inputs_dict["H2_input_demand"], n)
                e_product = float((demand_ts * weights).sum())
                price_ts = p_H2

            if any(k in plant.lower() for k in ["Methanol", "methanolisation", "meoh"]):
                bus_list = [f"Methanol {plant}", "Methanol delivery"]
                demand_ts = clean_series(inputs_dict["Methanol_input_demand"], n)
                e_product = float((demand_ts * weights).sum())
                price_ts = p_meoh

        elif driver == "demand":
            # ---- Import and align demand time series / max production limits ----
            if any(k in plant.lower() for k in ["biogas", "methanation"]):
                bus_list = ["bioCH4"]
                demand_ts = clean_series(inputs_dict["bioCH4_demand"], n)
            if any(k in plant.lower() for k in ["electrolysis"]):
                bus_list = ["H2"]
                demand_ts = clean_series(inputs_dict["H2_input_demand"], n)
            if any(k in plant.lower() for k in ["Methanol", "methanolisation", "meoh"]):
                bus_list = ["Methanol"]
                demand_ts = clean_series(inputs_dict["Methanol_input_demand"], n)

        else:
            raise ValueError("targets_dict['driver'] must be either 'price' or 'demand'.")

        return bus_list, demand_ts, price_ts, e_product

    def add_targets_per_product(n, driver, product, carrier, unit, bus_list, demand_ts, price_ts, e_product):
        """
        INPUTS
        driver : str , # 'price' or 'demand'
        product : str, # e.g. "bioCH4"
        carrier: str, # e.g. "gas"
        unit: str # e.g. "MW"
        """
        bus_dict = {
            "bus_list": bus_list,
            "carrier_list": [carrier] * len(bus_list),
            "unit_list": [unit] * len(bus_list),
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        if driver == "demand":
            # Load and store representing CH4 demand (
            if product not in n.loads.index:
                n.add("Load", product, bus=bus_list[-1], carrier=carrier)
                n.loads_t.p_set[product] = demand_ts.reindex(n.snapshots)

                # TODO (future work) constraint this for less flexible demand with a demand profile (for all products not only H2)
                if not (product == "H2" and H2_profile_flag):
                    n.add(
                        "Store",
                        f"{product} delivery",
                        bus=bus_list[-1],
                        e_nom_extendable=True,
                        e_cyclic=True,
                    )

        elif driver == "price":
            lk_name  = f"{plant}_to_delivery"
            ensure_carrier(n, product)

            if lk_name not in n.links.index:
                n.add(
                    "Link",
                    lk_name,
                    carrier=product,
                    bus0=bus_list[0],
                    bus1=bus_list[-1],
                    efficiency=1.0,
                    p_nom_extendable=True,
                )

                n.links_t.marginal_cost[f"{plant}_to_delivery"] = price_ts.reindex(n.snapshots)

                # tag to identiy the link for stochastich scenario creation
                n.links.loc[lk_name, "is_product_sale"] = True
                n.links.loc[lk_name, "product"] = product

                n.add(
                    "Store",
                    f"{product} delivery",
                    bus=bus_list[-1],
                    e_nom_extendable=True,
                    e_cyclic=False,
                    e_nom_max = e_product,
                )

        return n

    # ---- Driver-dependent settings ----
    driver = targets_dict["driver"]
    bus_list, demand_ts, price_ts, e_product = build_bus_list_demand_or_price(driver, plant)

    product_bus = bus_list[0]

    if bus_list is None:
        raise ValueError(f"Could not determine product for plant='{plant}' (driver={driver}).")
    # ==============================================================
    # 1. BIOCH4
    # ==============================================================
    if any(k in plant.lower() for k in ["biogas", "methanation"]):
        n = add_targets_per_product(n, driver = driver, product = 'bioCH4', carrier = 'gas', unit = 'MW', bus_list = bus_list, demand_ts = demand_ts, price_ts = price_ts, e_product = e_product)

    # ==============================================================
    # 2. HYDROGEN
    # ==============================================================
    elif any(k in plant.lower() for k in ["electrolysis"]):
        n = add_targets_per_product(n, driver = driver, product = 'H2', carrier = 'H2', unit = 'MW', bus_list = bus_list, demand_ts = demand_ts, price_ts = price_ts, e_product = e_product)

    # ==============================================================
    # 3. METHANOL
    # ==============================================================
    elif any(k in plant.lower() for k in ["Methanol", "methanolisation", "meoh"]):
        n = add_targets_per_product(n, driver = driver, product = 'Methanol', carrier = 'H2', unit = 'MW', bus_list = bus_list, demand_ts = demand_ts, price_ts = price_ts, e_product = e_product)

    else:
        raise ValueError("Plant not associate with a valid product/target")

    return n, product_bus


def add_targets_old(n, n_flags, inputs_dict, tech_costs, n_options, targets_dict):
    """Add exogenous energy demands (bioCH4, H2, Methanol) and corresponding delivery/storage links."""

    # Take a snapshot of network state
    n0_dict = get_network_status(n)

    # ---- Helper to process demand/price series safely (expects 1-col DataFrames) ----
    def clean_series(df, network):
        s = df.iloc[:, 0].astype(float)
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        return s.reindex(network.snapshots).fillna(0.0)

    # ---- Driver-dependent settings ----
    if targets_dict["driver"] == "price":
        ensure_carrier(n, "sales")

        # Import and align selling price time series
        p_H2   = clean_series(inputs_dict["price_H2"], n)
        p_meoh = clean_series(inputs_dict["price_meoh"], n)

        if "price_bioCH4" in inputs_dict:
            p_bioCH4 = clean_series(inputs_dict["price_bioCH4"], n)
        else:
            en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs, n_options)
            p_bioCH4 = en_market_prices["bioCH4_grid_sell_price"].reindex(n.snapshots).ffill()

        # ---- Import and align demand time series / max production limits ----
        d_bioCH4 = clean_series(inputs_dict["bioCH4_demand"], n)
        d_H2 = clean_series(inputs_dict["H2_input_demand"], n)
        d_meoh = clean_series(inputs_dict["Methanol_input_demand"], n)

        weights = n.snapshot_weightings["objective"]
        e_bioCH4 = float((d_bioCH4 * weights).sum())
        e_H2 = float((d_H2 * weights).sum())
        e_meoh = float((d_meoh * weights).sum())

        bus_list_H2   = ["H2", "H2 delivery"]
        bus_list_CH4  = ["bioCH4", "bioCH4 delivery"]
        bus_list_meoh = ["Methanol", "Methanol delivery"]

    elif targets_dict["driver"] == "demand":
        # ---- Import and align demand time series / max production limits ----
        d_bioCH4 = clean_series(inputs_dict["bioCH4_demand"], n)
        d_H2 = clean_series(inputs_dict["H2_input_demand"], n)
        d_meoh = clean_series(inputs_dict["Methanol_input_demand"], n)

        bus_list_H2   = ["H2"]
        bus_list_CH4  = ["bioCH4"]
        bus_list_meoh = ["Methanol"]

    else:
        raise ValueError("targets_dict['driver'] must be either 'price' or 'demand'.")

    # ==============================================================
    # 1. BIOCH4
    # ==============================================================
    if n_flags.get("biogas") or n_flags.get("methanation"):

        bus_dict = {
            "bus_list": bus_list_CH4,
            "carrier_list": ["gas"] * len(bus_list_CH4),
            "unit_list": ["MW"] * len(bus_list_CH4),
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        if targets_dict["driver"] == "demand":
            # Load representing CH4 demand
            n.add("Load", "bioCH4", bus=bus_list_CH4[-1], carrier="gas")
            n.loads_t.p_set["bioCH4"] = d_bioCH4.reindex(n.snapshots)

            n.add(
                "Store",
                "bioCH4 delivery",
                bus=bus_list_CH4[-1],
                e_nom_extendable=True,
                e_cyclic=True,
            )

        elif targets_dict["driver"] == "price":
            n.add(
                "Link",
                "bioCH4_to_delivery",
                carrier='sales',
                bus0=bus_list_CH4[0],
                bus1=bus_list_CH4[-1],
                efficiency=1.0,
                p_nom_extendable=True,
            )

            n.links_t.marginal_cost["bioCH4_to_delivery"] = p_bioCH4.reindex(n.snapshots)

            n.add(
                "Store",
                "bioCH4 delivery",
                bus=bus_list_CH4[-1],
                e_nom_extendable=True,
                e_cyclic=False,
                e_nom_max = e_bioCH4,
            )


    # ==============================================================
    # 2. HYDROGEN
    # ==============================================================

    if n_flags.get("electrolysis"):

        bus_dict = {
            "bus_list": bus_list_H2,
            "carrier_list": ["H2"] * len(bus_list_H2),
            "unit_list": ["MW"] * len(bus_list_H2),
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        if targets_dict["driver"] == "demand":
            # H2 demand (grid)
            n.add("Load", "H2 grid", bus=bus_list_H2[-1])
            n.loads_t.p_set["H2 grid"] = d_H2

            # H2 delivery storage
            n.add(
                "Store",
                "H2 delivery",
                bus=bus_list_H2[-1],
                e_nom_extendable=True,
                e_cyclic=True,
            )

        elif targets_dict["driver"] == "price":
            # Link from production (H2) to delivery (H2 delivery)
            n.add(
                "Link",
                "H2_to_delivery",
                bus0=bus_list_H2[0],
                bus1=bus_list_H2[-1],
                efficiency=1.0,
                p_nom_extendable=True,
            )

            if hasattr(n, "scenarios") and len(getattr(n, "scenarios", [])) > 0:
                for s in n.scenarios:
                    n.links_t.marginal_cost.loc[:, (s, "H2_to_delivery")] = p_H2.reindex(n.snapshots)
            else:
                n.links_t.marginal_cost["H2_to_delivery"] = p_H2.reindex(n.snapshots)

            # H2 delivery storage
            n.add(
                "Store",
                "H2 delivery",
                carrier='sales',
                bus=bus_list_H2[-1],
                e_nom_extendable=True,
                e_cyclic=False,
                e_nom_max = e_H2,
            )

    # ==============================================================
    # 3. METHANOL
    # ==============================================================
    if n_flags.get("meoh"):

        bus_dict = {
            "bus_list": bus_list_meoh,
            "carrier_list": ["Methanol"] * len(bus_list_meoh),
            "unit_list": ["MW"] * len(bus_list_meoh),
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        if targets_dict["driver"] == "demand":
            # Methanol demand
            n.add("Load", "Methanol", bus=bus_list_meoh[-1])
            n.loads_t.p_set["Methanol"] = d_meoh

            n.add(
                "Store",
                "Methanol delivery",
                bus=bus_list_meoh[-1],
                e_nom_extendable=True,
                e_cyclic=True,
            )

        elif targets_dict["driver"] == "price":
            n.add(
                "Link",
                "Methanol_to_delivery",
                carrier='sales',
                bus0=bus_list_meoh[0],
                bus1=bus_list_meoh[-1],
                efficiency=1.0,
                p_nom_extendable=True,
            )

            if hasattr(n, "scenarios") and len(getattr(n, "scenarios", [])) > 0:
                for s in n.scenarios:
                    n.links_t.marginal_cost.loc[:, (s, "Methanol_to_delivery")] = p_meoh.reindex(n.snapshots)
            else:
                n.links_t.marginal_cost["Methanol_to_delivery"] = p_meoh.reindex(n.snapshots)

            n.add(
                "Store",
                "Methanol delivery",
                bus=bus_list_meoh[-1],
                e_nom_extendable=True,
                e_cyclic=False,
                e_nom_max=e_meoh,
            )

    # ==============================================================
    # 4. Log newly added components
    # ==============================================================
    new_components = log_new_components(n, n0_dict)
    return n, new_components

# AGENTS
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

        def add_biogas_exp_cap(n, prefix, capital_cost, capacity, expansion, carrier):
            bus_dict = {'bus_list': ['Dig biomass', 'Digestate', 'biogas'],
                        'carrier_list': ['Dig biomass', 'Digestate', 'gas'],
                        'unit_list': ['t/h', 't/h DM', 'MW'],
                        }

            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict, symbiosis_n)

            name = prefix + 'biogas'
            n.add("Link",
                  name = name,
                  carrier = carrier,
                  bus0="Dig biomass",
                  bus1="biogas",
                  bus2=new_heat_buses[1],  # "Heat LT",
                  bus3=local_EL_bus,  # 'El_biogas',
                  bus4='Digestate',
                  efficiency=GL_eff.loc["bioCH4", "SkiveBiogas"],
                  efficiency2=GL_eff.loc["Heat LT", "SkiveBiogas"],
                  efficiency3=GL_eff.loc["El2 bus", "SkiveBiogas"] * 0.5,
                  efficiency4=GL_eff.loc["DM digestate", "SkiveBiogas"],
                  lifetime = tech_costs.at['biogas','lifetime'],
                  p_nom_extendable = expansion,
                  p_nom = capacity ,
                  p_nom_max = n_config.at['biogas', 'max capacity'],
                  capital_cost = capital_cost )
            return n

        def add_biogas_storage_exp_cap(n, prefix, capital_cost, capacity, expansion, carrier):
            bus_dict = {'bus_list': ['biogas'],
                        'carrier_list': ['gas'],
                        'unit_list': ['MW'],
                        }
            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict, symbiosis_n)

            n.add('Store',
                  name = prefix + 'biogas store',
                  carrier = carrier,
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

        def add_biogas_upgrading_exp_cap(n, product_bus, prefix, capital_cost, capacity, expansion, carrier):
            """ product_bus: str # is the bus for deliver of bioCH4, set by the add_targets function"""

            bus_dict = {'bus_list': ['NG', 'CO2 sep', 'biogas', 'bioCH4'],
                        'carrier_list': ['gas', 'CO2', 'gas', 'gas'],
                        'unit_list': ['MW', 't/h', 'MW', 'MW'],
                        }
            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict, symbiosis_n)

            n.add("Link",
                  name =  prefix + 'biogas upgrading',
                  carrier = carrier,
                  bus0="biogas",
                  bus1= product_bus, #"bioCH4",
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
                  lifetime = tech_costs.at['biogas upgrading', 'lifetime'],
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
                  carrier = carrier,
                  bus0="NG",
                  bus1=new_heat_buses[0],
                  efficiency=tech_costs.at['gas boiler steam', 'efficiency'],
                  p_nom_extendable=expansion,
                  p_nom = capacity_boiler,
                  p_nom_max = p_nom_max_boiler,
                  p_min_pu = p_min_pu_val,
                  lifetime = tech_costs.at['gas boiler steam','lifetime'],
                  capital_cost= tech_costs.at['gas boiler steam', 'fixed'] * n_config.at['NG boiler','cost factor'] * int(capital_cost > 0),
                  )
            n.links_t.marginal_cost.loc[:, name_lk] = en_market_prices["NG_grid_price"]

            return n

        def add_dewatering_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier):

            # Required buses
            bus_dict = {'bus_list': ['Digestate', 'moist biomass'],
                        'carrier_list': ['Digestate', 'moist biomass'],
                        'unit_list': ['t/h DM', 't/h DM'],
                        }

            # add required buses if not in the network
            n = add_requirements_buses(n, bus_dict, symbiosis_n)

            n.add('Link',
                  name =  prefix + 'dewatering',
                  carrier = carrier,
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
        ensure_carrier(n, t)

        if t in cap_to_add + exp_to_add:
            n = add_biogas_aux(n)

        if t in cap_to_add:
            capacity = n_config.at[t, 'initial capacity'] / GL_eff.loc["bioCH4", "SkiveBiogas"]
            n = add_biogas_exp_cap(n= n, prefix = 'EXI_', capital_cost = 0, capacity = capacity, expansion= False, carrier = t)

        if t in exp_to_add:
            capital_cost = tech_costs.at['biogas', 'fixed'] / GL_eff.loc["bioCH4", "SkiveBiogas"] * n_config.at[t,'cost factor']
            n = add_biogas_exp_cap(n=n, prefix = '', capital_cost=capital_cost, capacity = 0, expansion=True, carrier = t)

        # Add biogas storage
        t = 'biogas storage'
        ensure_carrier(n, t)

        if t in cap_to_add:
            capacity = n_config.at[t, 'initial capacity']
            n = add_biogas_storage_exp_cap(n= n, prefix = 'EXI_', capital_cost = 0, capacity= capacity, expansion= False, carrier = t)

        if t in exp_to_add:
            capital_cost = tech_costs.at['biogas storage', 'fixed'] * n_config.at[t,'cost factor']
            n = add_biogas_storage_exp_cap(n=n, prefix = '', capital_cost=capital_cost, capacity=0, expansion=True, carrier = t)

        # Biogas upgrading
        t = 'biogas upgrading'
        ensure_carrier(n, t)
        n , product_bus  = add_targets(n, plant = t, inputs_dict = inputs_dict, tech_costs = tech_costs, n_options = n_options, targets_dict = targets_dict)

        if t in cap_to_add + exp_to_add:
            n = add_biogas_upgrading_aux(n)

        if t in cap_to_add:
            capacity = n_config.at[t,'initial capacity']
            n = add_biogas_upgrading_exp_cap(n= n, product_bus = product_bus, prefix = 'EXI_', capital_cost = 0, capacity= capacity, expansion= False, carrier = t)

        if t in exp_to_add:
            capital_cost = tech_costs.at['biogas upgrading', 'fixed'] * n_config.at[t,'cost factor']
            n = add_biogas_upgrading_exp_cap(n=n, product_bus = product_bus, prefix = '', capital_cost=capital_cost, capacity=0, expansion=True, carrier = t)

        # dewatering of digestate fibers
        t = 'dewatering'
        ensure_carrier(n, t)

        if t in cap_to_add:
            capacity = n_config.at['dewatering', 'initial capacity']
            n = add_dewatering_cap_exp(n=n, prefix='EXI_', capital_cost=0, capacity=capacity, expansion=False, carrier = t)

        if t in exp_to_add:
            capital_cost = tech_costs.at['centrifugal dewatering', "fixed"] * n_config.at['dewatering', 'cost factor']
            n = add_dewatering_cap_exp(n=n, prefix='', capital_cost=capital_cost, capacity=0, expansion=True, carrier = t)

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

    def add_grid_connection_cap_exp(n, name, capital_cost, capacity, expansion, carrier):

        bus_dict = {'bus_list': ['El3', 'ElDK1 sell bus'],
                    'carrier_list': ['El', 'El'],
                    'unit_list': ['MW', 'MW']}
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        if 'Grid RE sell' not in n.stores.index:
            n.add("Store",
                  'Grid RE sell',
                  bus='ElDK1 sell bus',
                  e_nom_extendable=True,
                  e_cyclic=False,
                  e_nom_max=float('inf'),  # 657000
                  )

        n.add("Link",
              name=name,
              carrier = carrier,
              bus0="El3",
              bus1='ElDK1 sell bus',
              efficiency=1,
              p_nom_extendable=expansion,
              p_nom=capacity,
              p_nom_max=n_config.at['grid connection', 'max capacity'],
              capital_cost=capital_cost)
        n.links_t.marginal_cost[name] = en_market_prices["el_grid_sell_price"]

        return n

    def add_onwind_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier):
        n = add_requirements_buses(n, {
            'bus_list': ['El3'],
            'carrier_list': ['El'],
            'unit_list': ['MW']
        }, symbiosis_n)

        name = f"{prefix}onshorewind"
        n.add("Generator",
              name=name,
              bus="El3",
              carrier=carrier,
              p_nom_max=n_config.at['onwind', 'max capacity'],
              p_nom_extendable=expansion,
              p_nom=capacity,
              lifetime = tech_costs.at['onwind','lifetime'],
              capital_cost=capital_cost,
              marginal_cost=tech_costs.at['onwind', 'VOM'],
              p_max_pu=CF_wind["CF wind"])
        n.generators_t.p_max_pu[name] = CF_wind
        return n

    def add_solar_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier):
        n = add_requirements_buses(n, {
            'bus_list': ['El3'],
            'carrier_list': ['El'],
            'unit_list': ['MW']
        }, symbiosis_n)

        name = f"{prefix}solar"
        n.add("Generator",
              name=name,
              bus="El3",
              carrier=carrier,
              p_nom_max=n_config.at['solar', 'max capacity'],
              lifetime=tech_costs.at['solar', 'lifetime'],
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
    if 'onwind' in (cap_to_add or exp_to_add):
        t = 'wind'
        ensure_carrier(n, t)

    if 'onwind' in cap_to_add:
        cap = n_config.at['onwind', 'initial capacity']
        n = add_onwind_cap_exp(n, 'EXI_', 0, cap, False, carrier = t)
    if 'onwind' in exp_to_add:
        cost = tech_costs.at['onwind', 'fixed'] * n_config.at['onwind', 'cost factor']
        n = add_onwind_cap_exp(n, '', cost, 0, True, carrier = t)

    # Solar PV
    if 'solar' in (cap_to_add or exp_to_add):
        t = 'solar'
        ensure_carrier(n, t)

    if 'solar' in cap_to_add:
        cap = n_config.at['solar', 'initial capacity']
        n = add_solar_cap_exp(n, 'EXI_', 0, cap, False, carrier = t)
    if 'solar' in exp_to_add:
        cost = tech_costs.at['solar', 'fixed'] * n_config.at['solar', 'cost factor']
        n = add_solar_cap_exp(n, '', cost, 0, True, carrier = t)

    # Grid connection
    if 'grid connection' in (cap_to_add or exp_to_add):
        t = 'grid connection'
        ensure_carrier(n, t)

    if 'grid connection' in cap_to_add:
        cap = n_config.at['grid connection', 'initial capacity']
        n = add_grid_connection_cap_exp(n, 'EXI_El3_to_DK1', 0, cap, False, carrier = t)
    if 'grid connection' in exp_to_add:
        cost = tech_costs.at['electricity grid connection', 'fixed'] * n_config.at['grid connection', 'cost factor']
        n = add_grid_connection_cap_exp(n, 'El3_to_DK1', cost, 0, True, carrier = t)

    # ----------------------------------------------------------------------
    new_components = log_new_components(n, n0_dict)
    return n, new_components

def add_electrolysis(n, n_flags, inputs_dict, tech_costs):
    """Add electrolysis system (H2 production) to the network."""

    n0_dict = get_network_status(n)

    if not n_flags.get('electrolysis', False):
        empty = {k: [] for k in ['links', 'generators', 'loads', 'stores', 'buses']}
        return n, empty

    # inputs
    GL_eff = inputs_dict['GL_eff']
    H2_input_demand = inputs_dict['H2_input_demand']

    def electrolysis_aux(n, plant):
        # Local electricity hub
        local_EL_bus = f"El_{plant}"
        n = add_local_el_connections(n, local_EL_bus, inputs_dict, n_flags, tech_costs, n_config, n_options)
        link_rfnbos = f"DK1_to_{local_EL_bus}"

        # tag the link for creating stochastic scenario with UNIQUE CARRIER
        ensure_carrier(n,"rfnbos_grid_import")
        n.links.at[link_rfnbos, "carrier"] = "rfnbos_grid_import"  # or any unique carrier

        # ADD RFNBO constraint on use of grid electricity
        p_max_pu_rfnbos = add_el_grid_import_RFNBOs(inputs_dict, rfnbos_dict)
        p_max_pu_rfnbos = p_max_pu_rfnbos.reindex(n.snapshots).astype(float)
        n.links_t.p_max_pu[link_rfnbos] = p_max_pu_rfnbos

        return n, local_EL_bus, link_rfnbos

    # ---------- Choose CAPEX depending on H2 demand
    if H2_input_demand.iloc[:, 0].sum() > 0:
        electrolysis_cost = tech_costs.at['electrolysis', 'fixed'] * n_config.at['electrolysis', 'cost factor']
        electrolysis_lifetime = tech_costs.at['electrolysis', 'lifetime']
    else:
        electrolysis_cost = tech_costs.at['electrolysis small', 'fixed'] * n_config.at['electrolysis', 'cost factor']
        electrolysis_lifetime = tech_costs.at['electrolysis small', 'lifetime']

    # ---------- Electrolyzer component builder
    def add_H2_cap_exp(n, product_bus, prefix, capital_cost, capacity, expansion, carrier):

        n = add_requirements_buses(n, {
            'bus_list': ['El3', 'H2'],
            'carrier_list': ['El', 'H2'],
            'unit_list': ['MW', 'MW'],
        }, symbiosis_n)

        # ---------- Add local heat connections
        heat_bus_dict = {'Heat LT': 1}
        n, new_heat_buses = add_local_heat_connections(n, heat_bus_dict, plant_name=carrier, n_flags=n_flags,
                                                       tech_costs=tech_costs, n_config=n_config)

        name = f"{prefix}electrolysis"

        n.add("Link",
              name=name,
              bus0=local_EL_bus,
              bus1=product_bus,
              carrier = carrier,
              bus2=new_heat_buses[0],  # Heat LT
              efficiency=GL_eff.at['H2', 'GreenHyScale'],
              efficiency2=GL_eff.at['Heat LT', 'GreenHyScale'],
              lifetime = electrolysis_lifetime,
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

    for t in techs:
        if t in (cap_to_add or exp_to_add):
            ensure_carrier(n, t)

            # create connections for plant
            n, local_EL_bus, link_rfnbos = electrolysis_aux(n, plant=t)

            # add targets
            n, product_bus = add_targets(n, plant=t, inputs_dict=inputs_dict, tech_costs=tech_costs, n_options=n_options,
                                         targets_dict=targets_dict)
            # add plants
            if t in cap_to_add:
                cap = n_config.at[t, 'initial capacity']
                n = add_H2_cap_exp(n, product_bus, 'EXI_', 0, cap, False, carrier=t)
            if t in exp_to_add:
                n = add_H2_cap_exp(n, product_bus, '', electrolysis_cost, 0, True, carrier = t)

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
        empty = {k: [] for k in ["links", "generators", "loads", "stores", "buses", "storage_units"]}
        return n, empty

    GL_eff = inputs_dict['GL_eff']

    n0_dict = get_network_status(n)

    # ----------------------------------------------------------------------
    # Methanol synthesis reactor
    # ----------------------------------------------------------------------
    def add_methanolisation_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier, meoh_buses):
        # update methanation_buses
        meoh_buses.at['H2 in bus', 'methanolisation'] = 'H2 to methanolisation'
        meoh_buses.at['product bus', 'methanolisation'] = meoh_buses.at['product bus', 'meoh']
        meoh_buses.at['CO2 in bus', 'methanolisation'] = 'CO2 to methanolisation'
        meoh_buses.at['local EL bus', 'methanolisation'] = meoh_buses.at['local EL bus', 'meoh']
        meoh_buses.at['CO2 storage bus', 'methanolisation'] = meoh_buses.at['CO2 storage bus', 'meoh']
        meoh_buses.at['H2 storage bus', 'methanolisation'] = meoh_buses.at['H2 storage bus', 'meoh']

        # create wrapping buses if required
        s_bus = meoh_buses.loc[:, 'methanolisation'].iloc[1:]
        mask = s_bus.notna() & s_bus.astype(str).str.strip().ne("")

        bus_dict = {
            "bus_list": s_bus.loc[mask].tolist(),
            "carrier_list": meoh_buses.loc[:, "carrier"].iloc[1:].loc[mask].tolist(),
            "unit_list": meoh_buses.loc[:, "unit"].iloc[1:].loc[mask].tolist(),
        }

        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        # add Heat buses
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
            carrier = carrier,
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
            lifetime = tech_costs.at["methanolisation", "lifetime"],
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
                carrier = carrier,
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
        # initialize methanation_buses with existing bus: wrabbing buses for all techs
        idx = ['local EL bus','CO2 in bus', 'H2 in bus', 'product bus']
        carriers= ['El',"CO2", "H2", "Methanol"]
        units = ['MW', "t/h", "MW", "MW"]
        buses_meoh = ['El_meoh', 'CO2 distribution', 'H2 distribution', 'Methanol',]
        meoh_buses = pd.DataFrame(index =idx, columns=['meoh'] + techs, data = ''  )
        meoh_buses.loc[:,'meoh'] = buses_meoh
        meoh_buses.loc[:,'carrier'] = carriers
        meoh_buses.loc[:,'unit'] = units

        # add H2 and CO2 storage bus for all methanation techs if allowed
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
    n.add('Carrier', t)
    n, product_bus = add_targets(n, plant=t, inputs_dict=inputs_dict, tech_costs=tech_costs,
                                 n_options=n_options, targets_dict=targets_dict)
    meoh_buses.at['product bus', t] = product_bus

    if t in cap_to_add:
        cap = n_config.at[t, "initial capacity"]
        n, meoh_buses = add_methanolisation_cap_exp(n, "EXI_", 0, cap, False, carrier= t, meoh_buses= meoh_buses)

    if t in exp_to_add:
        cost = tech_costs.at["methanolisation", "fixed"] * n_config.at["methanolisation", "cost factor"]
        n, meoh_buses = add_methanolisation_cap_exp(n, "", cost, 0, True, carrier= t, meoh_buses = meoh_buses)

    new_components = log_new_components(n, n0_dict)

    return n, new_components


def add_methanation(n, n_flags, inputs_dict, tech_costs):
    """
    Add methanation facilities (biological and catalytic) to the network.
    Methanation can use biogas or CO2 as carbon source and requires H2.
    """

    n0_dict = get_network_status(n)

    if not n_flags.get("methanation", False):
        empty = {k: [] for k in ["links", "generators", "loads", "stores", "buses", "storage_units"]}
        return n, empty

    # ----------------------------------------------------------------------
    # BIOLOGICAL METHANATION (biogas)
    # ----------------------------------------------------------------------
    def add_biomethanation_biogas_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier, methanation_buses):

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
            carrier = carrier,
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
    def add_biomethanation_CO2_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier, methanation_buses):

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
        #print('vol_ratio', vol_ratio)

        name = f"{prefix}biomethanation CO2"

        n.add(
            "Link",
            name,
            carrier = carrier,
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
    def add_cat_methanation_biogas_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier, methanation_buses):

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
            carrier = carrier,
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
    def add_cat_methanation_CO2_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier, methanation_buses):

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
            carrier = carrier,
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
        # initialize methanation_buses with existing bus: wrabbing buses for all techs
        idx = ['local EL bus','CO2 in bus', 'H2 in bus', 'biogas in bus', 'product bus']
        carriers= ['El',"CO2", "H2", "gas", "gas"]
        units = ['MW', "t/h", "MW", "MW", "MW"]
        buses_methanation = ['El_methanation', 'CO2 distribution', 'H2 distribution', 'biogas', 'bioCH4']
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
        empty = {k: [] for k in ['links', 'generators', 'loads', 'stores', 'buses', 'storage_units']}
        return n, empty

    # add each technology with initial capacity or expansion
    for t, add_fn in [
        ("biomethanation biogas", add_biomethanation_biogas_cap_exp),
        ("biomethanation CO2", add_biomethanation_CO2_cap_exp),
        ("cat methanation biogas", add_cat_methanation_biogas_cap_exp),
        ("cat methanation CO2", add_cat_methanation_CO2_cap_exp),
    ]:
        if t in (cap_to_add or exp_to_add):
            n.add('Carrier', t)
            n, product_bus = add_targets(n, plant=t, inputs_dict=inputs_dict, tech_costs=tech_costs,
                                         n_options=n_options, targets_dict=targets_dict)

            # set product bus per technology (for price optimization)
            methanation_buses.at['product bus', t] = product_bus

        if t in cap_to_add:
            cap = n_config.at[t, "initial capacity"]
            n, methanation_buses = add_fn(n, "EXI_", 0, cap, False, carrier = t, methanation_buses= methanation_buses)
        if t in exp_to_add:
            cost = (tech_costs.at["biogas plus hydrogen", "fixed"]
                if "cat" in t
                else tech_costs.at["biomethanation", "fixed"]) * n_config.at[t, "cost factor"]
            n, methanation_buses = add_fn(n, "", cost, 0, True, carrier = t, methanation_buses= methanation_buses)

    new_components = log_new_components(n, n0_dict)
    return n, new_components


def add_central_heat_MT(n, n_flags, inputs_dict, tech_costs):
    """Add central heating technologies (biomass, gas, electric, pyrolysis) to the Heat MT bus."""

    n0_dict = get_network_status(n)

    # --- get energy prices from external markets
    en_market_prices = en_market_prices_w_CO2(inputs_dict, tech_costs, n_options)
    en_market_prices = {k: v.reindex(n.snapshots).ffill() for k, v in en_market_prices.items()}

    if not n_flags.get("central_heat", False):
        empty = {k: [] for k in ["links", "generators", "loads", "stores", "buses", "storage_units"]}
        return n, empty

    # Local electricity hub
    local_EL_bus = "El_central_heat"
    n = add_local_el_connections(n, local_EL_bus, inputs_dict, n_flags, tech_costs, n_config, n_options)

    # Add Heat MT bus (if symbiosis network active)
    if n_flags.get("symbiosis", False):
        if "Heat MT" not in n.buses.index:
            n.add("Bus", "Heat MT", carrier="Heat", unit="MW")

    # ---------------------------------------------------------
    # Pellet market
    # ---------------------------------------------------------
    if n_options.at["pellets market", "enable"]:

        techs = ["biomass boiler"]
        cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)
        if 'biomass boiler' in (cap_to_add or exp_to_add):
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

        ensure_carrier(n, "moist biomass")

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
    def add_biochar_sequestration(n):
        if n_options.at['biochar credits', 'enable']:
            bus_dict = {
                "bus_list": ["biochar", "biochar sequestration"],
                "carrier_list": ["CO2", "CO2"],
                "unit_list": ["t/h", "t/h"],
            }
            n = add_requirements_buses(n, bus_dict, symbiosis_n)
            c = 'biochar'
            ensure_carrier(n, c)

            co2_credits = pd.Series(float(inputs_dict["CO2 cost"]), index=n.snapshots)
            n.add(
                "Link",
                "biochar sequestration",
                carrier=c,
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

    def add_pyrolysis_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier):
        bus_dict = {
            "bus_list": ["pellets", "biochar"],
            "carrier_list": ["pellets", "CO2"],
            "unit_list": ["MW", "t/h"],
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        n.add(
            "Link",
            prefix + "pyrolysis",
            carrier = carrier,
            bus0="pellets",
            bus1="Heat MT",
            bus2=local_EL_bus,
            bus3="biochar",
            efficiency=tech_costs.at["biochar pyrolysis", "heat output"]
            / tech_costs.at["biochar pyrolysis", "biomass input"],
            efficiency2=-tech_costs.at["biochar pyrolysis", "electricity input"]
            / tech_costs.at["biochar pyrolysis", "biomass input"],
            efficiency3=1 / tech_costs.at["biochar pyrolysis", "biomass input"],
            marginal_cost=tech_costs.at['biomass boiler', "VOM"],
            lifetime = tech_costs.at["biochar pyrolysis", "lifetime"],
            p_nom_extendable=expansion,
            p_nom=capacity,
            p_nom_max=n_config.at["pyrolysis", "max capacity"],
            capital_cost=capital_cost,
        )
        return n

    techs = ["pyrolysis"]
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

    for t in techs:
        if t in (cap_to_add or exp_to_add):
            n.add('Carrier', t)
            n = add_biochar_sequestration(n)

            if t in cap_to_add:
                n = add_pyrolysis_cap_exp(n, "EXI_", 0, n_config.at["pyrolysis", "initial capacity"], False, carrier = t)
            if t in exp_to_add:
                cost = (
                    tech_costs.at["biochar pyrolysis", "fixed"]
                    / tech_costs.at["biochar pyrolysis", "biomass input"]
                    * n_config.at["pyrolysis", "cost factor"]
                )
                n = add_pyrolysis_cap_exp(n, "", cost, 0, True, carrier = t)

    # ---------------------------------------------------------
    # Biomass boiler (pellets)
    # ---------------------------------------------------------
    def add_C_biomass_boiler_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier):
        n = add_requirements_buses(
            n,
            {"bus_list": ["pellets", "Heat MT"], "carrier_list": ["pellets", "Heat"], "unit_list": ["MW", "MW"]}, symbiosis_n)
        n.add(
            "Link",
            prefix + "pellets boiler",
            carrier = carrier,
            bus0="pellets",
            bus1="Heat MT",
            efficiency=tech_costs.at['biomass boiler', "efficiency"] * symbiosis_n.at[n.buses.loc['pellets', 'properties'], 'LHV'],
            marginal_cost=tech_costs.at['biomass boiler', "VOM"],
            lifetime = tech_costs.at['biomass boiler', "lifetime"],
            p_nom_extendable=expansion,
            p_nom=capacity,
            p_nom_max=n_config.at["biomass boiler", "max capacity"],
            capital_cost=capital_cost,
        )
        return n

    techs = ["biomass boiler"]
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)
    for t in techs:
        if t in (cap_to_add or exp_to_add):
            n.add('Carrier', t)

            if t in cap_to_add:
                n = add_C_biomass_boiler_cap_exp(n, "EXI_", 0, n_config.at["biomass boiler", "initial capacity"], False, carrier = t)
            if t in exp_to_add:
                cost = tech_costs.at['biomass boiler', "fixed"] * n_config.at["biomass boiler", "cost factor"]
                n = add_C_biomass_boiler_cap_exp(n, "", cost, 0, True, carrier = t)

    # ---------------------------------------------------------
    # NG boiler
    # ---------------------------------------------------------
    def add_C_NG_boiler_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier):
        n = add_requirements_buses(
            n, {"bus_list": ["NG", "Heat MT"], "carrier_list": ["gas", "Heat"], "unit_list": ["MW", "MW"]}, symbiosis_n
        )
        n.add(
            "Link",
            prefix + "NG boiler",
            carrier = carrier,
            bus0="NG",
            bus1="Heat MT",
            efficiency=tech_costs.at["gas boiler steam", "efficiency"],
            p_nom_extendable=expansion,
            p_nom=capacity,
            p_nom_max=n_config.at["NG boiler", "max capacity"],
            capital_cost=capital_cost,
            marginal_cost=en_market_prices["NG_grid_price"]
            + tech_costs.at["gas boiler steam", "VOM"],
            lifetime=tech_costs.at['gas boiler steam', 'lifetime'],
        )
        return n

    techs = ["NG boiler"]
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)
    for t in techs:
        if t in (cap_to_add or exp_to_add):
            n.add('Carrier', t)

            if t in cap_to_add:
                n = add_C_NG_boiler_cap_exp(n, "EXI_", 0, n_config.at["NG boiler", "initial capacity"], False, carrier = t)
            if t in exp_to_add:
                cost = tech_costs.at["gas boiler steam", "fixed"] * n_config.at["NG boiler", "cost factor"]
                n = add_C_NG_boiler_cap_exp(n, "", cost, 0, True, carrier = t)

    # ---------------------------------------------------------
    # Electric boiler
    # ---------------------------------------------------------
    def add_C_El_boiler_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier):
        n = add_requirements_buses(
            n, {"bus_list": [local_EL_bus, "Heat MT"], "carrier_list": ["El", "Heat"], "unit_list": ["MW", "MW"]}, symbiosis_n
        )
        n.add(
            "Link",
            prefix + "El boiler",
            carrier = carrier,
            bus0=local_EL_bus,
            bus1="Heat MT",
            efficiency=tech_costs.at["electric boiler steam", "efficiency"],
            marginal_cost=tech_costs.at["electric boiler steam", "VOM"],
            p_nom_extendable=expansion,
            p_nom=capacity,
            p_nom_max=n_config.at["El boiler", "max capacity"],
            lifetime=tech_costs.at["electric boiler steam", 'lifetime'],
            capital_cost=capital_cost,
        )
        return n

    techs = ["El boiler"]
    cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)
    for t in techs:
        if t in (cap_to_add or exp_to_add):
            n.add('Carrier', t)

            if t in cap_to_add:
                n = add_C_El_boiler_cap_exp(n, "EXI_", 0, n_config.at["El boiler", "initial capacity"], False, carrier = t)
            if t in exp_to_add:
                cost = tech_costs.at["electric boiler steam", "fixed"] * n_config.at["El boiler", "cost factor"]
                n = add_C_El_boiler_cap_exp(n, "", cost, 0, True, carrier = t)

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
        empty = {k: [] for k in ["links", "generators", "loads", "stores", "buses", "storage_units"]}
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
        empty = {k: [] for k in ["links", "generators", "loads", "stores", "buses", "storage_units"]}
        return n , empty

    # ---------------------------------------------------------
    # Add carrier
    # ---------------------------------------------------------
    ensure_carrier(n, "symbiosys net")

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

        c = 'DH'
        ensure_carrier(n, c)

        n.add(
            "Link",
            "DH_to_DH_grid",
            carrier=c,
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
            carrier = 'symbiosys net',
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
            "carrier_list": ["CO2", "CO2"],
            "unit_list": ["MW", "MW"],
        }
        n = add_requirements_buses(n, bus_dict, symbiosis_n)

        n.add(
            "Link",
            "CO2_pipe",
            carrier='symbiosys net',
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
            capital_cost= loop_tol,  # Assumes plants can reject heat freely
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
            carrier='symbiosys net',
            bus0=b0,
            bus1=b1,
            efficiency=tech_costs.at["DH heat exchanger", "efficiency"],
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
    #network, comp_targets = add_targets(network, n_flags, inputs_dict, tech_costs, n_options, targets_dict)

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
        #"targets": comp_targets,
        "biogas": comp_biogas,
        "renewables": comp_renewables,
        "electrolysis": comp_electrolysis,
        "meoh": comp_meoh,
        "methanation": comp_methanation,
        "central_heat": comp_central_H,
        "symbiosis": comp_symbiosis,
        "storage": comp_storage,
    }

    # Store allocation inside network object
    network.network_comp_allocation = network_comp_allocation

    # ---------------------------------------------------------
    # 6. Return full network
    # ---------------------------------------------------------
    # fix for some efficiencies not assigned becoming NaN instead than 1 # TODO remove when issue with stochastic optimization is solved
    for col in [c for c in network.components["Link"].static.columns if c.startswith("efficiency")]:
        network.links[col] = network.links[col].fillna(1.0)

    return network
