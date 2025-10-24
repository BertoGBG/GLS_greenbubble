# how to call it
# --- Dependencies ---
dependencies = [n_flags.get("electrolysis", False)]
if not all(dependencies):
    return n



def add_compressor_and_storage(n, n_flags, tech_costs, n_config, comp_dict):
    """
    Add general compression and gas storage (cylinders/vessels/baloon) systems.
    Includes heat integration to LT/DH heat networks and auxiliary electric buses.


    return n and comp_dict (updated)
    """

    from scripts.technology_inputs import symbiosis_n, compress_multistage_with_Tcap, aftercomp_cool_duty

    comp_dict = {'plant' : plant_name, # ----> '' for centralized H2 compressor
                'local EL bus': local_EL_bus,
                'Heat DH bus' :local_heat_buses [0],
                'Heat LT bus' :local_heat_buses [1],
                'LP stream' : 'H2 production',
                'HP stream' : 'H2 to MeOH',
                'storage stream' : 'H2 storage',
                'comp initial capacity' :   0, # H2 compressor initial capacity
                'storage initial capacity' : 0, #H2  storage initial capacit
                 }
    # --- Snapshot network state ---
    n0_dict = get_network_status(n)

    # ==========================================================
    # 1. COMPRESSION LINK
    # ==========================================================

    def calculation_compressor(comp_dict, eta_s, r_max):
        ##### Calculation for compressor and heat exchangers:
        # --- H2 compression to MeOH
        LP_stream = comp_dict['LP stream']
        HP_stream = comp_dict['HP stream']
        storage_stream = comp_dict['H2 storage']

        ### main copression
        comp_res = compress_multistage_with_Tcap(
            fluid=symbiosis_n.at[LP_stream, 'fluid'],
            p_in_bar=symbiosis_n.at[LP_stream, 'P'],
            p_out_bar=symbiosis_n.at[HP_stream, 'P'],
            T_in_C=symbiosis_n.at[LP_stream, 'T'],
            eta_s=0.75,
            r_max=2.5,
            T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
            T_split_C=symbiosis_n.at['Heat DH min', 'T']  # default to T_in
        )

        ### second compression (for storage)

        if symbiosis_n.at[storage_stream, 'P'] > symbiosis_n.at[HP_stream, 'P']:
            # storage at higher pressure than delivery pressure,
            # requires additional compression for storage

            # ---- Cooling before compression for storage
            cooling_storage_pre_comp = aftercomp_cool_duty(
                fluid = symbiosis_n.at[HP_stream, 'fluid'],
                p_out_bar = symbiosis_n.at[HP_stream, 'P'],
                T_in_C = comp_res["stages"][-1]["T_out_C"],
                T_cool_C = symbiosis_n.at['Heat LT min', 'T'],
                T_split_C = symbiosis_n.at['Heat DH min', 'T'],
                clamp_to_zero=True)

            # ----- extra compression for storage ---
            comp_extra_res = compress_multistage_with_Tcap(
                fluid = symbiosis_n.at[HP_stream, 'fluid'],
                p_in_bar = symbiosis_n.at[HP_stream, 'P'],
                p_out_bar = symbiosis_n.at[storage_stream, 'P'],
                T_in_C =   symbiosis_n.at[LP_stream, 'T'],
                eta_s = 0.75,
                r_max = 2.5,
                T_cool_C = symbiosis_n.at['Heat LT min', 'T'],
                T_split_C = symbiosis_n.at['Heat DH min', 'T']  # default to T_in
            )

            # ---- final Cooling before storage ----
            cooling_storage_pre_comp = aftercomp_cool_duty(
                fluid=symbiosis_n.at[HP_stream, 'fluid'],
                p_out_bar=symbiosis_n.at[HP_stream, 'P'],
                T_in_C=comp_res["stages"][-1]["T_out_C"],
                T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
                T_split_C=symbiosis_n.at['Heat DH min', 'T'],
                clamp_to_zero=True)

        elif symbiosis_n.at[storage_stream, 'P'] < symbiosis_n.at[HP_stream, 'P']:






        return

    def add_compressor_aux(n):
        # add / check for required buses
        bus_dict = {
            "bus_list": [comp_dict['LP bus'],
                         comp_dict['HP bus'],
                         comp_dict['local Heat DH bus'],
                         comp_dict['local Heat LT bus'],
                         comp_dict['local EL bus']],
            "carrier_list": ["H2", "H2", 'Heat', 'Heat', 'AC'],
            "unit_list": ["MW", "MW", 'MW', 'MW', 'MW'],
        }
        n = add_requirements_buses(n, bus_dict)




        return n

    def add_compressor_cap_exp(n, prefix, capital_cost, capacity, expansion):

        n.add("Link",
              prefix + f"{plant_name}H2 compressor",
              bus0=comp_dict['H2 LP bus'],
              bus1=comp_dict['H2 HP bus'],
              bus2=comp_dict['local EL bus'],
              bus3=comp_dict['local Heat DH bus'],
              bus4=comp_dict['local Heat LT bus'],
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
        n = add_requirements_buses(n, bus_dict)

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
            H2_comp_dict['H2 storage bus'] = f"{plant_name}H2 storage"
            n = add_H2_storage_aux(n)

        if t in cap_to_add:
            n = add_H2_storage_cap_exp(n, prefix="EXI_", capital_cost=0, capacity=capacity[1], expansion=False)

        if t in exp_to_add:
            capital_cost = tech_costs.at["hydrogen storage tank type 1", "fixed"] * n_config.at['H2 storage', "cost factor"]
            n = add_H2_storage_cap_exp(n, prefix="", capital_cost=capital_cost, capacity=0, expansion=True)

    return n, H2_comp_dict
