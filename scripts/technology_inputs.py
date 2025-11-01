import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP
import math

# --- Network Inputs  T and P levels -------
T_max_comp = 160 # maximum discharge temperature for all compressors
T_ambient = 20 #

# biogas composition (used only locally)
biogas_mix = {"Methane": 0.65, "CarbonDioxide": 0.35}
M_CH4 = CP.PropsSI("M", "T", 300, "P", 1e5, "Methane")       # [kg/mol]
M_CO2 = CP.PropsSI("M", "T", 300, "P", 1e5, "CarbonDioxide") # [kg/mol]
M_mix = biogas_mix['Methane']*M_CH4 + biogas_mix['CarbonDioxide']*M_CO2
w_CH4 = biogas_mix['Methane']*M_CH4 / M_mix
w_CO2 = biogas_mix['CarbonDioxide']*M_CO2 / M_mix

lhv_ch4 = 13.9 # MWh/t
lhv_biogas = lhv_ch4 * w_CH4
lhv_h2 = 33.33 # MWh/t
lhv_meoh = 5.54 # MWh/t
lhv_pellets = 14.5/3.6 #  MWh/t @ moisture pellets (straw pellets)
lhv_chips = 2.3 #  MWh/t @ moisture moist biomass (straw pellets)

# CoolProp -  cache to avoid rebuilding phase envelopes for mixtures repeatedly (needed for not pure fluids)
_AS_cache = {}

# ------ List of stream in symbiosis network
# if a value is missing is either calculated or constrained by a global value (e.g. T_max_comp)
# fluid nomenclature must follow the Greenbube model, but it is matched to CoolProp standards: https://coolprop.org/fluid_properties/PurePseudoPure.html#list-of-fluids  within compressor_calculation
# fluid field must match the nomenclature in n_config (e.g. H2 --> H2 compressor, do not use Hydrogen --> H2 compressor # TODO : make it more general
# biogas is handles direclty from mixture defined in p

# network of fluid streams : index is a UNIQUE NAME and is used for look-up in the model and add buses
# T : Celsius
# P : bar(a)

symbiosis_data = {
    "Heat MT max": {"fluid": 'Water', "T": 180, 'P' : 10, }, # production
    'Heat MT min': {"fluid": 'Water', "T": 140, 'P': 3, 'buses' : ['Heat MT', "Heat MT storage"] }, # return
    'Heat DH min': {"fluid": 'Water', "T": 90, 'P': 1, 'buses' : ['Heat DH' ,'DH grid', "Heat DH storage"] },
    'Heat LT min': {"fluid": 'Water', "T": 50, 'P': 1, 'buses' : ['Heat LT'] },
    'Ambient': {"fluid": 'Air', "T": T_ambient, 'P': 1, 'buses' : ['Heat amb'] },
    'NG grid': {"fluid": 'CH4', "T": T_ambient, 'P': 40, 'LHV': lhv_ch4, 'buses' : ['NG'] },
    'H2 production': {"fluid": 'H2', "T": 50, 'P': 30, 'LHV': lhv_h2, 'buses' : ['H2' , 'H2 distribution', 'H2 delivery']},
    'H2 to methanolisation': {"fluid": 'H2', "T": T_max_comp, 'P': 80 , 'LHV': lhv_h2, 'buses' : ['H2 to methanolisation']},
    'H2 to biomethanation': {"fluid": 'H2', "T": T_ambient, 'P': 1, 'LHV': lhv_h2, 'buses' : ['H2 to biomethanatio']},
    'H2 to cat methanation': {"fluid": 'H2', "T": T_max_comp, 'P': 20, 'LHV': lhv_h2, 'buses' : ['H2 to cat methanation']},
    'H2 HP storage': {"fluid": 'H2', "T": T_ambient, 'P': 150, 'LHV': lhv_h2, 'buses' : ['H2 HP storage']},
    'CO2 biogas upgrading': {"fluid": 'CO2', "T": 50, 'P': 1, 'buses' : ["CO2 sep", "CO2 distribution", 'CO2 to biomethanation']},
    'biogas': {"fluid": "biogas", "T": 50, 'P': 1, 'LHV': lhv_biogas, 'buses' : ['biogas', 'biogas to biomethanation']}, # coolprop name assigned in function (as a mixture)
    'biogas to cat methanation': {"fluid": "biogas", "T": T_max_comp, 'P': 20, 'LHV': lhv_biogas, 'buses' :['biogas to cat methanation']},
    'bioCH4': {"fluid": "CH4", "T": 50, 'P': 1, 'LHV': lhv_ch4, 'buses' : ['bioCH4', 'biomethane', 'bio methane', 'bio CH4']},  # coolprop name assigned in function (as a mixture)
    'meoh': {"fluid": "Methanol", "T": 50, 'P': 1, 'LHV': lhv_meoh, 'buses' : ['Methanol']},
    'CO2 to methanolisation': {"fluid": 'CO2', "T": T_max_comp, 'P': 80, 'buses' : ['CO2 to methanolisation', 'CO2 to meoh']},
    'CO2 to cat methanation': {"fluid": 'CO2', "T": T_max_comp, 'P': 20, 'buses' :['CO2 to cat methanation']},
    'CO2 HP storage': {"fluid": 'CO2', "T": T_ambient, 'P': 60, 'buses' : ['CO2 HP storage']},
    'CO2 from HP storage': {"fluid": 'CO2', "T": T_ambient, 'P': 30},
    'CO2 from Liq storage': {"fluid": 'CO2', "T": T_ambient, 'P': 16},
    'CO2 Liq storage': {"fluid": 'CO2', "T": -26, 'P': 16, 'buses' : ['CO2 Liq sequestration', 'CO2 Liq storage']},
    'pellets': {"fluid": "pellets", "T": T_ambient, 'LHV': lhv_pellets, 'moisture' : 0.13, 'buses': ['pellets']}, # name NOT valid in coolprop
    'chips': {"fluid": "pellets", "T": T_ambient, 'LHV': lhv_chips, 'moisture' : 0.5, 'buses' :["moist biomass"]},  # name NOT valid in coolprop
}

symbiosis_n = pd.DataFrame.from_dict(symbiosis_data, orient="index")

# list of mixtures defined in the model
mixture_database ={'biogas' : biogas_mix}

# --- Component specific Calculations & HELPERS

def belt_dryer_investment(DM_flow_guess : float = 3, bed_height : float = 0.1, dry_bulk_density : float = 0.130, residence_time : float = 50 ):
    """"Calcualtes the cost of a biomass belt dryier using the correlation from:
    'Techno-economic evaluation of biomass drying in moving beds: The effect of drying kinetics
    on drying costs' 
    DOI: 10.1080/07373937.2018.1492615 

    inputs:
    -   biomass_moisture_content (kg_h2o /kg_tot)
    -   final_moisture (kg_h2o /kg_tot)
    -   DM_flow_guess (t/h)
    
    Outputs:
    -   investment : €/ MW dryed biomass
    -   A_belt : m3/MW dryed biomass

    industrial example: https://www.andritz.com/resource/blob/471276/94850607aea0f43b0f160ef4a1c827ed/pas-belt-dryers-for-sludge-biomass-rdf-en-web-data.pdf?utm_source=chatgpt.com
    """""
    # assumptions
    # bed_height = 0.1  # m
    # dry_bulk_density = 0.130  # t/m3 straw
    # residence_time = 50  # min
    # DM_flow_guess = 3 # t/h
    # LHV_final = 15/3.6 MWh/t

    # Cross-sectional Area of the continous running belt (m2)
    A_belt = DM_flow_guess / 3600 * residence_time * 60 / (dry_bulk_density * bed_height)

    if A_belt > 480:
        print('belt dryer area too large for one line')

    investment = (A_belt * (-3095 * np.log(A_belt / 480) + 5838)) / DM_flow_guess # €/(t/h DM)

    belt_dryer ={'investment' : investment, # €/(t/h DM)
                 'belt area' : A_belt/ DM_flow_guess } # m2/(t/h DM)
    return belt_dryer



def match_fluid_name_coolprop(fluid, mixture_db=None):
    """
    Return both a CoolProp AbstractState (ready to use) and the canonical fluid name.

    If a mixture (e.g. 'biogas') is defined in mixture_db, create the mixture state automatically.
    """
    fluid = str(fluid).strip()

    # Default mixture database (can be overridden)
    if mixture_db is None:
        mixture_db = {}

    # -----------------------------
    # 1. Check if fluid is a user-defined mixture
    # -----------------------------
    mixture = None
    for mix_name, mix_def in mixture_db.items():
        if fluid.lower() == mix_name.lower():
            mixture = mix_def
            break

    # -----------------------------
    # 2. If it’s a mixture, build the state
    # -----------------------------
    if mixture is not None:
        if isinstance(mixture, dict):
            comps, fracs = list(mixture.keys()), list(mixture.values())
        else:
            comps, fracs = zip(*mixture)
        fracs = [f / sum(fracs) for f in fracs]
        AS = CP.AbstractState("HEOS", "&".join(comps))
        AS.set_mole_fractions(fracs)

        # 🔧 build phase envelope to enable PS updates
        try:
            AS.build_phase_envelope("none")
        except Exception as e:
            print(f"⚠️ Could not build phase envelope for {comps}: {e}")

        resolved_name = "&".join(comps)
        return AS, resolved_name

    # -----------------------------
    # 3. Otherwise, handle as pure fluid
    # -----------------------------
    match = {
        "Hydrogen": ["H2", "hydrogen"],
        "Methane": ["CH4", "methane", "biomethane"],
        "CarbonDioxide": ["CO2", "carbon dioxide"],
        "Water": ["H2O", "water", "steam"],
        "Air": ["Air", "air"],
        "Methanol": ["methanol", "meoh", "ch3oh"],
        "Nitrogen": ["N2", "nitrogen"],
    }

    # Resolve alias
    for canonical, aliases in match.items():
        if fluid.lower() == canonical.lower() or fluid.lower() in [a.lower() for a in aliases]:
            fluid = canonical
            break

    # Build CoolProp state for the pure fluid
    AS = CP.AbstractState("HEOS", fluid)
    return AS, fluid


def compress_multistage_with_Tcap(fluid_state, fluid_name: str,
                                  p_in_bar: float, p_out_bar: float,
                                  T_in_C: float, T_max_C: float = 160,
                                  eta_s: float = 0.75, r_max: float = 2.5,
                                  T_cool_C: float = 50, T_split_C=None):
    """
    Multi-stage compressor with max stage ratio and max discharge temperature.
    Intercools to T_cool (default: T_in) after each stage.
    Works with both pure fluids (str) and CoolProp mixtures (AbstractState).

    Parameters
    ----------
    fluid_state : str or CoolProp.AbstractState
        Either a pure fluid name (e.g., "CO2", "Hydrogen")
        or a prebuilt CoolProp.AbstractState for mixtures.
    fluid_name : str
        Human-readable fluid name, used in output.
    """

    import math, CoolProp.CoolProp as CP

    # --- Temperature & pressure conversions ---
    if T_cool_C is None:
        T_cool_C = T_in_C
    T_in_K = T_in_C + 273.15
    T_max_K = T_max_C + 273.15
    T_cool_K = T_cool_C + 273.15
    T_split_K = T_split_C + 273.15 if T_split_C is not None else None
    p_in = p_in_bar * 1e5
    p_out_target = p_out_bar * 1e5

    # ===============================================================
    # Define unified CoolProp property functions
    # ===============================================================
    if isinstance(fluid_state, str):
        # --- Pure fluid path ---
        def get_s(T, P): return CP.PropsSI('S', 'T', T, 'P', P, fluid_state)
        def get_h(T, P): return CP.PropsSI('H', 'T', T, 'P', P, fluid_state)
        def get_T(P, H): return CP.PropsSI('T', 'P', P, 'H', H, fluid_state)
        def get_h_PS(P, S): return CP.PropsSI('H', 'P', P, 'S', S, fluid_state)

    else:
        # --- Mixture path ---
        AS = fluid_state

        def get_s(T, P):
            AS.update(CP.PT_INPUTS, P, T)
            return AS.smass()

        def get_h(T, P):
            AS.update(CP.PT_INPUTS, P, T)
            return AS.hmass()

        def get_T(P, H):
            """
            Robustly compute temperature from (H, P) for mixtures.
            Avoids CoolProp 'stationary point' errors by bounding and recovering gracefully.
            """
            try:
                AS.update(CP.HmassP_INPUTS, H, P)
                return AS.T()
            except Exception:
                # Hard fallback: monotonic temperature search with guarded CoolProp calls
                T_low, T_high = 80.0, T_max_K + 50.0
                last_good_T, last_good_h = None, None

                for _ in range(80):
                    T_mid = 0.5 * (T_low + T_high)
                    try:
                        AS.update(CP.PT_INPUTS, P, T_mid)
                        h_mid = AS.hmass()
                        last_good_T, last_good_h = T_mid, h_mid
                    except Exception as e:
                        # Handle the 'stationary point' and other failures
                        if "stationary" in str(e).lower() or "One stationary" in str(e):
                            # Move downwards in temperature to stay within stable region
                            T_high = T_mid - 10.0
                            continue
                        else:
                            # Non-recoverable CoolProp error → skip this T
                            T_mid = min(T_mid + 5.0, T_high)
                            continue

                    # Normal bisection logic
                    if h_mid > H:
                        T_high = T_mid
                    else:
                        T_low = T_mid

                    # Converged closely enough
                    if abs(h_mid - H) / max(abs(H), 1) < 1e-5:
                        return T_mid

                # Fallback: return last good temperature if available
                if last_good_T is not None:
                    return last_good_T
                raise ValueError(f"Failed to compute T from H={H:.3e}, P={P / 1e5:.2f} bar (mixture region unstable)")

        def get_h_PS(P, S_target, T_max_K=160 + 273.15):
            T_low, T_high = 80.0, min(1500.0, T_max_K + 100)
            h_last = None
            for _ in range(60):
                T_mid = 0.5 * (T_low + T_high)
                try:
                    AS.update(CP.PT_INPUTS, P, T_mid)
                    S_mid = AS.smass()
                except Exception as e:
                    if "stationary" in str(e):
                        T_mid = min(T_mid + 5.0, T_high)
                        continue
                    else:
                        raise
                if S_mid > S_target:
                    T_high = T_mid
                else:
                    T_low = T_mid
                h_last = AS.hmass()
            return h_last

    # ===============================================================
    # Stage compression logic
    # ===============================================================
    def stage_compress(pin, Tin, pout_desired):
        s1 = get_s(Tin, pin)
        h1 = get_h(Tin, pin)

        def discharge_T_and_work(pout):
            h2s = get_h_PS(pout, s1)
            ws = h2s - h1
            w = ws / eta_s
            h2 = h1 + w
            T2 = get_T(pout, h2)
            return T2, w, ws

        # --- respect max stage ratio ---
        pout_cap = min(pout_desired, pin * r_max)
        T2_try, w_try, ws_try = discharge_T_and_work(pout_cap)

        # --- limit discharge temperature ---
        if T2_try <= T_max_K + 1e-6:
            pout, T2, w, ws = pout_cap, T2_try, w_try, ws_try
        else:
            p_lo, p_hi = pin * (1.0 + 1e-9), pout_cap
            T2_lo, _, _ = discharge_T_and_work(p_lo)
            for _ in range(60):
                p_mid = 0.5 * (p_lo + p_hi)
                T2_mid, w_mid, ws_mid = discharge_T_and_work(p_mid)
                if T2_mid > T_max_K:
                    p_hi = p_mid
                else:
                    p_lo = p_mid
                    T2_lo = T2_mid
                    w_try, ws_try = w_mid, ws_mid
            pout, T2, w, ws = p_lo, T2_lo, w_try, ws_try

        # --- aftercooling ---
        h2 = h1 + w
        h_cool = get_h(T_cool_K, pout)
        Q_after = h2 - h_cool

        # --- split duty if needed ---
        Q_above, Q_below = 0.0, 0.0
        if T_split_K is not None and T2 > T_cool_K + 1e-9:
            T_hi = max(min(T2, T_split_K), T_cool_K)
            h_split = get_h(T_hi, pout)
            Q_above = max(h2 - h_split, 0.0)
            Q_below = max(h_split - h_cool, 0.0)
            if abs((Q_above + Q_below) - Q_after) > 1e-6 * max(1.0, Q_after):
                h_split = get_h(max(T_split_K, T_cool_K), pout)
                Q_above = max(h2 - h_split, 0.0)
                Q_below = Q_after - Q_above

        #print('T_in_C',Tin - 273.15 )
        #print('T_out_C', T2 - 273.15)
        #print('p_in_bar',pin / 1e5 )
        #print('p_out_bar', pout / 1e5)

        return {
            'p_in_bar': pin / 1e5,
            'T_in_C': Tin - 273.15,
            'p_out_bar': pout / 1e5,
            'T_out_C': T2 - 273.15,
            'w_actual_J_per_kg': w,
            'w_isentropic_J_per_kg': ws,
            'Q_aftercool_J_per_kg': Q_after,
            'Q_after_above_split_J_per_kg': Q_above,
            'Q_after_below_split_J_per_kg': Q_below,
            'ratio': pout / pin
        }

    # ===============================================================
    # Compute number of stages
    # ===============================================================
    total_ratio = p_out_target / p_in
    n_min_ratio = math.ceil(math.log(total_ratio, r_max)) if total_ratio > 1 else 0
    if n_min_ratio <= 0:
        raise ValueError("p_out must be greater than p_in for compression.")

    # ===============================================================
    # Run compression sequence
    # ===============================================================
    stages = []
    pin, Tin = p_in, T_in_K
    remaining_ratio = p_out_target / pin

    for _ in range(1, n_min_ratio + 20):
        stages_left_guess = max(1, math.ceil(math.log(remaining_ratio, r_max)))
        r_eq = remaining_ratio ** (1.0 / stages_left_guess)
        pout_desired = pin * min(r_max, r_eq)

        st = stage_compress(pin, Tin, pout_desired)
        stages.append(st)
        pin = st['p_out_bar'] * 1e5
        Tin = T_cool_K
        remaining_ratio = p_out_target / pin

        if pin >= p_out_target * (1 - 1e-9):
            break

    # ===============================================================
    # Aggregate results
    # ===============================================================
    eta_motor = 0.97
    w_total = sum(s['w_actual_J_per_kg'] for s in stages) / eta_motor
    ws_total = sum(s['w_isentropic_J_per_kg'] for s in stages) / eta_motor
    Q_total = sum(s['Q_aftercool_J_per_kg'] for s in stages)
    Q_above_total = sum(s['Q_after_above_split_J_per_kg'] for s in stages)
    Q_below_total = sum(s['Q_after_below_split_J_per_kg'] for s in stages)


    return {
        'fluid': fluid_name,
        'eta_s': eta_s,
        'r_max': r_max,
        'T_cool_K': T_cool_K,
        'T_split_K': T_split_K,
        'n_stages': len(stages),
        'stages': stages,
        'specific_work_J_per_kg': w_total,
        'specific_work_kWh_per_kg': w_total / 3.6e6,
        'specific_isentropic_work_J_per_kg': ws_total,
        'specific_isentropic_work_kWh_per_kg': ws_total / 3.6e6,
        'specific_aftercool_Q_J_per_kg': Q_total,
        'specific_aftercool_Q_kWh_per_kg': Q_total / 3.6e6,
        'specific_aftercool_Q_above_split_J_per_kg': Q_above_total if T_split_C else None,
        'specific_aftercool_Q_above_split_kWh_per_kg': (Q_above_total / 3.6e6) if T_split_C else None,
        'specific_aftercool_Q_below_split_J_per_kg': Q_below_total if T_split_C else None,
        'specific_aftercool_Q_below_split_kWh_per_kg': (Q_below_total / 3.6e6) if T_split_C else None,
    }


def aftercomp_cool_duty(fluid_state, fluid_name: str,
                        p_const: float, T_in_C: float, T_cool_C: float,
                        T_split_C: float = None, clamp_to_zero: bool = True):
    """
    Compression after-cooler duty at fixed outlet pressure, with optional split around T_split_C.

    Works with both:
      - pure fluids (fluid_state = "CO2", "Methane", etc.)
      - mixtures (fluid_state = CoolProp.AbstractState with mole fractions set)

    Parameters
    ----------
    fluid_state : str or CoolProp.AbstractState
        Either the CoolProp fluid name or an initialized mixture state.
    fluid_name : str
        Canonical name for logging or output.
    p_const : float
        Constant outlet pressure [bar].
    T_in_C, T_cool_C, T_split_C : float
        Temperatures in °C.
    clamp_to_zero : bool
        If True, negative (heating) duties are set to zero.

    Returns
    -------
    dict
        Heat duties and splits in J/kg and kWh/kg.
    """

    # Convert units
    T_discharge_K = T_in_C + 273.15
    T_cool_K = T_cool_C + 273.15
    T_split_K = T_split_C + 273.15 if T_split_C is not None else None
    P = p_const * 1e5  # bar → Pa

    # Unified property access
    if isinstance(fluid_state, str):
        def get_h(T, P): return CP.PropsSI('H', 'T', T, 'P', P, fluid_state)
    else:
        AS = fluid_state
        def get_h(T, P):
            AS.update(CP.PT_INPUTS, P, T)
            return AS.hmass()

    # Main enthalpy values
    h_hot = get_h(T_discharge_K, P)
    h_cold = get_h(T_cool_K, P)
    Q_total = h_hot - h_cold  # J/kg

    if clamp_to_zero and Q_total < 0.0:
        Q_total = 0.0

    # Default split outputs
    Q_above, Q_below = None, None

    if T_split_K is not None and T_discharge_K > T_cool_K:
        # Clamp T_split to range [T_cool, T_discharge]
        T_eff = min(max(T_split_K, T_cool_K), T_discharge_K)
        h_split = get_h(T_eff, P)

        Q_above_raw = h_hot - h_split
        Q_below_raw = h_split - h_cold

        if clamp_to_zero:
            Q_above = max(Q_above_raw, 0.0)
            Q_below = max(Q_below_raw, 0.0)
        else:
            Q_above, Q_below = Q_above_raw, Q_below_raw

        # Numerical consistency
        err = (Q_above + Q_below) - Q_total
        if abs(err) > 1e-6 * max(1.0, abs(Q_total)):
            Q_below -= err
    else:
        # No meaningful cooling
        Q_above = 0.0 if clamp_to_zero else None
        Q_below = 0.0 if clamp_to_zero else None

    # Assemble outputs
    return {
        'fluid': fluid_name,
        'specific_Q_J_per_kg': Q_total,
        'specific_Q_kWh_per_kg': Q_total / 3.6e6,
        'specific_Q_above_split_J_per_kg': Q_above,
        'specific_Q_above_split_kWh_per_kg': (Q_above / 3.6e6) if Q_above is not None else None,
        'specific_Q_below_split_J_per_kg': Q_below,
        'specific_Q_below_split_kWh_per_kg': (Q_below / 3.6e6) if Q_below is not None else None,
        'T_cool_C': T_cool_C,
    }


def compressor_calculation(comp_streams, symbiosis_n):
    # The function adresses HP storage with three different logic (Mode B, C, D) or simple compressor.
    # input comp_streams: dict (must match symbiosis_n)
    # returns
    # 1) main_compression: DF with electricity demand and heat rejected to DH and LT for the compression between Pin and Pout
    # 2) extra compression:  DF with electricity demand and heat rejected to DH and LT for the compression requested by a cycle of storage

    # NOTE: This function does not create any pypsa components,only executes the calculations for compressors, intercooling between stages and final cooling.

    ##### Calculation for compressor and heat exchangers:
    IN_stream = comp_streams['IN stream']
    OUT_stream = comp_streams['OUT stream']

    fluid = symbiosis_n.at[IN_stream, 'fluid']
    fluid_state, fluid_name = match_fluid_name_coolprop(fluid, mixture_db = mixture_database)

    Pin = symbiosis_n.at[IN_stream, 'P']
    Tin = symbiosis_n.at[IN_stream,'T']
    Pout = symbiosis_n.at[OUT_stream, 'P']

    if 'ST stream' in comp_streams.keys():
        ST_stream = comp_streams['ST stream']
        Pst = symbiosis_n.at[ST_stream, 'P']
    else:
        Pst = None

    if 'ST OUT stream' in comp_streams.keys():
        ST_OUT_stream = comp_streams['ST OUT stream'] # needed only in Mode D
        Pst_out = symbiosis_n.at[ST_OUT_stream, 'P']
    else:
        Pst_out = None


    # ---- Mode Zero : compressor not needed -------
    if (Pout / Pin) < 1 and (not Pst or Pst < Pout):
        compressor_data = pd.DataFrame()
        print("WARNING: Pout <= Pin and NO storage: compressor not needed")
        return compressor_data

    else:
        # apply pre-cooling if needed
        T_in_max = symbiosis_n.at['Heat LT min', 'T'] # compressor temp input limit
        if Tin > T_in_max:
            pre_cooling = aftercomp_cool_duty(
                fluid_state=fluid_state,
                fluid_name=fluid_name,
                p_const=Pin,
                T_in_C=Tin,
                T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
                T_split_C=symbiosis_n.at['Heat DH min', 'T'],
                clamp_to_zero=True)

            Tin = pre_cooling['T_cool_C']

        else:
            pre_cooling = {'specific_Q_above_split_kWh_per_kg' : 0,
                           'specific_Q_below_split_kWh_per_kg' : 0}

    # ----- Mode A : Pin < Pout,  Pst = None -------------
    if (Pout / Pin) > 1 and not Pst:
        # in this mode only compression is required
        #mode = 'A'
        #print('mode, Pin, Pout, fluid', mode , Pin, Pout, fluid)

        ### main compression:
        main_compression = compress_multistage_with_Tcap(
            fluid_state=fluid_state,
            fluid_name=fluid_name,
            p_in_bar=Pin,
            p_out_bar=Pout,
            T_in_C=Tin,
            T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
            T_split_C=symbiosis_n.at['Heat DH min', 'T']  # default to T_in
        )

        # ---- Cooling after compression if requested
        if symbiosis_n.at[OUT_stream, 'T'] <  main_compression["stages"][-1]["T_out_C"]:
            cooling_after_main_comp = aftercomp_cool_duty(
                fluid_state=fluid_state,
                fluid_name=fluid_name,
                p_const=main_compression["stages"][-1]["p_out_bar"],
                T_in_C=main_compression["stages"][-1]["T_out_C"],
                T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
                T_split_C=symbiosis_n.at['Heat DH min', 'T'],
                clamp_to_zero=True)
        else:
            cooling_after_main_comp = {'specific_Q_above_split_kWh_per_kg': 0,
                                        'specific_Q_below_split_kWh_per_kg': 0}

        main_compression['specific_aftercool_Q_above_split_kWh_per_kg'] += (cooling_after_main_comp['specific_Q_above_split_kWh_per_kg'] + pre_cooling['specific_Q_above_split_kWh_per_kg'])
        main_compression['specific_aftercool_Q_below_split_kWh_per_kg'] += (cooling_after_main_comp['specific_Q_below_split_kWh_per_kg'] + pre_cooling['specific_Q_below_split_kWh_per_kg'])
        extra_compression = {k : 0 for k in main_compression}


    # ----- Mode B : Pin = Pout < Pst -------------
    elif (Pout / Pin) == 1 and (Pst / Pout) > 1:
        # in this mode the compression is required only for storing the gas at HP
        #mode = 'B'
        #print('mode, Pin, Pout, Pst, fluid', mode , Pin, Pout, Pst, fluid)

        ### storage compression:
        extra_compression = compress_multistage_with_Tcap(
            fluid_state=fluid_state,
            fluid_name=fluid_name,
            p_in_bar=Pin,
            p_out_bar=Pst,
            T_in_C=Tin,
            T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
            T_split_C=symbiosis_n.at['Heat DH min', 'T']  # default to T_in
        )

        # ---- Cooling before storage
        cooling_after_main_comp = aftercomp_cool_duty(
            fluid_state=fluid_state,
            fluid_name=fluid_name,
            p_const=extra_compression["stages"][-1]["p_out_bar"],
            T_in_C=extra_compression["stages"][-1]["T_out_C"],
            T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
            T_split_C=symbiosis_n.at['Heat DH min', 'T'],
            clamp_to_zero=True)

        extra_compression['specific_aftercool_Q_above_split_kWh_per_kg'] += (cooling_after_main_comp['specific_Q_above_split_kWh_per_kg'] + pre_cooling['specific_Q_above_split_kWh_per_kg'])
        extra_compression['specific_aftercool_Q_below_split_kWh_per_kg'] += (cooling_after_main_comp['specific_Q_below_split_kWh_per_kg'] + pre_cooling['specific_Q_below_split_kWh_per_kg'])
        main_compression = {k : 0 for k in extra_compression}

    # ---- Mode C: Pin< Pout< Pst -------
    # in this mode the main compression (Pin -> Pout) followed by a second compression for the storage.
    elif (Pout / Pin) > 1 and (Pst / Pout) > 1:
        #mode = 'C'
        #print('mode, Pin, Pout, Pst, fluid', mode , Pin, Pout, Pst, fluid)

        ### main compression:
        main_compression = compress_multistage_with_Tcap(
            fluid_state=fluid_state,
            fluid_name=fluid_name,
            p_in_bar=Pin,
            p_out_bar=Pout,
            T_in_C=Tin,
            T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
            T_split_C=symbiosis_n.at['Heat DH min', 'T']  # default to T_in
        )

        # ---- Cooling after compression if requested
        cooling_after_main_comp = aftercomp_cool_duty(
            fluid_state=fluid_state,
            fluid_name=fluid_name,
            p_const=main_compression["stages"][-1]["p_out_bar"],
            T_in_C=main_compression["stages"][-1]["T_out_C"],
            T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
            T_split_C=symbiosis_n.at['Heat DH min', 'T'],
            clamp_to_zero=True)


        if symbiosis_n.at[OUT_stream, 'T'] >  main_compression["stages"][-1]["T_out_C"]:
            cooling_before_extra_compression = cooling_after_main_comp.copy()
            cooling_after_main_comp['specific_Q_above_split_kWh_per_kg'] = 0
            cooling_after_main_comp['specific_Q_below_split_kWh_per_kg'] = 0

        main_compression['specific_aftercool_Q_above_split_kWh_per_kg'] += (cooling_after_main_comp['specific_Q_above_split_kWh_per_kg'] + pre_cooling['specific_Q_above_split_kWh_per_kg'])
        main_compression['specific_aftercool_Q_below_split_kWh_per_kg'] += (cooling_after_main_comp['specific_Q_below_split_kWh_per_kg'] + pre_cooling['specific_Q_below_split_kWh_per_kg'])


        # ----- extra compression for storage ---
        extra_compression = compress_multistage_with_Tcap(
            fluid_state=fluid_state,
            fluid_name=fluid_name,
            p_in_bar=main_compression["stages"][-1]["p_out_bar"],
            p_out_bar=symbiosis_n.at[ST_stream, 'P'],
            T_in_C=cooling_after_main_comp['T_cool_C'],
            T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
            T_split_C=symbiosis_n.at['Heat DH min', 'T']  # default to T_in
        )

        # ---- final Cooling before storage ----
        final_cooling = aftercomp_cool_duty(
            fluid_state=fluid_state,
            fluid_name=fluid_name,
            p_const=extra_compression["stages"][-1]["p_out_bar"],
            T_in_C=extra_compression["stages"][-1]["T_out_C"],
            T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
            T_split_C=symbiosis_n.at['Heat DH min', 'T'],
            clamp_to_zero=True)

        extra_compression['specific_aftercool_Q_above_split_kWh_per_kg'] += (cooling_before_extra_compression['specific_Q_above_split_kWh_per_kg'] + final_cooling['specific_Q_above_split_kWh_per_kg'])
        extra_compression['specific_aftercool_Q_below_split_kWh_per_kg'] += (cooling_before_extra_compression['specific_Q_below_split_kWh_per_kg'] + final_cooling['specific_Q_below_split_kWh_per_kg'])


    # ---- Mode D: Pin < Pst_out < Pst < Pout -------
    # in this mode the main compression (Pin -> Pout) followed by a second compression for the storage. The return pressure is assumed > Pout
    elif (Pst/ Pin) > 1 and (Pout/ Pst) > 1:
        #mode = 'D'
        #print('mode, Pin, Pout, Pst, fluid', mode , Pin, Pout, Pst, fluid)

        ### main compression:
        main_compression = compress_multistage_with_Tcap(
            fluid_state=fluid_state,
            fluid_name=fluid_name,
            p_in_bar=Pin,
            p_out_bar=Pout,
            T_in_C=Tin,
            T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
            T_split_C=symbiosis_n.at['Heat DH min', 'T']  # default to T_in
        )

        # ---- Cooling after compression if requested
        cooling_after_main_comp = aftercomp_cool_duty(
            fluid_state=fluid_state,
            fluid_name=fluid_name,
            p_const=main_compression["stages"][-1]["p_out_bar"],
            T_in_C=main_compression["stages"][-1]["T_out_C"],
            T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
            T_split_C=symbiosis_n.at['Heat DH min', 'T'],
            clamp_to_zero=True)

        if symbiosis_n.at[OUT_stream, 'T'] <  main_compression["stages"][-1]["T_out_C"]:
            main_compression['specific_aftercool_Q_above_split_kWh_per_kg'] += cooling_after_main_comp['specific_Q_above_split_kWh_per_kg']
            main_compression['specific_aftercool_Q_below_split_kWh_per_kg'] += cooling_after_main_comp['specific_Q_below_split_kWh_per_kg']

        else:
            cooling_after_main_comp = {'specific_Q_above_split_kWh_per_kg' : 0,
                                       'specific_Q_below_split_kWh_per_kg' : 0}

            main_compression['specific_aftercool_Q_above_split_kWh_per_kg'] += (
                        cooling_after_main_comp['specific_Q_above_split_kWh_per_kg'] + pre_cooling[
                    'specific_Q_above_split_kWh_per_kg'])
            main_compression['specific_aftercool_Q_below_split_kWh_per_kg'] += (
                        cooling_after_main_comp['specific_Q_below_split_kWh_per_kg'] + pre_cooling[
                    'specific_Q_below_split_kWh_per_kg'])


        # ----- extra compression for storage (Pin - Pst)  ---
        extra_compression = compress_multistage_with_Tcap(
            fluid_state=fluid_state,
            fluid_name=fluid_name,
            p_in_bar=Pin,
            p_out_bar=Pst,
            T_in_C=Tin,
            T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
            T_split_C=symbiosis_n.at['Heat DH min', 'T']  # default to T_in
        )

        # ---- final Cooling before storage ----
        cooling_before_storage = aftercomp_cool_duty(
            fluid_state=fluid_state,
            fluid_name=fluid_name,
            p_const=extra_compression["stages"][-1]["p_out_bar"],
            T_in_C=extra_compression["stages"][-1]["T_out_C"],
            T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
            T_split_C=symbiosis_n.at['Heat DH min', 'T'],
            clamp_to_zero=True)

        # ----- extra compression2 for storage (Pst_out - Pout)  ---
        extra_compression2 = compress_multistage_with_Tcap(
            fluid_state=fluid_state,
            fluid_name=fluid_name,
            p_in_bar=Pst_out,
            p_out_bar=Pout,
            T_in_C=cooling_before_storage['T_cool_C'],
            T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
            T_split_C=symbiosis_n.at['Heat DH min', 'T']  # default to T_in
        )

        extra_compression['specific_work_kWh_per_kg'] += extra_compression2['specific_work_kWh_per_kg'] - main_compression['specific_work_kWh_per_kg']
        extra_compression['specific_aftercool_Q_above_split_kWh_per_kg'] +=  extra_compression2['specific_aftercool_Q_above_split_kWh_per_kg']- main_compression['specific_aftercool_Q_above_split_kWh_per_kg']
        extra_compression['specific_aftercool_Q_below_split_kWh_per_kg'] +=  extra_compression2['specific_aftercool_Q_below_split_kWh_per_kg']- main_compression['specific_aftercool_Q_below_split_kWh_per_kg']
    else:
        print("WARNING: compressor not installed")
        compressor_data = pd.DataFrame()
        return compressor_data

    # get LHV or keep mass based stream (e.g. CO2)
    val = symbiosis_n.at[IN_stream, 'LHV']
    if pd.notna(val):
        div_val = val  # energy-based stream
    else:
        div_val = 1  # mass-based stream

    data = [
        [main_compression['specific_work_kWh_per_kg'] / div_val,
         extra_compression['specific_work_kWh_per_kg'] / div_val],  # electricity-input
        [main_compression['specific_aftercool_Q_above_split_kWh_per_kg'] / div_val,
         extra_compression['specific_aftercool_Q_above_split_kWh_per_kg'] / div_val],  # heat-output DH
        [main_compression['specific_aftercool_Q_below_split_kWh_per_kg'] / div_val,
         extra_compression['specific_aftercool_Q_below_split_kWh_per_kg'] / div_val],  # heat-output LT
    ]

    compressor_data = pd.DataFrame(index=['electricity-input', 'heat-output DH', 'heat-output LT'],
                                   columns=['main compression', 'storage compression'], data=data)

    compressor_data.attrs['fluid'] = fluid
    compressor_data.loc['Pin',: ] = Pin
    compressor_data.loc['Pout', :] = Pout

    if Pst:
        compressor_data.loc['Pst', :] = Pst
    if Pst_out:
        compressor_data.loc['Pst_out', :] = Pst_out

    return compressor_data



def check_symbiosis_n(symbiosis_n):
    import itertools
    all_buses = list(itertools.chain.from_iterable(
        buses for buses in symbiosis_n['buses'].dropna() if isinstance(buses, list)
    ))

    # Find duplicates
    duplicates = [b for b in set(all_buses) if all_buses.count(b) > 1]

    if duplicates:
        print("⚠️ Duplicated bus names found:", duplicates)
    else:
        print("✅ symbiosis network buses OK")
    return


# ----- sanity check for col buses not having repetitions
check_symbiosis_n(symbiosis_n)

# ---- Technology and process specific inputs ---------
tech_inputs = {
    ('biomass belt dryer', 'T_min_heat'): {
        'value': 90,
        'unit': 'C',
        'further description': 'min temp for heat supply',
    },
    ('biomass belt dryer', 'DM flow reference'): {
        'value': 3,
        'unit': 't/h',
        'source': 'manual input',
        'further description': 'reference for investment calculation ',
    },
    ('biogas', 'DM feedstock input'): {
        'value': 0.12,
        'unit': '% DM ',
        'source': 'Danish Energy Agency, PFD for renewable fuels.xlsx',
        'further description': 'manure mix ',
    },
    ('biogas', 'DM conversion') : {
        'value' : 0.4632,
        'unit' : '% inut DM to biogas',
        'source': 'Danish Energy Agency, PFD renewable fuels.xlsx',
    },
    ('biogas', 'DM flow reference'): {
        'value': 3,
        'unit': 't/h DM',
        'source': 'Own assumption',
    },
    ('biogas', 'DM digestate'): {
    'value': 0.068,
    'unit': '% DM in Digestate',
    'source': 'Danish Energy Agency, PFD for renewable fuels.xlsx',
    },
    ('biogas', 'DM output'): {
        'value': 0.135,
        'unit': '(tDM) digestate / MWhCH4',
        'source': 'Calculation from: Danish Energy Agency, PFD for renewable fuels.xlsx',
    }
}

# inputs for pre-estimation of electricity demand for MeOh production.
IN_stream = 'H2 production'
OUT_stream = 'H2 to methanolisation'
fluid = symbiosis_n.at[IN_stream, 'fluid']
fluid_state, fluid_name = match_fluid_name_coolprop(fluid, mixture_db=mixture_database)

Pin = symbiosis_n.at[IN_stream, 'P']
Tin = symbiosis_n.at[IN_stream, 'T']
Pout = symbiosis_n.at[OUT_stream, 'P']

### main compression H2:
H2_comp_res = compress_multistage_with_Tcap(
    fluid_state=fluid_state,
    fluid_name=fluid_name,
    p_in_bar=Pin,
    p_out_bar=Pout,
    T_in_C=Tin,
    T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
    T_split_C=symbiosis_n.at['Heat DH min', 'T']  # default to T_in
)

### main compression CO2:
IN_stream = 'CO2 biogas upgrading'
OUT_stream = 'CO2 to methanolisation'
fluid = symbiosis_n.at[IN_stream, 'fluid']
fluid_state, fluid_name = match_fluid_name_coolprop(fluid, mixture_db=mixture_database)

Pin = symbiosis_n.at[IN_stream, 'P']
Tin = symbiosis_n.at[IN_stream, 'T']
Pout = symbiosis_n.at[OUT_stream, 'P']

CO2_comp_res = compress_multistage_with_Tcap(
    fluid_state=fluid_state,
    fluid_name=fluid_name,
    p_in_bar=Pin,
    p_out_bar=Pout,
    T_in_C=Tin,
    T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
    T_split_C=symbiosis_n.at['Heat DH min', 'T']  # default to T_in
)

# ---- Update tech_inputs:
tech_inputs['hydrogen storage compressor MeOH', 'electricity-input'] = {
    'value': H2_comp_res['specific_work_kWh_per_kg'] / lhv_h2,
    'unit': 'MW/MW_H2',
    'source': 'calculated ',
    'further description': 'calculated based on: Isoentropic efficiency and max comp ratio, max outlet temp. CoolProp',
}
tech_inputs['CO2 industrial compressor MeOH', 'electricity-input'] = {
    'value': CO2_comp_res['specific_work_kWh_per_kg'],
    'unit': 'MWh/t_CO2',
    'source': 'calculated ',
    'further description': 'calculated based on: Isoentropic efficiency and max comp ratio, max outlet temp. CoolProp',
    }

#  -----TECHNOLOGIES TO ADD TO TECHNOLOGY-DATA (ALSO AFTER CALCULATIONS) ----
#  PWH pipe
tech_inputs['PWH pipe', 'heat loss'] = {
    'value': 0.02,
    'unit': 'MW/MW',
    'source': 'Technology Data for transport of energy',
    'further description': 'Heat exchanged for DH transmission, sheet 113_11. constant for all years',
    }
tech_inputs['PWH pipe', 'distance'] = {
    'value': 5,
    'unit': 'km',
    'source': 'own assumption',
    'further description': 'based on estiamtion at GreenLab Skive',
    }

# CO2 pipe
tech_inputs['CO2 gas pipe', 'distance'] = {
    'value': 2,
    'unit': 'km',
    'source': 'own assumption',
    'further description': 'based on estimation at GreenLab Skive',
    }

# H2 pipe
tech_inputs['H2 pipe', 'distance'] = {
    'value': 2,
    'unit': 'km',
    'source': 'own assumption',
    'further description': 'based on estimation at GreenLab Skive',
    }

# ----- INVESTMENT Calculation for biomass dryer (based on size guess).
# Biomass dryer
DM_flow_guess = tech_inputs['biogas', 'DM flow reference']['value']
belt_dryer = belt_dryer_investment(DM_flow_guess = DM_flow_guess)
tech_inputs[('biomass belt dryer', 'investment')] = {
    'value': belt_dryer['investment'],
    'unit': '€/ (t/h DM)',
    'source': 'calculated based on: DOI: 10.1080/07373937.2018.1492615',
    'further description': 'calculated based on: DM flow reference',
    'currency_year': 2025}

