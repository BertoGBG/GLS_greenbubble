# SPDX-License-Identifier: MIT
"""Project-specific technology parameters not available in the technology-data catalogue.

This module defines techno-economic and physical parameters for technologies
that are specific to the GreenBubble / GreenLab Skive cluster.  Values are
compiled into the ``tech_inputs`` dict, which is merged into the standard
cost DataFrame by :func:`scripts.helpers.read_costs`.

Physical property calculations (compressor work, isentropic efficiency, etc.)
use `CoolProp <http://www.coolprop.org/>`_ for rigorous thermodynamic data.

.. note::
   Temperature and pressure levels for compressors are defined at the top of
   this module and referenced throughout.
"""

import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

# --- Process inputs: stream state (fluid / T / P / LHV) --------------------------
# These now live in config/p_config.default.yaml (+ gitignored p_config.yaml override),
# loaded by scripts/config.py. Only DERIVED quantities are computed here -- anything
# typed by a human belongs in the YAML.
#
# symbiosis_n keeps its name, shape and contents exactly: it is built from the YAML and
# is field-for-field identical to the dict it replaces, so every existing look-up
# (58 here, 67 in prepare_network.py) is untouched.
from scripts import config as _c

T_max_comp = _c.p_globals["T_max_comp"]   # maximum discharge temperature for all compressors
T_ambient  = _c.p_globals["T_ambient"]

# biogas composition (used only locally)
biogas_mix = _c.p_mixtures["biogas"]
M_CH4 = CP.PropsSI("M", "T", 300, "P", 1e5, "Methane")       # [kg/mol]
M_CO2 = CP.PropsSI("M", "T", 300, "P", 1e5, "CarbonDioxide") # [kg/mol]
M_mix = biogas_mix['Methane']*M_CH4 + biogas_mix['CarbonDioxide']*M_CO2
w_CH4 = biogas_mix['Methane']*M_CH4 / M_mix
w_CO2 = biogas_mix['CarbonDioxide']*M_CO2 / M_mix

_lhv = _c.p_globals["lhv"]
lhv_ch4     = _lhv["ch4"]
lhv_h2      = _lhv["h2"]
lhv_meoh    = _lhv["meoh"]
lhv_pellets = _lhv["pellets"]
lhv_chips   = _lhv["chips"]
lhv_biogas  = lhv_ch4 * w_CH4   # derived, mirrors config._p_derive_lhv_biogas

# CoolProp - cache to avoid rebuilding phase envelopes for mixtures repeatedly (needed for not pure fluids)
_AS_cache = {}

# ------ Streams in the symbiosis network (see config/p_config.default.yaml) -------
# index is a UNIQUE NAME used for look-up in the model and to add buses
# T : Celsius ; P : bar(a)
symbiosis_n = _c.p_streams.copy()

# list of mixtures defined in the model
mixture_database = dict(_c.p_mixtures)

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
    elif (Pout / Pin) == 1 and (Pst is not None) and (Pst / Pout) > 1:
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
    elif (Pout / Pin) > 1 and (Pst is not None) and (Pst / Pout) > 1:
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
    elif (Pst is not None) and (Pst / Pin) > 1 and (Pout / Pst) > 1:
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
        print(f"WARNING: {fluid} compressor not installed")
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
        print("✅ symbiosis network buses definition OK")
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
# ----------------------------------------------------------------------------------
# Methanol synthesis / distillation split  (option: n_options['meoh split','enable'])
# ----------------------------------------------------------------------------------
# SOURCES (cite these, with table, wherever these numbers are reported)
#
#  [DEA]  Danish Energy Agency, "Technology Data for Renewable Fuels",
#         sheet "98 Methanol from hydrogen". Reached via the technology-data fork
#         (BertoGBG/technology-data, branch pypsa-eur_AA) as technology 'methanolisation'.
#         Note F of that sheet: "Steam produced in the methanol reactor is reused for
#         heating purposes in the distillation section. The value provided states the
#         NET import steam." -> DEA never reports reactor and reboiler separately.
#
#  [ALA]  Alamia A., Partoon B., Rattigan E., Andresen G.B. (2024), "Optimizing hydrogen
#         and e-methanol production through Power-to-X integration in biogas plants",
#         Energy Conversion and Management 322:119175.
#         doi:10.1016/j.enconman.2024.119175
#         -> Table 4, column "Methanol Synthesis" (model inputs, 2030). The GreenBubble
#            base paper; its methanol column cites [NIE] + [DEA].
#
#  [OLI]  Lacerda de Oliveira Campos B., John K., Beeskow P., Herrera Delgado K.,
#         Pitter S., Dahmen N., Sauer J. (2022), "A Detailed Process and Techno-Economic
#         Analysis of Methanol Synthesis from H2 and CO2 with Intermediate Condensation
#         Steps", Processes 10(8):1535.  doi:10.3390/pr10081535
#         -> Table S14 of the Supplementary Material: per-exchanger heat duties.
#            Section E: one-step plant description (equipment temperatures).
#            Local copies: text_docs/meoh_distillation/processes-10-01535.pdf and
#                          text_docs/meoh_distillation/processes-1838784-supplementary.pdf
#         The ONLY source found that reports reactor and reboiler duties separately,
#         which is what a synthesis/distillation split needs.
#
#  [NIE]  Nieminen H., Laari A., Koiranen T. (2019), "CO2 Hydrogenation to Methanol by a
#         Liquid-Phase Process with Alcoholic Solvents: A Techno-Economic Analysis",
#         Processes 7(7):405.  doi:10.3390/pr7070405
#         -> cited by [ALA] Table 4 for the methanol column. Not yet mined.
#
#  [MBA]  Mbatha S., Everson R.C., Musyoka N.M., Langmi H.W., Lanzini A., Brilman W.
#         (2021), "Power-to-methanol process: a review...", Sustainable Energy & Fuels
#         5:3490-3569.  doi:10.1039/D1SE00635E
#         -> eqn (3): CO2 + 3H2 <-> CH3OH + H2O, dH_298 = -49.2 kJ/mol.
#
#  [MUC]  Mucci S., Mitsos A., Bongartz D. (2023), "Cost-Optimal Power-to-Methanol:
#         Flexible Operation or Intermediate Storage?", arXiv:2305.18338.
#         doi:10.48550/arXiv.2305.18338
#         -> source of the aggregate coefficients in Taslimi et al. Table 2; itemised
#            Biegler/Guthrie equipment costs exist in their model but are not published.
#
#  [MAG]  Methanol Magic LLC, "Methanol Plant Feasibility Study", ChE473k, University of
#         Texas at Austin, Spring 2015.
#         Local copy: text_docs/meoh_distillation/Methanol Magic Senior Design Report.pdf
#         -> Supplementary Tables A8-A14: itemised bare equipment costs, tagged by unit
#            number, for a 5000 t/d methanol plant. The only source found that costs the
#            synthesis loop and the distillation train SEPARATELY, which is what a
#            CAPEX split needs. Table 4: Chilton method installation factors.
#         CAVEAT: a student design report, not peer reviewed, and the plant is
#         shale-gas-to-syngas, not CO2 + H2. Used for RATIOS only, never for levels.
#
# ----------------------------------------------------------------------------------
# HEAT BALANCE -- [DEA] for levels, [OLI] for the split. RESOLVED.
#
# [DEA] reports only TWO plant-level numbers (Note F: "Steam produced in the methanol
# reactor is reused for heating purposes in the distillation section. The value provided
# states the NET import steam."):
#       net heat in   0.1047      heat out (district heating)   0.2562
#                     ^ the COMPILED value; the sheet's own rounding gives 0.1049
# A split needs THREE. The missing degree of freedom is X, the heat the synthesis block
# exports. The code below closes it structurally rather than by transcription:
#
#       synthesis    heat-output =  X
#       distillation heat-input  =  X + 0.1047      <- DERIVED in prepare_network.py
#       distillation heat-output =      0.2562
#
# so the COUPLED net is (X + 0.1047) - X = 0.1047 identically, for ANY X. The split
# therefore reproduces [DEA] exactly and can always collapse back to the monolithic
# solution -- which is the property that makes the flexibility result meaningful.
# X is the cost of DECOUPLING, and it is the one number [DEA] structurally cannot give.
#
# X is taken from [OLI], which is the right source: H2/CO2 = 3.000 feed, i.e. our exact
# chemistry and our exact water make (H2O/MeOH = 1.009 out). [OLI] Table S14, one-step,
# normalised over 801.8 MW_MeOH:
#       reactor (247.5 C, boiling water)       61.8 MW   0.0771  = dH, confirms [MBA]
#       HE5  product -> column feed           103.3 MW   0.1288  CROSSES the boundary
#       HE7  product -> cooling water          24.2 MW   0.0302  out at 30 C, unusable
#       HE4, HE6  internal recuperation       110.5 MW           never leaves synthesis
#       Col. reboiler  (99.6 C)                53.7 MW   0.0670
#       Col. condenser (53 C)                 143.6 MW   0.1791
#
#   X = 0.1288, the heat that MEASURABLY crosses the synthesis/distillation boundary.
#
# Two checks on the derived reboiler (0.1288 + 0.1047 = 0.2335):
#  1. [OLI]'s column needs reboiler + feed preheat = 0.0670 + 0.1288 = 0.1958 when run
#     standalone. A first-principles boil-up at reflux 2 (1100 kJ/kg x 3 / 5.54 MWh/t)
#     gives 0.166, within 18% ignoring feed subcooling. So decoupling roughly TRIPLES
#     the reboiler duty relative to [DEA]'s 0.0670-equivalent -- the store is not free.
#  2. 0.2335 sits 19% ABOVE [OLI]'s measured 0.1958. The gap is real and is [DEA]'s, not
#     ours: [OLI] burn purge gas in a fired heater (51 MW) driving a Rankine cycle, so
#     their column is fed by turbine exhaust rather than imported steam. [DEA] models no
#     purge burner, so [DEA] imports more. We model no purge burner either, so [DEA]'s
#     convention is the correct one to inherit. The choice is also the conservative one.
#
# SENSITIVITY on X, all three endpoints referenced, none invented:
#       0.0772  reaction enthalpy alone ([MBA] eqn 3 / [OLI] reactor) -- decoupling is
#               cheapest, the store looks best
#       0.0911  the value that makes the distillation import equal [OLI]'s measured
#               standalone 0.1958
#       0.1288  [OLI] HE5, used here -- decoupling is dearest, the store looks worst
# X cannot exceed 0.2059 (= 0.0771 + 0.1288), which is all the recoverable heat the
# synthesis block has; the remaining 0.0302 leaves at 30 C and is below every band.
#
# PENDING, deliberately not changed here (it is a wiring decision, not a data one):
# by temperature [OLI] puts the reboiler at 99.6 C (fed by 110 C steam -> Heat DH is
# enough; it is currently wired to Heat MT) and the condenser at 53 C (-> Heat LT, it is
# currently wired to Heat DH). The reactor at 247.5 C is above every band the model has.
# Note also that only 0.0772 of the 0.1288 synthesis export is genuinely MT-grade; the
# balance is the HE5 stream at roughly 150 -> 60 C, so wiring it all to Heat MT is
# generous to the model by about 40% of that export.
#
# [MAG] is deliberately ABSENT from this block. It is SMR syngas: its crude is 10.6 wt%
# water against our 36.0 wt%, its columns strip reformer inerts we do not have, and its
# reboiler duty (0.2916) reflects a Grade AA spec at reflux > 7. Useful for the CAPEX
# ratio and as a documented example of the decoupled architecture; useless for heat.
# ----------------------------------------------------------------------------------
#
# Basis: per MWh_MeOH of final product, matching 'methanolisation'.
# Electricity is split 50/50 synthesis/distillation. NOTE: an earlier 90/10 was justified
# as "compressor-dominated", which does NOT hold -- the H2 and CO2 compressors are modelled
# as separate components, so [ALA] Table 4 reports methanol electricity as 0.018
# MWel/MWmeoh "without compression of H2 and CO2". What remains is recycle circulation
# and pumps, which needs re-deriving from a flowsheet. (The compiled electricity-input
# 0.271 with the hardcoded 0.1x factor gives 0.0271, which does not reproduce [ALA]'s
# 0.018 either.)
#
# ----------------------------------------------------------------------------------
# STOICHIOMETRY -- derived here, never transcribed.
#
#   CO  + 2H2 -> CH3OH                dH_298 = -90.6 kJ/mol
#   CO2 +  H2 <-> CO + H2O   (RWGS)   dH_298 = +41.2 kJ/mol
#   ----------------------------------------------------------
#   CO2 + 3H2 -> CH3OH + H2O          dH_298 = -49.4 kJ/mol   ([MBA] eqn 3: -49.2)
#
# The CO2 route IS the CO route plus reverse water-gas shift. Water is not a side
# reaction that better catalysis could avoid: CO2 carries TWO oxygens and methanol
# contains ONE, so the spare oxygen must leave, and hydrogen is the only partner
# available. RWGS therefore does two things at once -- it eats 45% of the CO
# hydrogenation exotherm AND it makes the water. The modest reactor duty and the
# wet crude are the same phenomenon, not two independent facts.
#
# Consequence for the split: water out is an IDENTITY on the methanol PRODUCED,
#     t_H2O / t_MeOH = M_H2O / M_MeOH = 0.5622
# It is NOT a function of the CO2 fed -- CO2 that leaves in the purge makes no water,
# so scaling from the feed (0.253266/M_CO2*M_H2O = 0.1037 t/MWh) overstates it by the
# purge fraction. [DEA]'s Water row gives 0.55 t/t = 97.8% of stoichiometric; the ~2%
# is byproduct formation (DME and higher alcohols shift the H2O/MeOH ratio) plus the
# water that leaves dissolved in the product. Immaterial here -- nothing in the model
# consumes water-output; its only job is fixing the crude composition for tank sizing.
# ----------------------------------------------------------------------------------
M_MEOH_gmol = CP.PropsSI("M", "T", 300, "P", 1e5, "Methanol") * 1000.0
M_H2O_gmol  = CP.PropsSI("M", "T", 300, "P", 1e5, "Water") * 1000.0

meoh_water_t_per_t = M_H2O_gmol / M_MEOH_gmol          # 0.5622 t H2O per t MeOH
w_meoh_crude       = 1.0 / (1.0 + meoh_water_t_per_t)  # 0.6401 -> crude is 64.0 wt% MeOH
rho_crude          = 1.0 / (w_meoh_crude / 791.0 + (1.0 - w_meoh_crude) / 998.2)
crude_MWh_per_m3   = rho_crude * w_meoh_crude * lhv_meoh / 1000.0   # 3.032 MWh_MeOH/m3

# Cross-check against the compiled feed: the carbon that does NOT reach methanol is the
# purge, so the feed-based number must sit a few percent ABOVE the identity.
_co2_feed_check = 0.253266 / 44.0098 * M_H2O_gmol / (1.0 / lhv_meoh)   # t_H2O per t_MeOH
assert 1.00 <= _co2_feed_check / meoh_water_t_per_t <= 1.10, (
    f"CO2 feed implies {_co2_feed_check:.4f} t_H2O/t_MeOH against a stoichiometric "
    f"{meoh_water_t_per_t:.4f}; a ratio outside 1.00-1.10 means the feed is no longer "
    f"near-stoichiometric and the split's chemistry assumptions need revisiting.")

# ----------------------------------------------------------------------------------
# CAPEX SPLIT 90/10 -- derived from [OLI], our own chemistry.
#
# SUPERSEDES an earlier 73/27 taken from [MAG]. That was the wrong plant: [MAG] runs on
# SMR syngas, and its distillation solves two problems we do not have. Its syngas carries
# 3853 lbmol/h CH4 and 2846 lbmol/h N2 from the reformer, which need a dedicated topping
# column to strip, and it targets Grade AA (99.85 wt%) at a reflux ratio above 7 in a
# second, 83-tray column. CO2 + H2 brings no CH4 and only trace N2, so our separation is
# methanol/water in ONE column -- exactly [OLI]'s.
#
# [OLI] Table S17 itemises equipment for a CO2 + H2 plant fed at H2/CO2 = 3.000, i.e. our
# stoichiometry exactly. Summing the units that GreenBubble's methanol block represents
# (M EUR 2020, one-step process):
#
#   synthesis
#     Reactor (6 modules, 48600 m2)                       32.18
#     HE4  reactor feed/effluent recuperation               3.01
#     CP-REC  recycle compressor                            0.52
#     HE7  reactor product -> cooling water                 0.48
#     FLASH3 + FLASH4 + HE6                                 0.14
#                                                         -------
#                                                          36.33   90.2%
#   distillation
#     Packed column (5 m x 30 m, 2 units)                   2.06
#     HE5  reactor product -> column feed                   0.94   (see note)
#     Reboiler                                              0.50
#     Condenser                                             0.43
#                                                         -------
#                                                           3.93    9.8%
#
# HE5 is the exchanger that couples the two blocks in [OLI]'s flowsheet. In the DECOUPLED
# architecture this model represents, it survives as the distillation feed preheater on
# utility heat, so it is charged to distillation. Moving it to synthesis gives 92.6/7.4;
# dropping it entirely gives 92.4/7.6. The split is not sensitive to the choice.
#
# EXCLUDED, and why:
#   CP1/2/3-CO2 + CP-H2   37.12   GreenBubble models feed compression as its own
#                                 components, and [OLI] compress CO2 from 1 bar, which is
#                                 not our duty -- the cost would not transfer even if the
#                                 boundary matched.
#   HE1/HE2/HE3, FLASH1/2  1.13   part of that same compression train.
#   Fired heater + blower  4.10   [OLI] burn purge gas; we do not model a purge burner.
#   Turbine + generator +
#     HE8 + pump           2.90   [OLI]'s Rankine cycle; DEA has no such cycle.
#
# Why this differs from [MAG] so violently -- normalising each block per MW_MeOH:
#       reactor        [OLI] 40.1 kEUR/MW   [MAG] 33.3 kUSD/MW   ratio 1.20
#       column train   [OLI]  3.7 kEUR/MW   [MAG] 27.4 kUSD/MW   ratio 7.36
# The reactors agree within 20%; [MAG]'s columns cost seven times as much per MW. The
# disagreement is entirely in the separation, and entirely explained by the inerts and
# the product spec above -- not by chemistry or by the water content of the crude.
#
# KNOWN GAP: [DEA]'s 1364.7451 EUR/kW almost certainly INCLUDES feed compression (its
# electricity-input does, which is why 'electricity-input-no-compression' exists), while
# GreenBubble also adds separate compressor components. That double count predates the
# split and is not introduced by it. Folding [OLI]'s compressors onto the synthesis side
# would give 95/5; 90/10 is used instead because [OLI]'s compression duty is not ours.
#
# SCALE: [OLI] is an 801.8 MW_MeOH plant. Applying their per-equipment scaling exponents
# (reactor 0.44, packed column 0.86) down to 50-100 MW moves the split to roughly 94/6,
# i.e. further toward synthesis. Not applied -- the reactor is 6 PARALLEL modules, which
# scale nearer linearly than n=0.44 implies, so the exponent overstates the shift.
# ----------------------------------------------------------------------------------

tech_inputs['methanol synthesis', 'hydrogen-input'] = {
    'value': 1.138, 'unit': 'MWh_H2/MWh_MeOH',
    'source': '[DEA] sheet "98 Methanol from hydrogen", via technology-data tech methanolisation',
    'further description': 'all H2 enters the synthesis step; unchanged from the aggregate unit. [ALA] Table 4 gives 1.155',
}
tech_inputs['methanol synthesis', 'carbondioxide-input'] = {
    'value': 0.248, 'unit': 't_CO2/MWh_MeOH',
    'source': '[DEA] sheet "98 Methanol from hydrogen", via technology-data tech methanolisation',
    'further description': 'all CO2 enters the synthesis step; unchanged from the aggregate unit. [ALA] Table 4 gives 0.253',
}
tech_inputs['methanol synthesis', 'electricity-input'] = {
    'value': 0.009045, 'unit': 'MWh_e/MWh_MeOH',
    'source': '[DEA] electricity-input-no-compression (0.018090) x 0.5 -- ASSUMED SPLIT, no better basis',
    'further description': 'Split 50/50 as an explicit admission of ignorance rather than a false precision. An earlier 90/10 was justified as "compressor-dominated" (Taslimi et al. Table 2), which never applied and applies even less on the DEA basis, where feed compression is excluded outright: what remains is recycle circulation and pumps. Re-derive from a flowsheet when one is available.',
}
tech_inputs['methanol synthesis', 'heat-output'] = {
    'value': 0.1288, 'unit': 'MWh_th/MWh_MeOH',
    'source': '[OLI] Table S14 one-step, HE5 = 103.3 MW over 801.8 MW_MeOH; this is X, the heat crossing the synthesis/distillation boundary',
    'further description': 'This is the FREE PARAMETER of the split -- see the HEAT BALANCE block in the header. It is NOT the reaction enthalpy alone: [MBA] eqn (3) / [OLI] reactor give 0.0771 at 247.5 C, and the reactor product contributes a further 0.1288 as it cools. Sensitivity range 0.0772 - 0.1288, hard ceiling 0.2059. prepare_network.py DERIVES the distillation reboiler as this + methanolisation heat-input, so the coupled plant reproduces [DEA] for any value here.',
}
tech_inputs['methanol synthesis', 'investment'] = {
    'value': 1228.2706, 'unit': 'EUR/kW-methanol',
    'source': 'methanolisation investment (costs_2030, 1364.7451) x 0.90, split derived from [OLI] Table S17',
    'further description': 'Synthesis loop = 36.33 of 40.26 M EUR 2020 in-scope equipment, on a CO2 + H2 plant at H2/CO2 = 3.000 (our stoichiometry). See the CAPEX SPLIT block in the header. LEVEL is DEA, only the RATIO comes from [OLI].',
}
tech_inputs['methanol synthesis', 'lifetime'] = {
    'value': 30, 'unit': 'years',
    'source': '[DEA] sheet "98 Methanol from hydrogen"',
}

# NOTE: 'methanol distillation' has NO heat-input entry on purpose. The gross reboiler
# duty is currently DERIVED in prepare_network.py as
#     Q_reb = methanolisation['heat-input'] + methanol synthesis['heat-output'] = 0.1819
# That derivation is WRONG -- see item 1 in the header. [OLI] Table S14 measures 0.0670.
tech_inputs['methanol distillation', 'electricity-input'] = {
    'value': 0.009045, 'unit': 'MWh_e/MWh_MeOH',
    'source': '[DEA] electricity-input-no-compression (0.018090) x 0.5 -- ASSUMED SPLIT, no better basis',
    'further description': 'See the synthesis entry. 0.009045 + 0.009045 = 0.018090, i.e. the DEA total exactly.',
}
tech_inputs['methanol distillation', 'heat-output'] = {
    'value': 0.2562, 'unit': 'MWh_th/MWh_MeOH',
    'source': '[DEA] sheet "98 Methanol from hydrogen", district heating output row, rebased per MWh_MeOH (the sheet reports 0.2 per MWh of TOTAL INPUT; x 7.08/5.5278)',
    'further description': 'Matches [ALA] Table 4 (0.256). Was 0.1, which was neither DEA nor ALA. [OLI] Table S14 measure the column condenser at 143.6/801.8 = 0.1791 rejected at 53 C, i.e. Heat LT rather than the Heat DH this is wired to -- see PENDING in the HEAT BALANCE block. DEA is used for the level because DEA sets the plant-level heat out that the split must reproduce.',
}
tech_inputs['methanol distillation', 'investment'] = {
    'value': 136.4745, 'unit': 'EUR/kW-methanol',
    'source': 'methanolisation investment (costs_2030, 1364.7451) x 0.10, split derived from [OLI] Table S17',
    'further description': 'Column + reboiler + condenser + the HE5 feed preheater = 3.93 of 40.26 M EUR 2020 in-scope equipment. See the CAPEX SPLIT block in the header. Sums with synthesis to 1364.7451, i.e. the monolithic methanolisation investment exactly.',
}
tech_inputs['methanol distillation', 'lifetime'] = {
    'value': 30, 'unit': 'years',
    'source': '[DEA] sheet "98 Methanol from hydrogen"',
}

# ----------------------------------------------------------------------------------
# Crude methanol storage tank -- the buffer that lets synthesis and distillation
# run at different times. Without a cost here the store is FREE and the optimiser
# sizes it arbitrarily, which would make any flexibility result meaningless.
#
#  [MAG]  as cited in the header above. Here: Supplementary Table A12 (storage tanks)
#         and Table 4 (Chilton method factors).
#         CAVEAT beyond the one in the header: [MAG]'s tanks hold REFINED product;
#         ours holds 64.0 wt% crude (w_meoh_crude, derived). Used for the unit cost only.
#
#  [MI]   Methanol Institute, "Atmospheric Above Ground Tank Storage of Methanol".
#         https://methanol.org/wp-content/uploads/2016/06/AtmosphericAboveGroundTankStorageMethanol-1.pdf
#         Design/safety guidance only -- contains NO costs. Cited for the tank being
#         carbon steel with stainless cladding, and for methanol being hygroscopic
#         (a real tank needs dry-nitrogen padding of the free-board).
#
# Derivation, from [MAG] Table A12 (TK-3501/2/3, identical):
#     capacity      80,609 cuft            = 2282.6 m3   (geometry check: pi/4 x 61ft^2
#                                                         x 28ft = 2317 m3, within 1.5%)
#     bare cost     825,100 USD (2015)     = 361.5 USD/m3
#     x 4.952  Chilton fixed-capital factor ([MAG] Table 4:
#              1.43 x (1 + 0.40 + 0.07 + 0.15 + 0.50 + 0.10) x (1 + 0.35 + 0.20 + 0.01))
#                                          = 1790 USD/m3 installed
#     / 0.83   fluid volume ([MAG] Table A12)
#                                          = 2157 USD per m3 of USABLE volume
#     / 3.045  MWh_MeOH per m3 of crude at 64.5 wt% (methanol 791.0 kg/m3, water
#              998.2 kg/m3, LHV 19.9 GJ/t; ideal mixing)
#              NOTE: crude_MWh_per_m3, now derived from the stoichiometric identity,
#              gives 3.032 at 64.0 wt% -- 0.4% dearer per MWh. Below the noise of the
#              tank cost itself, so the technology-data entry is NOT re-issued.
#                                          = 708 USD/MWh_MeOH (2015 USD)
#     x 0.9015 USD->EUR (2015 average 1.1095 USD/EUR)
#     x 1.06   approximate 2015 -> 2020 EUR
#                                          = 677 EUR/MWh_MeOH (2020 EUR)
#
# The Chilton factor is corroborated independently: [OLI] Table S15 gives
# FCI = 4.8645 x equipment cost, against 4.952 here.
#
# This is a CONSERVATIVE (expensive) tank: [MAG]'s bare 361 USD/m3 is high for an
# atmospheric tank of this size because of the 0.5 in SS316 cladding. A plain carbon
# steel tank would be cheaper. Treat as an upper bound.

# The tank cost now lives in technology-data (pypsa-eur_AA, commit 99a268b) as
# technology 'methanol storage': investment 637.8 EUR/MWh_MeOH in 2015 EUR, lifetime 30
# years, FOM 2%/year, with the full derivation and both sources in its source field.
# prepare_network.py reads tech_costs.at['methanol storage','fixed'] directly, so there
# is nothing to define here -- a literal would be a second source of truth that drifts.

# ----------------------------------------------------------------------------------
# FOM 2.8 %/year -- DEA, supplied here as a WORKAROUND.
#
# [DEA] sheet "98 Methanol from hydrogen" gives Fixed O&M as 2.8 %/year of investment,
# and upstream PyPSA/technology-data compiles it as methanolisation,FOM = 2.8. Our fork
# does NOT: the value regressed on branch pypsa-eur_AA somewhere between commits 21479b6
# and de9012c and is now missing entirely (Fischer-Tropsch went 6.35 -> 0.0008 in the
# same regression, same sheet, same "Fixed O&M" row, with investment unaffected -- so the
# defect is in the fork's FOM percentage calculation, not the input data).
#
# Until technology-data is fixed, set it here for all THREE methanol technologies.
# FOM is a PERCENTAGE of investment, so it is scale-free: the same 2.8 on each half
# reproduces the monolithic total exactly, because the investment is already split 90/10.
#     fixed = (annuity(lifetime, r) + FOM/100) x investment          [helpers.py:382]
#     synthesis + distillation = (a + 0.028) x (1228.2706 + 136.4745)
#                              = (a + 0.028) x 1364.7451 = monolithic
#
# NOTE this CHANGES RESULTS: methanolisation fixed goes 109,980 -> 148,193 EUR/MW/yr
# (+35%). The model was previously charging NO fixed O&M on methanol capacity. It is set
# on 'methanolisation' too, deliberately: giving the split halves an FOM the monolithic
# link lacks would bias every split-vs-monolithic comparison in favour of the split.
# REMOVE all three once the fork's compile is fixed, or they will double up.
# ----------------------------------------------------------------------------------
tech_inputs['methanolisation', 'FOM'] = {
    'value': 2.8, 'unit': '%/year',
    'source': '[DEA] sheet "98 Methanol from hydrogen", Fixed O&M; = upstream technology-data methanolisation,FOM',
    'further description': 'WORKAROUND for a regression in the pypsa-eur_AA fork, which drops this parameter. Remove when the fork compiles it again.',
}
tech_inputs['methanol synthesis', 'FOM'] = {
    'value': 2.8, 'unit': '%/year',
    'source': '[DEA] sheet "98 Methanol from hydrogen", Fixed O&M, same rate as the aggregate unit',
    'further description': 'A percentage of investment, so the 90/10 CAPEX split carries it automatically; synthesis + distillation reproduce the monolithic fixed cost exactly.',
}
tech_inputs['methanol distillation', 'FOM'] = {
    'value': 2.8, 'unit': '%/year',
    'source': '[DEA] sheet "98 Methanol from hydrogen", Fixed O&M, same rate as the aggregate unit',
    'further description': 'See the methanol synthesis entry.',
}

tech_inputs['methanol distillation', 'water-output'] = {
    'value': meoh_water_t_per_t, 'unit': 't_H2O/t_MeOH',
    'source': 'DERIVED: M_H2O / M_MeOH (CoolProp), the stoichiometric identity of CO2 + 3H2 -> CH3OH + H2O -- see the STOICHIOMETRY block in the header',
    'further description': 'Was a transcribed [DEA] 0.55 (97.8% of stoichiometric; the ~2% is byproducts plus water dissolved in the product) and a matching literal 64.5 wt% crude. Both now follow from molar masses, so the tank sizing tracks the chemistry automatically. Confirmed by [OLI] Section E: column bottom water 4581 kmol/h vs methanol distillate 4542 kmol/h = 1.009, i.e. 1:1 molar. Not wired to a bus.',
}

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

