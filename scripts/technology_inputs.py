import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP
import math
from scripts.parameters import lhv_dict

# --- Component specific Calculations

def belt_dryer_investment(DM_flow_guess : float = 3, bed_height : float = 0.1, dry_bulk_density : float = 0.130, residence_time : float = 50, LHV_final : float = lhv_dict['pellets'] ):
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


def compress_multistage_with_Tcap(fluid: str, p_in_bar: float, p_out_bar: float, T_in_C: float, T_max_C: float = 150,
                                  eta_s: float = 0.75, r_max: float = 2.5, T_cool_C=None,
                                  T_split_C=None):  # optional split temperature for duty partition (e.g., 90 °C)

    """
    Multi-stage compressor with max stage ratio and max discharge temperature.
    Intercools to T_cool (default: T_in) after each stage. NOTE: No cooling on final stage.
    Uses real-gas properties via CoolProp.

    Adds split of aftercooling duty above/below a threshold temperature T_split_C.

    Inputs
        - fluid : 'Hydrogen', 'Methane', 'Water', 'Ammonia', 'CO2'
        - p_in_bar, p_out_bar : pressure levels
        - T_in_C : Inlet temperature in °C
        - T_max_C : maximum stage discharge temperature in °C
        - eta_s : isentropic efficiency
        - r_max : maximum compression ratio per stage
        - T_cool_C : interstage cooling setpoint in °C (defaults to T_in_C)
        - T_split_C : temperature threshold (°C) to split aftercooler duty
                      (e.g., 90 °C). If None, no split is computed.

    Returns dict with per-stage results and totals, including split duties.
    """

    if T_cool_C is None:
        T_cool_C = T_in_C

    # convert °C to K
    T_in_K = T_in_C + 273.15
    T_max_K = T_max_C + 273.15
    T_cool_K = T_cool_C + 273.15
    T_split_K = T_split_C + 273.15 if T_split_C is not None else None

    # Convert bar → Pa
    p_in = p_in_bar * 1e5
    p_out_target = p_out_bar * 1e5

    # Helper: compute stage outlet given pin, Tin, desired pout, with eta_s and T_max cap.
    def stage_compress(pin, Tin, pout_desired):
        # Inlet state
        s1 = CP.PropsSI('S', 'T', Tin, 'P', pin, fluid)
        h1 = CP.PropsSI('H', 'T', Tin, 'P', pin, fluid)

        # Given pout, return actual discharge T (K) and works (J/kg)
        def discharge_T_and_work(pout):
            h2s = CP.PropsSI('H', 'P', pout, 'S', s1, fluid)
            ws = h2s - h1
            w = ws / eta_s
            h2 = h1 + w
            T2 = CP.PropsSI('T', 'P', pout, 'H', h2, fluid)
            return T2, w, ws

        # Honor pressure-ratio cap
        pout_cap = min(pout_desired, pin * r_max)

        # Try capped outlet; if T_max exceeded, back off by bisection
        T2_try, w_try, ws_try = discharge_T_and_work(pout_cap)
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

        # Aftercool at same pout from T2 -> T_cool_K
        h2 = h1 + w
        h_cool = CP.PropsSI('H', 'T', T_cool_K, 'P', pout, fluid)
        Q_after = h2 - h_cool  # total aftercooler duty for this stage (J/kg)

        # Split the duty at T_split_K if provided
        Q_above = 0.0
        Q_below = 0.0
        if T_split_K is not None:
            # Clamp the split point between (T_cool_K, T2)
            if T2 > T_cool_K + 1e-9:
                T_hi = max(min(T2, T_split_K), T_cool_K)  # split temp within [T_cool, T2]
                h_split = CP.PropsSI('H', 'T', T_hi, 'P', pout, fluid)
                # Above = from T2 down to max(T_split, T_cool)
                Q_above = max(h2 - h_split, 0.0)
                # Below = from max(T_split, T_cool) down to T_cool
                Q_below = max(h_split - h_cool, 0.0)
                # Numerical guard: ensure sums match total within small tolerance
                if abs((Q_above + Q_below) - Q_after) > 1e-6 * max(1.0, Q_after):
                    # Fallback to exact split by ordering temps
                    Q_above = max(h2 - CP.PropsSI('H', 'T', max(T_split_K, T_cool_K), 'P', pout, fluid), 0.0)
                    Q_below = Q_after - Q_above
            else:
                Q_above = 0.0
                Q_below = 0.0

        stage = {
            'p_in_bar': pin / 1e5,
            'T_in_C': Tin - 273.15,
            'p_out_bar': pout / 1e5,
            'T_out_C': T2 - + 273.15,
            'w_actual_J_per_kg': w,
            'w_isentropic_J_per_kg': ws,
            'Q_aftercool_J_per_kg': Q_after,
            'Q_after_above_split_J_per_kg': Q_above,
            'Q_after_below_split_J_per_kg': Q_below,
            'ratio': pout / pin
        }
        return stage

    # Minimum stages from pressure-ratio cap
    total_ratio = p_out_target / p_in
    n_min_ratio = math.ceil(math.log(total_ratio, r_max)) if total_ratio > 1 else 0
    if n_min_ratio <= 0:
        raise ValueError("p_out must be greater than p_in for compression.")

    # Build stages (adaptive if T_max binds)
    stages = []
    pin = p_in
    Tin = T_in_K
    remaining_ratio = p_out_target / pin
    for _ in range(1, n_min_ratio + 20):  # safety margin
        stages_left_guess = max(1, math.ceil(math.log(remaining_ratio, r_max)))
        r_eq = remaining_ratio ** (1.0 / stages_left_guess)
        pout_desired = pin * min(r_max, r_eq)

        st = stage_compress(pin, Tin, pout_desired)
        stages.append(st)

        pin = st['p_out_bar'] * 1e5
        Tin = T_cool_K  # inter-cool
        remaining_ratio = p_out_target / pin

        if pin >= p_out_target * (1 - 1e-9):
            break

    # Totals (include motor efficiency if desired)
    eta_motor = 0.97
    w_total = sum(s['w_actual_J_per_kg'] for s in stages) / eta_motor
    ws_total = sum(s['w_isentropic_J_per_kg'] for s in stages) / eta_motor
    Q_total = sum(s['Q_aftercool_J_per_kg'] for s in stages)

    # Split totals
    Q_above_total = sum(s['Q_after_above_split_J_per_kg'] for s in stages)
    Q_below_total = sum(s['Q_after_below_split_J_per_kg'] for s in stages)

    result = {
        'fluid': fluid,
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

        'specific_aftercool_Q_above_split_J_per_kg': Q_above_total if T_split_C is not None else None,
        'specific_aftercool_Q_above_split_kWh_per_kg': (Q_above_total / 3.6e6) if T_split_C is not None else None,

        'specific_aftercool_Q_below_split_J_per_kg': Q_below_total if T_split_C is not None else None,
        'specific_aftercool_Q_below_split_kWh_per_kg': (Q_below_total / 3.6e6) if T_split_C is not None else None,
    }

    return result


def aftercomp_cool_duty(fluid: str, p_out_bar: float, T_in_C: float, T_cool_C: float, T_split_C: float, clamp_to_zero: bool = True):
    """
    Compression after-cooler duty at fixed outlet pressure, with optional split around T_split_C.

    Inputs
    ------
    fluid        : CoolProp fluid name (e.g., "Hydrogen", "CO2")
    p_out_bar    : discharge pressure [bar]
    T_in_C: actual discharge temperature [°C] entering the final cooler
    T_cool_C     : target cooled temperature [°C] after the cooler
    T_split_C    : split temperature [°C]; if None, no split is reported
    clamp_to_zero: if True, duty is clamped to >= 0 (no negative "heating")

    Returns (dict)
    --------------
    {
      'specific_Q_J_per_kg': ...,
      'specific_Q_kWh_per_kg': ...,
      'specific_Q_above_split_J_per_kg': ... or None,
      'specific_Q_above_split_kWh_per_kg': ... or None,
      'specific_Q_below_split_J_per_kg': ... or None,
      'specific_Q_below_split_kWh_per_kg': ... or None,
    }
    """
    # Convert units
    T_discharge_K = T_in_C + 273.15
    T_cool_K = T_cool_C + 273.15
    T_split_K = T_split_C + 273.15 if T_split_C is not None else None
    P = p_out_bar * 1e5  # bar -> Pa

    # Enthalpies at discharge and cooled states
    h_hot = CP.PropsSI('H', 'T', T_discharge_K, 'P', P, fluid)
    h_cold = CP.PropsSI('H', 'T', T_cool_K, 'P', P, fluid)

    # Total heat removed per kg
    Q_total = h_hot - h_cold  # J/kg

    if clamp_to_zero and Q_total < 0.0:
        Q_total = 0.0

    # Default split outputs
    Q_above = None
    Q_below = None

    if T_split_K is not None:
        # Handle edge cases: ensure split temperature lies between cool and discharge
        if T_discharge_K > T_cool_K:
            # Effective split point is clamped into [T_cool, T_discharge]
            T_eff = min(max(T_split_K, T_cool_K), T_discharge_K)
            h_split = CP.PropsSI('H', 'T', T_eff, 'P', P, fluid)

            # Above = from discharge down to max(T_split, T_cool)
            Q_above_raw = h_hot - h_split
            # Below  = from max(T_split, T_cool) down to T_cool
            Q_below_raw = h_split - h_cold

            if clamp_to_zero:
                Q_above = max(Q_above_raw, 0.0)
                Q_below = max(Q_below_raw, 0.0)
            else:
                Q_above = Q_above_raw
                Q_below = Q_below_raw

            # Numerical guard: enforce sum consistency if tiny rounding appears
            if Q_above is not None and Q_below is not None:
                err = (Q_above + Q_below) - Q_total
                if abs(err) > 1e-6 * max(1.0, abs(Q_total)):
                    # Push the tiny error into the 'below' portion
                    Q_below -= err
        else:
            # No cooling needed; both parts zero if clamped, else negative may appear
            if clamp_to_zero:
                Q_above = 0.0
                Q_below = 0.0
            else:
                # Keep None to signal "no meaningful split"
                Q_above = 0.0
                Q_below = 0.0

    out = {
        'specific_Q_J_per_kg': Q_total,
        'specific_Q_kWh_per_kg': Q_total / 3.6e6,
        'specific_Q_above_split_J_per_kg': Q_above,
        'specific_Q_above_split_kWh_per_kg': (Q_above / 3.6e6) if Q_above is not None else None,
        'specific_Q_below_split_J_per_kg': Q_below,
        'specific_Q_below_split_kWh_per_kg': (Q_below / 3.6e6) if Q_below is not None else None,
    }
    return out


# --- Network Inputs  T and P levels -------
T_max_comp = 160 # maximum dischrge temeprature for all compressors
T_ambient = 20 #

# ------ List of stream in symbiosis network
# if a value is missing is either calculated or constrained by a global value (e.g. T_max_comp)
# fluid nomenclature must follow CoolProp standards: https://coolprop.org/fluid_properties/PurePseudoPure.html#list-of-fluids

symbiosis_data = {
    "Heat MT max": {"fluid": 'Water', "T": 180, 'P' : 10,},
    'Heat MT min': {"fluid": 'Water', "T": 140, 'P': 3},
    'Heat DH min': {"fluid": 'Water', "T": 90, 'P': 1},
    'Heat LT min': {"fluid": 'Water', "T": 50, 'P': 1},
    'Ambient': {"fluid": 'Air', "T": T_ambient, 'P': 1},
    'H2 production': {"fluid": 'Hydrogen', "T": 50, 'P': 30},
    'H2 to MeOH': {"fluid": 'Hydrogen', "T": '', 'P': 80},
    'H2 to Methanation': {"fluid": 'Hydrogen', "T": '', 'P': 20},
    'H2 storage': {"fluid": 'Hydrogen', "T": '', 'P': 250},
    'CO2 biogas': {"fluid": 'CO2', "T": 50, 'P': 1},
    'CO2 to MeOH': {"fluid": 'CO2', "T": '', 'P': 80},
    'CO2 to Methanation': {"fluid": 'CO2', "T": '', 'P': 20},
    'CO2 to HP storage': {"fluid": 'CO2', "T": '', 'P': 80},
    'CO2 from HP storage': {"fluid": 'CO2', "T": T_ambient, 'P': 30},
}
symbiosis_n = pd.DataFrame.from_dict(symbiosis_data, orient="index")

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
    },
}

##### Calculation for compressor and heat exchangers:
# --- H2 compression to MeOH
in_stream = 'H2 production'
out_stream = 'H2 to MeOH'
storage_stream = 'H2 storage'

H2_comp_res = compress_multistage_with_Tcap(
    fluid=symbiosis_n.at[in_stream, 'fluid'],
    p_in_bar=symbiosis_n.at[in_stream, 'P'],
    p_out_bar=symbiosis_n.at[out_stream, 'P'],
    T_in_C=symbiosis_n.at[in_stream, 'T'],
    eta_s=0.75,
    r_max=2.5,
    T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
    T_split_C=symbiosis_n.at['Heat DH min', 'T']  # default to T_in
)

# ----- H2 extra compression for storage
H2_comp_extra_res = compress_multistage_with_Tcap(
    fluid=symbiosis_n.at[out_stream, 'fluid'],
    p_in_bar=symbiosis_n.at[in_stream, 'P'],
    p_out_bar=symbiosis_n.at[storage_stream, 'P'],
    T_in_C=symbiosis_n.at[in_stream, 'T'],
    eta_s=0.75,
    r_max=2.5,
    T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
    T_split_C=symbiosis_n.at['Heat DH min', 'T']  # default to T_in
)

# ---- extra Cooling before compression for storage
H2_cooling_storage_pre_comp = aftercomp_cool_duty(
    fluid=symbiosis_n.at[out_stream, 'fluid'],
    p_out_bar=symbiosis_n.at[out_stream, 'P'],
    T_in_C= H2_comp_res["stages"][-1]["T_out_C"],
    T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
    T_split_C=symbiosis_n.at['Heat DH min', 'T'],
    clamp_to_zero=True)

# ---- Update tech_inputs:
tech_inputs['hydrogen storage compressor', 'electricity-input'] = {
    'value': H2_comp_res['specific_work_kWh_per_kg'] / lhv_dict['H2'],
    'unit': 'MW/MW_H2',
    'source': 'calculated ',
    'further description': 'calculated based on: Isoentropic efficiency and max comp ratio, max outlet temp. CoolProp',
}
tech_inputs['hydrogen storage compressor', 'heat output LT'] = {
    'value': H2_comp_res['specific_aftercool_Q_below_split_kWh_per_kg'] / lhv_dict['H2'],
    'unit': 'MW/MW_H2',
    'source': 'calculated ',
    'further description': 'calculated based on: Isoentropic efficiency and max comp ratio, max outlet temp. CoolProp',
}
tech_inputs['hydrogen storage compressor', 'heat output DH'] = {
    'value': H2_comp_res['specific_aftercool_Q_above_split_kWh_per_kg'] / lhv_dict['H2'],
    'unit': 'MW/MW_H2',
    'source': 'calculated ',
    'further description': ' calculated based on: Isoentropic efficiency and max comp ratio, max outlet temp. CoolProp',
}
tech_inputs['hydrogen storage compressor', 'extra electricity-input'] = {
    'value': H2_comp_extra_res['specific_work_kWh_per_kg'] / lhv_dict['H2'],
    'unit': 'MW/MW_H2',
    'source': 'calculated for final compression to storage pressure ',
    'further description': 'calculated based on: Isoentropic efficiency and max comp ratio, max outlet temp. CoolProp',
}
tech_inputs['hydrogen storage compressor', 'extra heat output LT'] = {
    'value': (H2_comp_extra_res['specific_aftercool_Q_below_split_kWh_per_kg'] + H2_cooling_storage_pre_comp['specific_Q_below_split_kWh_per_kg']) / lhv_dict['H2'],
    'unit': 'MW/MW_H2',
    'source': 'calculated ',
    'further description': 'calculated based on: Isoentropic efficiency and max comp ratio, max outlet temp. CoolProp',
}
tech_inputs['hydrogen storage compressor', 'extra heat output DH'] = {
    'value': (H2_comp_extra_res['specific_aftercool_Q_above_split_kWh_per_kg'] + H2_cooling_storage_pre_comp['specific_Q_above_split_kWh_per_kg'] )/ lhv_dict['H2'],
    'unit': 'MW/MW_H2',
    'source': 'calculated ',
    'further description': ' calculated based on: Isoentropic efficiency and max comp ratio, max outlet temp. CoolProp',
}


# --- CO2 Compression and storage
# 1. CO2 Compression for MeOH
in_stream = 'CO2 biogas'
out_stream = 'CO2 to MeOH'
storage_in_stream = 'CO2 to HP storgae'
storage_out_stream = 'CO2 from HP storage'


CO2_comp_res = compress_multistage_with_Tcap(
    fluid=symbiosis_n.at[in_stream, 'fluid'],
    p_in_bar=symbiosis_n.at[in_stream, 'P'],
    p_out_bar=symbiosis_n.at[out_stream, 'P'],
    T_in_C=symbiosis_n.at[in_stream, 'T'],
    eta_s=0.75,
    r_max=2.5,
    T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
    T_split_C=symbiosis_n.at['Heat DH min', 'T']  # default to T_in
)

# ----- CO2 extra compression for storage
CO2_comp_extra_res = compress_multistage_with_Tcap(
    fluid=symbiosis_n.at[storage_out_stream, 'fluid'],
    p_in_bar=symbiosis_n.at[storage_out_stream, 'P'],
    p_out_bar=symbiosis_n.at[out_stream, 'P'],
    T_in_C=symbiosis_n.at[storage_out_stream, 'T'],
    eta_s=0.75,
    r_max=2.5,
    T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
    T_split_C=symbiosis_n.at['Heat DH min', 'T']  # default to T_in
)

CO2_final_cooling = aftercomp_cool_duty(
    fluid=symbiosis_n.at[out_stream, 'fluid'],
    p_out_bar = symbiosis_n.at[out_stream, 'P'],
    T_in_C= CO2_comp_res["stages"][-1]["T_out_C"],
    T_cool_C=symbiosis_n.at['Heat LT min', 'T'],
    T_split_C=symbiosis_n.at['Heat DH min', 'T'],
    clamp_to_zero= True)

tech_inputs['CO2 industrial compressor', 'electricity-input'] = {
    'value': CO2_comp_res['specific_work_kWh_per_kg'],
    'unit': 'MWh/t_CO2',
    'source': 'calculated ',
    'further description': 'calculated based on: Isoentropic efficiency and max comp ratio, max outlet temp. CoolProp',
    }
tech_inputs['CO2 industrial compressor', 'extra electricity-input'] = {
    'value': CO2_comp_extra_res['specific_work_kWh_per_kg'],
    'unit': 'MWh/t_CO2',
    'source': 'calculated ',
    'further description': 'calculated based on: Isoentropic efficiency and max comp ratio, max outlet temp. CoolProp',
    }
tech_inputs['CO2 industrial compressor', 'heat output LT'] = {
    'value': CO2_comp_res['specific_aftercool_Q_below_split_kWh_per_kg'],
    'unit': 'MWh/t_CO2',
    'source': 'calculated ',
    'further description': 'calculated based on: Isoentropic efficiency and max comp ratio, max outlet temp. CoolProp',
    }
tech_inputs['CO2 industrial compressor', 'heat output DH'] = {
    'value': CO2_comp_res['specific_aftercool_Q_above_split_kWh_per_kg'],
    'unit': 'MWh/t_CO2',
    'source': 'calculated ',
    'further description': ' calculated based on: Isoentropic efficiency and max comp ratio, max outlet temp. CoolProp',
    }
tech_inputs['CO2 industrial compressor', 'extra heat output LT'] = {
    'value': CO2_comp_extra_res['specific_aftercool_Q_below_split_kWh_per_kg'] + CO2_final_cooling['specific_Q_below_split_kWh_per_kg'],
    'unit': 'MWh/t_CO2',
    'source': 'calculated ',
    'further description': 'calculated based on: Isoentropic efficiency and max comp ratio, max outlet temp. CoolProp',
    }
tech_inputs['CO2 industrial compressor', 'extra heat output DH'] = {
    'value': CO2_comp_extra_res['specific_aftercool_Q_above_split_kWh_per_kg'] + CO2_final_cooling['specific_Q_above_split_kWh_per_kg'],
    'unit': 'MWh/t_CO2',
    'source': 'calculated ',
    'further description': ' calculated based on: Isoentropic efficiency and max comp ratio, max outlet temp. CoolProp',
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
belt_dryer = belt_dryer_investment(DM_flow_guess = DM_flow_guess, LHV_final=lhv_dict['pellets'])
tech_inputs[('biomass belt dryer', 'investment')] = {
    'value': belt_dryer['investment'],
    'unit': '€/ (t/h DM)',
    'source': 'calculated based on: DOI: 10.1080/07373937.2018.1492615',
    'further description': 'calculated based on: DM flow reference',
    'currency_year': 2025}

