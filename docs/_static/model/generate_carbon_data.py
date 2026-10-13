# SPDX-FileCopyrightText: Contributors to GreenBubble
# SPDX-License-Identifier: MIT
"""Write carbon_data.js for the interactive carbon balance (carbon_balance.html).

The model carries most carbon implicitly: methane and methanol in MWh, biomass in MWh
or t DM, and only CO2 and biochar in tonnes. This script collects every coefficient the
carbon balance needs from the same sources the model uses (technology-data through
prepare_costs, the GreenLab input workbook, p_config), so the page stays consistent
with the network. Optional solved networks become presets with their annual flows.

Run from the repository root:

    python docs/_static/model/generate_carbon_data.py
    python docs/_static/model/generate_carbon_data.py --network NAME=path/to/network.nc ...
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts import config as c, parameters as p  # noqa: E402
from scripts.helpers import prepare_costs  # noqa: E402
from scripts.preprocessing import GL_inputs_to_eff  # noqa: E402
from scripts.technology_inputs import tech_inputs  # noqa: E402

OUT = Path(__file__).with_name("carbon_data.js")
CO2_PER_C = 44.0 / 12.0

DEFAULT_PRESETS = {
    "Tutorial 1 (price mode)": "outputs/single_analysis/tut1_price/networks/"
    "B_H_RE_H2_MEOH_METH_SN_ST_CO2_100_tP_H2_120_MeOH_200_CH4_80_2024_El_0.3_DET_8h_tut1_price_OPT.nc",
    "Tutorial 2 (brownfield)": "outputs/single_analysis/tut2_brownfield/networks/"
    "B_H_RE_H2_MEOH_METH_SN_ST_CO2_100_tP_H2_120_MeOH_200_CH4_80_2024_El_0.3_DET_8h_tut2_brownfield_OPT.nc",
}


def coefficients():
    tc = prepare_costs(
        latitude=c.latitude, longitude=c.longitude, tech_inputs=tech_inputs,
        USD_to_EUR=c.USD_to_EUR, discount_rate=c.discount_rate,
        cost_path_EU=p.cost_path, cost_path_US=p.cost_path_US, dict_tech_US_EU=p.dict_tech_US_EU,
    )
    v = lambda t, k: float(tc.at[t, k])  # noqa: E731
    gl = GL_inputs_to_eff(pd.read_excel(ROOT / p.GL_input_file, sheet_name="Overview_2", index_col=0))
    sb = gl["SkiveBiogas"]
    streams = c.p_streams
    pel_lhv, pel_moist = float(streams.at["pellets", "LHV"]), float(streams.at["pellets", "moisture"])

    k_gas, k_meoh, k_bio = v("gas", "CO2 intensity"), v("methanol", "CO2 intensity"), v("solid biomass", "CO2 intensity")
    co2_per_ch4 = float(sb["CO2 pure"] / sb["bioCH4"])
    co = {
        # carbon content of carriers, tCO2 per unit
        "k_gas": [k_gas, "tCO2/MWh CH4", "technology-data: gas, CO2 intensity"],
        "k_meoh": [k_meoh, "tCO2/MWh MeOH", "technology-data: methanol, CO2 intensity"],
        "k_bio": [k_bio, "tCO2/MWh biomass", "technology-data: solid biomass, CO2 intensity (18 GJ/t DM, 50 % C)"],
        "k_moist": [0.5 * CO2_PER_C, "tCO2/t DM", "50 % C in dry matter, as in solid biomass CO2 intensity"],
        # biogas plant (GreenLab workbook, per MWh of biogas on the 'biogas' bus = CH4 energy)
        "bg_co2": [co2_per_ch4, "tCO2/MWh CH4", "GreenLab_Input_file.xlsx: SkiveBiogas CO2 pure / bioCH4"],
        "bg_dig": [float(sb["DM digestate"] / sb["bioCH4"]), "t DM/MWh CH4",
                   "GreenLab_Input_file.xlsx: SkiveBiogas DM digestate / bioCH4"],
        # conversions
        "esmr_meoh": [v("methanol from biogas", "methanol-output") / v("methanol from biogas", "biogas-input"),
                      "MWh MeOH/MWh biogas", "technology-data: methanol from biogas"],
        "methbg_ch4": [v("methanation biogas", "methane-output") / v("methanation biogas", "biogas-input"),
                       "MWh CH4/MWh biogas", "technology-data: methanation biogas"],
        "biometh_ch4": [v("biomethanation", "methane-output") / v("biomethanation", "biogas-input"),
                        "MWh CH4/MWh biogas", "technology-data: biomethanation"],
        "methco2_ch4": [1.0 / v("methanation", "carbondioxide-input"), "MWh CH4/tCO2",
                        "technology-data: methanation, carbondioxide-input"],
        "biomethco2_ch4": [v("biomethanation CO2", "methane-output") / v("biomethanation CO2", "CO2-input"),
                           "MWh CH4/tCO2", "technology-data: biomethanation CO2"],
        "meoh_co2": [1.0 / v("methanolisation", "carbondioxide-input"), "MWh MeOH/tCO2",
                     "technology-data: methanolisation, carbondioxide-input"],
        "meohsyn_co2": [1.0 / v("methanol synthesis", "carbondioxide-input"), "MWh MeOH/tCO2",
                        "technology_inputs.py: methanol synthesis, carbondioxide-input"],
        "dewater_dm": [v("centrifugal dewatering", "DM separation"), "t DM fibre/t DM digestate",
                       "technology-data: centrifugal dewatering, DM separation"],
        "dryer_mwh": [pel_lhv / (1 - pel_moist), "MWh pellets/t DM",
                      "p_config: pellets LHV / (1 - moisture), as the belt dryer link"],
        "pyro_char": [1.0 / v("biochar pyrolysis", "biomass-input"), "tCO2 biochar/MWh pellets",
                      "technology-data: biochar pyrolysis, biomass-input"],
        "liq_seq": [float(c.n_options.at["CO2 Liq credits", "efficiency"]), "-",
                    "n_config options: CO2 Liq credits, efficiency (boil-off to air)"],
    }
    return co


def enabled_defaults():
    """n_config default: a technology exists if it has initial capacity or may expand."""
    raw = yaml.safe_load(open(ROOT / "config/n_config.default.yaml"))
    on = lambda k: bool(raw[k].get("initial capacity", 0) or raw[k].get("expansion", False))  # noqa: E731
    o = raw["options"]
    return {
        "upgrading": on("biogas upgrading"), "engine": on("biogas engine"),
        "esmr": on("methanol from biogas"), "methanolisation": on("methanolisation"),
        "meoh_split": bool(o["meoh split"]["enable"]),
        "methBiogas": on("methanation biogas"), "biometh": on("biomethanation"),
        "methCO2": on("methanation CO2"), "biomethCO2": on("biomethanation CO2"),
        "liq": on("CO2 Liq storage"), "dewatering": on("dewatering"), "dryer": on("biomass belt dryer"),
        "pyrolysis": on("pyrolysis"), "boiler": on("biomass boiler"), "ngboiler": on("NG boiler"),
        "biochar_credits": bool(o["biochar credits"]["enable"]),
        "liq_credits": bool(o["CO2 Liq credits"]["enable"]),
        "pellets_market": bool(o["pellets market"]["enable"]),
        "moist_market": bool(o["moist biomass market"]["enable"]),
    }


# page toggle -> link name in the network (EXI_ prefix stripped)
TECH_LINKS = {
    "upgrading": "biogas upgrading", "engine": "biogas engine", "esmr": "methanol from biogas",
    "methanolisation": "methanolisation", "meoh_split": "methanol synthesis",
    "methBiogas": "methanation biogas", "biometh": "biomethanation", "methCO2": "methanation CO2",
    "biomethCO2": "biomethanation CO2", "liq": "CO2 liquefaction", "dewatering": "dewatering",
    "dryer": "biomass belt dryer", "pyrolysis": "pyrolysis", "boiler": "pellets boiler", "ngboiler": "NG boiler",
}
FLAG_TOKENS = {"B": "biogas", "H": "central_heat", "RE": "renewables", "H2": "electrolysis",
               "MEOH": "meoh", "METH": "methanation", "SN": "symbiosis", "ST": "storage"}


def flags_from_name(name):
    """n_flags from the file-name prefix, e.g. B_H_RE_H2_MEOH_METH_SN_ST_CO2_..."""
    tokens = name.split("_CO2_")[0].split("_")
    return {flag: tok in tokens for tok, flag in FLAG_TOKENS.items()}


def tech_present(bases):
    """Which page toggles have a link in the network. Each link counts for its longest
    matching technology name, so 'biomethanation CO2 ...' does not switch on 'biomethanation'."""
    names = sorted(TECH_LINKS.items(), key=lambda kv: -len(kv[1]))
    found = set()
    for b in bases:
        for key, name in names:
            if b == name or b.startswith(name + " "):
                found.add(key)
                break
    return {key: key in found for key in TECH_LINKS}


def preset_from_network(path):
    """Annual flows of a solved network expressed as the page's inputs and route shares."""
    import pypsa
    n = pypsa.Network(path)
    w = n.snapshot_weightings.generators
    links = n.links.copy()
    links["base"] = links.index.str.replace(r"^EXI_", "", regex=True)

    def port_flow(bus, sign=1):
        """Annual withdrawal from `bus` per link base name (sign=-1 for injection)."""
        out = {}
        for k in range(10):
            col = f"bus{k}"
            if col not in links or f"p{k}" not in n.links_t:
                continue
            for l in links.index[links[col] == bus]:
                if l in n.links_t[f"p{k}"]:
                    f = float((n.links_t[f"p{k}"][l] * w).sum()) * sign
                    if f > 1e-6:
                        out[links.at[l, "base"]] = out.get(links.at[l, "base"], 0) + f
        return out

    gen = lambda g: float((n.generators_t.p[g] * w).sum()) if g in n.generators_t.p else 0.0  # noqa: E731
    bg_in = port_flow("biogas", -1)              # producers of biogas
    bg_out = port_flow("biogas")                 # consumers of biogas
    co2sep = port_flow("CO2 sep")
    dist = port_flow("CO2 distribution")
    dig = port_flow("Digestate")
    pel = port_flow("pellets")
    ng = port_flow("NG")
    biogas_gwh = bg_in.get("biogas", 0) / 1e3
    dig_total = sum(port_flow("Digestate", -1).values())
    return {
        "inputs": {
            "biogas": round(biogas_gwh, 1),
            "pellets": round(gen("pellets market") / 1e3, 1),
            "moist": round(gen("moist biomass market") / 1e3, 2),
            "ng": round(sum(ng.values()) / 1e3, 1),
        },
        "biogas": {"upgrading": bg_out.get("biogas upgrading", 0), "engine": bg_out.get("biogas engine", 0),
                   "esmr": bg_out.get("methanol from biogas", 0), "methBiogas": bg_out.get("methanation biogas", 0),
                   "biometh": bg_out.get("biomethanation", 0)},
        "co2": {"vent": co2sep.get("CO2 sep to atm", 0), "dist": co2sep.get("CO2_pipe", 0)},
        "dist": {"liq": dist.get("CO2 liquefaction", 0),
                 "meoh": sum(f for k, f in dist.items() if k.startswith("methanolisation CO2 compressor")
                             or k.startswith("methanol synthesis CO2 compressor")),
                 "methCO2": sum(f for k, f in dist.items() if k.startswith("methanation CO2")
                                and "storage" not in k),
                 "biomethCO2": sum(f for k, f in dist.items() if k.startswith("biomethanation CO2")
                                   and "storage" not in k)},
        "dig": {"dewatering": dig.get("dewatering", 0),
                "field": max(dig_total - dig.get("dewatering", 0), 0)},
        "pel": {"boiler": pel.get("pellets boiler", 0), "pyrolysis": pel.get("pyrolysis", 0)},
        "enabled": tech_present(set(links["base"])),
        "flags": flags_from_name(Path(path).name),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--network", action="append", default=[], metavar="NAME=PATH",
                    help="solved network to add as a preset (repeatable)")
    args = ap.parse_args()
    nets = dict(a.split("=", 1) for a in args.network) or DEFAULT_PRESETS
    presets = {}
    for name, path in nets.items():
        if (ROOT / path).exists():
            presets[name] = preset_from_network(str(ROOT / path))
        else:
            print(f"skipped preset {name!r}: {path} not found")
    flags = yaml.safe_load(open(ROOT / "config/config.default.yaml"))["n_flags"]
    data = {"coef": coefficients(), "enabled": enabled_defaults(), "presets": presets,
            "flags": {k: bool(v) for k, v in flags.items() if k in FLAG_TOKENS.values()},
            "year_investment": c.year_investment}
    OUT.write_text("// generated by generate_carbon_data.py, do not edit\n"
                   f"window.GB_CARBON = {json.dumps(data, indent=1)};\n", encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
