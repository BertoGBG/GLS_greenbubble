"""Snakemake wrapper: build technology-cost DataFrame and save as pickle."""
from pathlib import Path
import sys
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.helpers import prepare_costs
from scripts.technology_inputs import tech_inputs
from scripts import parameters as p

cfg = snakemake.config

tech_costs = prepare_costs(
    latitude        = cfg["latitude"],
    longitude       = cfg["longitude"],
    tech_inputs     = tech_inputs,
    USD_to_EUR      = cfg["USD_to_EUR"],
    discount_rate   = cfg["discount_rate"],
    cost_path_EU    = snakemake.input.costs_eu,
    cost_path_US    = p.cost_path_US,
    dict_tech_US_EU = p.dict_tech_US_EU,
)

with open(snakemake.output.tech_costs, "wb") as fh:
    pickle.dump(tech_costs, fh)
