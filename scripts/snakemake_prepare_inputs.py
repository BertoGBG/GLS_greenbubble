# SPDX-License-Identifier: MIT
"""Snakemake wrapper: assemble network input dictionary and save as pickle."""
from pathlib import Path
import sys
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.preprocessing import prepare_all_inputs
from scripts import config as c

inputs_dict = prepare_all_inputs(
    targets_dict      = c.targets_dict,
    CO2_cost          = c.CO2_cost,
    CO2_cost_ref_year = c.CO2_cost_ref_year,
    max_RE_to_grid    = c.max_RE_to_grid,
)

with open(snakemake.output.inputs, "wb") as fh:
    pickle.dump(inputs_dict, fh)
