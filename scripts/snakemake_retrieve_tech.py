# SPDX-License-Identifier: MIT
"""Snakemake wrapper: download technology-cost CSV (retrieve_technology_data)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.retrieve import retrieve_technology_data
from scripts import parameters as p

for _out_path in snakemake.output.costs:
    retrieve_technology_data(_out_path, p.technology_data_url)
