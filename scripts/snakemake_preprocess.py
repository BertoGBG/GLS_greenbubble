# SPDX-License-Identifier: MIT
"""Snakemake wrapper: download and preprocess energy-market input data.

Runs unconditionally when the marker file is absent; Snakemake's DAG
ensures this only executes once unless --forcerun preprocess_inputs is used.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.preprocessing import pre_processing_energy_data

year = int(snakemake.wildcards.year)
pre_processing_energy_data(year=year)

Path(snakemake.output.done).touch()
