# SPDX-License-Identifier: MIT
"""Snakemake wrapper: download and preprocess energy-market input data.

Runs unconditionally when the marker file is absent; Snakemake's DAG
ensures this only executes once unless --forcerun preprocess_inputs is used.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.preprocessing import pre_processing_energy_data
from scripts import config as c

year = int(snakemake.wildcards.year)

# Read DH peak capacity from n_config (options.DH.peak capacity).
# Falls back to parameters.DH_Skive_Capacity if the key is absent (e.g. old
# user override files that pre-date this parameter).
try:
    dh_peak_mw = float(c.n_options.at["DH", "peak capacity"])
except (KeyError, TypeError, ValueError):
    dh_peak_mw = None

pre_processing_energy_data(year=year, dh_peak_capacity=dh_peak_mw)

Path(snakemake.output.done).touch()
