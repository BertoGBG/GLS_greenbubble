"""Snakemake wrapper: download and preprocess energy-market input data.

If config preprocess_flag is False the marker file is created immediately
so downstream rules can use already-existing CSVs in data/Inputs_{year}/.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

if snakemake.config.get("preprocess_flag", False):
    from scripts.retrieve import pre_processing_energy_data
    pre_processing_energy_data()

Path(snakemake.output.done).touch()
