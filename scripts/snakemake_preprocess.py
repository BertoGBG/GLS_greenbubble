# SPDX-License-Identifier: MIT
"""Snakemake wrapper: download and preprocess energy-market input data.

Runs unconditionally when the marker file is absent; Snakemake's DAG
ensures this only executes once unless --forcerun preprocess_inputs is used.

When ``pypsa_eur_link.enabled`` is true, branches to the soft-link CSV writer
(``scripts.pypsa_eur_link.write_softlink_inputs``) instead of the normal
API-download path — same output files, same marker, so the rest of the
Snakemake DAG (``prepare_inputs`` etc.) needs no changes either way.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import config as c

year = int(snakemake.wildcards.year)

if c.pypsa_eur_link["enabled"]:
    from scripts import pypsa_eur_link as pel
    from scripts import parameters as p

    # config.py already enforces clustering.temporal.resolution is set (not
    # False) when pypsa_eur_link is enabled, but can't cheaply verify it
    # matches the linked network's own resolution without loading it — done
    # here instead, inside write_softlink_inputs, which loads the network
    # anyway (avoids loading a large sector-coupled network twice).
    expected_res = float(str(c.clustering["temporal"]["resolution"]).rstrip("hH"))

    pel.write_softlink_inputs(
        year=year,
        network_path=c.pypsa_eur_link["network_path"],
        regions_path=c.pypsa_eur_link["regions_path"],
        latitude=p.latitude,
        longitude=p.longitude,
        co2_stored_price_mode=c.pypsa_eur_link["co2_stored_price_mode"],
        run_id=c.pypsa_eur_link["id"],
        expected_resolution_hours=expected_res,
    )
else:
    from scripts.preprocessing import pre_processing_energy_data

    # Read DH peak capacity from n_config (options.DH.peak capacity).
    # Falls back to parameters.DH_Skive_Capacity if the key is absent (e.g. old
    # user override files that pre-date this parameter).
    try:
        dh_peak_mw = float(c.n_options.at["DH", "peak capacity"])
    except (KeyError, TypeError, ValueError):
        dh_peak_mw = None

    pre_processing_energy_data(year=year, dh_peak_capacity=dh_peak_mw)

Path(snakemake.output.done).touch()
