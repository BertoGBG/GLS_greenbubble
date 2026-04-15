.. _rules:

Rules Overview
==============

The GreenBubble workflow is managed by `Snakemake <https://snakemake.readthedocs.io>`_.
Rules are defined in the ``rules/`` folder and assembled by the ``Snakefile``.

DAG
---

.. code-block:: text

                          retrieve_tech_data
                                 │
        preprocess_inputs ───────┤   (one job per year, runs in parallel)
        preprocess_inputs ───────┤
        preprocess_inputs ───────┤
        preprocess_inputs ───────┘
                 │
         prepare_costs ──────────────────────┐
         prepare_inputs ──────────────────────┤
                                              ▼
                                       build_network
                                             │
                                       solve_network
                                             │
                                       plot_results


rules/retrieve.smk
------------------

**retrieve_tech_data**

Downloads the technology cost CSV from the
`technology-data <https://github.com/BertoGBG/technology-data>`_ repository
for the ``year_EU`` set in ``config.yaml``.

- Output: ``data/technology-data/outputs/costs_{year_EU}.csv``
- Script: ``scripts/snakemake_retrieve_tech.py``

**preprocess_inputs**

Downloads and preprocesses all market input data for a given ``{year}``:
electricity spot prices, CO₂ emission intensities, natural gas prices,
renewable capacity factors (wind, solar), and district heating demand.

Runs once per scenario year. With ``-j4``, all years are downloaded in parallel.

- Output: ``data/Inputs_{year}/.preprocessed`` (marker file)
- Script: ``scripts/snakemake_preprocess.py``
- Wildcard: ``{year}`` — see :ref:`wildcards`


rules/build.smk
---------------

**prepare_costs**

Builds the ``tech_costs`` DataFrame from the retrieved cost CSV,
blending EU and US cost assumptions based on the project location.

- Input: ``data/technology-data/outputs/costs_{year_EU}.csv``
- Output: ``resources/tech_costs.pkl``
- Script: ``scripts/snakemake_prepare_costs.py``

**prepare_inputs**

Loads all preprocessed CSV files for all scenario years and assembles
the ``inputs_dict`` passed to the network builder.
Waits for **all** ``preprocess_inputs`` jobs to complete before running.

- Input: ``data/Inputs_{year}/.preprocessed`` for all years in ``PREPROCESS_YEARS``
- Output: ``resources/inputs_{year}.pkl``
- Script: ``scripts/snakemake_prepare_inputs.py``

**build_network**

Constructs the PyPSA network with all active technologies (controlled by ``n_flags``).
In stochastic mode, couples all scenario networks into a single LP.
Saves the pre-optimisation network for inspection.

- Input: ``resources/tech_costs.pkl``, ``resources/inputs_{year}.pkl``
- Output: ``resources/{network}_PRE.nc``, ``resources/{network}_comp_alloc.pkl``
- Script: ``scripts/snakemake_build_network.py``


rules/solve.smk
---------------

**solve_network**

Runs the capacity expansion + dispatch linear programme via
`Linopy <https://linopy.readthedocs.io>`_.
Solver and profile are configured in ``config.yaml`` under ``optimization``.

- Input: ``resources/{network}_PRE.nc``
- Output: ``outputs/.../{network}/networks/{network}_OPT.nc``
- Script: ``scripts/snakemake_solve_network.py``


rules/plot.smk
--------------

**plot_results**

Exports dispatch time series plots, optimal capacity bar charts,
and shadow price tables. Components to plot are configured in ``plots_config.yaml``.

- Input: ``outputs/.../{network}/networks/{network}_OPT.nc``
- Output: ``outputs/.../{network}/plots/.done`` (marker)
- Script: ``scripts/snakemake_plot.py``
