.. _rules:

Rules Overview
==============

The GreenBubble workflow is managed by `Snakemake <https://snakemake.readthedocs.io/en/stable/>`_.
Rules are defined in the ``rules/`` folder and assembled by the ``Snakefile``.

Two path variables appear throughout: ``{run_name}`` is the ``run_name`` set in
``config.yaml``, and ``{network}`` is the long descriptive file name built by
``build_network_name()`` (see :ref:`wildcards`).

DAG
---

.. code-block:: text

                          retrieve_tech_data
                       (all cost years, 2020-2050)
                                 │
        preprocess_inputs ───────┤   (one job per year, runs in parallel)
        preprocess_inputs ───────┤
        preprocess_inputs ───────┤
        preprocess_inputs ───────┘
                 │
           prepare_inputs
                 │
                 ├──────────── costs_{year}.csv
                 ▼
           build_network
                 │
           solve_network
                 │
           plot_results

There is no separate cost-preparation rule: the cost CSVs are read directly by
``build_network`` and ``solve_network``.

When ``rolling_horizon.enabled`` is true, ``solve_rolling_horizon`` and
``plot_rolling_horizon`` replace the last two steps (see below).

rules/retrieve.smk
------------------

**retrieve_tech_data**

Downloads the technology cost CSVs from the
`technology-data <https://github.com/BertoGBG/technology-data>`_ repository.
All years are fetched, not just ``year_investment``, because brownfield
components look up their cost at their own ``construction_year``.

- Output: ``data/technology-data/outputs/costs_{year}.csv`` for 2020, 2025, 2030, 2035, 2040, 2045, 2050
- Script: ``scripts/snakemake_retrieve_tech.py``

**preprocess_inputs**

Downloads and preprocesses all market input data for a given ``{year}``:
electricity spot prices, CO₂ emission intensities, natural gas prices,
renewable capacity factors (wind, solar), and district heating demand.

Runs once per scenario year. With ``-j4``, all years are downloaded in parallel.

- Output: ``data/Inputs_{year}/.preprocessed`` (marker file)
- Script: ``scripts/snakemake_preprocess.py``
- Wildcard: ``{year}``; see :ref:`wildcards`

.. note::

   The marker file is the rule's only output. The per-year CSVs underneath are
   downloaded by guarded code that skips any file already on disk, so forcing
   this rule does **not** refresh a stale CSV. Delete the CSV first.

rules/build.smk
---------------

**prepare_inputs**

Loads all preprocessed CSV files for all scenario years and assembles
the ``inputs_dict`` passed to the network builder.
Waits for **all** ``preprocess_inputs`` jobs to complete before running.
Re-runs whenever ``config.default.yaml`` or ``config.yaml`` changes.

- Input: ``data/Inputs_{year}/.preprocessed`` for all years in ``PREPROCESS_YEARS``; both config files
- Output: ``resources/inputs_{year}.pkl``
- Script: ``scripts/snakemake_prepare_inputs.py``

**build_network**

Constructs the PyPSA network with all active technologies (controlled by ``n_flags``).
In stochastic mode, couples all scenario networks into a single LP.
Saves the pre-optimisation network for inspection.

- Input: ``resources/inputs_{year}.pkl``; ``costs_{year_investment}.csv``; all ``costs_{year}.csv``
- Output: ``resources/{run_name}/{network}_PRE.nc``, ``resources/{run_name}/{network}_comp_alloc.pkl``
- Script: ``scripts/snakemake_build_network.py``

rules/solve.smk
---------------

**solve_network**

Runs the capacity expansion + dispatch linear programme via
`Linopy <https://linopy.readthedocs.io>`_.
Solver and profile are configured in ``config.yaml`` under ``optimization``.
Also exports the EVPI CSV when stochastic EVPI is enabled.

- Input: ``resources/{run_name}/{network}_PRE.nc``, ``costs_{year_investment}.csv``, ``{network}_comp_alloc.pkl``
- Output: ``outputs/single_analysis/{run_name}/networks/{network}_OPT.nc``
- Script: ``scripts/snakemake_solve.py``

rules/plot.smk
--------------

**plot_results**

Exports dispatch time series plots, optimal capacity bar charts,
and shadow price tables. Components to plot are configured in
``config/plots_config.default.yaml``.

- Input: ``{network}_OPT.nc``, ``{network}_comp_alloc.pkl``
- Output: ``outputs/single_analysis/{run_name}/plots/{network}.done`` (marker)
- Script: ``scripts/snakemake_plot.py``

rules/rolling_horizon.smk
-------------------------

These two rules exist only when ``rolling_horizon.enabled: true`` in
``config.yaml``. They run dispatch on a fixed-capacity network; capacity
expansion is bypassed. See :doc:`guide_rolling_horizon`.

**solve_rolling_horizon**

Dispatch-only solve on the network given by ``rolling_horizon.network_path``.
When ``rh_year`` differs from ``En_price_year``, a fresh network is built from
the second year's inputs and the capacities are transferred across before
dispatch runs.

- Input: ``rolling_horizon.network_path``, ``costs_{year_investment}.csv``
- Output: ``outputs/single_analysis/{run_name}/networks/{network}_OPT.nc``
- Script: ``scripts/snakemake_rolling_horizon.py``

**plot_rolling_horizon**

Full plots plus the perfect-foresight vs rolling-horizon comparison.

- Input: the RH ``{network}_OPT.nc`` and the original fixed-capacity network
- Output: ``outputs/single_analysis/{run_name}/plots_rh/{network}.done`` (marker)
- Script: ``scripts/snakemake_plot_rolling_horizon.py``

.. note::

   ``solve_rolling_horizon`` and ``solve_network`` both produce
   ``{network}_OPT.nc``. The ``ruleorder: solve_rolling_horizon > solve_network``
   directive at the end of the file tells Snakemake which to prefer, so the DAG
   stays unambiguous.

rule all
--------

The default target is the plot marker: ``plots_rh/{network}.done`` when rolling
horizon is enabled, otherwise ``plots/{network}.done``. Asking for that one file
pulls the whole DAG.
