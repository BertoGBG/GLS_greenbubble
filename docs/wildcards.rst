.. _wildcards:

Wildcards
=========

Snakemake wildcards are placeholder values resolved at runtime to determine
which files to build. GreenBubble currently uses the following wildcards.

``{year}``
----------

**Used in:** ``preprocess_inputs``

Represents an energy price year for which market data is downloaded and preprocessed.

In deterministic mode, ``{year}`` resolves to ``En_price_year`` from ``config.yaml``.
In stochastic mode, it resolves to each key in ``stochastic.scenarios``.

**Example values:** ``2022``, ``2023``, ``2024``, ``2025``

**Constraint:** ``\d{4}`` (exactly four digits)

**Output:** ``data/Inputs_{year}/.preprocessed``

``{network}``
-------------

**Used in:** ``build_network``, ``solve_network``, ``plot_results``

A short string that uniquely identifies a model run. It is constructed by
``build_network_name()`` in ``Snakefile`` before any rule executes, inspired by
the PyPSA-EUR convention of keeping output paths short to avoid Windows'
260-character path limit.

**Format:**

.. code-block:: text

   {run_name}_{year}_{det|stc}_{res}

Rolling-horizon runs append ``_RH``: ``{run_name}_{year}_{det}_{res}_RH``

**Examples:**

.. code-block:: text

   tut1_demand_2024_det_3h
   tut4_stoch_2024_stc_3h
   high_co2_demand_2024_det_1h
   my_scenario_2025_det_3h_RH

The segments encode: ``run_name`` from ``config.yaml``, the energy price year
(``En_price_year``), deterministic/stochastic mode, and temporal resolution
(``1h`` when ``clustering.temporal.resolution`` is not set).

The **full configuration** — flags, CO₂ cost, product targets, and all other
parameters — is saved to ``outputs/.../{network}/networks/config_run.yaml``
after every solve, providing the complete reproducibility record.

**Constraint:** literal match against the pre-computed ``NETWORK`` string
(via ``wildcard_constraints: network = NETWORK_PATTERN``).

**Outputs:** ``resources/{network}_PRE.nc``, ``outputs/.../{network}/``
