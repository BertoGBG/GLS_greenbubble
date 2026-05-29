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

A string that uniquely identifies a model run based on the active ``n_flags``,
demands, CO₂ cost, year, and run name. It is constructed by ``build_network_name()``
in ``Snakefile`` before any rule executes.

**Format:**

.. code-block:: text

   {flags}CO2_{co2}_{tD|tP}_H2_{h2}_MeOH_{meoh}_CH4_{ch4}_{year}_El_{el}_{DET|STC}_{res}_{run_name}

**Example — full resolution (default, 1 h):**

.. code-block:: text

   B_H_RE_H2_MEOH_METH_SN_ST_CO2_100_tD_H2_0_MeOH_4_CH4_300_2023_El_0.1_DET_1h_H2_meth_dmd_DK

**Example — 4-hour temporal resolution:**

.. code-block:: text

   B_H_RE_H2_MEOH_METH_SN_ST_CO2_100_tD_H2_0_MeOH_4_CH4_300_2023_El_0.1_DET_4h_H2_meth_dmd_DK

The segments encode (in order): active technology flags, CO₂ cost, demand driver,
demand targets, energy year, grid export share, stochastic mode, time resolution
(``1h`` when ``clustering.temporal.resolution: false``; otherwise the configured
offset string), and run name.

**Constraint:** literal match against the pre-computed ``NETWORK`` string
(via ``wildcard_constraints: network = NETWORK_PATTERN``).

**Outputs:** ``resources/{network}_PRE.nc``, ``outputs/.../{network}/``
