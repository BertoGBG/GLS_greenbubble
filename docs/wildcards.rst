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

**Example value:**

.. code-block:: text

   B_H_RE_H2_MEOH_METH_SN_ST_CO2_100_tD_H2_0_MeOH_4_CH4_300_2023_El_0.1_DET_H2_meth_dmd_DK

The segments encode (in order): active technology flags, CO₂ cost, demand driver,
demand targets, energy year, grid export share, stochastic mode, and run name.

**Constraint:** literal match against the pre-computed ``NETWORK`` string
(via ``wildcard_constraints: network = NETWORK_PATTERN``).

**Outputs:** ``resources/{network}_PRE.nc``, ``outputs/.../{network}/``
