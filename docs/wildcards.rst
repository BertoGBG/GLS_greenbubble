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

A descriptive string used as the **file-name prefix** for all outputs of a
model run (``_OPT.nc``, ``_PRE.svg``, duals, etc.). It is constructed by
``build_network_name()`` in ``Snakefile`` before any rule executes.

**Format:**

.. code-block:: text

   {flag_prefix}CO2_{co2}_{tD|tP}_H2_{h2}_MeOH_{meoh}_CH4_{ch4}_{year}_El_{el}_{DET|STC}_{res}_{run_name}

Rolling-horizon runs append ``_RH``.

**Example** (all flags on, demand driver, 3 h resolution):

.. code-block:: text

   B_H_RE_H2_MEOH_METH_SN_ST_CO2_100_tD_H2_200_MeOH_9_CH4_350_2024_El_0.1_DET_3h_my_scenario

Flag abbreviations: ``B`` biogas · ``H`` central heat · ``RE`` renewables ·
``H2`` electrolysis · ``MEOH`` methanol · ``METH`` methanation · ``SN``
symbiosis · ``ST`` storage.

The ``{network}`` name is used only for files *inside* the run folder, keeping
individual file names informative. The **output folder** is just
``outputs/single_analysis/{run_name}/`` so the path stays short enough for
Windows' 260-character limit.

The **full configuration** is also saved to
``outputs/{run_name}/networks/config_run.yaml`` after every solve.

**Constraint:** literal match against the pre-computed ``NETWORK`` string
(via ``wildcard_constraints: network = NETWORK_PATTERN``).

**Outputs:** ``resources/{network}/{network}_PRE.nc``,
``outputs/single_analysis/{run_name}/networks/{network}_OPT.nc``
