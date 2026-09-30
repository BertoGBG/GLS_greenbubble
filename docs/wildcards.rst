.. _wildcards:

Wildcards
=========

Snakemake wildcards are placeholder values resolved at runtime to determine
which files to build. GreenBubble currently uses the following wildcards.

``{folder}``
------------

**Used in:** ``preprocess_inputs``

The input-data folder of one energy price year, for which market data is
downloaded and preprocessed. The folder depends on the site's bidding zone:
``data/Inputs_{year}`` for DK_1 and ``data/{zone}/Inputs_{year}`` for other
zones. The year is taken from its last four digits.

In deterministic mode there is one folder, for ``En_price_year`` from
``config.yaml``. In stochastic mode there is one per key in
``stochastic.scenarios``. ``preprocessed_marker()`` in ``Snakefile`` builds the
paths.

**Example values:** ``data/Inputs_2024``, ``data/GB/Inputs_2024``

**Constraint:** ``data/(\w+/)?Inputs_\d{4}``

**Output:** ``{folder}/.preprocessed``

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
