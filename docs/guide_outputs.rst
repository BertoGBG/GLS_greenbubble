.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _guide-outputs:

Outputs & Results Reference
===========================

Every run writes to ``outputs/single_analysis/{run_name}/``
(e.g. ``my_scenario/``).

.. _outputs-naming:

Output naming structure
-----------------------

GreenBubble uses a **hybrid naming scheme** that separates the *folder* from
the *file names* inside it:

- **Folder** — just ``run_name`` from ``config.yaml``.  Short by design to
  stay within Windows' 260-character path limit.

- **File names** inside — a long descriptive prefix that encodes the full
  configuration detail::

     {flags}CO2_{co2}_{tD|tP}_H2_{h2}_MeOH_{meoh}_CH4_{ch4}_{year}_El_{el}_{DET|STC}_{res}_{run_name}

  Example::

     B_H_RE_H2_MEOH_METH_SN_ST_CO2_100_tD_H2_200_MeOH_9_CH4_350_2024_El_0.1_DET_3h_my_scenario

  Flag abbreviations: ``B`` biogas · ``H`` central heat · ``RE`` renewables ·
  ``H2`` electrolysis · ``MEOH`` methanol · ``METH`` methanation · ``SN``
  symbiosis · ``ST`` storage.  Rolling-horizon runs append ``_RH`` to the file
  name prefix.

Multiple configurations sharing the same ``run_name`` (different years, modes,
or flag sets) coexist in the same folder — the descriptive file names
distinguish them. The complete configuration is also captured in
``networks/config_run.yaml`` (see :ref:`outputs-config-run`).

.. code-block:: text

   outputs/single_analysis/<run_name>/
   ├── networks/    # solved network, run fingerprint (config_run.yaml), topology diagrams
   ├── csv/         # numeric results (the data behind the plots)
   ├── plots/       # figures + network diagrams
   └── duals/       # raw constraint duals (if collect_all_duals: true)

Plots and CSVs are produced from the same post-processing pass, so most figures
have a CSV twin; they are described together below.

---

.. _outputs-networks:

``networks/`` — the solved model
--------------------------------

================================  ====================================================
File                              Contents
================================  ====================================================
``<name>_OPT.nc``                 The solved PyPSA network (capacities, dispatch, duals).
                                  Re-load with ``pypsa.Network(path)`` for custom analysis.
``config_run.yaml``               **Run fingerprint** — the complete merged configuration
                                  used for this run (see :ref:`outputs-config-run`).
``network_comp_allocation.pkl``   Mapping of components to technology agents (internal use).
================================  ====================================================

The network **topology diagrams** live in ``plots/``:

- ``<name>_PRE.svg`` — the network *before* solving (all candidate components).
- ``<name>_OPT.svg`` — the *optimal* network (only built components, sized).
- ``*.dot`` — Graphviz source for the SVGs.

.. _outputs-config-run:

``config_run.yaml`` — the run fingerprint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``config_run.yaml`` is the **single authoritative record** of every parameter
used in a run.  It is the merged result of all three config-file pairs
(default + user override) written to disk after the solve completes.

**Structure:**

.. code-block:: yaml

   config:                        # merged config.default.yaml + config.yaml
     run_name: my_scenario
     CO2_cost: 100
     targets:
       driver: demand
       demand_H2: 200000
       ...
     n_flags:
       biogas: true
       renewables: true
       ...
     optimization:
       solver: highs
       solver_profile: highs-fast
     clustering:
       temporal:
         resolution: 3h
     ...

   n_config:                      # merged n_config.default.yaml + n_config.yaml
     battery:                     # keyed by component name
       initial capacity: 0
       expansion: true
       cost factor: 1
       max hours: 2.0
     biomass boiler:
       initial capacity: 30
       expansion: false
       remaining_investment_fraction: 0.1
       construction_year: 2020
     ...

   n_options:                     # options: section of n_config (DH, biomass markets, …)
     DH:
       enable: true
       price: 80.0
       peak capacity: 80.0
     pellets market:
       enable: true
       price: 60.0
     ...

   plots_config:                  # merged plots_config.default.yaml
     carrier_colors: ...
     plotting: ...

**How to use it:**

- **Reproduce a run** — copy ``config:`` back to ``config/config.yaml`` and
  ``n_config:`` + ``n_options:`` back to ``config/n_config.yaml`` (keep only
  keys that differ from the defaults).
- **Compare runs** — diff two ``config_run.yaml`` files to see exactly what
  changed between scenarios.
- **Audit brownfield settings** — ``n_config`` entries with ``initial capacity > 0``
  and ``expansion: false`` are the fixed brownfield assets; ``remaining_investment_fraction``
  tells you how much residual CAPEX was charged.
- **Check solver** — the ``optimization`` block records which solver and profile
  were actually used.

---

.. _outputs-capacities:

Capacities — *what gets built*
------------------------------

- **Plot** ``Opt_capacities_SP_vs_WS.png`` — optimal installed capacity per
  technology. In stochastic runs it contrasts the single shared design (**SP**,
  stochastic program) against the per-scenario **WS** (wait-and-see) optima.
- **CSV** ``optimal_capacities.csv`` — capacity, fixed cost (€/y), unit and energy
  capacity per component; ``opt_capacities_SP_vs_WP.csv`` — the SP-vs-WS table
  behind the plot.

---

.. _outputs-operation:

Operation — *how it runs*
-------------------------

- **Plot** ``CF_operation_by_scenario.png`` — capacity factor / utilisation of
  each technology (how hard each asset works).
- **Plot** ``Operation_heat_maps_by_scenario.png`` — hour-of-day × day-of-year
  dispatch heat maps, revealing daily and seasonal operating patterns.
- **Plot** ``CF_operation_heat_maps_by_scenario.png`` — the same as heat maps but
  normalised to capacity factor.
- **CSV** the underlying time series are in ``full_component_table.csv`` and the
  network ``.nc``.

---

.. _outputs-inputs:

Inputs — *the exogenous drivers*
--------------------------------

- **Plot** ``inputs_LDC_by_scenario.png`` — load-duration curves of the exogenous
  inputs (electricity price, gas price, renewable capacity factors). Sorting each
  series high→low shows how often prices/resources are favourable.

---

.. _outputs-shadow-prices:

Internal-market shadow prices
-----------------------------

The dual of each carrier's nodal balance is its **shadow price** (€/MWh) — the
marginal value of that energy/material inside the plant.

- **Plot** ``shd_prices_mean_bar.png`` — the **energy-weighted mean** shadow price
  per internal bus (the headline "what is H₂/CO₂/heat worth here" number).
- **CSV** ``shadow_prices_mean.csv`` — columns ``bus``,
  ``energy weighted mean (EUR/MWh)`` (the data behind that bar chart).
- **Plot** ``shd_prices_ldc.png`` — duration curve of each shadow price (how
  often it is high or low).
- **Plot** ``shd_prices_violin.png`` — violin showing the full snapshot
  distribution. The **crimson line** is the *snapshot-weighted* mean (the plain
  arithmetic average over all hours); the **blue diamond** is the
  *energy-weighted* mean (same value as ``shadow_prices_mean.csv``/ the bar
  chart, weighted by the energy flowing through the bus). The gap between the two
  is informative: if the energy-weighted mean sits above the snapshot mean, the
  carrier is worth most precisely when it flows most.

---

.. _outputs-srmc:

SRMC & merit order
------------------

- **Plot** ``srmc_by_technology.png`` — the **short-run marginal cost** time
  series per producing technology vs the product shadow price; where SRMC ≤ price
  the unit is *in merit* and runs.
- **CSV** ``srmc_by_technology.csv`` — per snapshot & link:
  ``SRMC_EUR_per_MWh``, ``dispatch_MW``, ``π_product_bus`` (product shadow price),
  ``in_merit`` flag.

---

.. _outputs-costs:

System cost & levelised cost
----------------------------

- **Plot** ``TSC_by_carrier.png`` — **total system cost** split by carrier /
  technology (annualised CAPEX + OPEX); ``TSC_by_agents.png`` — the same split by
  plant *agent*.
- **CSV** ``TSC_by_carrier.csv`` / ``TSC_by_agent.csv`` — columns ``scenario``,
  ``group``, ``capex``, ``opex``, ``total``, ``probability``.
- **Plot/CSV** ``lcop_by_technology.png`` / ``lcop_by_technology.csv`` — the
  **levelised cost of production** per product, broken into CAPEX, OPEX, indirect
  OPEX, by-product revenue, annual production and annual profit.
  ``lcop_kkt_by_technology.csv`` is the dual/KKT cross-check (zero-profit
  condition — see :ref:`guide-economic-analysis`).
- **CSV** ``pypsa_statistics.csv`` — PyPSA's standard statistics (optimal/installed
  capacity, supply, capacity factor, CAPEX, OPEX, revenue, market value).
- **CSV** ``cost_assumptions.csv`` — the techno-economic inputs actually used.

---

.. _outputs-full-table:

The master data table
----------------------

``full_component_table.csv`` is the **one-stop data source**: one row per
component with its plant, carrier, expandability, initial vs optimal capacity,
capacity factor, curtailment, specific and total fixed/variable costs, production
and revenue. Most figures above are aggregations of this table — start here when
a plot raises a question.

.. seealso::

   :ref:`guide-economic-analysis` (theory) · :ref:`wildcards` (network-name encoding)
   · :ref:`tutorial-1-greenfield` (worked example)

---

.. _outputs-rerun:

Repeating or resetting an analysis
-----------------------------------

To cleanly re-run an analysis from scratch, delete **both** the output folder
and the corresponding intermediate resources folder, then re-run Snakemake:

.. code-block:: bash

   # replace <run_name> with your run_name (e.g. my_scenario)
   rm -rf outputs/single_analysis/<run_name>/
   rm -rf resources/<run_name>/
   snakemake --cores 4

The ``resources/<run_name>/`` folder holds the pre-built ``.nc`` and the
component-allocation ``.pkl`` that Snakemake uses as intermediate inputs.
Deleting it forces ``build_network`` and ``solve_network`` to re-run in full.

If you only changed the configuration and want to **re-solve without
rebuilding** the network, delete only the output folder and the OPT network
inside resources (Snakemake tracks the PRE network as an input to solve):

.. code-block:: bash

   rm -rf outputs/single_analysis/<run_name>/
   snakemake --forcerun solve_network --cores 4

.. tip::

   The long file-name prefix (for manual inspection) is printed by
   ``snakemake -n`` (dry-run) and is also visible as the file-name stem inside
   ``outputs/single_analysis/<run_name>/networks/``.
