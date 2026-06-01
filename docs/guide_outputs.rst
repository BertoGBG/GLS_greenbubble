.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _guide-outputs:

Outputs & Results Reference
===========================

Every run writes to ``outputs/single_analysis/<network_name>/``, where
``<network_name>`` encodes the active flags, targets, year and mode (see
:ref:`wildcards`). This guide is a **map of that folder** — what each file is and
how the CSVs and plots pair up. For the *theory* behind the economic quantities
(LCOP, SRMC, shadow prices, KKT) see :ref:`guide-economic-analysis`.

.. code-block:: text

   outputs/single_analysis/<network_name>/
   ├── networks/   # solved network + the exact config used
   ├── csv/        # numeric results (the data behind the plots)
   └── plots/      # figures + network diagrams

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
``config_run.yaml``               The *full* configuration used for this run — the
                                  reproducibility record (also read by the near-optimal tool).
``network_comp_allocation.pkl``   Mapping of components to technology agents (internal use).
================================  ====================================================

The network **topology diagrams** live in ``plots/``:

- ``<name>_PRE.svg`` — the network *before* solving (all candidate components).
- ``<name>_OPT.svg`` — the *optimal* network (only built components, sized).
- ``*.dot`` — Graphviz source for the SVGs.

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
- **Plot** ``shd_prices_ldc.png`` / ``shd_prices_violin.png`` — the full
  *distribution* of each shadow price over time (duration curve and violin).

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
