.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _tutorial-1-greenfield:

Tutorial 1 — Greenfield: Demand vs Price
========================================

.. note::

   **Tutorial format.** Each tutorial follows the same loop: *(1) read the
   concept → (2) copy a ready-made config into* ``config/`` *→ (3) run and
   interpret the results.* Config files live in ``tutorials/<name>/`` and are
   copied over the (gitignored) ``config/config.yaml`` and
   ``config/n_config.yaml`` user overrides.

This first tutorial builds the plant **from scratch** (*greenfield*: every
capacity is an investment decision, nothing pre-exists) and contrasts the two
ways GreenBubble can be driven:

* **1.1 demand-driven** — you fix annual production of the three products
  (H₂ to the grid, biomethane, methanol) and minimise total system cost.
* **1.2 price-driven** — you fix sale *prices* and let the model decide *how
  much* of each product to make to maximise profit.

In both cases every investment must **pay back within 10 years**
(``amortization_period: 10``) and only **biological methanation**
(*biomethanation* of biogas and of CO₂) is available — the catalytic (Sabatier)
routes are switched off — so we can watch the competition between
**biomethanation** and **biogas upgrading** for supplying the biomethane demand.

.. contents:: On this page
   :local:
   :depth: 1

---

1 · The economic basis
----------------------

GreenBubble minimises (demand mode) or maximises the negative of (price mode)
the **annualised total system cost**

.. math::

   \text{cost} = \sum_i \text{CAPEX}_i^{\text{ann}} \cdot P_i
               + \sum_{i,t} \text{VOM}_i \cdot p_{i,t}
               + \text{(imports)} - \text{(revenue)} .

Each technology's investment is turned into a yearly charge with an **annuity**:

.. math::

   \text{CAPEX}^{\text{ann}} = \text{investment} \times
   \frac{r\,(1+r)^{n}}{(1+r)^{n}-1},

with discount rate :math:`r` (``discount_rate: 0.07``) and lifetime :math:`n`.
By default :math:`n` is each technology's *technical lifetime*; here we set
``amortization_period: 10`` so **every** technology is amortised over 10 years.

.. admonition:: Why this matters
   :class: tip

   A shorter amortisation period raises the annual capital charge, so the model
   only builds a technology if it earns its (steeper) payback within 10 years.
   This is the single most important economic lever in the tutorial — see
   :ref:`economics` for the full treatment.

---

2 · Run it
----------

**1.1 — demand-driven**

.. code-block:: bash

   cp tutorials/1_greenfield_demand/config.yaml   config/config.yaml
   cp tutorials/1_greenfield_demand/n_config.yaml config/n_config.yaml
   snakemake --cores 4

Key settings (:download:`config.yaml <../tutorials/1_greenfield_demand/config.yaml>`,
:download:`n_config.yaml <../tutorials/1_greenfield_demand/n_config.yaml>`):

.. code-block:: yaml

   targets:
     driver: demand
     demand_H2:   200000     # MWh/y to the grid
     demand_CH4:  350000     # MWh/y biomethane
     demand_meoh:   9000     # MWh/y methanol
     CH4_demand_mode:  flat       # constant demand
     MeOH_demand_mode: bins_flat  # stepwise (bins) demand
   amortization_period: 10

The two demand **shapes** shown — ``flat`` (constant every hour) and
``bins_flat`` (a few constant steps) — are the two simplest of the four modes;
see :ref:`guide-demands` for ``profile`` and ``bins_profile``.

**1.2 — price-driven**

.. code-block:: bash

   cp tutorials/1_greenfield_price/config.yaml   config/config.yaml
   cp tutorials/1_greenfield_price/n_config.yaml config/n_config.yaml
   snakemake --cores 4

.. code-block:: yaml

   targets:
     driver: price
     price_H2:      100        # EUR/MWh
     price_bioCH4:  NG_based   # natural-gas-linked price
     price_meoh:    200        # EUR/MWh
     demand_H2:   200000       # now an UPPER BOUND on production
     ...

In **demand mode** the ``demand_*`` values are *equality* constraints (you must
deliver exactly that much). In **price mode** they become *upper bounds*: the
model produces a product only while its sale price exceeds its marginal +
annualised capital cost, so price mode reads out the **break-even** of each route.

.. note::

   Both runs use ``clustering.temporal.resolution: 3h`` and the default HiGHS
   solver so they finish in a few minutes on a laptop. Outputs land in
   ``outputs/single_analysis/<network name>/`` (``plots/``, ``networks/``,
   ``csv/``).

---

3 · Interpret the results
-------------------------

This is the most detailed walkthrough in the series — later tutorials only
revisit what changes. For the full map of the output folder and every file see
:ref:`guide-outputs`; here we read six figures in order. *(Figures from the 3 h
reference runs will be embedded; filenames are given so you can open them now.)*

**(a) Inputs — the drivers** (``inputs_LDC_by_scenario.png``,
:ref:`outputs-inputs`). Start here: the load-duration curves of electricity
price, gas price and wind/solar capacity factors set the economics. How often
electricity is cheap determines how attractive electrolysis (and hence
biomethanation) is.

**(b) Capacities — what gets built** (``Opt_capacities_SP_vs_WS.png``,
:ref:`outputs-capacities`; data in ``optimal_capacities.csv``). The headline
result: which technologies are built and how large.

.. admonition:: Biomethanation vs biogas upgrading [REVIEW]
   :class: caution

   *Draft reading — verify before publishing.* Both routes deliver pipeline-grade
   biomethane: **upgrading** strips CO₂ from biogas (cheap, but discards carbon →
   lower CH₄ yield); **biomethanation** reacts that CO₂ with green H₂ into
   *extra* CH₄ (higher yield, but needs an electrolyser + electricity).

   - **Demand case**: ``demand_CH4`` is met by the cheapest mix under a 10-year
     payback — expect upgrading to cover the base and biomethanation to grow only
     while cheap renewable H₂ makes the extra CH₄ competitive. Read the split off
     the capacities plot.
   - **Price case**: biomethanation appears only if ``price_bioCH4`` (NG-linked)
     clears the H₂ + reactor cost; otherwise production sits below the cap — the
     produced-vs-cap gap is the economic signal.

**(c) Operation — how it runs** (``CF_operation_by_scenario.png`` and
``Operation_heat_maps_by_scenario.png``, :ref:`outputs-operation`). Capacity
factors show how hard each asset works; the heat maps show *when* (daily/seasonal
patterns) — e.g. the electrolyser tracking cheap-power hours.

**(d) Internal-market shadow prices** (``shd_prices_mean_bar.png``,
:ref:`outputs-shadow-prices`; data in ``shadow_prices_mean.csv``). The
energy-weighted marginal value of H₂, CO₂, heat, … inside the plant — what each
internal carrier is "worth". The time-resolved view is
``srmc_by_technology.png`` (:ref:`outputs-srmc`): where a technology's
short-run marginal cost sits below the product shadow price, it runs (*in merit*).

**(e) Total system cost** (``TSC_by_carrier.png``, :ref:`outputs-costs`; data in
``TSC_by_carrier.csv``). The annualised CAPEX + OPEX split by technology — the
clearest view of where the money goes and whether the 10-year payback binds.
``lcop_by_technology.csv`` gives the levelised cost per product.

.. admonition:: Reading the economics [REVIEW]
   :class: caution

   *Draft.* H₂-to-grid (``price_H2 = 100``) and methanol (``price_meoh = 200``)
   are produced only while profitable — check whether either hits its cap in (b),
   and confirm in (e)/LCOP that built technologies clear their annualised cost
   within 10 years. *(Fill exact capacities, LCOP and TSC from the runs.)*

**(f) The data behind it all.** Every number above aggregates
``csv/full_component_table.csv`` (:ref:`outputs-full-table`) — one row per
component with capacity, capacity factor, costs, production and revenue. Open it
when a figure raises a question.

---

What you learned
----------------

- The annuity / ``amortization_period`` mechanism and why payback length drives
  what gets built.
- The difference between **demand** (fixed production, minimise cost) and
  **price** (fixed prices, maximise profit) optimisation.
- The **biomethanation vs biogas-upgrading** trade-off for biomethane supply.

Next: :ref:`tutorial-2-brownfield` adds *existing* assets (brownfield) with
residual investment costs, and process constraints (committable, ramping,
min-load).

.. seealso::

   :ref:`guide-outputs` (every result file) · :ref:`guide-economic-analysis` (theory)
   · :ref:`economics` · :ref:`guide-demands` · :ref:`config-targets` · :ref:`config-economics`
