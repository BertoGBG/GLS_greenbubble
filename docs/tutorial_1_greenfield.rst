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
(``amortization_period: 10``). Only **biological methanation** is available, of
biogas and of CO₂; the catalytic Sabatier routes are switched off. This leaves
**biomethanation** and **biogas upgrading** competing to supply the biomethane
demand.

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
     price_H2:      120        # EUR/MWh
     price_bioCH4:  200        # EUR/MWh
     price_meoh:    200        # EUR/MWh
     demand_H2:   200000       # now an UPPER BOUND on production

In **demand mode** the ``demand_*`` values are *equality* constraints (you must
deliver exactly that much). In **price mode** they become *upper bounds*: the
model produces a product only while its sale price exceeds its marginal +
annualised capital cost, so price mode reads out the **break-even** of each route.

.. note::

   Both runs use ``clustering.temporal.resolution: 8h`` and the default HiGHS
   solver, so each finishes in about five minutes on a laptop. Outputs land in
   ``outputs/single_analysis/{run_name}/`` (e.g. ``tut1_demand/``). File names
   inside encode the full configuration (see :ref:`wildcards`). The full
   configuration is also saved to ``networks/config_run.yaml`` inside that
   folder.

---

3 · Interpret the results (demand case)
---------------------------------------

This is the most detailed walkthrough in the series. Later tutorials only
revisit what changes. For the full map of the output folder and every file see
:ref:`guide-outputs`. We read six figures in order. The numbers quoted are from
the 8 h reference run.

**(a) Inputs, the drivers** (:ref:`outputs-inputs`). Load-duration curves of
electricity price, gas price and wind/solar capacity factors set the economics.
How often electricity is cheap decides how attractive electrolysis is.

.. figure:: /_static/tutorials/tut1_demand_inputs_LDC_by_scenario.png
   :width: 95%

**(b) Capacities, what gets built** (:ref:`outputs-capacities`; data in
``optimal_capacities.csv``). The model builds **170 MW onshore wind** (CF 0.34)
and no solar. An **86 MW alkaline electrolyser** (AEC, CF 0.56) supplies the
hydrogen. The 350 GWh/y biomethane demand is met by **40 MW of biogas upgrading
running flat out**. Biomethanation is not built.

.. figure:: /_static/tutorials/tut1_demand_Opt_capacities_SP_vs_WS.png
   :width: 95%

.. admonition:: Biomethanation vs biogas upgrading, the key result
   :class: important

   Both routes deliver pipeline-grade biomethane. **Upgrading** strips CO₂ out of
   biogas. It is cheap, but the carbon is vented, so the CH₄ yield is lower.
   **Biomethanation** reacts that CO₂ with green H₂ into *extra* CH₄. The yield is
   higher, but it needs an electrolyser and electricity.

   **When the biomethane quantity is fixed, upgrading wins outright: biomethanation
   is not built** (0 MW). Upgrading is the cheapest way to deliver a fixed
   350 GWh/y. The electrolyser that *is* built serves the H₂ and methanol demands,
   not methanation.

   The price case below reverses this result. Once extra CH₄ can be sold at
   200 €/MWh, the higher yield of biomethanation pays off.

**(c) Operation, how it runs** (:ref:`outputs-operation`). Capacity factors show
how hard each asset works; the heat maps show *when*. The electrolyser runs at
CF 0.56, following cheap-power periods. Upgrading runs constantly.

.. figure:: /_static/tutorials/tut1_demand_CF_operation_by_scenario.png
   :width: 95%

.. figure:: /_static/tutorials/tut1_demand_Operation_heat_maps_by_scenario.png
   :width: 95%

**(d) Internal-market shadow prices** (:ref:`outputs-shadow-prices`; data in
``shadow_prices_mean.csv``). In demand mode these are the **marginal cost of
meeting each product's demand**: H₂ ≈ **106 €/MWh**, biomethane ≈ **119 €/MWh**,
methanol ≈ **155 €/MWh**. Internal CO₂ costs ≈ 2.7 €/MWh and medium-temperature
heat ≈ 21 €/MWh. The time-resolved ``srmc_by_technology.png``
(:ref:`outputs-srmc`) shows which units are *in merit* in each period.

.. figure:: /_static/tutorials/tut1_demand_shd_prices_mean_bar.png
   :width: 80%

**(e) Total system cost** (:ref:`outputs-costs`; data in ``TSC_by_carrier.csv``).
Net total **≈ €64.3 M/y**. The **biogas plant** dominates (€36.3 M/y), followed by
**wind** (€22.7 M/y), the **electrolyser** (€8.2 M/y) and **upgrading**
(€3.9 M/y). The grid connection nets about €10 M/y from electricity sales. In demand mode
each product's LCOP equals its delivery shadow price (bioCH₄ 119, H₂ 106,
MeOH 155 €/MWh). This is the zero-profit signature of a cost-minimising solve.

.. figure:: /_static/tutorials/tut1_demand_TSC_by_carrier.png
   :width: 95%

**(f) The data behind it all.** Every number above aggregates
``csv/full_component_table.csv`` (:ref:`outputs-full-table`). It has one row per
component with capacity, capacity factor, costs, production and revenue.

---

4 · The price case
------------------

Re-run with the price-driven config (Section 2): ``price_H2 = 120``,
``price_bioCH4 = 200`` and ``price_meoh = 200`` €/MWh. Production is now optional
and driven by profitability. The model maximises **profit**, net ≈ **€10.1 M/y**.

.. figure:: /_static/tutorials/tut1_price_Opt_capacities_SP_vs_WS.png
   :width: 95%

.. admonition:: Price-case results
   :class: important

   Compare each product's price against its break-even LCOP from the demand case:

   - **Biomethane (price 200 ≫ LCOP 119)**: sold up to its 350 GWh/y cap. The
     route flips: **biomethanation (20 MW)** replaces biogas upgrading, which is
     not built at all. At 200 €/MWh the extra CH₄ from turning biogas CO₂ into
     methane is worth more than the hydrogen it consumes. The biogas plant also
     shrinks, because each tonne of feedstock now yields more methane.
   - **H₂ to grid (price 120 > LCOP 106)**: profitable, so it is sold up to its
     200 GWh/y cap. Together with biomethanation this needs a **154 MW AEC** and
     **300 MW of wind**, both much larger than in the demand case.
   - **Methanol (price 200 vs LCOP 155)**: *not* produced. The demand-case LCOP
     was set by a small, steady 9 GWh/y. In the price case hydrogen and CO₂ are
     worth more in biomethanation, so methanol no longer pays. This is the
     clearest "price reveals the opportunity cost" signal.

.. figure:: /_static/tutorials/tut1_price_TSC_by_carrier.png
   :width: 95%

The cost-by-carrier plot now shows **revenue bars** (products sold) against
technology costs. The net is a profit rather than a pure cost.

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
