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
     price_H2:      120        # EUR/MWh
     price_bioCH4:  200        # EUR/MWh
     price_meoh:    200        # EUR/MWh
     demand_H2:   200000       # now an UPPER BOUND on production

In **demand mode** the ``demand_*`` values are *equality* constraints (you must
deliver exactly that much). In **price mode** they become *upper bounds*: the
model produces a product only while its sale price exceeds its marginal +
annualised capital cost, so price mode reads out the **break-even** of each route.

.. note::

   Both runs use ``clustering.temporal.resolution: 3h`` and the default HiGHS
   solver so they finish in a few minutes on a laptop. Outputs land in
   ``outputs/single_analysis/{run_name}_{year}_{det|stc}_{res}/``
   (e.g. ``tut1_demand_2024_det_3h/``). The full configuration — flags, CO2
   cost, product targets — is saved to ``networks/config_run.yaml`` inside
   that folder.

---

3 · Interpret the results (demand case)
---------------------------------------

This is the most detailed walkthrough in the series — later tutorials only
revisit what changes. For the full map of the output folder and every file see
:ref:`guide-outputs`. We read six figures in order; the numbers quoted are from
the 3 h reference run.

**(a) Inputs — the drivers** (:ref:`outputs-inputs`). Load-duration curves of
electricity price, gas price and wind/solar capacity factors set the economics:
how often electricity is cheap decides how attractive electrolysis is.

.. figure:: /_static/tutorials/tut1_demand_inputs_LDC_by_scenario.png
   :width: 95%

**(b) Capacities — what gets built** (:ref:`outputs-capacities`; data in
``optimal_capacities.csv``). The model builds **127 MW onshore wind** (CF 0.32),
**10 MW solar** (CF 0.11), a **51.5 MW electrolyser** (CF 0.69) and meets the
350 GWh/y biomethane demand with **40 MW of biogas upgrading running near
flat-out**. Biomethanation is not built — at default costs biogas upgrading is
the cheaper biomethane route within a 10-year payback window.

.. figure:: /_static/tutorials/tut1_demand_Opt_capacities_SP_vs_WS.png
   :width: 95%

.. admonition:: Biomethanation vs biogas upgrading — the key result
   :class: important

   Both routes deliver pipeline-grade biomethane: **upgrading** strips CO₂ out of
   biogas (cheap, but carbon is vented → lower CH₄ yield); **biomethanation**
   reacts that CO₂ with green H₂ into *extra* CH₄ (higher yield, but needs an
   electrolyser and electricity).

   **At default costs with a 10-year payback, biogas upgrading wins outright:
   biomethanation is not built at all** (0 MW). The electrolyser that *is* built
   (52 MW) serves the H₂-to-grid and methanol demands, not methanation. So the
   "competition" resolves decisively in favour of upgrading here — the extra CH₄
   from biomethanation does not pay back the H₂ + reactor cost within 10 years.

   To make biomethanation competitive: increase ``price_bioCH4``, lower
   ``amortization_period`` further, or add CO₂ utilisation incentives.

**(c) Operation — how it runs** (:ref:`outputs-operation`). Capacity factors show
how hard each asset works; the heat maps show *when*. The electrolyser runs at
CF 0.69, following cheap-power hours; upgrading runs near-constantly.

.. figure:: /_static/tutorials/tut1_demand_CF_operation_by_scenario.png
   :width: 95%

.. figure:: /_static/tutorials/tut1_demand_Operation_heat_maps_by_scenario.png
   :width: 95%

**(d) Internal-market shadow prices** (:ref:`outputs-shadow-prices`; data in
``shadow_prices_mean.csv``). In demand mode these are the **marginal cost of
meeting each product's demand**: H₂ ≈ **130 €/MWh**, biomethane ≈ **119 €/MWh**,
methanol ≈ **175 €/MWh**; internal CO₂ ≈ 2.7 €/MWh and medium-temperature heat
≈ 20 €/MWh. The time-resolved ``srmc_by_technology.png`` (:ref:`outputs-srmc`)
shows which units are *in merit* hour by hour.

.. figure:: /_static/tutorials/tut1_demand_shd_prices_mean_bar.png
   :width: 80%

**(e) Total system cost** (:ref:`outputs-costs`; data in ``TSC_by_carrier.csv``).
Net total **≈ €69.5 M/y**, dominated by the **biogas plant CAPEX**, then **wind**,
**electrolysis** and **upgrading**, with a small grid-export revenue. In demand
mode each product's LCOP equals its delivery shadow price (bioCH₄ 119, H₂ 130,
MeOH 175 €/MWh) — the zero-profit signature of a cost-minimising solve.

.. figure:: /_static/tutorials/tut1_demand_TSC_by_carrier.png
   :width: 95%

**(f) The data behind it all.** Every number above aggregates
``csv/full_component_table.csv`` (:ref:`outputs-full-table`) — one row per
component with capacity, capacity factor, costs, production and revenue.

---

4 · The price case
------------------

Re-run with the price-driven config (Section 2): prices ``price_H2 = 120``,
``price_bioCH4 = 200``, ``price_meoh = 200`` €/MWh. Now production is optional
and driven by profitability — the model maximises **profit** (net ≈ **€28.3 M/y**).

.. figure:: /_static/tutorials/tut1_price_Opt_capacities_SP_vs_WS.png
   :width: 95%

.. admonition:: Price-case results
   :class: important

   Compare each product's price against its break-even LCOP from the demand case:

   - **Biomethane (price 200 ≫ LCOP 119)** — the big winner: produced up to its
     cap (350 GWh/y) entirely via **biogas upgrading (40 MW, flat)**. Even at this
     high price biomethanation is not built — upgrading remains the cheaper route,
     so the profit-maximiser never needs it.
   - **Methanol (price 200 > LCOP 175)** — produced at its full cap (9 GWh/y); a
     small electrolyser (2.9 MW), wind (12.6 MW) and solar (2.6 MW) are built
     essentially to feed methanol synthesis.
   - **H₂ to grid (price 120 < LCOP 130)** — *not* profitable at the greenfield
     cost, so very little is sold: the electrolyser is sized for methanol's H₂
     need only. This is the clearest "price reveals break-even" signal — raise
     ``price_H2`` above ~130 €/MWh and H₂-to-grid immediately becomes attractive.

.. figure:: /_static/tutorials/tut1_price_TSC_by_carrier.png
   :width: 95%

The cost-by-carrier plot now shows **revenue bars** (products sold) against
technology costs; the net is a profit rather than a pure cost.

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
