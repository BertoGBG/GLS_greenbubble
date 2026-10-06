.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _tutorial-4-stochastic:

Tutorial 4 — Two-Stage Stochastic Optimisation
===============================================

The previous tutorials optimise against a **single** year. But the right
investment under 2023 prices may be wrong under 2025 prices. **Two-stage
stochastic optimisation** finds *one* set of investment decisions
(*here-and-now*) that performs best in expectation across several scenarios,
while **dispatch adapts per scenario** (*wait-and-see*).

We reuse the **brownfield** plant from :ref:`tutorial-2-brownfield` and optimise
it across three weather/market scenario years (2022 / 2023 / 2024) at once.

.. contents:: On this page
   :local:
   :depth: 1

---

1 · The two-stage idea
----------------------

.. math::

   \min_{\text{capacities}}\; \sum_{s} p_s \,
   \big[\, \text{CAPEX}^{\text{ann}} + \text{OPEX}_s \,\big]

Investment variables are **shared** across scenarios :math:`s` (you build one
plant); operational variables are **scenario-specific** (each year is dispatched
on its own prices and renewable profiles). Scenarios carry a probability
:math:`p_s` summing to 1. The objective is the **expected** annual cost.

---

2 · Run it
----------

.. code-block:: bash

   cp tutorials/4_stochastic/config.yaml   config/config.yaml
   cp tutorials/4_stochastic/n_config.yaml config/n_config.yaml
   snakemake --cores 4     # downloads/preprocesses all scenario years first

.. code-block:: yaml

   stochastic:
     stochastic: true
     EVPI: false
     # scenarios 2022/2023/2024 and their weights are set in config.default.yaml

.. admonition:: Stochastic needs a pure LP — two required changes
   :class: caution

   The ``tutorials/4_stochastic/n_config.yaml`` already applies these; they are
   the reason it differs from the Tutorial 2 brownfield n_config:

   #. **No unit commitment** — ``committable: false`` everywhere (stochastic
      cannot use binary variables).
   #. **No ramp limits** — set ``ramp limit up`` / ``ramp limit down`` to
      ``null`` for electrolysis, methanolisation, biomethanation and
      biomethanation CO2. PyPSA cannot build ramp-limit constraints on a
      scenario network — confirmed on both 1.0.7 (pinned) and the latest
      release (1.2.4), and a value such as ``1`` still builds them — only
      ``null`` disables them. See :ref:`guide-stochastic` → *Limitations*.

The output folder uses the ``STC`` token instead of ``DET``.
Temporal resolution is set to ``24h``. The stochastic LP holds all three years
at once, and at 8 h HiGHS did not converge within two hours. At 24 h it solves
in about four minutes. Daily snapshots cannot show within-day cycles, so
batteries and short-term arbitrage are not represented in this tutorial.

---

3 · Interpret the results
-------------------------

The three scenarios span very different market conditions (2022 = energy-crisis
year; 2023/2024 = post-crisis normalisation), with weights 10 % / 40 % / 50 %:

.. figure:: /_static/tutorials/tut4_inputs_LDC_by_scenario.png
   :width: 95%

   Input duration curves for all three scenarios. **2022** was the European
   energy-crisis year, with high gas and electricity prices. It is the most
   profitable scenario (€29.2 M/y net). **2023** and **2024** follow with
   €24.1 M/y and €23.5 M/y. The 90 % combined weight on 2023/2024 governs the
   final design.

The optimiser finds **one** investment that is robust across all three years:

.. figure:: /_static/tutorials/tut4_Opt_capacities_SP_vs_WS.png
   :width: 95%

   Stochastic-programme (SP) optimal investments: **biogas upgrading 28.7 MW**,
   an **AEC electrolyser of 14.7 MW** and **biomethanation 6.6 MW**. Existing
   brownfield assets (52 MW wind, 30 MW solar, 62.85 t/h DM biogas digester)
   carry over from Tutorial 2.

The expected cost breakdown shows which revenue streams justify the design:

.. figure:: /_static/tutorials/tut4_TSC_by_carrier.png
   :width: 95%

   Expected total system cost (probability-weighted) by carrier.
   **Biomethane (bioCH4)** dominates revenues at ≈ €31.5 M/y, followed by
   e-methane from biomethanation (€7.8 M/y) and district heating (€5.3 M/y).
   Fixed CAPEX is €18.5 M/y in every scenario, mostly the brownfield biogas
   digester, wind and solar. **Expected net profit: €24.3 M/y.**

Shadow prices show the internal marginal value of each carrier:

.. figure:: /_static/tutorials/tut4_shd_prices_mean_bar.png
   :width: 95%

   Energy-weighted mean shadow prices and annual throughput, shown for the
   first scenario (2022). E-methane collection sits at 200 €/MWh, its sale
   price, and biomethane collection at 120 €/MWh. Electricity (El3) is
   ≈ 106 €/MWh in the crisis year. Buses that carry no energy are marked
   "no flow".

.. note::

   The stochastic objective is the probability-weighted sum of the scenario
   costs, so PyPSA returns each scenario's bus duals multiplied by the scenario
   weight. The plotting step divides them by the weight, so the charts and CSVs
   show prices in EUR/MWh.

The same fixed-capacity plant dispatches differently in each scenario:

.. figure:: /_static/tutorials/tut4_CF_operation_by_scenario.png
   :width: 95%

   Per-scenario utilisation duration curves. Biogas upgrading runs at
   CF 0.84-0.87 in all three scenarios. The electrolyser runs *less* in 2022
   (CF 0.67), when electricity is expensive, and more in 2023 and 2024
   (CF 0.74-0.76).

.. admonition:: Key results
   :class: important

   - **Expected net profit: €24.3 M/y**
     (0.10 × €29.2 M + 0.40 × €24.1 M + 0.50 × €23.5 M). The 2022 crisis
     scenario is the most profitable but carries only 10 % weight. The design
     is governed by 2023-2024 conditions.
   - The stochastic design is a **hedge**. Biogas upgrading (28.7 MW) earns
     robust revenue in all three years. The electricity-sensitive electrolyser
     (14.7 MW) and biomethanation (6.6 MW) are sized for the normal years,
     without over-betting on any single price environment.
   - The **same capacity** runs differently across years. The electrolyser
     backs off when power is expensive (2022) and runs harder when it is cheap.
     This wait-and-see dispatch flexibility is what makes one design work in
     all three scenarios.
   - Here ``EVPI: false``. Set ``EVPI: true`` to also solve each year with
     perfect foresight and quantify the **Expected Value of Perfect
     Information**, the annual value of knowing next year's market in advance.

---

What you learned
----------------

- The **here-and-now vs wait-and-see** two-stage structure and expected-cost objective.
- How to enable stochastic mode and the **pure-LP requirements** (no committable,
  null ramp limits).
- Reading a stochastic design as a **robust hedge** across scenarios, and
  interpreting scenario-level profitability spreads.

This is the final tutorial in the core sequence. For exploring *near-optimal*
alternatives to a single design, see the near-optimal (MGA) guide.

.. seealso::

   :ref:`guide-stochastic` · :ref:`guide-outputs` · :ref:`config-stochastic` · :ref:`tutorial-2-brownfield`
