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
it across three weather/CO₂ scenario years at once.

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
     # scenarios 2023/2024/2025 inherited from config.default.yaml

.. admonition:: Stochastic needs a pure LP — two required changes
   :class: caution

   The ``tutorials/4_stochastic/n_config.yaml`` already applies these; they are
   the reason it differs from the Tutorial 2 brownfield n_config:

   #. **No unit commitment** — ``committable: false`` everywhere (stochastic
      cannot use binary variables).
   #. **No ramp limits** — set ``ramp limit up`` / ``ramp limit down`` to
      ``null`` for electrolysis, methanolisation and biomethanation. PyPSA
      (≤ 1.2.2) cannot build ramp-limit constraints on a scenario network, and a
      value such as ``1`` still builds them — only ``null`` disables them. See
      :ref:`guide-stochastic` → *Limitations*.

The output folder uses the ``STC`` token instead of ``DET``.

---

3 · Interpret the results
-------------------------

.. figure:: /_static/tutorials/tut4_Opt_capacities_SP_vs_WS.png
   :width: 95%

   The single shared (stochastic-program) investment, sized for all three years.

.. admonition:: Interpretation [REVIEW]
   :class: important

   *Draft reading — verify before publishing.*

   - **Expected profit ≈ €17.7M/y** across 2023/2024/2025 — far below the
     single-year **2024** brownfield result (≈ €61.8M/y in :ref:`tutorial-2-brownfield`).
     2024 is an unusually profitable year; averaging in the leaner 2023/2025
     scenarios pulls the expectation down. This is exactly why optimising on one
     good year is misleading.
   - The stochastic design is a **hedge**: it keeps the cheap, always-useful
     **biogas upgrading (28 MW)** but under-builds the electricity-dependent
     capacity whose value swings between years — **electrolyser 4.3 MW and
     biomethanation 1.6 MW**, versus 25 MW and 11.3 MW in the (rosy) single-year
     2024 design. It invests boldly only where the payoff is robust across
     scenarios.
   - Inspect the **per-scenario dispatch** (``CF_operation_by_scenario.png``): the
     same plant runs differently each year, revealing which assets are stressed in
     which conditions.
   - Here ``EVPI: false``. Set ``EVPI: true`` to also solve each year with perfect
     foresight and quantify the **Expected Value of Perfect Information** — the
     annual value of knowing next year's prices in advance.

.. figure:: /_static/tutorials/tut4_CF_operation_by_scenario.png
   :width: 95%

---

What you learned
----------------

- The **here-and-now vs wait-and-see** two-stage structure and expected-cost objective.
- How to enable stochastic mode and the **pure-LP requirements** (no committable,
  null ramp limits).
- Reading a stochastic design as a **robust hedge** across scenarios.

This is the final tutorial in the core sequence. For exploring *near-optimal*
alternatives to a single design, see the near-optimal (MGA) guide.

.. seealso::

   :ref:`guide-stochastic` · :ref:`guide-outputs` · :ref:`config-stochastic` · :ref:`tutorial-2-brownfield`
