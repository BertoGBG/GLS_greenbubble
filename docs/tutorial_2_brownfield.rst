.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _tutorial-2-brownfield:

Tutorial 2 — Brownfield & Process Constraints
==============================================

Building on :ref:`tutorial-1-greenfield`, this tutorial adds **existing assets**
(*brownfield*) and introduces the **process constraints** that shape realistic
dispatch: committable operation, ramping, and minimum load.

We stay **price-driven**, drop hydrogen-to-grid (``demand_H2: 0``), fix a few
plant sizes as pre-existing, and enable **district heating** as a heat off-take.

.. contents:: On this page
   :local:
   :depth: 1

---

1 · Greenfield vs brownfield
----------------------------

A technology's investment mode is set by three ``n_config`` keys:

.. list-table::
   :widths: 22 14 14 50
   :header-rows: 1

   * - initial capacity
     - expansion
     - rif
     - meaning
   * - 0
     - true
     - 0
     - **Greenfield**: built from scratch (Tutorial 1)
   * - >0
     - false
     - 0
     - **Brownfield, sunk**: fixed size, no capital charge
   * - >0
     - false
     - >0
     - **Brownfield, residual**: fixed size, *partial* capital charge
   * - >0
     - true
     - any
     - **Mixed**: existing block + expandable new capacity

``rif`` = ``remaining_investment_fraction``: the share of the *original*
investment still being paid off. The annual charge is

.. math::

   \text{rif} \times \text{investment}(\text{construction\_year})
   \times \text{annuity}(r, \text{amortization\_period}).

Because ``rif`` is set **per technology**, each existing asset carries its own
residual cost independently.

---

2 · Run it
----------

.. code-block:: bash

   cp tutorials/2_brownfield/config.yaml   config/config.yaml
   cp tutorials/2_brownfield/n_config.yaml config/n_config.yaml
   snakemake --cores 4

Existing assets (:download:`n_config.yaml <../tutorials/2_brownfield/n_config.yaml>`):

.. code-block:: yaml

   biogas:   {initial capacity: 30, expansion: false, construction_year: 2020, remaining_investment_fraction: 0.3}
   onwind:   {initial capacity: 52, expansion: false, construction_year: 2020, remaining_investment_fraction: 0.5}
   solar:    {initial capacity: 30, expansion: false, construction_year: 2020, remaining_investment_fraction: 0.5}
   options:
     DH: {enable: true, price: 30}   # district heating off-take at 30 €/MWh

The biogas plant (30 MW CH₄), wind (52 MW) and solar (30 MW) are now **fixed**;
the optimiser sizes only the remaining expandable technologies (electrolyser,
biomethanation, storage, …) around them. Each existing asset carries a different
residual fraction: the biogas plant recovers 30 % of its original investment,
wind and solar 50 %.

---

3 · Process constraints
-----------------------

Three constraints make dispatch physically realistic. They are configured per
technology in ``n_config``:

* **Committable** (``committable: true``) — binary on/off unit commitment. Only
  valid for **fixed-size** components (``expansion: false``) or rolling horizon,
  because it turns the problem into a MILP. Leave ``false`` for expandable units.
* **Ramp limits** (``ramp limit up`` / ``ramp limit down``) — max fractional
  change in output per hour (e.g. ``0.9`` = 90 %/h for the electrolyser).
* **Minimum load** (``min load``) — fraction of capacity that must run when the
  unit is on (e.g. ``0.15`` for the electrolyser); below it the unit shuts off.

The common result figures (capacities, operation, shadow prices, system cost)
are described once in :ref:`guide-outputs` and read as in
:ref:`tutorial-1-greenfield`; below we focus only on **what brownfield changes**.

---

4 · Interpret the results
-------------------------

.. figure:: /_static/tutorials/tut2_Opt_capacities_SP_vs_WS.png
   :width: 95%

   Optimal capacities. The ``EXI_`` assets (biogas 62.85 t/h DM ≈ 30 MW CH₄,
   wind 52 MW, solar 30 MW) are fixed; everything else is sized around them.

.. admonition:: The key change: biomethanation now competes
   :class: important

   - **Both biomethane routes are built**: biogas upgrading **23.9 MW** *and*
     biomethanation **8.0 MW**. In Tutorial 1 with a fixed demand, only upgrading
     was built. The reason is the **fixed, cheap existing renewables**: 52 MW wind
     and 30 MW solar power a 20.8 MW new electrolyser. Its H₂ turns biogas CO₂
     into extra CH₄, which sells at ``price_bioCH4 = 200 €/MWh``. The brownfield
     context turns the upgrading-vs-biomethanation *competition* into a *mix*.
   - **District heating** adds value to waste heat. The DH bus shadow price
     clears at ≈ 24 €/MWh, and biomethanation and heat-exchanger links export
     heat to it.
   - Net profit ≈ **€25.4 M/y**. The existing assets are largely sunk, so only
     their residual CAPEX is charged rather than a full greenfield investment.
   - Electrolyser CF ≈ 0.73, biomethanation CF ≈ 0.90 (running almost wherever
     H₂ is available), upgrading CF ≈ 0.91.

The process constraints shape *how* units run. The electrolyser follows cheap
renewable periods (visible in the operation LDCs below), while upgrading acts as
baseload because the biogas supply is continuous.

.. figure:: /_static/tutorials/tut2_CF_operation_by_scenario.png
   :width: 95%

.. figure:: /_static/tutorials/tut2_shd_prices_mean_bar.png
   :width: 80%

   Energy-weighted mean shadow prices at internal carrier buses. The e-methane
   collection bus sits at 200 €/MWh, its sale price. H₂ collection clears at
   ≈ 144 €/MWh and the heat buses at 23–24 €/MWh.

---

5 · Payback by agent
---------------------

Brownfield changes more than dispatch. It changes how fast each agent pays back
what is still owed on it. This tutorial also sets ``amortization_period: 10``,
shorter than most technologies' technical lifetime. This matters for how the
payback numbers below should be read (see :ref:`economics-payback` for the full
formulas).

.. figure:: /_static/tutorials/tut2_payback_by_agent.png
   :width: 95%

   Payback and capital cost coverage by agent (price mode; gated on
   ``targets.driver == 'price'``).

.. admonition:: Reading brownfield vs. cross-subsidised agents
   :class: important

   - **Brownfield-heavy agents pay back fast, by construction**: ``biogas``
     (552 % coverage, 1.4-year discounted payback), ``renewables`` (211 %,
     3.9 y) and ``symbiosis`` (756 %, 1.0 y). Only 30-50 % of the original
     biogas, wind and solar investment is still outstanding, so modest cash flow
     clears it easily. This is the payback-side view of the
     ``remaining_investment_fraction`` mechanism from Section 1.
   - **``electrolysis`` and ``methanation`` are the interesting pair**. Both are
     greenfield and freely sized by the optimiser, yet coverage is only 77 % and
     59 % (discounted payback 15 and 29 years). This is *not* a sign that they are
     mis-sized. It is a **cross-subsidy**: their value shows up on other agents'
     books. Electrolysis H₂ lets methanation turn biogas CO₂ into extra methane,
     and the biogas agent earns on the larger CH₄ output. Checking a
     low-coverage, freely sized agent against the agents around it is the
     general diagnostic; see :ref:`guide-economic-analysis`.
   - **``amortization_period: 10`` sets the bar these agents are compared
     against.** Coverage is cash flow ÷ *effective*-period annuity: here 10 years,
     not each technology's own 20-30-year technical lifetime. The technical
     lifetime is still shown separately (black tick / "technical lifetime"
     column). A shorter amortization period demands faster capital recovery.
     This is why ``central_heat`` (45 %) sits below 100 % without being a loss.

---

What you learned
----------------

- Greenfield vs brownfield vs mixed, and independent **residual cost** per asset.
- The **committable / ramping / min-load** process constraints and when each applies.
- Adding a heat off-take (district heating) as a revenue stream.
- Why brownfield context changes the biomethanation break-even.
- Reading per-agent payback and capital cost coverage, and spotting a
  cross-subsidised agent versus a genuine brownfield fast-payback.

Next: :ref:`tutorial-3-rolling-horizon` re-dispatches this fixed plant hour-by-hour
over the full year using rolling horizon, and :ref:`tutorial-2b-brownfield-heat`
extends the brownfield plant with a larger electrolysis and heat-network integration.

.. seealso::

   :ref:`guide-outputs` · :ref:`config-economics` · :ref:`economics` ·
   :ref:`economics-payback` · :ref:`guide-economic-analysis` ·
   :ref:`tutorial-1-greenfield`
