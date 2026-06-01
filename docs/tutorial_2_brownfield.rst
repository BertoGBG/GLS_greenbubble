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
     - **Greenfield** — built from scratch (Tutorial 1)
   * - >0
     - false
     - 0
     - **Brownfield, sunk** — fixed size, no capital charge
   * - >0
     - false
     - >0
     - **Brownfield, residual** — fixed size, *partial* capital charge
   * - >0
     - true
     - any
     - **Mixed** — existing block + expandable new capacity

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
     DH: {enable: true, price: 30}   # district heating off-take

The biogas plant (30 MW CH₄), wind (52 MW) and solar (30 MW) are now **fixed**;
the optimiser sizes only the remaining expandable technologies (electrolyser,
biomethanation, storage, …) around them.

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

.. figure:: /_static/tutorials/tut2_Opt_capacities_SP_vs_WS.png
   :width: 95%

   Optimal capacities. The ``EXI_`` assets (biogas 30 MW CH₄, wind 52 MW, solar
   30 MW) are fixed; everything else is sized around them.

.. admonition:: The key change — biomethanation now competes [REVIEW]
   :class: important

   *Draft reading — verify before publishing.*

   - **Both biomethane routes are built**: biogas upgrading **22 MW** *and*
     biomethanation **11.3 MW** (vs Tutorial 1, where biomethanation never built).
     The reason is the **fixed, cheap existing renewables**: 52 MW wind + 30 MW
     solar power a 25 MW electrolyser whose H₂ makes the extra biomethanation CH₄
     worthwhile at ``price_bioCH4 = 200``. So the brownfield context is exactly
     where the upgrading-vs-biomethanation *competition* turns into a *mix*.
   - **District heating** adds value to waste heat: the DH bus clears at
     ≈ 20 €/MWh and methanolisation/biomethanation export heat to it.
   - Profit ≈ **€61.8M/y** (higher than the greenfield price case — the existing
     assets are largely sunk, so only their residual cost is charged).
   - ``rif`` (here 0.3 for all existing assets) only changes the *annual capital
     charge* on those assets — it shifts the LCOP/profit, not the dispatch.

The process constraints above shape *how* these units run: inspect the
electrolyser dispatch (``CF_operation_by_scenario.png``) to see ramping and
minimum-load behaviour against the wind/solar profile.

.. figure:: /_static/tutorials/tut2_CF_operation_by_scenario.png
   :width: 95%

---

What you learned
----------------

- Greenfield vs brownfield vs mixed, and independent **residual cost** per asset.
- The **committable / ramping / min-load** process constraints and when each applies.
- Adding a heat off-take (district heating) as a revenue stream.

Next: :ref:`tutorial-3-rolling-horizon` re-dispatches this fixed plant hour-by-hour
over the full year.

.. seealso::

   :ref:`guide-outputs` · :ref:`config-economics` · :ref:`economics` · :ref:`tutorial-1-greenfield`
