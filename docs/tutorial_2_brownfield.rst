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

.. admonition:: Interpretation [REVIEW]
   :class: caution

   *Draft — verify before publishing.*

   - With wind+solar+biogas fixed, the electrolyser and biomethanation are sized
     to the **residual** opportunity: compare built electrolyser/biomethanation
     capacity here vs the greenfield Tutorial 1.
   - District heating gives waste heat a value (30 EUR/MWh); expect more heat
     recovered and a small profit improvement — check the heat balance plot and
     the DH revenue line in ``csv/``.
   - Ramp + min-load on the electrolyser smooth its operation against the
     wind/solar profile; inspect the electrolyser dispatch time series.
   - ``rif`` only changes the *annual cost* of the fixed assets, not the dispatch
     — it shifts the LCOP, not the operation. *(Fill exact numbers from the run.)*

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
