.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _tutorial-3-rolling-horizon:

Tutorial 3 — Rolling Horizon Dispatch
=====================================

Capacity expansion decides *what to build*. **Rolling horizon** decides *how to
operate* a fixed plant through the year, one sliding window at a time. This is
closer to how a plant is run, because the operator never knows the whole year in
advance.

This tutorial takes the **solved network from** :ref:`tutorial-2b-brownfield-heat`
and re-solves only its **dispatch**.

.. contents:: On this page
   :local:
   :depth: 1

---

1 · How rolling horizon works
-----------------------------

The year is solved in overlapping windows (here **3 weeks** with a **9-day**
overlap). Each window starts from the storage
levels left by the previous one, and its overlap is discarded to avoid
end-of-horizon effects. Capacity expansion is **bypassed**: capacities are read
from the ``network_path`` network and held fixed.

``horizon`` and ``overlap`` count **snapshots**, not hours. Rolling horizon keeps
the time resolution of the loaded network. Tutorial 2b solves at 8 h, so the
63-snapshot windows below cover three weeks each, and the 27-snapshot overlap
covers nine days.

.. important::

   Rolling horizon is **dispatch-only**. It needs a previously solved
   ``*_OPT.nc``, so **run Tutorial 2b first**.

Three adjustments are made to the loaded network before the windows are solved:

- Each capacity is fixed at its optimum plus a tiny margin (10\ :sup:`-6`
  relative, 10\ :sup:`-3` absolute). Without it, solver round-off in the
  original solve can leave a link a fraction of a kW short of the load it must
  carry.
- Free, non-cyclic stores are annual sink counters (CO₂ vent, ambient heat, DH
  sales, digestate, grid sales). Their optimal size is only the cumulative total
  of the original year, so they are left uncapped.
- Cyclic storage constraints are switched off, because no single window spans
  the whole year.

If any window is infeasible, the rule stops with an error rather than exporting
an incomplete dispatch.

---

2 · Run it
----------

.. code-block:: bash

   # 1) run Tutorial 2b, which writes its *_OPT.nc
   # 2) network_path is pre-filled; edit it if your path differs
   cp tutorials/3_rolling_horizon/config.yaml   config/config.yaml
   cp tutorials/3_rolling_horizon/n_config.yaml config/n_config.yaml
   snakemake --cores 4

Targets, flags, resolution and ``n_config.yaml`` match Tutorial 2b, so the output
file name describes the same plant. Only ``run_name`` and the ``rolling_horizon``
block differ:

.. code-block:: yaml

   run_name: tut3_rh

   rolling_horizon:
     enabled:      true
     horizon:       63       # window length (snapshots; 3 weeks at 8 h)
     overlap:       27       # overlap (snapshots; 9 days)
     rh_year:      2024
     network_path: 'outputs/single_analysis/tut2_brownfield_heat/networks/B_H_RE_H2_METH_SN_ST_CO2_100_tD_H2_580_MeOH_0_CH4_262_2024_El_0.3_DET_8h_tut2_brownfield_heat_OPT.nc'

.. admonition:: Committable units are supported in rolling horizon
   :class: tip

   Unlike stochastic mode, rolling horizon **can** use unit commitment. Each
   window is solved as its own small MILP, so ``committable: true`` on a
   fixed-capacity asset is valid here. The ``min load`` / ``committable`` note in
   ``n_config.default.yaml`` reads *"only for initial capacity or RH"*. This
   tutorial leaves committable at its default ``false``. Switch it on for a fixed
   (``expansion: false``) asset to see on/off cycling in the dispatch.

The output network name gets an ``_RH`` suffix. Plots land in
``outputs/single_analysis/tut3_rh/plots_rh/`` and CSVs in ``csv_rh/``. The
agent-level outputs use the component allocation of the Tutorial 2b run.

---

3 · Interpret the results
-------------------------

The RH plot suite adds two **PF-vs-RH comparison** figures.

.. figure:: /_static/tutorials/tut3_PF_vs_RH_total_cost.png
   :width: 95%

   Total cost by carrier: perfect foresight (PF) vs rolling horizon (RH).
   CAPEX is identical, so the whole difference is OPEX.

.. admonition:: PF vs RH: how much does limited foresight cost?
   :class: important

   - **PF ≈ €69.1 M/y**, **RH ≈ €71.5 M/y**. Rolling horizon costs
     **€2.37 M/y (3.4 %) more** than perfect foresight.
   - Both runs deliver exactly the same 580,000 MWh of hydrogen and 262,000 MWh
     of methane. Only the timing of production changes.
   - Perfect foresight is the lower bound on cost, because it sees the whole
     year at once. A positive gap is what a correct implementation must show.

.. figure:: /_static/tutorials/tut3_PF_vs_RH_opex_delta.png
   :width: 100%

   OPEX difference per carrier (RH minus PF). Red bars cost more under RH.

The extra cost is almost all electricity. RFNBO grid imports rise by €2.72 M/y
and other grid electricity by €0.33 M/y. Biogas upgrading OPEX falls by
€0.60 M/y, which offsets a small part.

.. figure:: /_static/tutorials/tut3_CF_operation_by_scenario.png
   :width: 95%

   Capacity-factor duration curves under rolling horizon. Same capacities as
   Tutorial 2b, re-dispatched window by window.

---

4 · Where the gap comes from: the hydrogen buffer
-------------------------------------------------

Tutorial 2b delivers hydrogen through a free ``H2 delivery store``. The annual
demand is fixed, but the store lets production run ahead of delivery. With
perfect foresight the optimiser uses it as a **seasonal buffer**. It produces
hydrogen when power is cheap and holds 38-76 GWh from February to November.

.. figure:: /_static/tutorials/tut3_H2_buffer_PF_vs_RH.png
   :width: 100%

   Hydrogen held in the delivery store. Perfect foresight builds a seasonal
   stock; rolling horizon never holds more than about 7 GWh.

A three-week window cannot see a cheap season months ahead. Hydrogen left in the
store at the end of a window has no value inside that window, so rolling horizon
produces almost just in time. It must then buy electricity in expensive weeks
that perfect foresight avoided. This is the 3.4 % gap.

Short storage behaves differently. The battery and the DH heat store cycle within
hours or days, well inside one window, so foresight beyond three weeks adds
little for them. The gap grows with the time scale of the plant's flexibility,
not with its size.

To explore further, shorten or lengthen ``horizon`` and watch the gap. A longer
window lets rolling horizon use more of the hydrogen buffer.

---

What you learned
----------------

- The difference between perfect-foresight expansion and **rolling-horizon dispatch**.
- ``horizon`` / ``overlap`` window mechanics and the ``network_path`` input.
- Why a seasonal buffer makes the PF–RH gap large, while short storage barely affects it.

Next: :ref:`tutorial-4-stochastic` optimises the *investment* against several
scenarios at once.

.. seealso::

   :ref:`guide-rolling-horizon` · :ref:`guide-outputs` · :ref:`config-rolling-horizon` · :ref:`tutorial-2b-brownfield-heat`
