.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _tutorial-3b-rolling-horizon-bins:

Tutorial 3b — Rolling Horizon with Delivery Bins
================================================

:ref:`tutorial-3-rolling-horizon` re-dispatches a plant with a flat hydrogen
demand. Every window sees its own demand there. This tutorial delivers the
hydrogen in **two-weekly bins** instead. Most windows then contain no delivery,
and the rolling horizon relies on the stock targets described in
:ref:`rh-deliveries`.

The run also shows the one risk those targets carry: they assume the plant can
always run at its rating, and in a calm winter month it cannot.

.. contents:: On this page
   :local:
   :depth: 1

---

1 · The plant and the demand
----------------------------

A greenfield plant with renewables, biogas, electrolysis, storage and
symbiosis. Heat, methanol and methanation are off.

- **Hydrogen:** 200 GWh/y in 26 two-weekly ``bins_profile`` bins. The bins
  follow the Danish gas demand profile, so the 3 December bin (14,745 MWh) is
  more than three times the July ones.
- **Methane:** 350 GWh/y, flat.
- 2024 at 8 h resolution, HiGHS.

The symbols :math:`d_k`, :math:`t_k`, :math:`e(t)` and :math:`E` are those of
:ref:`demand-formulation`.

---

2 · Step 1: perfect foresight
-----------------------------

.. code-block:: bash

   cp tutorials/3b_rh_bins_pf/config.yaml   config/config.yaml
   cp tutorials/3b_rh_bins_pf/n_config.yaml config/n_config.yaml
   snakemake --cores 4

The optimiser builds 83.5 MW of alkaline electrolysis, 160 MW of onshore wind
and a 54.4 MW grid connection. The delivery store fills to its limit
:math:`E = 1.5 \cdot \max_k d_k = 22{,}117` MWh. It is free, so the plant makes
hydrogen when the wind blows and holds it for the winter bins.

The grid connection is smaller than the electrolyser's 83.5 MW. Running at
full load therefore needs wind on top of the grid. With perfect foresight this
costs nothing: the plant simply produces ahead of the calm weeks.

---

3 · Step 2: rolling horizon with stock targets
----------------------------------------------

.. code-block:: bash

   cp tutorials/3b_rh_bins/config.yaml   config/config.yaml
   cp tutorials/3b_rh_bins/n_config.yaml config/n_config.yaml
   snakemake --cores 4

Windows of 30 days (90 snapshots) with 7 days of overlap. Each window sees at
most two deliveries, and the target at its end tells it what the later bins
need.

.. code-block:: yaml

   rolling_horizon:
     horizon:  90              # 30 days at 8 h
     overlap:  21              # 7 days
     delivery_target: pro_rata

The log reports the targets and any shortfall:

.. code-block:: text

   [rolling_horizon] delivery targets 'H2 delivery store' (pro_rata): 200,000 MWh in 26
   deliveries, largest 14,745 MWh, inflow capacity 41.44 MW, ... target set in 14/16 windows
   [rolling_horizon]   WARNING: 662.8 MWh of 'H2 delivery' deliveries unmet

.. figure:: /_static/tutorials/tut3b_H2_store.png
   :width: 100%
   :alt: Hydrogen delivery store over the year, perfect foresight and three rolling-horizon runs

   Stock in the hydrogen delivery store. Dots are the targets at the window
   ends. Perfect foresight stocks up through the autumn; 30-day windows hold
   little more than the next bin needs.

---

4 · The risk: the targets assume the rating
-------------------------------------------

663 MWh, 0.3 % of the year, came from the ``unmet delivery`` generator, all of
it between 19 November and 3 December.

The targets are computed before the first window, from the deliveries and the
inflow capacity :math:`C` = 41.4 MW. The rolling horizon may not know the wind,
so it cannot do better. It assumes the electrolyser can make 41.4 MW of hydrogen
in every hour. That needs 83.5 MW of electricity, and the grid supplies only
54.4 MW of it.

.. figure:: /_static/tutorials/tut3b_electrolysis_late_year.png
   :width: 100%
   :alt: Hydrogen production in November and December against the rating and against what grid plus wind allowed

   Hydrogen made in the 30-day run. The dashed line is the most the grid
   connection and the available wind allowed. In the shaded weeks before the
   3 December delivery the plant ran at that limit, below :math:`C`, and still
   fell short.

The window that holds the 3 December delivery had to make that bin and also
build stock for the two late-December bins. Its target assumed 41.4 MW in every
hour. With less wind it could not reach it, and the window before had no reason
to produce more than its own target.

.. warning::

   The stock targets guarantee every delivery only if the plant can run at its
   rating whenever needed. When its supply depends on the weather, for instance
   an electrolyser larger than its grid connection, a calm spell before a large
   bin can leave a shortfall. The rolling horizon reports it and charges
   ``unmet_cost`` for it. Read the log for ``deliveries unmet``.

---

5 · Compare targets and windows
-------------------------------

Change ``delivery_target`` and ``horizon`` / ``overlap`` in
``config/config.yaml`` and rerun step 2. Give each run its own ``run_name``.

.. list-table::
   :header-rows: 1
   :widths: 22 20 18 20 20

   * - Windows
     - ``delivery_target``
     - Hydrogen unmet
     - OPEX vs PF
     - OPEX vs PF, without the unmet penalty
   * - 30 d / 7 d
     - ``none``
     - 1,145 MWh
     - +21.3 %
     - +3.7 %
   * - 30 d / 7 d
     - ``e_min``
     - 663 MWh
     - +14.1 %
     - +3.9 %
   * - 30 d / 7 d
     - ``pro_rata``
     - 663 MWh
     - +13.5 %
     - +3.3 %
   * - 60 d / 14 d
     - ``pro_rata``
     - 638 MWh
     - +12.5 %
     - +2.7 %
   * - 90 d / 21 d
     - ``pro_rata``
     - 0
     - +1.7 %
     - +1.7 %

- **Targets cut the shortfall by 40 %.** Without them a window does not prepare for
  bins beyond its horizon.
- **The rest is the risk of section 4.** Both target rules leave the same
  663 MWh, because both assume the rating.
- **Longer windows remove it.** A 90-day window sees the December bins from
  September. It produces ahead through the windy autumn, as perfect foresight
  does, and meets every delivery.

Redraw the figures for your own runs with:

.. code-block:: bash

   python tutorials/3b_rh_bins/plot_rh_deliveries.py \
       "tut3b_rh=30-day windows, pro_rata" "tut3b_rh_none=30-day windows, no target"

---

What you learned
----------------

- Binned deliveries leave most rolling-horizon windows without demand; stock
  targets at the window ends give them one.
- The targets use the plant rating, because the rolling horizon must not know
  the weather. A weather-limited plant can still fall short before a large bin.
- Reading the ``deliveries unmet`` warning, and removing the shortfall with a
  longer window.

.. seealso::

   :ref:`rh-deliveries` · :ref:`demand-formulation` · :ref:`tutorial-3-rolling-horizon` · :ref:`config-rolling-horizon`
