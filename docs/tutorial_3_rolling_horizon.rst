.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _tutorial-3-rolling-horizon:

Tutorial 3 — Rolling Horizon Dispatch
=====================================

Capacity expansion decides *what to build*; **rolling horizon** decides *how to
operate* a fixed plant through the year, one sliding window at a time — closer to
how a plant is actually run, and without perfect foresight over all 8760 hours.

This tutorial takes the **solved brownfield network from** :ref:`tutorial-2-brownfield`
and re-solves only its **dispatch**.

.. contents:: On this page
   :local:
   :depth: 1

---

1 · How rolling horizon works
-----------------------------

The full year is solved in overlapping windows (here **168 h** windows with a
**72 h** overlap). Each window is optimised with the end-of-previous-window state
as its start; the overlap is discarded to avoid end-of-horizon artefacts.
Capacity expansion is **bypassed entirely** — capacities are read from the
``network_path`` network and held fixed.

.. important::

   Rolling horizon is **dispatch-only** and runs at full **hourly** resolution
   (temporal clustering is ignored). It needs a previously solved, fixed-capacity
   ``*_OPT.nc`` — so **run Tutorial 2 first**.

---

2 · Run it
----------

.. code-block:: bash

   # 1) ensure Tutorial 2 has produced its *_OPT.nc
   # 2) point network_path at it (already pre-filled; edit if your path differs)
   cp tutorials/3_rolling_horizon/config.yaml   config/config.yaml
   cp tutorials/3_rolling_horizon/n_config.yaml config/n_config.yaml
   snakemake --cores 4

.. code-block:: yaml

   rolling_horizon:
     enabled:      true
     horizon:      168       # window length (h)
     overlap:       72       # overlap (h)
     rh_year:      2024
     network_path: 'outputs/single_analysis/<brownfield>/networks/<brownfield>_OPT.nc'

.. admonition:: Committable units are supported in rolling horizon
   :class: tip

   Unlike stochastic mode, rolling horizon **can** use unit commitment: each
   window is solved as its own (small) MILP, so ``committable: true`` on a
   fixed-capacity asset is valid here (the ``min load`` / ``committable`` note in
   ``n_config.default.yaml`` reads *"only for initial capacity or RH"*). This
   tutorial leaves committable at its default ``false``; you can switch it on for
   a fixed (``expansion: false``) asset to see on/off cycling in the dispatch.

The output network name gets an ``_RH`` suffix, and plots land in
``outputs/single_analysis/<…_RH>/plots_rh/``.

---

3 · Interpret the results
-------------------------

.. admonition:: Interpretation [REVIEW]
   :class: caution

   *Draft — verify before publishing.*

   - Compare the **rolling-horizon dispatch** against the perfect-foresight
     dispatch from Tutorial 2 (the RH plot suite includes a PF-vs-RH comparison).
     RH cost is typically **slightly higher** because it cannot see beyond the
     window — quantify the gap from the comparison CSV.
   - Storage behaviour is the clearest difference: with a 168 h window the model
     can shift energy within a week but not across seasons — look at the storage
     state-of-charge trace.
   - Verify production targets/prices are still met window-by-window.
     *(Fill exact PF-vs-RH cost delta from the run.)*

---

What you learned
----------------

- The difference between perfect-foresight expansion and **rolling-horizon dispatch**.
- ``horizon`` / ``overlap`` window mechanics and the ``network_path`` input.
- Why committable is dropped for RH in this setup.

Next: :ref:`tutorial-4-stochastic` optimises the *investment* against several
scenarios at once.

.. seealso::

   :ref:`guide-rolling-horizon` · :ref:`guide-outputs` · :ref:`config-rolling-horizon` · :ref:`tutorial-2-brownfield`
