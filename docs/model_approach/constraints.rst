.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _additional-constraints:

Additional constraints
======================

On top of the standard PyPSA constraints listed in :doc:`/design`, GreenBubble
adds four of its own. They are built in ``scripts/helpers.py`` and added to the
Linopy model before solving.

Renewable export cap
--------------------

.. math::

   \sum_{t} w_t \, p_{\text{export},t}
   \;\le\; \texttt{max\_RE\_to\_grid} \cdot \sum_{t} w_t \, p_{\text{RE consumed},t}

**What it prevents.** A wind farm with a chemical plant attached. Without it the
cheapest way to satisfy any product target is to over-build renewables and sell
the surplus, and the model answers a question about electricity trading rather
than about industrial symbiosis.

Note that it binds on **annual energy, not hourly power**: the site may export
freely in any single hour as long as the yearly total stays within the share.
One constraint per scenario, built by ``add_max_RE_sales_constraint``.

Shared grid-connection capacity
-------------------------------

.. math::

   P_{\text{nom},\;\text{import}} \;=\; P_{\text{nom},\;\text{export}}

**What it prevents.** Paying twice for one cable. Import and export are separate
links in the model but one physical connection on site, so their capacities are
tied and only one is given a capital cost.

It is a single equality on the two capacity variables, not one per snapshot —
each link's own PyPSA bound already caps its hourly flow once the capacities are
equal. Built by ``add_grid_connection_shared_capacity_constraint``; the cost
side is :ref:`grid-connection-capex`.

Store power-to-energy ratio
---------------------------

.. math::

   P_{\text{nom},\;\text{link}} \;-\; f \cdot E_{\text{nom},\;\text{store}} \;\le\; 0

**What it prevents.** A store with free power. Left alone, the optimiser would
give a battery a very large charger and a very small store, because the charger
is cheap per MW and the store is what costs money — producing a device that can
absorb enormous power for one minute.

:math:`f` comes from ``min_max_hours`` in ``n_config``, so a value of 0.25
means the charger cannot exceed a quarter of the store's energy capacity, i.e.
at least four hours to fill. Built by ``add_custom_constraints_stores``.

RFNBO compliance
----------------

**What it prevents.** Hydrogen counted as renewable that was made from fossil
electricity. The constraint restricts when grid electricity may feed the
electrolysers — by price threshold, by hourly correlation with on-site
renewables, or not at all — according to ``rfnbos_dict.limit``.

This one changes the answer more than any other constraint in the list, and it
is the one most worth stating in a results table. The variants and their
formulations are in :ref:`methods-rfnbo`.
