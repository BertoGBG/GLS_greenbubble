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

Electricity exported to the grid is limited to a share of the renewable
electricity consumed on site. ``max_RE_to_grid`` sets the share.

Without this limit, the cheapest way to meet a product target is often to build
more wind and solar than the site needs and sell the surplus. The model would
then describe an electricity trading business rather than an industrial cluster.

The limit applies to annual energy, not to hourly power. The site may export any
amount in a given hour, provided the yearly total stays within the share. The
constraint is built by ``add_max_RE_sales_constraint``, once per scenario.

Shared grid-connection capacity
-------------------------------

.. math::

   P_{\text{nom},\;\text{import}} \;=\; P_{\text{nom},\;\text{export}}

Import and export are two links in the model but one cable on site. Their
capacities are therefore forced equal, and only one of them carries a capital
cost. Without this, the site would pay for two connections where it needs one.

This is a single equality between the two capacity variables. No constraint per
snapshot is needed, because each link's own PyPSA bound already limits its
hourly flow once the capacities match. The constraint is built by
``add_grid_connection_shared_capacity_constraint``. For how the cost is
assigned, see :ref:`grid-connection-capex`.

Store power-to-energy ratio
---------------------------

.. math::

   P_{\text{nom},\;\text{link}} \;-\; f \cdot E_{\text{nom},\;\text{store}} \;\le\; 0

A charger's power is limited by the size of the store it fills. Without this
limit, the optimiser builds a large charger and a small store, because power
capacity is cheap per MW and energy capacity is not. The result is a store that
can absorb a great deal of power for a few minutes, which is not a device anyone
would build.

:math:`f` is derived from ``min_max_hours`` in ``n_config``. A value of 0.25
allows a charger of at most a quarter of the store's energy capacity, so the
store takes at least four hours to fill. The constraint is built by
``add_custom_constraints_stores``.

RFNBO compliance
----------------

Grid electricity may only feed the electrolysers under conditions set by
``rfnbos_dict.limit``: below a price threshold, in hours correlated with on-site
renewable generation, or without restriction. The purpose is to prevent hydrogen
made from fossil electricity from counting as renewable.

This constraint has a larger effect on results than the other three, so the
setting used should be reported alongside them. The variants and their
formulations are described in :ref:`methods-rfnbo`.
