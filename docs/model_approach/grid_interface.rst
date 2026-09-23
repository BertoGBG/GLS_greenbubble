.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _grid-connection-capex:

Modelling the electrical grid interface
=======================================

.. figure:: /_static/model/grid_interface.svg
   :width: 100%
   :alt: One capital-costed import link feeding a shared bus, per-agent branches with no capex, and the export link tied to the same capacity

   One cable, paid for once. The thick link is the only one carrying grid
   connection capital cost; the per-agent branches carry none, and the export
   link's capacity is tied to the import link's by an equality constraint.

The site connects to the external DK1 grid in two directions: import (one
``DK1_to_El_{agent}`` link per on-site agent, e.g. ``DK1_to_El_electrolysis``)
and export (``El3_to_DK1``, priced at real spot prices). Physically these
share one connection asset, so the model prices it as one shared capacity
rather than paying for import and export capacity separately:

- Every agent's import branch draws from a single bus, ``"ElDK1 buy bus"``,
  fed by one extendable link, ``"DK1_to_ElDK1_buy"`` (built in
  ``add_local_el_connections``, ``scripts/prepare_network.py``). This link
  alone carries the "electricity grid connection" capital cost; every branch
  from ``"ElDK1 buy bus"`` to an agent's local bus has ``capital_cost = 0``
  and keeps its own ``marginal_cost`` (purchase price), so per-agent
  operating-cost reporting is unaffected — only capital cost moved.
- When both the shared import link and ``El3_to_DK1`` exist for a given
  ``n_flags`` configuration, ``build_network()`` zeros the export link's own
  capital cost and flags the network (``n.meta["consolidated_grid_connection"]
  = True``) so a custom constraint,
  ``add_grid_connection_shared_capacity_constraint`` (``scripts/helpers.py``),
  ties the two links' ``p_nom`` together — one equality constraint, not one
  per snapshot, since each link's own native PyPSA bound already caps its
  hourly dispatch once the two ``p_nom`` variables are equal. Only one
  physical capacity is ever paid for.
- If a configuration has only one side (e.g. a pure-import site with no
  export capability, or a fully self-sufficient renewables site that never
  buys from the grid), there is nothing to share — that link's own capital
  cost is left untouched, and the constraint silently skips (same defensive,
  never-raise style as :func:`add_max_RE_sales_constraint`).
- For reporting, ``reallocate_grid_connection_capex`` (``scripts/helpers.py``,
  called from ``snakemake_plot.py`` after the network is solved) splits the
  shared link's total capex back onto the individual import/export links, in
  proportion to each link's share of flow at the year's peak-usage hour(s) —
  see :ref:`shared grid-connection capex <payback-cost-allocation>` in :doc:`/economics` for the method. This
  is a reporting-only, in-memory step; it never touches the optimisation.
