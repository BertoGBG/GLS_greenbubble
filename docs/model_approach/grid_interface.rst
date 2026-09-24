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

The site connects to the external DK1 grid in two directions. Import uses one
``DK1_to_El_{agent}`` link per on-site agent, such as
``DK1_to_El_electrolysis``. Export uses a single link, ``El3_to_DK1``, priced at
spot prices.

Physically both directions share one connection asset. The model therefore prices
one shared capacity instead of paying for import and export capacity separately.

**One link carries the capital cost.** Every agent's import branch draws from a
single bus, ``ElDK1 buy bus``, fed by one extendable link,
``DK1_to_ElDK1_buy`` (built in ``add_local_el_connections``,
``scripts/prepare_network.py``). That link alone carries the grid-connection
capital cost. Each branch from ``ElDK1 buy bus`` to an agent's local bus has
``capital_cost = 0`` and keeps its own ``marginal_cost``, which is the purchase
price. Only capital cost moved, so per-agent operating-cost reporting is
unaffected.

**The two capacities are tied.** When both the shared import link and
``El3_to_DK1`` exist for a given ``n_flags`` configuration, ``build_network()``
zeros the export link's capital cost and sets
``n.meta["consolidated_grid_connection"] = True``. A custom constraint,
``add_grid_connection_shared_capacity_constraint`` (``scripts/helpers.py``), then
forces the two links' ``p_nom`` to be equal. One equality is enough. Each link's
own PyPSA bound already caps its hourly dispatch once the capacities match, so no
per-snapshot constraint is needed, and only one physical capacity is ever paid
for.

**A site with only one direction needs no sharing.** Some configurations import
without exporting, or generate enough on site never to buy. There is then nothing
to share: the remaining link keeps its own capital cost, and the constraint skips
silently. This is the same defensive style as
:func:`add_max_RE_sales_constraint`.

**Reporting splits the cost back.** After the solve,
``reallocate_grid_connection_capex`` (``scripts/helpers.py``, called from
``snakemake_plot.py``) divides the shared link's capex between the individual
import and export links, in proportion to each link's share of flow in the
year's peak-usage hours. See
:ref:`shared grid-connection capex <payback-cost-allocation>` in
:doc:`/economics` for the method. This step is in-memory and for reporting only.
It never touches the optimisation.
