.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _design:

Design
======

GreenBubble is built on PyPSA, and inherits its data model unchanged. This page
records which parts of that model the project uses and how, so that the rest of
the documentation can talk about buses, links and snapshots without redefining
them.

It follows the structure of PyPSA's own design page, which is the authoritative
reference and worth reading first:

   `PyPSA — Design <https://docs.pypsa.org/v1.0.2/user-guide/design/>`_
   (PyPSA v1.0.2 user guide). Component attributes are documented under
   `Components <https://docs.pypsa.org/v1.0.2/user-guide/components/>`_.

Nothing below replaces those pages. GreenBubble departs from stock PyPSA in three
ways: extra constraints, a process-state layer, and the agent structure. All three
are described in :doc:`model_approach`.

----

Network
-------

A single ``pypsa.Network`` object holds the components, their static attributes
and the time series attached to them. GreenBubble builds one network per run in
``build_network()`` (``scripts/prepare_network.py``) and passes it to Linopy to
solve.

The network is also the unit of everything downstream. It is written to disk as
NetCDF, once before the solve (``*_PRE.nc``) and once after (``*_OPT.nc``). Every
plot, CSV export and economic figure is derived from that file rather than
recomputed.

Buses
-----

Buses are the fundamental nodes of the network. Every component attaches to one
or more buses, and energy balances at every bus in every snapshot. Each bus
carries one energy carrier. By project convention it also carries one
thermodynamic state, described in :doc:`guide_process_streams`.

Buses are not geographic in GreenBubble. The model represents a single site, so a
bus is a header or a point in the process rather than a location. For example,
``H2 distribution`` is the shared 30 bar hydrogen header, and
``CO2 to methanolisation`` is that plant's own compressed inlet.

Components
----------

GreenBubble uses six of PyPSA's component types. Full attribute lists are in the
`PyPSA components documentation <https://docs.pypsa.org/v1.0.2/user-guide/components/>`_;
this table records only what each one is used *for* in this project.

.. list-table::
   :header-rows: 1
   :widths: 16 24 60

   * - Component
     - Used for
     - Notes
   * - ``Bus``
     - Carrier headers, plant inlets, storage states
     - One carrier and one state each. Tagged with a ``properties`` field naming
       its stream in ``p_config``.
   * - ``Carrier``
     - Labelling energy carriers
     - Drives plot colours; no physics attached.
   * - ``Generator``
     - Wind, solar, and external purchases
     - Renewables use a ``p_max_pu`` time series from the capacity factors. The
       natural-gas and biomass markets are generators with a price.
   * - ``Load``
     - Fixed exogenous demand
     - Used sparingly: most demand is modelled as a delivery link into a store
       with an annual target, which is what makes price mode possible.
   * - ``Link``
     - Every conversion, transport and compression step
     - The workhorse. See multi-port links below.
   * - ``Store``
     - Energy and mass inventories
     - Hydrogen, CO₂, methanol, biogas, heat, and the product delivery stores.
   * - ``StorageUnit``
     - Battery and the thermal stores
     - Used where a single component with charge/discharge efficiency and a
       power-to-energy ratio is the natural description.

Multi-port links
~~~~~~~~~~~~~~~~

Almost every plant in GreenBubble is a **single link with several ports**. PyPSA
allows a link to connect ``bus0`` to ``bus1`` … ``bus4`` with an ``efficiency``,
``efficiency2`` … ``efficiency4`` on each, and the project uses this to represent
a whole unit operation as one component.

The sign convention matters and is easy to get wrong: ``p0`` is the flow into
``bus0``, and for every other port a **positive efficiency produces** into that
bus while a **negative efficiency consumes** from it. So methanolisation is one
link whose ``bus0`` is hydrogen, which produces methanol and district heat and
consumes CO₂, electricity and medium-temperature heat:

.. code-block:: python

   n.add("Link", "methanolisation",
         bus0=...,  # H2            (p0, the rated flow)
         bus1=...,  # Methanol      efficiency  > 0  produced
         bus2=...,  # CO2           efficiency2 < 0  consumed
         bus3=...,  # electricity   efficiency3 < 0  consumed
         bus4=...,  # Heat MT       efficiency4 < 0  consumed
         bus5=...)  # Heat DH       efficiency5 > 0  produced

This has one consequence that runs through the whole model: **a plant's capacity
is rated on the carrier at** ``bus0``. Methanolisation is sized in MW of hydrogen
input, not in MW of methanol output, and its costs are rebased to that basis.
Per-MW figures from ``technology-data`` therefore often have to be divided by an
input coefficient before they can be used. See :doc:`economics`.

Snapshots
---------

Snapshots are the model's time index: 8 760 hourly steps for a full year by
default, optionally clustered to a coarser resolution (see
:doc:`guide_temporal_resolution`). Time-varying data is attached per snapshot:
prices, capacity factors and demand profiles.

``snapshot_weightings`` holds the number of hours each snapshot represents, so a
clustered run still integrates to a full year. Every energy total and every cost
in the objective is weighted by it. Results read directly from a solved network
must be weighted the same way.

.. note::

   GreenBubble does **not** use PyPSA's investment periods. The model optimises a
   single investment decision against one weather and price year, and multi-year
   questions are handled differently: by solving several years as stochastic
   scenarios (:ref:`methods-stochastic`), or by fixing capacities and re-running
   dispatch on another year (:doc:`guide_rolling_horizon`). Where the PyPSA design
   page discusses ``investment_periods`` and multi-horizon planning, none of it
   applies here.

----

Basic constraints
-----------------

These come from PyPSA and hold in every GreenBubble run. They are stated here
because the rest of the documentation assumes them; the formulations are
PyPSA's, documented under
`Optimisation <https://docs.pypsa.org/v1.0.2/user-guide/optimal-power-flow/>`_.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Constraint
     - Meaning
   * - **Nodal balance**
     - At every bus and every snapshot, everything produced equals everything
       consumed. This is the only constraint the model truly cannot violate, and
       it is what makes the shadow price of a bus meaningful.
   * - **Capacity limits**
     - ``p_nom_min ≤ p_nom ≤ p_nom_max`` for extendable components; ``p_nom``
       fixed otherwise. Dispatch is bounded by ``p_min_pu·p_nom ≤ p ≤
       p_max_pu·p_nom``, with ``p_max_pu`` a time series for renewables.
   * - **Store energy balance**
     - ``e(t) = e(t-1)·(1-standing_loss) + inflow``, with ``e_nom`` bounds.
       Stores are cyclic over the year unless configured otherwise.
   * - **Ramp limits**
     - ``|p(t) − p(t−1)| ≤ ramp_limit · p_nom`` where set in ``n_config``. Used
       on electrolysis, methanation and methanol synthesis.
   * - **Minimum load**
     - ``p(t) ≥ min_load · p_nom`` when a unit is online. With
       ``committable: true`` this becomes a binary on/off decision, which makes
       the problem a MILP. See :ref:`config-committable`.

GreenBubble adds a small number of constraints of its own on top of these. They
are listed in :doc:`model_approach`.

Objective function
------------------

The objective is PyPSA's: minimise total annualised system cost over the year.

.. math::

   \min \; \sum_{i} c_i \, P_{\text{nom},i}
        \;+\; \sum_{t} w_t \sum_{i} o_{i,t} \, p_{i,t}

where :math:`c_i` is the annualised capital cost of component *i*,
:math:`o_{i,t}` its marginal cost, and :math:`w_t` the snapshot weighting.
Capital costs are annualised from the investment cost, lifetime and discount
rate; revenues enter as negative marginal costs on the links that sell a
carrier. The derivation and the cost conventions are in :doc:`economics`.

Only what happens **inside the system boundary** is priced. Interfaces to the
outside world carry no capital cost. Buying and selling across them appears in
the objective only as the price of the carrier that crosses. That boundary is
drawn explicitly in :doc:`model_approach`.

Demand or price
---------------

The same network can be solved to answer two different questions, set by
``targets.driver`` in ``config.yaml``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - ``driver``
     - Question
   * - ``demand``
     - *What is the cheapest way to deliver this much product?* The annual
       ``demand_*`` figures become requirements the model must meet, and the
       objective is pure cost minimisation. The shadow price of the product bus
       is then the marginal cost of supplying it.
   * - ``price``
     - *Given what the products are worth, what is worth building?* The
       ``demand_*`` figures become **sale caps** rather than requirements, the
       ``price_*`` figures set the revenue, and the model may choose to produce
       nothing at all if no route clears its cost.

The distinction is not a switch on the objective so much as on what the annual
targets *mean*, and it catches people out: in price mode a ``demand_*`` of zero
does not free the model, it forbids the sale. Both modes, the demand profile
shapes, and the delivery-store mechanism that implements them are described in
:doc:`guide_demands`.

Solver
------

The LP is built and solved through Linopy, with HiGHS (open source) or Gurobi.
Named option sets live in ``scripts/solver_profiles.py`` and are selected by
``optimization.solver_profile``; the default for local runs is ``highs-fast``.
Committable components make the problem a MILP, which is incompatible with
stochastic mode.
