.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _agents:

Agents
======

The model is organised into **agents**: broad categories of plant, each defined
by a function rather than by a technology. They are switched on and off
individually by the ``n_flags`` block in ``config.yaml``, and each corresponds to
one builder function in ``scripts/prepare_network.py``.

An agent is the tier-1 object of the model. It is self-standing: it owns its
plant-local buses and components, and it must leave the network feasible whether
or not any other agent was built.

.. figure:: /_static/model/agents_technologies.svg
   :width: 100%
   :alt: The seven agents and the competing technologies inside each

   Each agent holds several technologies that perform the same function. The
   optimiser sizes any subset, including none.

**Inside an agent, technologies compete.** This is the point of the structure.
``electrolysis`` is not "an electrolyser" — it offers alkaline, PEM and solid
oxide, with different capital costs, efficiencies, and outlet pressures, all
producing into the same hydrogen header. The optimiser picks. The same is true of
methanol, where two different chemistries feed one collection bus, and of heat,
where four boilers and a heat pump bid against each other hour by hour.

What each agent connects to
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 17 27 24 32

   * - Agent
     - Shared buses used
     - External interfaces
     - Products
   * - ``biogas``
     - biogas, CO₂, electricity, heat
     - biomass markets, gas grid
     - methane
   * - ``renewables``
     - electricity
     - electricity market
     - —
   * - ``electrolysis``
     - electricity, H₂, heat
     - electricity market
     - hydrogen
   * - ``meoh``
     - H₂, CO₂, biogas, electricity, heat
     - —
     - methanol
   * - ``methanation``
     - H₂, CO₂, biogas, heat
     - gas grid
     - methane
   * - ``central_heat``
     - electricity, heat, gas
     - gas grid, biomass markets, DH, biochar
     - —
   * - ``storage``
     - electricity, H₂, CO₂, heat
     - —
     - —
   * - ``symbiosis``
     - *builds them all*
     - —
     - —

The technologies inside each agent are described in :doc:`/technologies`.

.. _symbiosis-network:

Symbiosis is what makes it a hub
--------------------------------

The ``symbiosis`` flag builds the shared distribution buses — electricity,
hydrogen, CO₂, biogas and the three heat circuits. Without it, every agent can
still trade with the external interfaces, but not with its neighbours.

.. figure:: /_static/model/symbiosis_on_off.svg
   :width: 100%
   :alt: The same agents with symbiosis on, sharing carrier buses, and off, standing alone

   With ``symbiosis`` the plants form one hub; without it they are separate
   projects that happen to share a site.

Two agents cannot exist without it at all. ``meoh`` needs hydrogen from the
electrolyser and CO₂ from the upgrader; ``methanation`` needs the same. Both
check for ``electrolysis``, ``biogas`` **and** ``symbiosis`` before building
anything, and return an empty component set if any is missing. ``renewables`` is
gated differently — it needs ``symbiosis`` *and* at least one on-site consumer,
so that a wind farm with no customer cannot be built purely to export.

**Any combination of flags is a valid run.** The dependency rules above resolve
first, and a blocked agent simply contributes nothing; infeasibility, if it
comes, comes from the targets and constraints rather than from the flag
combination itself. The resolution logic is ``network_dependencies()``.
