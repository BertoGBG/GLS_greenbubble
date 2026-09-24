.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _agents:

Agents
======

The model is organised into **agents**. An agent is a broad category of plant,
defined by the function it performs rather than by a specific technology. Each
agent is switched on or off by its entry in the ``n_flags`` block of
``config.yaml``, and each corresponds to one builder function in
``scripts/prepare_network.py``.

An agent is self-standing. It owns its plant-local buses and components, and it
must leave the network feasible whether or not any other agent was built.

.. figure:: /_static/model/agents_technologies.svg
   :width: 100%
   :alt: The seven agents and the competing technologies inside each

   Each agent holds several technologies that perform the same function. The
   optimiser may build any subset of them, including none.

**Technologies compete inside an agent.** The ``electrolysis`` agent does not
represent one electrolyser. It offers alkaline, PEM and solid oxide, which differ
in capital cost, efficiency and outlet pressure but all produce into the same
hydrogen header. The optimiser chooses between them, and may build more than one.

The same applies elsewhere. Two chemistries feed the methanol collection bus, and
four boilers and a heat pump compete to supply heat in each hour.

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

The ``symbiosis`` flag builds the shared distribution buses for electricity,
hydrogen, CO₂, biogas and the three heat circuits. Without it, each agent can
still trade with the external interfaces, but not with its neighbours.

.. figure:: /_static/model/symbiosis_on_off.svg
   :width: 100%
   :alt: The same agents with symbiosis on, sharing carrier buses, and off, standing alone

   With ``symbiosis`` the plants form one hub. Without it they operate as
   separate projects on the same site.

Two agents cannot be built without it. ``meoh`` needs hydrogen from the
electrolyser and CO₂ from the upgrader, and ``methanation`` needs the same. Both
check for ``electrolysis``, ``biogas`` and ``symbiosis`` before building
anything, and return an empty set of components if any of the three is missing.

``renewables`` is gated differently. It requires ``symbiosis`` and at least one
on-site consumer, so that renewable capacity cannot be built purely to export.

**Any combination of flags is a valid run.** The dependency rules resolve first,
and a blocked agent contributes nothing. Infeasibility comes from the targets and
constraints, not from the combination of flags. The rules are implemented in
``network_dependencies()``.
