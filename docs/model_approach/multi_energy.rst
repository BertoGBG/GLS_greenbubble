.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _multi-energy:

Multi-energy representation
===========================

GreenBubble represents an industrial cluster as a multi-energy network: several
carriers modelled at once, with conversion between them as the object of study
rather than an afterthought.

Each carrier has its own buses, and a plant that consumes one and produces
another is a single multi-port link across them. The carriers are:

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Carrier
     - Unit
     - Role
   * - Electricity
     - MW
     - On-site renewables, grid import and export, and every electrical load.
   * - Hydrogen
     - MW (LHV)
     - Produced by electrolysis, consumed by methanol and methanation, or sold.
   * - CO₂
     - t/h
     - Separated by biogas upgrading; a feedstock, not only an emission.
   * - Biogas and methane
     - MW (LHV)
     - Raw biogas, upgraded biomethane, and synthetic methane, tracked apart.
   * - Methanol
     - MW (LHV)
     - The product of two competing routes.
   * - Heat
     - MW
     - Three temperature circuits, not one: MT, DH and LT.
   * - Biomass
     - MW / t
     - Wet biomass, pellets and digestate through the drying and pyrolysis chain.

Modelling heat as three circuits rather than one carrier is the choice that costs
the most and buys the most. It means a reactor's waste heat can only serve a
demand at or below its own temperature, which is the difference between a
plausible integration figure and an optimistic one.
