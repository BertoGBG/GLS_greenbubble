.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _multi-energy:

Multi-energy representation
===========================

GreenBubble models an industrial cluster as a multi-energy network. Seven
carriers are represented at the same time, and conversion between them is what
the model optimises.

Each carrier has its own buses. A plant that consumes one carrier and produces
another is a single multi-port link between those buses.

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

Heat is modelled as three separate circuits instead of one carrier. A heat
stream can therefore only serve a demand at or below its own temperature. This
adds buses and constraints, but it prevents the model from using a reactor's
low-grade waste heat to meet a high-temperature steam demand, which would
overstate how much heat integration the site can achieve.
