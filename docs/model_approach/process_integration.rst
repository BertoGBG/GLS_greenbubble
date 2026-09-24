.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _process-integration:

Process integration
===================

GreenBubble represents process integration in more detail than a typical
energy-system model, but in less detail than a process simulator. This section
states where the boundary falls.

**What is represented.** Every stream has a declared physical state: fluid,
temperature and pressure. These are held in ``p_config``, separately from the
techno-economic magnitudes in ``technology-data``. Because the state is declared
rather than assumed, a compressor's duty is computed from its actual inlet and
outlet conditions, and waste heat can only be delivered to a circuit it is hot
enough to serve.

**What is not.** Heat integration is not a heat-exchanger network. There is no
pinch analysis, and individual hot and cold streams are not matched to each other.
Heat is instead carried by three pressurised hot-water circuits, each defined by a
temperature band:

.. list-table::
   :header-rows: 1
   :widths: 22 20 20 38

   * - Circuit
     - T range [°C]
     - P [bar]
     - Typical role
   * - ``Heat MT``
     - 140 – 180
     - 12
     - Process steam duty: reboilers, reactor preheat.
   * - ``Heat DH``
     - 90 – 140
     - 6
     - District heating, and the useful part of compressor aftercooling.
   * - ``Heat LT``
     - 50 – 90
     - 3
     - Low-grade heat, and the floor that cooling duties reject to.

.. figure:: /_static/model/heat_circuits.svg
   :width: 100%
   :alt: Three stacked heat circuits with their temperature floors, and heat degrading downward only

   The tiers are floors, and heat only ever moves down them.

The tiers are **floors**: a stream may be delivered to any circuit whose minimum
temperature it exceeds, so heat degrades downward but never upward. A reactor at
160 °C can serve all three; a 60 °C cooling duty can serve only ``Heat LT``.

Temperatures and pressures of the circuits are set in ``p_config``. **The number
of circuits is fixed at three** — the plant builders name ``Heat MT``,
``Heat DH`` and ``Heat LT`` directly, so adding or removing one is a code change,
not a configuration change.

The consequence to keep in mind when reading heat results: a duty is placed in
one circuit by a single temperature cut, so a stream spanning two bands
contributes wholly to one of them. ``scripts/heat_bands.py`` generalises this to
contiguous bands, but the plant code does not use it yet. See
:doc:`/guide_process_streams`.

.. _physics-based:

Physics-based calculations
--------------------------

Two quantities in the model are not catalogue numbers but calculated
thermodynamics, using `CoolProp <http://www.coolprop.org/>`_ for real fluid
properties.

.. figure:: /_static/model/pressure_ladder.svg
   :width: 100%
   :alt: A logarithmic pressure scale from 1 to 150 bar with each carrier level and the compressors bridging them

   Each arrow is one compressor. Every lift is different, which is why none of
   them is redundant.

**Compressor duty.** Electricity and waste heat are computed stage by stage from
the declared inlet and outlet states: isentropic work from enthalpy and entropy,
a discharge-temperature cap that sets the number of stages, intercooling between
them. Pure fluids are looked up by name; biogas is handled as a real CH₄/CO₂
mixture.

**Cooling duty.** The heat rejected after each stage is integrated at constant
pressure and split by temperature between the ``Heat DH`` and ``Heat LT``
circuits, so a compressor is a component that buys electricity and sells two
grades of heat.

Both are evaluated **before the optimisation runs**, in
``scripts/technology_inputs.py``, and enter the LP as fixed coefficients on the
relevant links. The optimiser therefore sizes and dispatches a compressor whose
efficiency was set by physics rather than by assumption — but it cannot trade off
pressure levels, because those are an input. Changing a pressure in ``p_config``
changes the coefficients and requires a re-solve.

The method, the parameters and where each sits in the model are documented under
:ref:`technologies-compression`.
