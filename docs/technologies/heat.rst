.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _tech-heat:

Heat system
===========

Agent: ``n_flags.central_heat``. Supplies the three heat circuits from several competing units, buying gas and biomass externally and optionally selling to district heating.

Three heat buses represent temperature levels in the cluster:

- **Heat MT** (medium temperature) — process waste heat, biomass boilers,
  biogas engine CHP output
- **Heat DH** (district heating) — optional connection to an external DH
  network (price and load set in ``options.DH``)
- **Heat LT** (low temperature) — low-grade cooling and heat pump source

A ``heat pump`` link can upgrade LT to DH heat (extendable, disabled by
default).  ``El boiler`` and ``NG boiler`` provide backup heat.
