.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _tech-methanation:

Methanation
===========

Agent: ``n_flags.methanation``. Consumes hydrogen, CO₂ and biogas from the shared buses and sells methane to the gas grid. Requires ``electrolysis``, ``biogas`` and ``symbiosis``.

.. figure:: /_static/model/tech_methanation.svg
   :width: 100%
   :alt: methanation links and bus connections

   The four methanation routes and their bus connections. All four compete for ``eCH4 collection``. Each port shows the bus number and the efficiency per unit of ``bus0``; negative
   values are inputs. Notes on a bus list its connectors, compressors, stores and generators.
   Drawn from the network with every technology enabled.


Two technology pathways are available.  Both convert H₂ + CO₂ (or H₂ +
biogas) into synthetic methane:

**Catalytic methanation** (``methanation CO2``, ``methanation biogas``)
  — Sabatier reaction.  Strict ramp limits (8 %/h) and minimum load (40 %)
  reflect the thermal inertia of the catalyst bed.

**Biomethanation** (``biomethanation CO2``, ``biomethanation``)
  — Biological hydrogenotrophic process (trickle-bed reactor).
  Faster ramp response than catalytic (ramp limit 100 %/h); no minimum load
  constraint in the default configuration.

Both are modelled as ``Link`` components.  The feed can be CO₂ (from the
CO₂ distribution bus) or raw biogas (from the biogas bus).
