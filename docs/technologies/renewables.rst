.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _tech-renewables:

Renewable electricity
=====================

Agent: ``n_flags.renewables``. Connects to the shared electricity bus and to the external electricity market. Builds only when ``symbiosis`` is on and at least one on-site consumer is active; see :ref:`agents`.

.. figure:: /_static/model/tech_renewables.svg
   :width: 100%
   :alt: renewables links and bus connections

   Links of the ``renewables`` agent and the buses they connect. Each port shows the bus number and the efficiency per unit of ``bus0``; negative
   values are inputs. Notes on a bus list its connectors, compressors, stores and generators.
   Drawn from the network with every technology enabled.


**Onshore wind** and **solar PV** are modelled as ``Generator`` components
with capacity-factor time series retrieved from
`Renewables.ninja <https://www.renewables.ninja>`_ for the configured site
coordinates.  Both are extendable by default (greenfield) with costs from the
technology-data database.

Up to ``max_RE_to_grid`` fraction of total renewable output can be exported
to the electricity grid; the remainder must be consumed internally.
