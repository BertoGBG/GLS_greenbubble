.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _tech-electrolysis:

Electrolysis
============

Agent: ``n_flags.electrolysis``. Draws electricity from the shared bus and the external market, produces into the shared hydrogen header, and rejects heat to the heat circuits. Hydrogen may be sold directly or consumed by ``meoh`` and ``methanation``.

**Three** electrolyser technologies are modelled, and they compete: all three are
offered to the optimiser whenever the ``electrolysis`` flag is on, and it sizes
each one (possibly to zero).  Each is a ``Link`` from the shared local
``El_electrolysis`` bus to ``H2 collection``.

.. list-table::
   :header-rows: 1
   :widths: 12 32 28 28

   * - Tech
     - Full name
     - Heat
     - Electrical efficiency
   * - ``AEC``
     - Alkaline electrolysis
     - **rejects** to Heat LT
     - 0.4962 MWh_H2 / MWh_el
   * - ``PEMEC``
     - Proton-exchange-membrane electrolysis
     - **rejects** to Heat LT
     - 0.4645
   * - ``SOEC``
     - Solid-oxide (high-temperature steam) electrolysis
     - **consumes** from Heat MT
     - 0.6651

``SOEC`` is the structurally different one.  Being a high-temperature process it
*imports* heat — its ``efficiency2`` is negative, drawing from ``Heat MT`` — and in
exchange converts electricity to hydrogen far more efficiently than the two
low-temperature routes.  Whether that trade is worth making depends on what else on
the site wants MT heat, which is exactly the kind of question the symbiosis network
exists to answer.  ``AEC`` and ``PEMEC`` instead *reject* waste heat to ``Heat LT``.

All three share ramp limits of 0.9 /h, a 15 % minimum load, and
``committable: false`` by default (commitment is intended for brownfield or
rolling-horizon dispatch).  Efficiency is fixed — there is no part-load curve, and
an improved part-load model is still planned.

Costs are **size-dependent**.  ``prepare_network.py`` appends a size suffix, chosen
from whether an external H₂ demand exists, and looks the result up in
technology-data:

.. code-block:: text

   tech_name = f"{t}{size_suffix}"        # e.g. "AEC large" / "AEC small"

   AEC large     691,534 EUR/MW      AEC small    1,100,168 EUR/MW
   PEMEC large   817,268 EUR/MW      PEMEC small  1,194,468 EUR/MW
   SOEC large  1,210,477 EUR/MW      SOEC small   1,952,383 EUR/MW

(2030 values, lifetime 25 years.)  The component itself is always named ``AEC`` /
``PEMEC`` / ``SOEC``; the suffix exists only for the cost lookup, and
``_build_comp_tech_map`` re-applies it when mapping components back to their
technology.
