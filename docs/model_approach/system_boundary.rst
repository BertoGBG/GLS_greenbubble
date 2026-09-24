.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _system-boundary:

The system boundary
===================

The model prices everything inside one boundary. Outside it are the markets and
sinks that the site trades with.

.. figure:: /_static/model/system_boundary.svg
   :width: 100%
   :alt: The GreenBubble system boundary, with external interfaces crossing a dashed green line

   The bubble and its interfaces. Only what is inside carries capital cost.

The site meets the outside world at these interfaces:

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Interface
     - Direction
     - What it represents
   * - Electricity
     - buy and sell
     - The DK1 spot market, at an hourly price series, with tariffs applied on
       top and an export cap.
   * - Natural gas
     - buy and sell
     - The gas grid: a purchase price for the boilers, and a sale route for
       biomethane and e-methane at their own premiums.
   * - Hydrogen
     - sell
     - Delivery to an off-taker, as an annual target or at a price.
   * - Methanol
     - sell
     - As above, and the destination of both methanol routes.
   * - District heating
     - sell
     - Surplus heat off-take, disabled by default (``options.DH``).
   * - CO₂ liquid
     - sell *(optional)*
     - Liquefied CO₂ leaving for sequestration. When
       ``options['CO2 Liq credits']`` is enabled it earns the **CO₂ tax value**
       per tonne sequestered. The sequestered share is
       ``options['CO2 Liq credits'].efficiency`` (0.95 by default). Off by
       default.
   * - Biochar
     - sell *(optional)*
     - Carbon leaving as a solid. When ``options['biochar credits']`` is enabled
       it earns the **CO₂ tax value** per tonne sequestered. Off by default —
       see the note below on what "per tonne sequestered" means here.
   * - Ambient heat
     - sink
     - Where heat too cold to be useful goes. Unpriced, unlimited.
   * - Biomass markets
     - buy
     - Pellets, wood chips and digestible biomass.

.. note::

   **Both sequestration routes are paid at the CO₂ tax**, not at a separate
   price: the credit is the value of the emission avoided, so it moves with
   ``CO2_cost``. Both are **options**, off by default, and switching either on
   can change which technologies are worth building.

   The two are de-rated differently, and the difference is easy to misread.

   For **liquid CO₂** the de-rating is in the model. The sequestration link has
   the efficiency set in ``options['CO2 Liq credits'].efficiency`` (0.95 by
   default, an assumption that includes boil-off). The credit is paid per tonne
   of liquid CO₂ leaving the site, multiplied by this efficiency. With the
   default, 5 % of the liquefied stream earns nothing.

   For **biochar** the de-rating is already in the data. DEA's slow-pyrolysis
   sheet notes that *"only 70 % of carbon in biochar is assumed to be sequestered
   when spread on soil"*, and expresses every figure **per tonne of CO₂
   sequestered** rather than per tonne of biochar — ``biomass-input`` is
   7.6748 MWh_biomass/t_CO₂ on that basis. So the model pays the full CO₂ price
   on a quantity that has already had the 70 % applied to it. There is no 0.7
   factor anywhere in GreenBubble, and adding one would count it twice.

**Interfaces carry no capital cost.** The model does not charge for the existence
of a grid connection to the market, or of a pipeline to an off-taker. Only the
price of the carrier crossing the boundary enters the objective: electricity
bought, methanol sold, gas purchased.

This is a deliberate accounting choice. It keeps the objective equal to the cost
of the site itself, so any change in the objective corresponds to something the
project could build or operate.

There is one exception, and it is internal. The on-site electrical connection has
a real capacity that the site must size and pay for. See
:ref:`grid-connection-capex`.
