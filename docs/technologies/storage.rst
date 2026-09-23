.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _tech-storage:

Storage
=======

Agent: ``n_flags.storage``. Buffers electricity, hydrogen, CO₂ and heat on the shared buses. Every store is optional and sized by the optimiser.

All storage components are activated together by the ``storage`` flag.
Individual technologies can be disabled by setting ``expansion: false`` and
``initial capacity: 0`` in ``n_config.default.yaml``.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Component
     - Type
     - Notes
   * - Battery (Li-ion)
     - StorageUnit
     - Includes inverter losses; default 2-hour duration
   * - H₂ HP storage
     - Store
     - High-pressure steel vessel; extendable
   * - CO₂ HP storage
     - Store
     - Pressurised cylinders; extendable
   * - CO₂ Liq storage
     - Store
     - Liquefaction + insulated tank; e_nom and liquefaction capacity
       optimised separately
   * - TES concrete
     - StorageUnit
     - Concrete thermal store charged from ``Heat MT``; 10-hour duration;
       standing losses
   * - TES concrete El
     - Store + 2 Links
     - Concrete store charged **electrically**, discharging as heat — see below
   * - TES district heating
     - StorageUnit
     - Hot-water buffer for DH; 50-hour duration

Electrically charged thermal storage (``TES concrete El``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This one is not a heat buffer but a **power-to-heat store**: electricity heats a
solid medium, and the heat is returned later. Concrete is the medium costed here,
but the idea is generic — crushed rock, ceramic and refractory brick designs all
do the same thing, and several can return heat at 250–350 °C. It is the way to
move cheap electricity into a high-temperature process heat demand at a different
hour.

Unlike the other stores it is built from three components rather than a
``StorageUnit``, because charge and discharge are priced separately:

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - Component
     - Path
     - technology-data row
   * - ``TES concrete El charger``
     - ``El3`` → store
     - ``Concrete-charger``
   * - ``TES concrete El storage``
     - the store itself
     - ``Concrete-store``
   * - ``TES concrete El discharger``
     - store → ``Heat MT``
     - ``Concrete-discharger``

Costs come from ``technology-data`` (Viswanathan 2022 for the store and the
efficiencies, Georgiou 2018 for the split of power-equipment cost between charge
and discharge). Duration limits are set in ``n_config`` by
``min_max_hours_charge`` (4) and ``min_max_hours_discharge`` (15), with a 2 %
standing loss.

.. warning::

   **The discharge efficiency is wrong for a heat discharge.** With no override
   in ``n_config``, the discharger takes ``Concrete-discharger``'s efficiency of
   **0.4343**, giving a round trip of 0.99 × 0.4343 ≈ **0.43** from electricity
   to ``Heat MT``. That figure is Viswanathan's **electrical** discharge — its
   own note reads "RTE assume 99% for charge and other for discharge", i.e. a
   *power-to-power* round trip through a steam cycle. Returning the heat *as
   heat*, through a heat exchanger, should be close to unity, so the model
   currently discards about 57 % of the stored energy for no physical reason.

   Nothing is affected today: the technology ships with ``initial capacity: 0``
   and ``expansion: false``, so it is never built. But it must be corrected
   before enabling it — set ``efficiency discharge`` for ``TES concrete El`` in
   ``n_config``, which ``_eff`` will use in preference to the catalogue value.

.. note::

   A second limit to keep in mind: the discharger feeds ``Heat MT``, which
   ``p_config`` declares as a 140–180 °C circuit. A store able to deliver
   250–350 °C therefore cannot show that advantage in this model — there is no
   hotter circuit for it to serve.
