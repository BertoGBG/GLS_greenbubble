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
     - Store + 1 Link
     - Concrete store charged from ``Heat MT`` and discharged back to it;
       8 hours; 2 %/h standing loss — see below
   * - TES concrete El
     - Store + 2 Links
     - The same store charged **electrically** instead, discharging as heat;
       4 h charge / 15 h discharge — see below
   * - TES district heating
     - Store + 1 Link
     - Hot-water buffer for ``Heat DH``; 10 hours; 0.2 %/h standing loss

.. note::

   None of the thermal stores is a PyPSA ``StorageUnit``. Each is a ``Store``
   plus one or two ``Link`` s, which is what allows charge and discharge to be
   priced and limited separately. The duration figures are ``min_max_hours`` in
   ``n_config``: the constraint is
   :math:`P_{\text{nom,link}} \le E_{\text{nom,store}} / \text{hours}`, so the
   number is the minimum time to fill or empty the store.

The two concrete stores
-----------------------

``TES concrete`` and ``TES concrete El`` are the **same medium reached two
different ways**, and they are separate ``n_config`` entries that can be enabled
independently.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - ``TES concrete``
     - ``TES concrete El``
   * - Charged from
     - ``Heat MT`` (heat)
     - ``El3`` (electricity)
   * - Discharged to
     - ``Heat MT``
     - ``Heat MT``
   * - Components
     - one bidirectional heat exchanger + store
     - separate electric charger + heat discharger + store
   * - Charge efficiency
     - 1.0
     - 0.99 (``Concrete-charger``)
   * - Discharge efficiency
     - 1.0 (same link)
     - 0.4343 (``Concrete-discharger``) — see the note below
   * - Duration
     - ``min_max_hours`` 8
     - 4 h charge, 15 h discharge

Use ``TES concrete`` to shift *heat* in time — storing a reactor's surplus for a
later steam demand. Use ``TES concrete El`` to turn *cheap electricity* into heat
for later, which is a different economic proposition: it competes with the
``El boiler`` plus a heat store, not with a heat buffer.

Both draw their costs from the same three ``technology-data`` rows
(``Concrete-store``, ``Concrete-charger``, ``Concrete-discharger``; Viswanathan
2022 and Georgiou 2018).

.. warning::

   **``TES concrete``'s heat exchanger is priced as a power block.** Its single
   link takes its capital cost from ``Concrete-discharger``, which is
   725 192 EUR/MW on Georgiou's basis of *"80 % of capital costs of power
   components for sensible thermal storage"* — i.e. the turbine-side equipment
   for converting stored heat back to **electricity**. ``TES concrete``'s link
   is a heat exchanger returning heat as heat, so this is likely a large
   overestimate and makes the technology look more expensive than it is.

   Its *efficiency* is handled correctly: it defaults to 1.0 rather than
   inheriting ``Concrete-discharger``'s 0.4343, so no energy is lost on the
   round trip. Only the capital cost is questionable.

   Both stores ship with ``expansion: false``, so neither is built by default and
   no current result depends on either figure.

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

.. note::

   **On the discharge efficiency.** With no override in ``n_config``, the
   discharger takes ``Concrete-discharger``'s efficiency of **0.4343**, giving a
   round trip of 0.99 × 0.4343 ≈ **0.43** from electricity to ``Heat MT``, on top
   of a 2 %/h standing loss.

   A real electrically-charged heat battery does lose substantially on discharge,
   so a figure well below unity is the right *kind* of number: heat has to be
   extracted through an air or steam loop at a useful temperature, not simply
   handed over. For comparison, a study of a RONDO brick heat battery in this
   same cluster [DHAR]_ measured a **discharge efficiency of 0.76** against an
   electric boiler's 0.95, and an **observed annual round trip of 63–65 %** with
   1.2 %/h self-discharge.

   So 0.43 is not implausible in kind, but it is materially lower than a
   comparable real device — and its provenance is worth knowing: Viswanathan's
   row note reads *"RTE assume 99% for charge and other for discharge"*, which is
   a **power-to-power** split through a steam cycle rather than a heat discharge.
   Treat it as a conservative placeholder rather than a measured heat-side figure,
   and revisit it against [DHAR]_ or a vendor datasheet before drawing
   conclusions about this technology.

   To change it, set ``efficiency discharge`` for ``TES concrete El`` in
   ``n_config``; ``_eff`` uses that in preference to the catalogue value. Nothing
   is affected today, since the technology ships with ``expansion: false`` and is
   never built.

.. [DHAR] P. Dhar, *Modelling optimal storage operation with limited foresight in
   an industrial cluster*, MSc thesis, DTU Department of Wind and Energy Systems,
   April 2026.

.. note::

   A second limit to keep in mind: the discharger feeds ``Heat MT``, which
   ``p_config`` declares as a 140–180 °C circuit. A store able to deliver
   250–350 °C therefore cannot show that advantage in this model — there is no
   hotter circuit for it to serve.
