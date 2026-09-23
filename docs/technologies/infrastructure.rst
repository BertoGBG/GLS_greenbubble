.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _tech-infrastructure:

Compression, heat exchange and shared infrastructure
====================================================

These components belong to no single agent: compressors and pipes are built
wherever a plant needs them, and the grid connection and external markets are
the site's interfaces. Their duties are computed from physics rather than read
from a catalogue — see :ref:`physics-based`.

.. _technologies-compression:

Compression and pressure levels
-------------------------------


Every carrier in the model has a declared pressure, and a compressor exists
wherever a plant needs its feed above the pressure of the header it draws from.
The states live in ``p_config`` — see :doc:`/guide_process_streams` — and the
compressor duties are computed from them with CoolProp, not hardcoded.

**Shared headers.** These are the conditions a carrier sits at between plants:

.. list-table::
   :header-rows: 1
   :widths: 34 12 12 42

   * - Header
     - Fluid
     - P [bar]
     - Note
   * - ``H2 production``
     - H₂
     - 30
     - The shared hydrogen header. AEC and PEMEC inject directly; DEA states
       both "can deliver hydrogen at pressures as high as 30 bar".
   * - ``H2 SOEC outlet``
     - H₂
     - 3.5
     - SOEC alone delivers low-pressure and is lifted to the header by its own
       compressor. Manufacturer figure (Topsoe); DEA gives no SOEC pressure.
   * - ``H2 HP storage``
     - H₂
     - 150
     - Buffer storage, filled from the 30 bar header.
   * - ``biogas``
     - biogas
     - 1
     - Leaves the digester at ambient.
   * - ``CO2 biogas upgrading``
     - CO₂
     - 1
     - Separated CO₂, ambient.
   * - ``CO2 HP storage``
     - CO₂
     - 60
     - Returned to consumers at 30 bar.
   * - ``CO2 Liq storage``
     - CO₂
     - 16
     - Liquefied, at −26 °C.
   * - ``NG grid``
     - CH₄
     - 40
     - External gas grid.

**Plant inlets.** Each plant declares the pressure it needs. The gap between
that and the header is what the compressor pays for:

.. list-table::
   :header-rows: 1
   :widths: 30 22 22 26

   * - Plant
     - Feed
     - Needs [bar]
     - Compressor
   * - ``biomethanation``
     - H₂
     - 1
     - none — the reactor is atmospheric
   * - ``methanation``
     - H₂, CO₂, biogas
     - 20
     - yes, on each feed
   * - ``methanolisation``
     - H₂, CO₂
     - 80
     - yes — DEA sheet 98 excludes feed compression
   * - ``methanol from biogas``
     - H₂, biogas
     - taken as supplied
     - none — DEA sheet 97 includes it, and reforming is atmospheric

This is why the same carrier can reach two plants through different components:
``biomethanation`` takes hydrogen straight off the header because it runs at
ambient pressure, while ``methanolisation`` pays to lift the same hydrogen to
80 bar. Tying the duty to a declared state rather than a constant is what keeps
those two consistent.

How the compressor duty is calculated
-------------------------------------


Compressor electricity and waste heat are **not** catalogue numbers and not
fixed ratios. They are computed from the thermodynamics of the actual fluid,
between the actual inlet and outlet states, using
`CoolProp <http://www.coolprop.org/>`_ for the property data. Pure fluids are
looked up by name; biogas is handled as a real CH₄/CO₂ mixture built from
``globals.mixtures.biogas``.

The calculation lives in ``compress_multistage_with_Tcap``
(``scripts/technology_inputs.py``). Per stage it is the textbook isentropic
route:

.. math::

   w_s = h(p_\text{out}, s_\text{in}) - h_\text{in}, \qquad
   w = \frac{w_s}{\eta_s}, \qquad
   h_\text{out} = h_\text{in} + w

with the discharge temperature read back from :math:`(p_\text{out},
h_\text{out})`. Both enthalpies and the entropy come from CoolProp, so real-gas
behaviour is included rather than assumed ideal.

Staging is set by two limits, whichever binds first:

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Parameter
     - Value
     - Meaning
   * - ``r_max``
     - 2.5
     - Maximum pressure ratio per stage. The stage count is
       :math:`\lceil \log(p_\text{out}/p_\text{in}) / \log r_\text{max} \rceil`,
       and the ratio is then shared equally between stages.
   * - ``T_max_C``
     - 160 °C
     - Maximum discharge temperature. A stage is cut short if it would exceed
       this, adding another stage instead.
   * - ``eta_s``
     - 0.75
     - Isentropic efficiency.
   * - motor efficiency
     - 0.97
     - Applied once to the summed shaft work.
   * - intercooling
     - to 50 °C
     - Between stages, back down to the ``Heat LT`` floor.

Because the duty is derived rather than tabulated, the same component gives
different numbers for different fluids and lifts. Hydrogen from the SOEC outlet
to the header (3.5 → 30 bar) needs 3 stages and 1.22 kWh/kg, which is 3.66 % of
the hydrogen LHV; from 1 bar it would need 5 stages and 1.93 kWh/kg, 5.78 %.

``compressor_calculation`` (same file) is the orchestrator: it reads the inlet
and outlet states from ``p_config``, applies pre-cooling if the feed arrives
above the compressor inlet limit, handles the high-pressure storage cases, and
returns electricity and heat per unit of throughput.

``globals.T_max_comp`` in ``p_config`` is the single source of this limit. It
does two jobs: it sets the **declared temperature** of streams downstream of a
compressor (the ``${T_max_comp}`` references in the port states), and it is
passed to ``compress_multistage_with_Tcap`` as the discharge cap, so the staging
follows it. Lowering it adds stages and lowers the work, because more
intercooling moves the machine closer to isothermal compression:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - ``T_max_comp``
     - Stages
     - Work [kWh/kg_H2]
   * - 160 °C (default)
     - 3
     - 1.2212
   * - 120 °C
     - 5
     - 1.1861
   * - 90 °C
     - 7
     - 1.1505

(hydrogen, 3.5 → 30 bar, 50 °C inlet — the SOEC lift).

Aftercooling and the heat exchangers
------------------------------------


Every stage rejects heat, and that heat is not thrown away: it is split by
temperature and sold into the heat circuits. ``aftercomp_cool_duty``
(``scripts/technology_inputs.py``) integrates the enthalpy drop at constant
pressure and divides it at the ``Heat DH`` floor:

- above the split → ``Heat DH`` (usable district heat)
- below the split → ``Heat LT``

Those two numbers become ``efficiency3`` and ``efficiency4`` on the compressor
link, so the same PyPSA component buys electricity and sells both heat grades.
``DH heat exchanger`` is the component that couples a plant's local heat bus to
the shared circuit; its cost and efficiency come from ``technology-data`` like
any other technology, and it is attached by ``add_local_heat_connections``
(``scripts/prepare_network.py``).

.. note::

   The split is a single cut at one temperature, so a stream leaving a
   compressor at 160 °C contributes its 160–140 °C slice to ``Heat DH`` even
   though that slice is MT-grade. ``scripts/heat_bands.py`` generalises this to
   contiguous temperature bands, but the compressor code does not use it yet.
   See :doc:`/guide_process_streams`.

Where the compressors sit in the model
--------------------------------------


All of them are built by ``add_compressor_and_storage``
(``scripts/prepare_network.py``), which supports two placements:

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Placement
     - Component name
     - Status
   * - **Per plant**
     - ``methanolisation H2 compressor``,
       ``methanation biogas compressor``, ``SOEC`` …
     - **What the model actually builds.** Every compressor belongs to the plant
       that needs the lift, and is sized from that plant's throughput.
   * - **Centralised**
     - ``H2 compressor``, ``CO2 compressor``, …
     - Supported by the code but **not currently used** — no caller requests it.
       It would be one shared machine per fluid, sized from ``n_config``.

The distinction is the ``plant`` key of the dict passed to
``add_compressor_and_storage``: empty means centralised, a plant name means
per-plant. Every call site today passes a name.

This matters for reading results: there is no single "H2 compressor" row to look
at. Compression shows up distributed across the plants — ``SOEC`` lifting to the
header, ``methanolisation H2 compressor`` lifting to 80 bar, and so on — so the
site's total compression cost is the sum of those, not one line item.

**Per-plant does not mean duplicated.** A greenfield build with methanation and
both methanol routes active contains exactly these feed compressors:

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Component
     - Lift
   * - ``SOEC H2 compressor``
     - ``H2 SOEC LP`` → ``H2 collection``, 3.5 → 30 bar
   * - ``methanolisation H2 compressor``
     - ``H2 distribution`` → ``H2 to methanolisation``, 30 → 80 bar
   * - ``methanolisation CO2 compressor``
     - ``CO2 distribution`` → ``CO2 to methanolisation``, 1 → 80 bar
   * - ``methanation CO2 CO2 compressor``
     - ``CO2 distribution`` → ``CO2 to methanation``, 1 → 20 bar
   * - ``methanation biogas biogas compressor``
     - ``biogas`` → ``biogas to methanation``, 1 → 20 bar

Each one performs a **different** lift, so none is redundant. Carbon dioxide
leaves the same header for two plants at two pressures — 20 bar for methanation,
80 bar for methanolisation — and a single shared machine could not serve both
without over-compressing one feed. The per-plant split is what the pressure
ladder requires, not an oversight.

Where several plants *do* want the same lift, they share one component rather
than building two: the biomethanation, methanation-biogas and methanation-CO₂
builders all pass the same ``methanation`` label for their hydrogen, the
component name collides deliberately, and a guard skips anything already in the
network. (For that particular case no compressor is built at all — methanation
takes hydrogen at 20 bar from a 30 bar header, which is a reduction, so
``compressor_calculation`` returns its "not needed" case.)

That is also why the centralised placement has never been needed: it assumes one
lift per fluid, and this model does not have that. It would only pay off if two
plants wanted the same lift from the same header.

.. note::

   Component names are ``{plant} {fluid} compressor``, so a plant whose own name
   already contains the fluid reads oddly — ``methanation CO2 CO2 compressor`` is
   the CO₂ compressor of the ``methanation CO2`` plant, not a typo.

The generic ``H2 compressor`` / ``CO2 compressor`` / ``CH4 compressor`` rows in
``n_config`` are still doing work in the per-plant case: they carry the
``expansion`` permission and the cost lookup that every per-plant instance
inherits. The capacity, though, comes from the calling plant.

All compressors are ordinary PyPSA links, so the optimiser sizes them and
dispatches them hour by hour like any other component.

**Other shared components.** ``H2 pipe`` and ``CO2 pipe`` carry a carrier
between plants when ``symbiosis`` is on. All are configured in ``n_config`` like
any other technology.

Grid connection
---------------


A ``Link`` representing the point of common coupling (PCC) with the
electricity grid.  Capacity is extendable by default.  Grid import carries
the full Danish tariff stack (TSO, DSO, state levies) defined in
``tariffs_dict``; grid export earns the sell tariff.

External markets
----------------


Configured under the ``options:`` section of ``n_config.default.yaml``:

- **Pellets market** — biomass pellet purchase at a fixed price; optional
  capacity cap
- **Moist biomass market** — co-substrate purchase for biogas plant
- **Biochar credits** — revenue credit for biochar sequestration from
  pyrolysis
- **CO₂ Liq credits** — revenue credit for liquefied CO₂ export
- **District heating** — heat sale to external DH grid (``options.DH``)
