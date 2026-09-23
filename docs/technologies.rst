.. _technologies:

Technologies
============

This page describes each technology group modelled in GreenBubble: how it is
represented in PyPSA, which parameters control its behaviour, and any
modelling assumptions worth knowing.  Each group corresponds to one or more
entries in ``config/n_config.default.yaml`` and is activated by a flag in
``n_flags``.

.. note::
   This page documents the current technology set.  New technologies follow
   the pattern described in :ref:`guide-new-technology`.

---

Renewable electricity  (``n_flags.renewables``)
-------------------------------------------------

**Onshore wind** and **solar PV** are modelled as ``Generator`` components
with capacity-factor time series retrieved from
`Renewables.ninja <https://www.renewables.ninja>`_ for the configured site
coordinates.  Both are extendable by default (greenfield) with costs from the
technology-data database.

Up to ``max_RE_to_grid`` fraction of total renewable output can be exported
to the electricity grid; the remainder must be consumed internally.

---

Electrolysis  (``n_flags.electrolysis``)
------------------------------------------

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

---

Biogas and biomass chain  (``n_flags.biogas``)
------------------------------------------------

**Biogas plant** — modelled as a ``Generator`` producing raw biogas (CH₄
equivalent) from digestible biomass inputs (manure, co-substrate).  Feedstock
cost and availability are set in the ``options.Dig biomass`` block of
``n_config.default.yaml``.

**Biogas upgrading** — pressure-swing adsorption (PSA) or equivalent, strips
CO₂ from raw biogas to yield pipeline-quality biomethane.  Modelled as a
``Link`` (raw biogas → biomethane + CO₂).

**Biogas compressor** — compresses raw biogas upstream of methanation;
modelled as a ``Link`` with electricity input.

**Biogas engine** — combined heat-and-power unit; a multi-output ``Link``
(biogas → electricity + heat) with minimum load and optional committable
mode for dispatch-only runs.

**Biomass belt dryer** — a hot-air belt dryer that takes wet biomass from about
**50 % moisture down to 13 %**, so it can be used by a downstream thermal
process. Drying is what makes wet biomass usable: at 50 % moisture roughly half
the mass is water that would otherwise be evaporated inside the conversion
process itself, at the cost of its own heat. In GreenBubble the dried product
feeds the ``pellets boiler`` (combustion) and ``pyrolysis`` (biochar); the same
step would precede gasification in a plant that had one.

The duty is a mass and energy balance, not a fitted curve
(``mass_energy_balance_drying`` in ``scripts/prepare_network.py``). Water is
counted on a **dry basis**, which is what makes the two moisture figures
comparable:

.. math::

   w = \frac{M_\text{in}}{1 - M_\text{in}} - \frac{M_\text{out}}{1 - M_\text{out}}

With the model's values (``chips`` at 0.50, ``pellets`` at 0.13, both declared in
``p_config``) that is 0.8506 t of water removed per tonne of dry matter. Heat and
electricity then follow from technology-data at 1.0 MWh_th and 0.025 MWh_e per
tonne of water evaporated.

The dryer is a ``Link`` from ``moist biomass`` to ``pellets``, drawing ``Heat MT``
and electricity. ``expansion: false`` by default — treated as a fixed brownfield
asset.

**Dewatering** — screw press for digestate solid/liquid separation.
Fixed capacity, no expansion.

**Pelletization** — biomass pellets production from dried fibres.

---

Methanation routes  (``n_flags.methanation``)
----------------------------------------------

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

---

Methanol synthesis  (``n_flags.meoh``)
----------------------------------------

CO₂ hydrogenation to methanol (``methanolisation``): CO₂ + H₂ → MeOH + H₂O.
Modelled as a multi-input ``Link`` with ramp limits (8 %/h) and minimum
load (15 %).  Waste heat is recovered to the medium-temperature heat bus.

The water is not a side reaction that better catalysis could avoid: CO₂ carries
two oxygens and methanol contains one, so the spare oxygen must leave with
hydrogen.  Equivalently, the CO₂ route is the CO route plus reverse water-gas
shift, and RWGS both consumes 45 % of the CO exotherm and creates the water —
the modest reactor duty and the wet crude are the same phenomenon.

Optionally this link is **split** into ``methanol synthesis`` + ``methanol
distillation`` with a crude-methanol tank between them, so the reactor and the
column can run at different times.  Enable with ``options['meoh split']`` in
``n_config``; see :doc:`meoh_split`.

**Methanol from biogas** (``methanol from biogas``) is an *alternative* to
``methanolisation``, not an addition to it: both produce into the same methanol
collection bus, and the optimiser picks. Instead of feeding separated CO₂ and
hydrogen to a synthesis reactor, raw biogas is reformed with oxygen in an
autothermal reformer and the resulting syngas is hydrogenated.

The carbon comes from the biogas itself, so the route needs **2.5x less
hydrogen** per MWh of methanol (0.449 MWh_H2/MWh_MeOH against 1.138) — but it
burns biomethane that could have been sold instead. Which one wins is exactly
the trade-off the model is set up to answer.

Costs and coefficients come from DEA sheet 97, "Methanol from biogas and
hydrogen", and are stored per MWh of hydrogen:

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Quantity
     - Value
     - Unit
   * - Methanol output
     - 2.229
     - MWh_MeOH / MWh_H2
   * - Biogas input
     - 1.6516
     - MWh_biogas / MWh_H2
   * - Electricity input
     - 0.1371
     - MWh_e / MWh_H2
   * - Heat input (MT)
     - 0.0617
     - MWh_th / MWh_H2
   * - Investment
     - 6423.8
     - EUR / kW_H2

Two streams are deliberately **not** wired. Oxygen: the reformer needs
0.1729 t_O2/MWh_H2 while the electrolysis feeding it co-produces 0.2381 t, so
oxygen never binds on a site with its own electrolyser and is treated as free.
Water output is left unwired for the same reason as in ``methanolisation``.

This route also has **no feed compressors at all**, unlike ``methanolisation``.
Two independent reasons agree.

*Physically*, there is nothing to compress at the battery limit. The pilot this
route is based on reforms biogas at ambient pressure and compresses the
**syngas** afterwards, once, with the hydrogen already mixed in [FRITSCH]_:

   "we designed a system for simple, atmospheric conversion of biogas to SynGas
   in a single-stage ATR-unit ... SynGas is cooled and compressed for subsequent
   introduction into the MeOH-synthesis loop"

The 20 bar quoted for that plant is the synthesis **loop** pressure, downstream
of both feeds — and it is a pilot-scale ceiling rather than an optimum, set by
the EU Pressure Equipment Directive on the basis of system gas volume.
Industrial methanol loops run at 50-100 bar.

*In the accounting*, DEA agrees: sheet 97 bundles "compressors prior to reformer
and methanol reactor" into its CAPEX and its electricity, while sheet 98
excludes feed compression (hence ``electricity-input-no-compression`` and the
separate compressor components on the other route). Both plants pay for
compression; only the bookkeeping differs. A GreenBubble compressor here would
charge it twice.

Because this plant creates no compressed bus of its own, it has no ``processes:``
entry in ``p_config`` — see :doc:`guide_process_streams` for why that is the
correct form rather than an omission.

.. note::

   **Known simplification.** Hydrogen is drawn from the shared 30 bar header
   while the reference plant doses it at ambient, so this route pays for
   compression it does not physically need. Left in place deliberately:
   correcting it needs a low-pressure hydrogen tap, and it would only make the
   route cheaper.

.. note::

   DEA costed the **oxygen-fired autothermal (tri-reforming)** configuration,
   which is what is implemented. The electrically heated **eSMR** (bi-reforming)
   variant described in the same DEA chapter has no cost data and is not
   implemented.

.. [FRITSCH] C. Fritsch, J. Blankenstein, B. Bender, J. Dornseiffer, M. Haep and
   K. Ooms, *Design, implementation and piloting of an integrated hydrogen- and
   oxygen-added process for conversion of biogas to methanol*, Sustainable Energy
   & Fuels, 2025. https://doi.org/10.1039/d5se00691k

---

Compression and pressure levels
--------------------------------

Every carrier in the model has a declared pressure, and a compressor exists
wherever a plant needs its feed above the pressure of the header it draws from.
The states live in ``p_config`` — see :doc:`guide_process_streams` — and the
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
   See :doc:`guide_process_streams`.

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

---

Storage  (``n_flags.storage``)
--------------------------------

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

---

Heat system  (``n_flags.central_heat``)
-----------------------------------------

Three heat buses represent temperature levels in the cluster:

- **Heat MT** (medium temperature) — process waste heat, biomass boilers,
  biogas engine CHP output
- **Heat DH** (district heating) — optional connection to an external DH
  network (price and load set in ``options.DH``)
- **Heat LT** (low temperature) — low-grade cooling and heat pump source

A ``heat pump`` link can upgrade LT to DH heat (extendable, disabled by
default).  ``El boiler`` and ``NG boiler`` provide backup heat.

---

Industrial symbiosis  (``n_flags.symbiosis``)
-----------------------------------------------

The symbiosis flag adds all internal distribution links between plants:
electricity, H₂, CO₂, and heat exchange connections across plant boundaries.
Disabling it isolates each plant to its own buses, useful for benchmarking
individual plant economics.

The ``symbiosis El transformer`` (configured under ``options`` in
``n_config.default.yaml``) controls whether the internal electrical
transformer between plants is extendable.

---

Grid connection
----------------

A ``Link`` representing the point of common coupling (PCC) with the
electricity grid.  Capacity is extendable by default.  Grid import carries
the full Danish tariff stack (TSO, DSO, state levies) defined in
``tariffs_dict``; grid export earns the sell tariff.

---

External markets
-----------------

Configured under the ``options:`` section of ``n_config.default.yaml``:

- **Pellets market** — biomass pellet purchase at a fixed price; optional
  capacity cap
- **Moist biomass market** — co-substrate purchase for biogas plant
- **Biochar credits** — revenue credit for biochar sequestration from
  pyrolysis
- **CO₂ Liq credits** — revenue credit for liquefied CO₂ export
- **District heating** — heat sale to external DH grid (``options.DH``)
