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

**Belt dryer** — hot-air dryer for biomass pellet production.  Semi-empirical
sizing based on moisture content.  ``expansion: false`` by default — treated
as a fixed brownfield asset.

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

This route also has **no separate H₂ compressor**, unlike ``methanolisation``.
That is a difference in the DEA battery limits, not an oversight: sheet 97
includes the reformer and reactor compressors in its CAPEX and electricity,
while sheet 98 excludes feed compression (hence
``electricity-input-no-compression`` and the separate compressor components on
the other route). Both plants pay for compression; only the bookkeeping differs.

.. note::

   DEA costed the **oxygen-fired autothermal (tri-reforming)** configuration,
   which is what is implemented. The electrically heated **eSMR** (bi-reforming)
   variant described in the same DEA chapter has no cost data and is not
   implemented.

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
     - Concrete thermal store; 10-hour duration; standing losses
   * - TES district heating
     - StorageUnit
     - Hot-water buffer for DH; 50-hour duration

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
