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

Alkaline electrolysis is a ``Link`` (electricity → hydrogen) with:

- Ramp limits (up/down) to reflect the electrolyser's response speed
- Minimum load to avoid operating at inefficient part-load
- Committable mode available for brownfield dispatch runs

The electrolyser cost is sourced from the IEA via the technology-data
database.  Efficiency is fixed (no part-load curve); an improved part-load
model is planned.

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

**Biomethanation** (``biomethanation CO2``, ``biomethanation biogas``)
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

An eSMR + methanol synthesis route is planned but not yet implemented.

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
