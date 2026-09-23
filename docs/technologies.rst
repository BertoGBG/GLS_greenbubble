.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _technologies:

Technologies
============

Every technology in the model belongs to one **agent** — the broad category of
plant it performs its function within — and each agent is switched on by its
``n_flags`` entry. The agent structure and what it means for the model is in
:doc:`model_approach`; this section is the catalogue.

.. toctree::
   :maxdepth: 1
   :caption: By agent

   technologies/renewables
   technologies/electrolysis
   technologies/biogas
   technologies/methanation
   technologies/meoh
   technologies/heat
   technologies/storage
   technologies/infrastructure
   meoh_split

----

Where a technology is characterised
-----------------------------------

A technology's numbers come from up to three places, and knowing which one holds
what saves a lot of searching:

.. list-table::
   :header-rows: 1
   :widths: 10 26 64

   * - Key
     - Source
     - Holds
   * - **TD**
     - ``technology-data``
     - The downloaded cost catalogue: investment, FOM, VOM, lifetime,
       efficiencies. Fetched per year into
       ``data/technology-data/outputs/costs_{year}.csv``. Project-specific rows
       that do not belong upstream are added in ``scripts/technology_inputs.py``.
   * - **NC**
     - ``n_config``
     - What the optimiser may do with it: initial capacity, whether it can
       expand, maximum size, ramp limits, minimum load, cost factor, and the
       brownfield parameters. Every technology has an entry.
   * - **PC**
     - ``p_config``
     - The physical state of its streams — fluid, temperature, pressure — where
       the technology has streams of its own. This is what makes compressor
       duties and heat-circuit assignment physical rather than assumed.

Because **every** technology has an ``n_config`` entry, the table below marks
only where **TD** and **PC** additionally apply.

----

All technologies
----------------

Alphabetical. "Connects to" names the shared buses the technology touches when
``symbiosis`` is on, plus any external interface or product bus.

.. list-table::
   :header-rows: 1
   :widths: 22 15 45 18

   * - Technology
     - Agent
     - Connects to
     - Data
   * - ``AEC``
     - electrolysis
     - electricity → H₂ header, heat out
     - TD · PC
   * - ``battery``
     - storage
     - electricity
     - TD
   * - ``biogas``
     - biogas
     - biomass in → biogas bus, heat, electricity
     - TD · PC
   * - ``biogas compressor``
     - *shared*
     - biogas bus → a plant's compressed inlet
     - TD · PC
   * - ``biogas engine``
     - biogas
     - biogas → electricity + heat
     - TD
   * - ``biogas storage``
     - biogas
     - biogas bus
     - TD
   * - ``biogas upgrading``
     - biogas
     - biogas → CH₄ (sale) + CO₂ header
     - TD · PC
   * - ``biomass belt dryer``
     - central_heat
     - moist biomass → pellets, heat MT, electricity
     - TD
   * - ``biomass boiler``
     - central_heat
     - pellets → heat MT
     - TD
   * - ``biomethanation``
     - methanation
     - H₂ + biogas → CH₄ (sale)
     - TD · PC
   * - ``biomethanation CO2``
     - methanation
     - H₂ + CO₂ → CH₄ (sale)
     - TD
   * - ``CH4 compressor``
     - *shared*
     - CH₄ bus → delivery pressure
     - TD
   * - ``CO2 compressor``
     - *shared*
     - CO₂ header → a plant's compressed inlet
     - TD · PC
   * - ``CO2 HP storage``
     - storage
     - CO₂ header
     - TD · PC
   * - ``CO2 Liq storage``
     - storage
     - CO₂ header → liquid CO₂ export
     - TD · PC
   * - ``CO2 pipe``
     - *shared*
     - CO₂ between plants
     - TD
   * - ``crude MeOH storage``
     - meoh
     - crude methanol, between synthesis and distillation
     - TD
   * - ``dewatering``
     - biogas
     - digestate solid/liquid separation
     - TD
   * - ``DH heat exchanger``
     - *shared*
     - plant heat bus ↔ shared heat circuit
     - TD
   * - ``El boiler``
     - central_heat
     - electricity → heat MT
     - TD
   * - ``grid connection``
     - renewables
     - electricity ↔ external market
     - TD
   * - ``H2 compressor``
     - *shared*
     - H₂ header → a plant's compressed inlet
     - TD · PC
   * - ``H2 HP storage``
     - storage
     - H₂ header
     - TD · PC
   * - ``H2 pipe``
     - *shared*
     - H₂ between plants
     - TD
   * - ``heat pump``
     - central_heat
     - electricity + low-grade heat → heat DH
     - TD
   * - ``methanation biogas``
     - methanation
     - H₂ + biogas → CH₄ (sale), catalytic
     - TD · PC
   * - ``methanation CO2``
     - methanation
     - H₂ + CO₂ → CH₄ (sale), catalytic
     - TD · PC
   * - ``methanol distillation``
     - meoh
     - crude methanol → methanol (sale), heat MT in
     - TD
   * - ``methanol from biogas``
     - meoh
     - H₂ + biogas → methanol (sale), via reforming
     - TD · PC
   * - ``methanol synthesis``
     - meoh
     - H₂ + CO₂ → crude methanol, heat out
     - TD
   * - ``methanolisation``
     - meoh
     - H₂ + CO₂ → methanol (sale), heat out
     - TD · PC
   * - ``NG boiler``
     - central_heat
     - natural gas → heat MT
     - TD
   * - ``onwind``
     - renewables
     - → electricity
     - TD
   * - ``pelletization``
     - central_heat
     - dried biomass → pellets
     - TD
   * - ``PEMEC``
     - electrolysis
     - electricity → H₂ header, heat out
     - TD · PC
   * - ``pyrolysis``
     - central_heat
     - pellets → heat MT + biochar (export)
     - TD
   * - ``SOEC``
     - electrolysis
     - electricity → low-pressure H₂ → header, heat out
     - TD · PC
   * - ``solar``
     - renewables
     - → electricity
     - TD
   * - ``TES concrete``
     - storage
     - heat MT
     - TD
   * - ``TES concrete El``
     - storage
     - electricity → heat MT
     - TD
   * - ``TES DH``
     - storage
     - heat DH
     - TD

.. note::

   ``base`` is not a technology. It is the row in ``n_config`` that every other
   entry inherits its defaults from — change it and you change every technology
   that has not overridden that field.
