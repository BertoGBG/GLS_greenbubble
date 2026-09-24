.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _tech-biogas:

Biogas and biomass
==================

Agent: ``n_flags.biogas``. Buys biomass from the external markets, sells biomethane to the gas grid, and supplies raw biogas and separated CO₂ to the shared buses. It has no dependencies: it can run standalone.

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

**Biomass belt dryer**, a hot-air belt dryer that takes wet biomass from about
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
