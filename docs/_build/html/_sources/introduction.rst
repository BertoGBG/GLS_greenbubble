.. _introduction:

Introduction
============

**GreenBubble** is an open techno-economic optimisation model for Power-to-X industrial
clusters, built on the `PyPSA <https://pypsa.readthedocs.io>`_ framework.
It is inspired by the `GreenLab Skive <https://www.greenlab.dk>`_ industrial park in Denmark —
an agricultural-industrial hub that co-locates biogas, electrolysis, methanation,
and methanol production.

.. image:: front_image.png
   :alt: GreenBubble network diagram
   :width: 100%
   :align: center

|

The model simultaneously optimises **capacity expansion** and **hourly dispatch** of a
multi-energy network (electricity, hydrogen, CO₂, heat, biomethane, methanol) over a
full year at 1-hour resolution. It is used in:

   *Optimizing hydrogen and e-methanol production through Power-to-X integration in
   biogas plants*
   — https://doi.org/10.1016/j.enconman.2024.119175


What the model can do
---------------------

- Greenfield and brownfield capacity expansion via linear programming
- Simultaneous capacity and dispatch optimisation (no decomposition)
- Multi-energy networks: electricity, H₂, CO₂, biomethane, methanol, heat (3 levels)
- Internal hydrogen, CO₂, electricity and heat distribution networks
- Stochastic optimisation across multiple price/weather scenarios
- Shapley value cost allocation across industrial partners
- Shadow price analysis for all internal energy and material flows
- RFNBO compliance constraints (price-based or emission-based)


Technologies and processes
--------------------------

**Hydrogen production**

- Alkaline electrolysis

**Methane production**

- Biogas + upgrading
- Biomethanation of biogas (with H₂)
- Biomethanation of CO₂ (with H₂)
- Catalytic methanation of biogas (with H₂)
- Catalytic methanation of CO₂ (with H₂)

**Methanol production**

- CO₂ hydrogenation
- eSMR + methanol synthesis *(coming soon)*

**Renewable electricity**

- Onshore wind
- Solar PV

**Storage**

- Lithium-ion batteries
- H₂ in steel vessels
- CO₂ liquefaction and storage
- CO₂ pressurised cylinders
- Hot water thermal storage (district heating style)
- Concrete-based thermal energy storage

**Biomass handling**

- Hot air belt dryer
- Dewatering of digestate fibres


External markets
----------------

The model is a **price taker** with respect to external markets.
Exogenous inputs include:

- CO₂ tax on fossil emissions
- Electricity spot prices, emission intensities, TSO/DSO grid tariffs
- Natural gas prices
- District heating price
- Biomass pellets and chips
- Digestible biomass (manure)


Data sources
------------

- **Electricity prices, CO₂ intensities, NG prices** — `Energi Data Service <https://www.energidataservice.dk>`_
- **Renewable capacity factors** — `Renewables.ninja <https://www.renewables.ninja>`_
- **Technology costs** — `technology-data <https://technology-data.readthedocs.io>`_ (extended for industrial clusters)
- **GreenLab Skive plant data** — ``data/GreenLab_Input_file.xlsx``
- **Technology exceptions** (compressors, biomass drying) — ``scripts/technology_inputs.py``


Licence
-------

The code is released under the MIT licence.
The documentation is released under CC-BY-4.0.
