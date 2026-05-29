.. _n-options-yaml:

n_options.yaml
==============

Controls external market connections and optional revenue streams.
These options are independent of the network topology (``n_flags``) and act as
boundary conditions for the optimisation.

Location: ``config/n_options.yaml``

Options
-------

**pellets market**
   Purchase of wood pellets for biomass boiler or drying.
   ``enable: true`` opens the market; ``price`` sets the purchase price (€/MWh);
   ``max capacity`` caps the annual volume (MWh/y).

**moist biomass market**
   Purchase of wet biomass (e.g. wood chips). Disabled by default.

**Dig biomass**
   Supply of digestible biomass (manure + co-substrates) to the biogas plant.
   Price is set to 0 by default (own supply); ``max capacity`` in t/y.

**DH** (district heating sale)
   Enables revenue from selling surplus heat to the district heating grid.
   ``price`` in €/MWh; ``load multiplier`` scales the demand profile.
   Disabled by default — enable to model DH as an additional revenue stream.

**biochar credits**
   Revenue from CO₂ sequestration via biochar. Disabled by default.

**CO2 Liq credits**
   Revenue from liquefied CO₂ sequestration. Disabled by default.

**symbiosis El transformer**
   If ``expansion: true``, allows the electrical transformer connecting internal
   and external grids to be sized by the optimiser.
