.. _plots-config-yaml:

plots_config.yaml
=================

Controls which network components are exported and plotted after optimisation.

Location: ``config/plots_config.yaml``

Structure
---------

Each entry under ``capacity_items`` describes one panel in the results plots:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Field
     - Description
   * - ``label``
     - Display name used in plots and tables.
   * - ``kind``
     - PyPSA component type: ``Generator``, ``Link``, ``Store``, or ``StorageUnit``.
   * - ``field``
     - Variable to extract: ``p`` / ``p0`` (power), ``e`` (energy/mass store),
       ``state_of_charge`` (SOC for storage).
   * - ``selector``
     - Substring matched against component names in the network.
       All matching components are aggregated.
   * - ``th``
     - Capacity threshold alias (see ``thresholds``). Components below this value
       are filtered out to reduce clutter.

Thresholds
----------

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Alias
     - Default
     - Applied to
   * - ``GEN_TH``
     - 0.5 MW
     - Generators (wind, solar)
   * - ``LINK_TH``
     - 0.5 MW
     - Power links
   * - ``LINK_MASS_TH``
     - 0.2 t/h
     - Mass-flow links
   * - ``STORE_TH``
     - 1.0 MWh
     - Energy stores
   * - ``STORE_MASS_TH``
     - 0.1 t
     - Mass stores
   * - ``SU_TH``
     - 0.2 MW
     - StorageUnits
   * - ``NO_TH``
     - 0
     - No filtering (used for demand-side links)

bus_list_mp
-----------

List of internal energy/material buses for which marginal prices (shadow prices)
are extracted and plotted. Default buses:

- ``El3 bus`` — electricity at plant level
- ``H2`` — hydrogen
- ``bioCH4`` — biomethane
- ``Methanol`` — methanol
- ``Heat MT`` — medium-temperature heat
- ``Heat DH`` — district heating
- ``Heat LT`` — low-temperature heat
- ``CO2 distribution`` — CO₂ distribution bus
