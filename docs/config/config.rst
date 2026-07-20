.. _config-yaml:

config.yaml
===========

Main configuration file. Controls the optimisation problem, scenario settings,
economics, and solver choice.

Location: ``config/config.yaml``

General
-------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``run_name``
     - ``H2_meth_dmd_DK``
     - Short label appended to the output folder name.
   * - ``CO2_cost``
     - ``100``
     - CO₂ price assumed for the scenario (€/t). Applied as the full price to
       natural gas (boiler combustion, bioCH4 sale price), and as the delta
       over ``CO2_cost_ref_year`` to electricity, since historical spot
       prices already embed a reference-year carbon price but historical NG
       commodity prices do not (that liability sits with the combusting
       plant, not the gas supplier).
   * - ``CO2_cost_ref_year``
     - ``0``
     - CO₂ price already embedded in the historical electricity price for
       the reference year (€/t). Not used for natural gas.
   * - ``En_price_year``
     - ``2023``
     - Year used to download electricity prices, CO₂ intensities, and NG prices.
   * - ``latitude`` / ``longitude``
     - ``56.566`` / ``9.033``
     - Location for renewable capacity factor retrieval (default: Skive, Denmark).
   * - ``max_RE_to_grid``
     - ``0.1``
     - Maximum share of renewable generation that can be exported to the grid.
   * - ``outputs_folder``
     - ``outputs/single_analysis``
     - Root folder for all run outputs.

targets
-------

Controls whether the model is demand-driven or price-driven.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Example
     - Description
   * - ``driver``
     - ``'demand'``
     - ``'demand'``: annual demands are fixed targets.
       ``'price'``: demands become upper bounds; model maximises profit.
   * - ``demand_H2``
     - ``0``
     - Annual H₂ demand (MWh\ :sub:`H2`/y). Only used when ``driver: 'demand'``.
   * - ``demand_CH4``
     - ``300000``
     - Annual biomethane demand (MWh\ :sub:`CH4`/y).
   * - ``demand_meoh``
     - ``4000``
     - Annual methanol demand (MWh\ :sub:`MeOH`/y).
   * - ``price_H2``
     - ``90``
     - H₂ price target (€/MWh). Only used when ``driver: 'price'``.
   * - ``price_bioCH4``
     - ``95``
     - BioCH₄ price target. Set to ``'NG_based'`` to derive from NG price + CO₂ tax.
   * - ``price_meoh``
     - ``110``
     - Methanol price target (€/MWh).

n_flags — technology activation
--------------------------------

Boolean switches that add or remove technology groups from the network.
Setting a flag to ``false`` excludes the corresponding components entirely.

.. list-table::
   :widths: 20 60
   :header-rows: 1

   * - Flag
     - Effect
   * - ``biogas``
     - Includes the biogas plant and all downstream biogas routes.
   * - ``central_heat``
     - Adds central heat supply and district heating (DH) connection.
   * - ``renewables``
     - Adds onshore wind and solar PV with capacity expansion.
   * - ``electrolysis``
     - Adds alkaline electrolysis for H₂ production.
   * - ``meoh``
     - Adds the methanol synthesis route (CO₂ hydrogenation).
   * - ``methanation``
     - Adds catalytic and biological methanation routes.
   * - ``symbiosis``
     - Enables all internal energy/material exchange links between plants.
   * - ``storage``
     - Adds all storage technologies defined in :ref:`n-config-yaml`.
   * - ``print``
     - Saves an SVG of the pre-optimisation network.
   * - ``export``
     - Exports the pre-optimisation network to ``.nc``.

n_flags_opt — output flags
---------------------------

.. list-table::
   :widths: 20 60
   :header-rows: 1

   * - Flag
     - Effect
   * - ``print``
     - Saves SVG of the optimal network.
   * - ``export``
     - Exports the optimal network to ``.nc``.
   * - ``plot``
     - Generates dispatch and capacity plots.

stochastic
----------

Enables multi-scenario stochastic optimisation.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``stochastic``
     - ``false``
     - Enable stochastic mode. All scenario years are preprocessed and coupled.
   * - ``scenarios``
     - see file
     - Dict of ``{year: probability}``. Years must sum to 1.
   * - ``CO2_cost_s``
     - see file
     - Per-scenario CO₂ cost (€/t).
   * - ``CO2_cost_ref_year_s``
     - see file
     - Per-scenario reference-year CO₂ cost.
   * - ``EVPI``
     - ``true``
     - If ``true``, runs one deterministic solve per scenario to compute EVPI.
       Automatically set to ``false`` when ``stochastic: false``.

optimization
------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``solver``
     - ``'gurobi'``
     - ``'gurobi'`` or ``'highs'``.
   * - ``solver_profile``
     - ``'gurobi-barrier-fast'``
     - Solver parameter preset. Defined in ``scripts/solver_profiles.py``.
   * - ``collect_all_duals``
     - ``true``
     - Saves dual variables for all constraints (needed for shadow price analysis).
   * - ``return_model``
     - ``true``
     - Returns the Linopy model object after solving.
   * - ``overrides``
     - ``null``
     - Optional dict of raw solver parameters (e.g. ``OutputFlag``, ``BarConvTol``).

Economics
---------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``year_investment``
     - ``2030``
     - Target year for technology cost projections from the cost database.
   * - ``discount_rate``
     - ``0.2``
     - Annual discount rate for annualised capital costs.
   * - ``DKK_Euro``
     - ``7.46``
     - DKK → EUR exchange rate.

tariffs_dict
------------

Danish electricity grid tariffs (€/MWh) applied to grid imports and exports.
Values sourced from Energinet and DSO tariff sheets. See inline comments in
``config.yaml`` for source links.

rfnbos_dict
-----------

Controls RFNBO (Renewable Fuels of Non-Biological Origin) compliance constraints.

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``limit``
     - ``'price'``
     - Compliance method: ``'price'`` (threshold on el. price), ``'emissions'``
       (threshold on grid CO₂ intensity), ``'unlimited'``, or ``'disconnected'``.
   * - ``price_threshold``
     - ``20``
     - Maximum electricity price (€/MWh) for RFNBO-compliant operation.
   * - ``emission_threshold``
     - ``0.0648``
     - Maximum grid CO₂ intensity (tCO₂/MWh) for RFNBO compliance.
