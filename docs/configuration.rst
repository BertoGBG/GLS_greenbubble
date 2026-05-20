.. _configuration:

Configuration
=============

GreenBubble is configured via three YAML files in the ``config/`` folder.
Each file has a committed ``*.default.yaml`` base and an optional
``*.yaml`` user-override that is merged on top at runtime.
See :ref:`guide-snakemake` for the full override workflow.

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Default file
     - Purpose
   * - ``config/config.default.yaml``
     - Main settings: demands, technology flags, economics, solver, stochastic scenarios
   * - ``config/n_config.default.yaml``
     - Per-technology capacity bounds, ramp limits, brownfield initial capacities,
       and external market options (``options:`` section)
   * - ``config/plots_config.default.yaml``
     - Which components to export and plot after optimisation

----

config.default.yaml
--------------------

.. _config-general:

General
^^^^^^^

.. code-block:: yaml

   run_name: H2_meth_dmd_DK

Short label appended to the output folder name. Keep it concise.

.. code-block:: yaml

   CO2_cost: 100         # €/t — CO₂ tax on fossil emissions
   CO2_cost_ref_year: 0  # €/t — CO₂ cost already embedded in energy prices

.. code-block:: yaml

   En_price_year: 2023   # year used to download electricity/NG prices and CO₂ intensities

.. code-block:: yaml

   latitude:  56.566     # Skive, Denmark (used for renewable CF retrieval)
   longitude:  9.033

.. code-block:: yaml

   max_RE_to_grid: 0.1   # max share of renewable output that can be exported to grid

.. code-block:: yaml

   outputs_folder: outputs/single_analysis

.. _config-targets:

targets
^^^^^^^

Controls whether the model is **demand-driven** or **price-driven**.

.. code-block:: yaml

   targets:
     driver: 'demand'       # 'demand' | 'price'
     demand_H2:   0         # MWh_H2/y  — annual H₂ demand (demand mode)
     demand_CH4:  300000    # MWh_CH4/y — annual biomethane demand
     demand_meoh: 4000      # MWh_MeOH/y — annual methanol demand
     price_H2:    90        # €/MWh — H₂ price target (price mode)
     price_bioCH4: 95       # €/MWh — 'NG_based' derives from NG price + CO₂ tax
     price_meoh:  110       # €/MWh — methanol price target

In **demand mode** (``driver: 'demand'``), annual production targets are fixed constraints.
In **price mode** (``driver: 'price'``), demands become upper bounds and the model
maximises revenue at the given prices.

.. _config-nflags:

n_flags — technology activation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Boolean switches that include or exclude technology groups from the network.
Setting a flag to ``false`` removes the corresponding components entirely.

.. code-block:: yaml

   n_flags:
     biogas:       true   # biogas plant and all downstream biogas routes
     central_heat: true   # central heat supply and district heating connection
     renewables:   true   # onshore wind + solar PV with capacity expansion
     electrolysis: true   # alkaline electrolysis
     meoh:         true   # methanol synthesis (CO₂ hydrogenation)
     methanation:  true   # catalytic and biological methanation
     symbiosis:    true   # all internal energy/material exchange links
     storage:      true   # all storage technologies (from n_config.default.yaml)
     print:        true   # save SVG of pre-optimisation network
     export:       false  # export pre-optimisation network to .nc

.. note::
   The ``n_flags`` combination is encoded into the output folder name,
   making each run uniquely identifiable.

.. _config-stochastic:

stochastic
^^^^^^^^^^

Enables multi-scenario stochastic optimisation.

.. code-block:: yaml

   stochastic:
     stochastic: false    # true → stochastic LP across all scenario years
     scenarios:
       '2022': 0.05       # year: probability (must sum to 1)
       '2023': 0.25
       '2024': 0.35
       '2025': 0.35
     CO2_cost_s:          # per-scenario CO₂ cost (€/t)
       '2022': 100
       ...
     CO2_cost_ref_year_s: # per-scenario reference-year CO₂ cost
       '2022': 0
       ...
     EVPI: true           # compute Expected Value of Perfect Information

When ``stochastic: true``, input data is downloaded for **all scenario years**
in parallel (one Snakemake job per year) before building the coupled network.
``EVPI: true`` adds one deterministic solve per scenario to compute the EVPI;
automatically disabled when ``stochastic: false``.

.. _config-optimization:

optimization
^^^^^^^^^^^^

.. code-block:: yaml

   optimization:
     solver: 'gurobi'                  # 'gurobi' | 'highs'
     solver_profile: 'gurobi-barrier-fast'  # preset from scripts/solver_profiles.py
     collect_all_duals: true           # save dual variables for shadow price analysis
     return_model: true                # return Linopy model object after solving
     overrides: null                   # optional raw solver parameters

Solver profiles are defined in ``scripts/solver_profiles.py``.
Common Gurobi profiles: ``gurobi-barrier-fast``, ``gurobi-simplex``.

.. _config-economics:

Economics
^^^^^^^^^

.. code-block:: yaml

   year_EU:       2030   # target year for technology cost projections
   discount_rate: 0.2    # annual discount rate for annualised capital costs
   DKK_Euro:      7.46   # DKK → EUR exchange rate
   USD_to_EUR:    0.85   # USD → EUR exchange rate

.. _config-tariffs:

tariffs_dict
^^^^^^^^^^^^

Danish electricity grid tariffs (€/MWh). Applied to all grid imports/exports.

.. code-block:: yaml

   tariffs_dict:
     el_transmission_tariff: 9.92   # TSO tariff (Energinet)
     el_system_tariff:        6.84
     el_afgift:               45    # state electricity tax (øre/kWh)
     el_net_tariff_low:        2    # DSO tariff — off-peak
     el_net_tariff_high:       6    # DSO tariff — shoulder
     el_net_tariff_peak:      12    # DSO tariff — peak
     el_tariff_sell:          1.4   # tariff on electricity export
     NG_dso_tariff:           1.8   # natural gas DSO tariff
     NG_tso_tariff:           0.01  # natural gas TSO tariff

.. _config-rfnbos:

rfnbos_dict
^^^^^^^^^^^

Controls RFNBO (Renewable Fuels of Non-Biological Origin) compliance constraints.

.. code-block:: yaml

   rfnbos_dict:
     limit: 'price'              # 'price' | 'emissions' | 'unlimited' | 'disconnected'
     price_threshold:    20      # €/MWh — max electricity price for RFNBO compliance
     emission_threshold: 0.0648  # tCO₂/MWh — max grid intensity for RFNBO compliance

----

n_config.default.yaml
----------------------

Per-technology configuration for greenfield/brownfield optimisation.
Each entry sets the initial installed capacity, expansion allowance, cost factor,
and operational constraints (ramp limits, minimum load) for one technology group.

Key investment parameters: ``initial capacity``, ``expansion``, ``cost factor``,
``residual cost factor``, ``max capacity``.

The ``options:`` section at the bottom of this file controls external market
connections: biomass purchase markets, district heating sales, biochar and CO₂
sequestration credits, and electrical transformer sizing.

.. _brownfield-greenfield:

Greenfield and Brownfield configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three parameters jointly determine the investment mode for each technology:

.. list-table::
   :header-rows: 1
   :widths: 18 12 22 48

   * - ``initial capacity``
     - ``expansion``
     - ``residual cost factor``
     - Result
   * - ``0``
     - ``false``
     - any
     - **Technology absent.** Not added to the model. ``residual cost factor`` is ignored.
   * - ``0``
     - ``true``
     - ``0``
     - **Pure greenfield.** Only a new expandable component is built; the optimizer decides the capacity.
   * - ``0``
     - ``true``
     - ``> 0``
     - **Degenerate.** No existing plant exists, so ``residual cost factor`` has no effect — same as pure greenfield.
   * - ``> 0``
     - ``false``
     - ``0``
     - **Pure brownfield (sunk cost).** Existing capacity is fixed in the model, capital cost is fully sunk — no CAPEX charged to the objective.
   * - ``> 0``
     - ``false``
     - ``> 0``
     - **Brownfield with residual CAPEX.** Existing capacity is fixed; a fraction of the annualised capital cost is charged to the system cost (e.g. remaining loan repayments).
   * - ``> 0``
     - ``true``
     - ``0``
     - **Mixed — existing free, expandable.** Existing capacity dispatches at zero capital cost; additional capacity can be built at full cost.
   * - ``> 0``
     - ``true``
     - ``> 0``
     - **Mixed — existing with residual CAPEX, expandable.** Existing capacity carries a residual capital charge; additional capacity can be built on top at full cost.

**Parameter meanings**

``cost factor``
   Multiplier applied to the capital cost of **new** capacity (``expansion=true`` component).
   Used for cost sensitivity analysis: ``1.0`` = tech-data value, ``0.5`` = 50% cost reduction scenario.
   Does not affect existing capacity.

``residual cost factor``
   Fraction of the technology capital cost charged for **existing** (``EXI_``) capacity.
   Represents annualised residual CAPEX still to be recovered (e.g. a plant halfway through
   its financing period has ``residual cost factor ≈ 0.5``).
   ``0`` = sunk cost (default); ``1`` = full annualised cost charged.

**PyPSA implementation**

Internally, an ``EXI_<tech>`` component with ``residual cost factor > 0`` is built as
``p_nom_extendable=True`` with ``p_nom_min = p_nom_max = initial_capacity``.
This forces the LP variable to its fixed value, moving the capital cost term into
``n.objective_constant``.  ``n.statistics.capex()`` reads ``capital_cost × p_nom_opt``
after solving and correctly accounts for the residual charge in LCOP and TSC outputs.

**Example** — existing biogas plant, 40 % CAPEX still to recover, no new capacity allowed:

.. code-block:: yaml

   # config/n_config.yaml  (user override)
   biogas:
     initial capacity: 5.0   # MW CH4
     expansion: false
     residual cost factor: 0.4

----

plots_config.default.yaml
--------------------------

.. note::
   Full documentation coming soon. See inline comments in ``config/plots_config.default.yaml``.

Defines which network components are extracted and plotted after optimisation,
including capacity thresholds and the list of internal buses for shadow price plots.
