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

Each product also has a demand **shape** and optional **flexibility store**:

.. code-block:: yaml

   targets:
     CH4_demand_mode: flat          # flat | profile | bins_flat | bins_profile
     CH4_bins:         1            # number of equal bins (bins_* modes only)
     CH4_flexibility:  0.00         # fraction of annual demand used as store e_nom_max
     CH4_profile:      null         # path to CSV seasonal profile; null = built-in NG_DK
     H2_demand_mode:   profile
     H2_bins:          12
     H2_flexibility:   0.1
     H2_profile:       data/common/NG_demand_DK_profile.csv
     MeOH_demand_mode: bins_profile
     MeOH_bins:        2
     MeOH_flexibility: 0.0
     MeOH_profile:     data/common/NG_demand_DK_profile.csv
     demand_store_buffer: 0.5       # extra headroom for bins_profile stores

See :ref:`guide-demands` for a full explanation of demand modes, flexibility stores,
and seasonal profiles.

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

.. _config-nflags-opt:

n_flags_opt — output control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Controls which post-solve artefacts are generated.

.. code-block:: yaml

   n_flags_opt:
     print:  true    # save SVG of the optimal network topology
     export: true    # export the solved network to a .nc file
     plot:   true    # run the full post-processing plot suite

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

.. _config-clustering:

clustering
^^^^^^^^^^

Reduces the number of snapshots by resampling hourly input data to a coarser
time interval before the network is built.

.. code-block:: yaml

   clustering:
     temporal:
       resolution: false   # false | "4h" | "8h" | "24h" | ...

``false`` (default) keeps native 1-hour resolution.
Any pandas offset string that represents an interval ≥ 1 hour is accepted
(``"4h"``, ``"8h"``, ``"24h"`` are the most common choices).
Sub-hourly strings raise ``NotImplementedError``.

Effect on snapshot count:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - resolution
     - snapshots
     - typical speed-up
   * - ``false`` (1 h)
     - 8 760
     - baseline
   * - ``"4h"``
     - 2 190
     - ~4×
   * - ``"8h"``
     - 1 095
     - ~8×
   * - ``"24h"``
     - 365
     - ~20×

**Resampling rules**

- Prices, capacity factors, and demands → ``mean()`` over the interval.
- Store upper bounds (``e_max_pu``) → ``min()`` over the interval (conservative).
- Store lower bounds (``e_min_pu``) → ``max()`` over the interval (conservative).
- Snapshot weightings → ``sum()`` so annual energy totals are preserved.

**Incompatibilities**

- ``rolling_horizon.enabled: true`` — resampling is skipped with a warning;
  the RH solver operates on the full hourly network provided via ``network_path``.
- ``stochastic.stochastic: true`` — allowed but issues a warning; each scenario
  is resampled independently, so inter-scenario sub-period correlations are lost.

See :ref:`guide-temporal-resolution` for worked examples and advice on choosing
a resolution.

.. _config-rolling-horizon:

rolling_horizon
^^^^^^^^^^^^^^^

Dispatch-only solve on a pre-existing fixed-capacity network.
When enabled, the capacity-expansion ``solve_network`` rule is bypassed entirely.

.. code-block:: yaml

   rolling_horizon:
     enabled:      false
     horizon:      168    # window size in hours (e.g. 168 = 1 week)
     overlap:       72    # overlap in hours between consecutive windows
     network_path: ''     # REQUIRED — path to a solved .nc network
     rh_year:      null   # null = same as En_price_year; or an integer year

``network_path`` is required when ``enabled: true``.
Setting ``rh_year`` to a different year than ``En_price_year`` replaces all
time-varying inputs (prices, capacity factors) with data from that year.

See :ref:`guide-rolling-horizon` for the full workflow, cross-year analysis,
committable dispatch, and cost comparison outputs.

.. _config-optimization:

optimization
^^^^^^^^^^^^

.. code-block:: yaml

   optimization:
     solver: 'gurobi'                  # 'gurobi' | 'highs'
     solver_profile: 'gurobi-barrier-fast'  # preset from scripts/solver_profiles.py
     collect_all_duals: true           # save dual variables for shadow price analysis
     return_model: true                # return Linopy model object after solving
     overrides: null                   # optional raw solver parameters (dict)
     zero_threshold_MW: 0.01           # MW — components built below this are zeroed out

Solver profiles are defined in ``scripts/solver_profiles.py``.
Common Gurobi profiles: ``gurobi-barrier-fast``, ``gurobi-simplex``.
``zero_threshold_MW`` removes solver-noise artefacts: any extendable component
whose ``p_nom_opt`` (or ``e_nom_opt``) is strictly below this value is treated
as "not built" — its optimal capacity and result time series are zeroed before
export.

.. _config-economics:

Economics
^^^^^^^^^

.. code-block:: yaml

   year_investment:    2030   # target year for new-capacity technology costs
   amortization_period: null  # null = use each technology's technical lifetime
   discount_rate:      0.07   # real discount rate (constant EUR, excludes inflation)
   EUR_to_DKK:         7.46   # EUR → DKK exchange rate
   USD_to_EUR:         0.85   # USD → EUR exchange rate

See :ref:`economics` for a full explanation of how these parameters interact with
technology costs, brownfield initial conditions, and the annuity formula.

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
``construction_year``, ``remaining_investment_fraction``, ``max capacity``.

The ``options:`` section at the bottom of this file controls external market
connections: biomass purchase markets, district heating sales, biochar and CO₂
sequestration credits, and electrical transformer sizing.

See :ref:`economics-brownfield` for a full explanation of how brownfield
parameters are combined to compute residual annual capital charges.

.. _brownfield-greenfield:

Greenfield and Brownfield configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Three parameters jointly determine the investment mode for each technology:

.. list-table::
   :header-rows: 1
   :widths: 18 12 28 42

   * - ``initial capacity``
     - ``expansion``
     - ``remaining_investment_fraction``
     - Result
   * - ``0``
     - ``false``
     - any
     - **Technology absent.** Not added to the model.
   * - ``0``
     - ``true``
     - ``0``
     - **Pure greenfield.** Only a new expandable component is built.
   * - ``> 0``
     - ``false``
     - ``0``
     - **Pure brownfield (sunk cost).** Existing capacity is fixed; no CAPEX charged.
   * - ``> 0``
     - ``false``
     - ``> 0``
     - **Brownfield with residual CAPEX.** Existing capacity is fixed; annual charge = ``rif × investment(construction_year) × annuity(r, amortization_period)``.
   * - ``> 0``
     - ``true``
     - ``0``
     - **Mixed — existing free, expandable.** Existing capacity at zero capital cost; additional capacity can be built at full cost.
   * - ``> 0``
     - ``true``
     - ``> 0``
     - **Mixed — existing with residual CAPEX, expandable.**

**Parameter meanings**

``cost factor``
   Multiplier applied to the capital cost of **new** capacity (``expansion=true`` component).
   Used for cost sensitivity analysis: ``1.0`` = tech-data value, ``0.5`` = 50% cost reduction scenario.

``construction_year``
   Year the existing plant was built.  Used to look up the investment cost at the actual
   build year (technology costs differ between years due to learning curves).
   ``null`` defaults to ``year_investment - 10``, capped at 2020.

``remaining_investment_fraction``
   Fraction of ``investment(construction_year)`` still to be financially recovered.
   ``0`` = fully amortised / sunk cost (default); ``1`` = the full original investment
   is still outstanding.

**PyPSA implementation**

Internally, an ``EXI_<tech>`` component with ``remaining_investment_fraction > 0`` is built as
``p_nom_extendable=True`` with ``p_nom_min = p_nom_max = initial_capacity``.
This forces the LP variable to its fixed value, moving the capital cost into
``n.objective_constant``.  ``n.statistics.capex()`` correctly accounts for the
residual charge in LCOP and TSC outputs.

**Example** — existing biogas plant, 40 % of original investment still outstanding,
built in 2022, no new capacity allowed:

.. code-block:: yaml

   # config/n_config.yaml  (user override)
   biogas:
     initial capacity: 5.0          # MW CH4
     expansion: false
     construction_year: 2022
     remaining_investment_fraction: 0.4

----

plots_config.default.yaml
--------------------------

.. note::
   Full documentation coming soon. See inline comments in ``config/plots_config.default.yaml``.

Defines which network components are extracted and plotted after optimisation,
including capacity thresholds and the list of internal buses for shadow price plots.
