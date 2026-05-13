.. _guide-rolling-horizon:

Rolling Horizon Dispatch Optimisation
======================================

The rolling horizon (RH) pipeline runs a **dispatch-only** optimisation on a
network whose capacities have already been fixed by a previous capacity
expansion run.  It is useful for:

- Evaluating how a fixed plant configuration performs in a different weather or
  price year (cross-year analysis).
- Running high-resolution dispatch on a large network without solving a full
  investment problem.

When rolling horizon is enabled, Snakemake targets the RH output directly and
the ``solve_network`` (capacity expansion) rule is **never triggered**.

---

How it works
------------

1. The OPT network produced by the capacity expansion run is loaded.
2. All capacities are fixed: ``p_nom = p_nom_opt``, ``p_nom_extendable = False``.
3. *(Cross-year only)* All year-dependent time series (wind/solar capacity
   factors, electricity and gas prices, RFNBO constraints, product sale prices)
   are replaced with data from ``rh_year``.
4. A rolling horizon dispatch solve is run: the year is split into overlapping
   windows of ``horizon`` hours, each solved sequentially with ``overlap`` hours
   of carry-over to avoid end-of-window artefacts.
5. The result is saved as a NetCDF network file containing full hourly dispatch
   time series.

---

Configuration
-------------

All settings live under the ``rolling_horizon`` key in ``config/config.yaml``:

.. code-block:: yaml

   rolling_horizon:
     enabled: false          # set true to activate the RH pipeline
     horizon: 168            # window size in hours (168 = 1 week)
     overlap: 24             # overlap between consecutive windows in hours
     network_path: null      # REQUIRED: path to the .nc OPT network to load
     rh_year: null           # optional: dispatch year (defaults to En_price_year)

.. list-table:: Parameters
   :header-rows: 1
   :widths: 20 10 70

   * - Parameter
     - Type
     - Description
   * - ``enabled``
     - bool
     - Activates the RH pipeline. When ``true``, ``rule all`` targets the RH
       output instead of the standard plots.
   * - ``horizon``
     - int
     - Rolling window size in hours. Typical values: 168 (week), 720 (month).
   * - ``overlap``
     - int
     - Hours of overlap between consecutive windows. Reduces boundary
       artefacts. A value of 24 is usually sufficient.
   * - ``network_path``
     - string
     - Absolute or relative path to the optimised ``.nc`` network.
       **Required** when ``enabled: true``.
   * - ``rh_year``
     - int / null
     - Year whose weather and price data are used for dispatch. If ``null``
       or equal to ``En_price_year``, the same data as the capacity expansion
       run is used.

---

Same-year mode
--------------

Use this when you want to run dispatch on the same year as the capacity
expansion without re-solving the full investment problem.

.. code-block:: yaml

   rolling_horizon:
     enabled: true
     horizon: 168
     overlap: 24
     network_path: outputs/single_analysis/.../networks/..._OPT.nc
     rh_year: null

---

Cross-year mode
---------------

Use this to evaluate the optimised capacities under a different year's
weather and energy prices.  The network topology and component sizes remain
exactly as in the OPT network; only the time series data is replaced.

.. code-block:: yaml

   rolling_horizon:
     enabled: true
     horizon: 168
     overlap: 24
     network_path: outputs/single_analysis/.../networks/..._OPT.nc
     rh_year: 2019   # dispatch with 2019 wind/solar CFs and electricity prices

The preprocessed inputs for ``rh_year`` (``data/Inputs_2019/``) must exist.
If they do not, Snakemake will run ``preprocess_inputs`` for that year
automatically before the RH solve.

---

Running
-------

With ``enabled: true`` and ``network_path`` set, run Snakemake as normal:

.. code-block:: bash

   snakemake --cores 4

Snakemake's DAG will target only the RH output.  No capacity expansion rule
is triggered.

---

Output
------

The result is saved as a NetCDF file at:

.. code-block:: text

   {outputs_folder}/{network_name}/networks/rolling_horizon/{network_name}_RH.nc

Load it with PyPSA to access the full dispatch time series:

.. code-block:: python

   import pypsa
   n = pypsa.Network("path/to/network_RH.nc")

   n.generators_t.p                    # generator dispatch [MW]
   n.links_t.p0                        # link flow at bus0 [MW]
   n.stores_t.e                        # store energy content [MWh]
   n.storage_units_t.state_of_charge   # storage state of charge [MWh]

---

What changes between years
--------------------------

When ``rh_year`` differs from ``En_price_year``, the following are replaced
with data from ``data/Inputs_{rh_year}/``:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Data
     - Source file
   * - Wind capacity factors
     - ``CF_wind.csv``
   * - Solar capacity factors
     - ``CF_solar.csv``
   * - Electricity spot price (buy / sell)
     - ``Elspotprices_input.csv``
   * - Natural gas price
     - ``NG_price_year_input.csv``
   * - RFNBO hourly import constraint
     - derived from electricity price
   * - Product sale prices (bioCH4, etc.)
     - derived from NG price + CO₂ cost

Capital costs, efficiencies, network topology, and component capacities are
**not** changed — they come from the OPT network.

----

Demands and flexibility stores in rolling horizon
--------------------------------------------------

See also: :ref:`guide-demands`.

Flat and profile modes
^^^^^^^^^^^^^^^^^^^^^^

The delivery store carries a cyclic SOC across the full year in the capacity
expansion run. In RH the store is made **non-cyclic** (``e_cyclic = False``)
so each window starts from the SOC carried forward from the previous window
rather than an arbitrary optimised value. The store capacity is left at the
optimised value.

Annual point-load products (bins_flat / bins_profile with n_bins = 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a product has a single annual delivery the delivery buffer store
accumulates the full year's production before releasing it at year-end. This
is incompatible with rolling horizon:

- With ``e_cyclic = False`` the optimizer front-loads all production in the
  first window, ending with a massive surplus.
- With ``e_cyclic = True`` PyPSA treats the initial SOC as a free optimisation
  variable, injecting phantom energy at window start.

The RH solver automatically detects these annual point-load stores (>95% of
throughput concentrated in a single timestep) and:

1. **Redistributes** the demand to a flat hourly rate for the duration of the
   RH run — the load shape changes but the annual total is preserved.
2. **Caps** the delivery store to ``2 × (annual / n_windows)`` MWh, preventing
   multi-window carry-over while keeping a one-window buffer.

Weekly / monthly bins (n_bins ≥ 12)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For higher delivery frequencies no redistribution is needed. Each bin's
delivery fits within a single rolling window, so the store fills and empties
naturally within each window. The store cap is the per-bin size set during
network building.
