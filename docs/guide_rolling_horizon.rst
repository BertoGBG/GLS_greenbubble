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
