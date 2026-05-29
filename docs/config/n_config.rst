.. _n-config-yaml:

n_config.yaml
=============

Technology-level configuration for greenfield and brownfield optimisation.
Each entry defines the expansion settings and operational constraints for one
technology group.

Location: ``config/n_config.yaml``

All technologies inherit from a common ``base`` block unless overridden:

.. code-block:: yaml

   base:
     initial capacity: 0     # pre-installed capacity (MW or t/h), no annualised cost
     expansion: true          # allow capacity expansion
     cost factor: 1           # multiplier on technology cost from the database
     max capacity: .inf       # upper bound on total installed capacity

Units refer to bus0 of the component in the PyPSA network (except for ``biogas``,
which is in MW\ :sub:`CH4`).

Key parameters
--------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``initial capacity``
     - Pre-installed capacity added at zero annualised cost (brownfield asset).
   * - ``expansion``
     - If ``false``, the technology is fixed at ``initial capacity``.
   * - ``cost factor``
     - Scales the capital and/or O&M cost from the technology database.
   * - ``max capacity``
     - Hard upper bound on total capacity (MW or t/h).
   * - ``max hours``
     - Energy-to-power ratio for storage (MWh/MW). Used for batteries and stores.
   * - ``min load``
     - Minimum part-load fraction (0–1). Enforces a minimum dispatch level.
   * - ``ramp limit up/down``
     - Maximum ramp rate (fraction of capacity per hour).

Notable entries
---------------

**biogas** — fixed at 30 MW\ :sub:`CH4` (``expansion: false``), representing the
existing biogas plant at GreenLab Skive.

**electrolysis** — fully expandable alkaline electrolyser.

**biomethanation / methanation** — biological and catalytic routes with ramp limits
and minimum loads reflecting real operational constraints.

**battery** — includes the inverter; ``max hours: 10`` sets the energy-to-power ratio.

**biomass belt dryer** — has a ``cost factor: 1.2`` override and a minimum load of 20%.

All other technologies default to greenfield (``initial capacity: 0``,
``expansion: true``, ``max capacity: inf``).
