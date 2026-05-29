.. _guide-snakemake:

Running the Model with Snakemake
=================================

This guide walks through how to configure and run GreenBubble using the
Snakemake workflow, explains the default/override config pattern, and shows
a concrete example of creating a scenario run.

---

How the config system works
----------------------------

GreenBubble uses **three configuration files**, each with a ``*.default.yaml``
version committed to the repository:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - File
     - Purpose
   * - ``config/config.default.yaml``
     - Run settings: demands or price targets, technology flags, economics, solver
   * - ``config/n_config.default.yaml``
     - Per-technology capacity bounds, ramp limits, operational options
   * - ``config/plots_config.default.yaml``
     - Which components to plot and export after optimisation

The ``*.default.yaml`` files are the **committed base** — they define a working
scenario out of the box and should not be edited for individual runs.

To customise a run, you create a plain ``config/config.yaml`` (and optionally
``config/n_config.yaml``) containing **only the keys you want to change**.
Snakemake merges the default and override files at startup: every key in
``config.yaml`` replaces the corresponding key in ``config.default.yaml``;
everything else keeps its default value.

.. note::
   ``config/config.yaml`` and ``config/n_config.yaml`` are listed in
   ``.gitignore`` — they are your local workspace and are not committed by default.


Loading order (inside ``Snakefile``)::

   configfile: "config/config.default.yaml"   # always loaded

   if Path("config/config.yaml").exists():
       configfile: "config/config.yaml"        # merged on top if present


---

Quick start: run with defaults
--------------------------------

If no ``config/config.yaml`` exists, the model runs with the committed defaults.

**1. Dry-run** — preview what Snakemake would execute without running anything::

   snakemake -n

**2. Full run** with 4 parallel jobs::

   snakemake -j4

**3. Run with a single job** (useful for debugging)::

   snakemake -j1

Outputs land in ``outputs/single_analysis/<network_name>/``.

---

Example: creating a custom scenario
-------------------------------------

Suppose you want to run a **demand-driven** scenario for a site in southern
Denmark with a higher CO₂ cost, keeping everything else at the defaults.

**Step 1 — create your override file**

Create ``config/config.yaml`` with only the keys that differ:

.. code-block:: yaml

   # config/config.yaml  — only keys that differ from config.default.yaml

   run_name: high_co2_demand

   CO2_cost: 150          # €/t  (default: 100)

   latitude:  55.2        # Vejle
   longitude:  9.5

   targets:
     driver: 'demand'
     demand_CH4: 400000   # MWh_CH4/y
     demand_H2:  50000    # MWh_H2/y
     demand_meoh: 0

You do **not** need to repeat ``n_flags``, ``tariffs_dict``, ``optimization``,
or any other key — those are inherited from ``config.default.yaml``.

**Step 2 — dry-run to verify the scenario name**::

   snakemake -n

Snakemake prints the planned jobs and the output path, which encodes the
key parameters (flags, CO₂ cost, targets, year, run name).

**Step 3 — run the scenario**::

   snakemake -j4

Results are written to::

   outputs/single_analysis/B_H_RE_H2_MEOH_METH_SN_ST_CO2_150_tD_H2_50_MeOH_0_CH4_400_2024_El_0.1_DET_high_co2_demand/

---

Switching between scenarios
-----------------------------

Each scenario is just a different ``config/config.yaml``.  A simple way to
manage multiple scenarios is to keep named copies and swap them in:

.. code-block:: bash

   # save the current scenario
   cp config/config.yaml config/config.high_co2.yaml

   # switch to a different scenario
   cp config/config.price_mode.yaml config/config.yaml

   snakemake -j4

Because outputs are keyed to the scenario parameters in the folder name,
results from different scenarios never overwrite each other.

---

Overriding network component settings
---------------------------------------

If you also need to change a technology setting — for example, to allow
the heat pump to expand or to increase the biogas storage cap — create
``config/n_config.yaml`` with only those entries:

.. code-block:: yaml

   # config/n_config.yaml  — only keys that differ from n_config.default.yaml

   heat pump:
     expansion: true

   biogas storage:
     max capacity: 500    # MWh  (default: 200)

The ``options:`` block (district heating, biomass markets, biochar credits)
lives at the bottom of ``n_config.default.yaml`` and can be overridden the
same way:

.. code-block:: yaml

   options:
     DH:
       enable: true
       price: 35          # €/MWh  (default: 30)

---

Useful Snakemake flags
------------------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Flag
     - Effect
   * - ``-n`` / ``--dryrun``
     - Show planned jobs without executing
   * - ``-j N``
     - Use N parallel jobs (set to number of CPU cores)
   * - ``--forcerun <rule>``
     - Force re-execution of a specific rule
   * - ``--until <rule>``
     - Stop after a specific rule completes
   * - ``--config key=value``
     - Override a single top-level config key on the command line
   * - ``--rerun-incomplete``
     - Re-run jobs whose output files are incomplete

**Example — override a single key without editing any file**::

   snakemake -j4 --config CO2_cost=200

This is convenient for quick parameter sweeps but does not affect keys nested
under a parent (e.g. ``targets.driver``); use ``config/config.yaml`` for those.

---

See also
---------

- :ref:`configuration` — full reference for every config key
- :ref:`rules` — description of each Snakemake rule
- :ref:`wildcards` — how output paths are constructed from config values
- :ref:`guide-rolling-horizon` — dispatch-only runs on a fixed network
