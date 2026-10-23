.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _guide-stochastic:

Stochastic Optimisation
========================

GreenBubble supports **two-stage stochastic capacity expansion**: a single LP
couples multiple annual scenarios (each with its own time series) so that
investment decisions are shared across scenarios while dispatch is
scenario-specific.

For the mathematical formulation see the
`PyPSA stochastic optimisation documentation <https://docs.pypsa.org/v1.0.0/user-guide/optimization/stochastic/#mathematical-formulation>`_.

---

Configuration
-------------

Enable stochastic mode in ``config/config.yaml``:

.. code-block:: yaml

   stochastic:
     stochastic: true
     scenarios:
       '2022': 0.10    # year: probability (must sum to 1)
       '2023': 0.30
       '2024': 0.30
       '2025': 0.30
     CO2_cost_s:       # CO₂ cost (€/t) per scenario
       '2022': 80
       '2023': 100
       '2024': 100
       '2025': 120
     CO2_cost_ref_year_s:  # CO₂ cost already embedded in the historical
                            # electricity price; see :ref:`config-co2-pricing`
       '2022': 0
       '2023': 0
       '2024': 0
       '2025': 0
     EVPI: true        # also compute Expected Value of Perfect Information
     VSS: false        # also compute the Value of the Stochastic Solution

Each scenario year must have preprocessed input data.  Snakemake downloads
and preprocesses all scenario years automatically before building the coupled
network.

See :ref:`config-stochastic` for the full parameter reference.

---

Running
-------

No special flags are needed — just run Snakemake as usual::

   snakemake -n      # verify all scenario years appear in the plan
   snakemake --cores 4

The stochastic token ``STC`` is appended to the output folder name instead
of ``DET``, so stochastic and deterministic results never overwrite each other.

---

EVPI and VSS
------------

Both compare the stochastic solution with deterministic ones. All networks
are taken from the stochastic network before it is solved, with
``n.get_scenario()``, so they use exactly its scenario data.

.. list-table::
   :header-rows: 1
   :widths: 12 58 30

   * - Value
     - Problem solved
     - Extra solves
   * - RP
     - The stochastic problem: one design for all scenarios
     - –
   * - E_WS
     - Wait-and-see: each scenario on its own, with its own design;
       probability-weighted mean of the costs
     - one per scenario (``EVPI``)
   * - EV
     - Expected value: one deterministic problem whose time series are the
       probability-weighted mean over scenarios
     - one (``VSS``)
   * - EEV
     - The EV design, capacities fixed, dispatched in each scenario;
       probability-weighted mean of the costs
     - one per scenario (``VSS``)

.. math::

   \mathrm{EVPI} = \mathrm{RP} - \mathrm{E\_WS} \ge 0, \qquad
   \mathrm{VSS} = \mathrm{EEV} - \mathrm{RP} \ge 0

EVPI is what perfect foresight of the scenario would be worth. VSS is what
the stochastic design saves against a design for average conditions. Both
use total annual system cost. In EEV, load shedding at
``targets.load_shedding_cost`` covers any demand the EV design cannot meet in
a scenario.

The results go to ``csv/value_of_information.csv``: WS and EEV objectives
per scenario, then RP, E_WS, EVPI, EV, EEV and VSS. The WS, EV and EEV
networks are exported next to the stochastic one (``*_WS_<year>_OPT.nc``,
``*_EV_OPT.nc``, ``*_EEV_<year>_OPT.nc``). Both options are disabled when
``stochastic: false``. With unit commitment the solves are MILPs; set a MIP
gap well below the values you expect, or EVPI and VSS are lost in the gap.

---

Limitations
-----------

- **Unit commitment (committable links)** works in stochastic mode from
  PyPSA 1.4, also for extendable technologies. The problem becomes a MILP that
  grows with the number of scenarios. EVPI is only meaningful when the MIP gap
  is well below it: set ``optimization.overrides: {MIPGap: 1e-4}`` or tighter.
- **Ramp limits** work in stochastic mode from PyPSA 1.3 on. Earlier
  releases (1.0.7 to 1.2.4) failed to build ramp-limit constraints on a
  scenario network, so older configurations set ``ramp limit up`` /
  ``ramp limit down`` to ``null`` for stochastic runs. That is no longer
  needed.
- **Temporal resampling** with stochastic mode issues a warning (scenarios are
  resampled independently; see :ref:`guide-temporal-resolution`).

---

See also
--------

- `PyPSA stochastic optimisation <https://docs.pypsa.org/v1.0.0/user-guide/optimization/stochastic/#mathematical-formulation>`_ — mathematical background
- :ref:`config-stochastic` — full configuration reference
- :func:`scripts.create_stoch_scenarios.create_scenarios` — API reference
