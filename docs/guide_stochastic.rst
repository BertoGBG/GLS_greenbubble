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
                            # electricity price — see :ref:`config-co2-pricing`
       '2022': 0
       '2023': 0
       '2024': 0
       '2025': 0
     EVPI: true        # also compute Expected Value of Perfect Information

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

EVPI
----

When ``EVPI: true``, GreenBubble runs one additional deterministic solve per
scenario (with perfect foresight for that year) and computes the
**Expected Value of Perfect Information**:

.. code-block:: text

   EVPI = E[cost under perfect information] − cost of stochastic solution

EVPI is written to ``networks/evpi.yaml`` in the output folder.
It is automatically disabled when ``stochastic: false``.

---

Limitations
-----------

- **MILP (committable links)** is incompatible with stochastic mode — PyPSA's
  stochastic multi-scenario LP requires a pure LP (no binary variables).
  Set ``committable: false`` for all technologies when using stochastic mode.
  Note: committable + extendable *is* supported in deterministic capacity
  expansion via PyPSA's big-M formulation; see
  `committable-extendable example <https://docs.pypsa.org/latest/examples/committable-extendable/>`_.
- **Temporal resampling** with stochastic mode issues a warning (scenarios are
  resampled independently — see :ref:`guide-temporal-resolution`).

---

See also
--------

- `PyPSA stochastic optimisation <https://docs.pypsa.org/v1.0.0/user-guide/optimization/stochastic/#mathematical-formulation>`_ — mathematical background
- :ref:`config-stochastic` — full configuration reference
- :func:`scripts.create_stoch_scenarios.create_scenarios` — API reference
