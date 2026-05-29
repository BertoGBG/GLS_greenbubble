.. _methods:

Analytical Methods
==================

This page describes the analytical and mathematical methods available in
GreenBubble beyond the core LP.  Each method is optional and controlled by
its own configuration block.

---

.. _methods-stochastic:

Stochastic optimisation
------------------------

GreenBubble can solve a **two-stage stochastic linear programme** across
multiple price/weather scenarios (e.g. different market years).

**Motivation**

Capacity expansion decisions must be made before knowing which price year
will materialise.  A stochastic formulation captures this uncertainty
by coupling all scenarios in a single LP, ensuring that the first-stage
capacity decisions are feasible and optimal across all scenarios.

**Formulation**

- **First stage** — capacity variables (``p_nom``) are shared across all
  scenarios; they represent investment decisions made before uncertainty resolves.
- **Second stage** — dispatch variables (``p(t)``) are scenario-specific;
  the optimiser can dispatch differently in each scenario year.
- **Objective** — minimise expected total cost:
  :math:`\sum_s \pi_s \cdot C_s(x, y_s)`, where :math:`\pi_s` is the
  probability of scenario *s*, *x* are first-stage capacity decisions, and
  *y_s* are second-stage dispatch decisions.

The coupled network is built by ``scripts/create_stoch_scenarios.py``:
scenario-specific time series are attached as extra ``snapshots`` in separate
PyPSA sub-networks that share the same capacity variables.

**EVPI**

When ``stochastic.EVPI: true``, an additional deterministic solve is run
for each scenario individually (perfect-information runs).  The
**Expected Value of Perfect Information** is then:

:math:`\text{EVPI} = \text{EV(perfect info)} - \text{EV(stochastic solution)}`

A large EVPI indicates that the uncertainty in scenario realisations
strongly influences the optimal design.

**Configuration** — see :ref:`config-stochastic` for all parameters.

---

.. _methods-rolling-horizon:

Rolling horizon dispatch
-------------------------

See the dedicated guide: :ref:`guide-rolling-horizon`.

The rolling horizon method runs a **dispatch-only** optimisation on a
capacity-fixed network.  It is used to evaluate how a designed plant performs
in a different weather or price year, or to obtain high-resolution dispatch
profiles for a given design.

The year is divided into overlapping windows of ``horizon`` hours, solved
sequentially.  The ``overlap`` parameter avoids end-of-window artefacts by
carrying forward the storage state from the overlapping portion.

---

.. _methods-rfnbo:

RFNBO compliance
-----------------

The EU Renewable Fuels of Non-Biological Origin (RFNBO) delegated acts
impose requirements on electrolytic hydrogen to be certified as renewable.
GreenBubble implements two constraint modes, controlled by ``rfnbos_dict.limit``:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Mode
     - Constraint applied
   * - ``price``
     - Electrolysis can only operate when the electricity spot price is below
       ``price_threshold`` (€/MWh) — a proxy for high-renewable-share hours
   * - ``emissions``
     - Electrolysis is restricted to hours when the grid emission intensity
       is below ``emission_threshold`` (tCO₂/MWh) — direct temporal correlation
   * - ``unlimited``
     - No RFNBO constraint; electrolysis can draw from the grid freely
   * - ``disconnected``
     - Electrolysis is fully disconnected from the grid; must be covered
       entirely by on-site renewables

The price and emission time series are downloaded per year from
`Energi Data Service <https://www.energidataservice.dk>`_.

---

.. _methods-shadow-prices:

Shadow prices and internal market
-----------------------------------

The dual variables of the **bus balance constraints** in the LP are the
**shadow prices** (marginal values) of each energy carrier at each hour.
In economic terms they represent the internal market clearing price: what
it would cost to produce one additional MWh (or tonne) of that carrier at
that moment.

GreenBubble extracts and exports shadow prices for all internal buses
(electricity, H₂, CO₂, biomethane, methanol, heat levels).  These are used
in the economic analysis module (``scripts/plots.py``) to:

- Decompose the system value created at each node
- Identify binding constraints (hours where prices spike)
- Assess the implicit cross-subsidies between co-located partners

The list of buses to analyse is configured in ``plots_config.default.yaml``
under ``plotting.bus_list_mp``.

See :ref:`guide-economic-analysis` for interpretation and post-processing.

---

.. _methods-shapley:

Shapley value cost allocation
-------------------------------

.. note::
   Full documentation coming soon.

When multiple industrial partners share infrastructure (grid connection,
storage, compression), the total system cost is lower than the sum of each
partner building independently.  The **Shapley value** from cooperative game
theory provides a fair allocation of these savings.

GreenBubble computes the Shapley value by:

1. Enumerating all coalitions (subsets) of partners
2. Running a separate optimisation for each coalition (with only the
   corresponding ``n_flags`` active)
3. Computing marginal contributions and averaging over all orderings

The number of optimisation runs grows as :math:`2^N` in the number of
partners *N*, so this is typically run for N ≤ 5.  Results are used to
allocate shared infrastructure costs fairly among partners.
