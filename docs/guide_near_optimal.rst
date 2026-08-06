.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _guide-near-optimal:

Near-Optimal Space Exploration (MGA)
====================================

A cost-optimal model returns a *single* design. In practice there are usually
many **near-optimal** designs — within a few percent of the optimal cost — that
differ substantially in their technology mix. Exploring this *near-optimal
feasible space* reveals **investment flexibility** ("how much could this
capacity change while staying cost-competitive?") and identifies **must-have**
and **must-avoid** technologies.

GreenBubble implements this via **Modelling to Generate Alternatives (MGA)**,
building on PyPSA's native MGA API. The methodology follows
`Neumann & Brown (2021) <https://doi.org/10.1016/j.epsr.2020.106690>`_
(near-optimal space) and
`Grochowicz et al. (2023) <https://doi.org/10.1016/j.eneco.2022.106496>`_
(robustness via intersection of per-year spaces).

The feature runs **after** a network has been solved (it consumes the
``*_OPT.nc`` network) and never alters the normal solve.

---

Three tiers
-----------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Tier
     - What it produces
   * - **1 — ranges**
     - Per-technology min/max installed capacity within the cost budget.
       ``min > 0`` ⇒ *must-have*; ``max ≈ 0`` ⇒ *must-avoid*.
   * - **2 — hull**
     - Many search directions in the selected capacity space → extreme
       points → convex hull (the approximated near-optimal polytope) and
       2-D projection plots.
   * - **3 — robustness**
     - Intersect the per-year near-optimal hulls and take the **Chebyshev
       centre** (deepest interior point) as a robust design that stays
       near-optimal across all years.

---

Mathematical background
-----------------------

Write the capacity-expansion model as the linear program

.. math::

   \min_{x}\; c^\top x \quad\text{s.t.}\quad Ax \le b,

where :math:`x` collects all decision variables (investment **and** operation),
:math:`c^\top x` is the total system cost, and :math:`c_\mathrm{opt}` is its optimal
value. Following Neumann & Brown (2021) and Grochowicz et al. (2023), the
**ε-near-optimal feasible space** is the slice of the feasible region whose cost is
within a slack :math:`\varepsilon` of the optimum:

.. math::

   X_\varepsilon \;=\; \{\, x : Ax \le b,\; c^\top x \le (1+\varepsilon)\,c_\mathrm{opt} \,\}.

GreenBubble adds exactly this budget to the model (``slack`` = :math:`\varepsilon`),
with a sign-robust form so it is also valid when the optimum is negative — which
happens in **price-driven** runs where net sales revenue exceeds cost
(:math:`c_\mathrm{opt} < 0`):

.. math::

   c^\top x \;\le\; c_\mathrm{opt} + \varepsilon\,\lvert c_\mathrm{opt}\rvert
   \;=\;
   \begin{cases}
     (1+\varepsilon)\,c_\mathrm{opt}, & c_\mathrm{opt} \ge 0 \quad\text{(demand mode)}\\
     (1-\varepsilon)\,c_\mathrm{opt}, & c_\mathrm{opt} < 0 \quad\text{(price mode)}
   \end{cases}

Here :math:`c_\mathrm{opt}` is the **optimised objective** ``n.objective``. For a
**stochastic** network it is automatically the probability-weighted *expected* cost
:math:`\sum_s p_s\, c_s^\top x_s`, so the same expression defines the near-optimal
space consistently for demand, price and stochastic runs (and needs no per-scenario
cost statistics). The anchor is the LP objective itself — the quantity the model
minimises — not a re-derived ``capex + opex``, matching the cited papers' definition.

**Projection (dimension reduction).** We do not work with the full-dimensional
:math:`X_\varepsilon`. We project onto a handful of *technology capacity* coordinates
:math:`y = \sigma(x)` (each :math:`y_i` = summed installed capacity of one technology;
the σ-map of Grochowicz et al. §2.2). Exploring a single direction :math:`d` in this
reduced space means solving the original model once with the objective swapped to
:math:`\pm d^\top \sigma(x)` subject to the budget above (MGA / *modelling to generate
alternatives*). Tier 1 uses the unit axes (min/max each technology); Tier 2 sweeps many
directions to trace the projected polytope; Tier 3 intersects per-year polytopes and
takes the **Chebyshev centre** :math:`\max\{r : a_i^\top y + r\lVert a_i\rVert \le b_i\}`
as the most robust design (Grochowicz et al. §2.3–2.4).

---

Dimensions
----------

Each *dimension* is a technology, measured as its **installed capacity (MW)**.
The set of selectable dimensions is **auto-derived** from the network: a
technology is offered if and only if it has at least one *extendable*, costed
(``capital_cost > 0``) component and its name is a known technology in
``n_config``. Nothing is hardcoded — flipping a technology to ``expansion: true``
in ``n_config.yaml`` automatically makes it available as a dimension.

Select a subset (typically 4–5) by name. An unknown name raises an error that
lists the dimensions available for your network setup.

---

Configuration
-------------

Add an ``mga`` block to ``config/config.yaml``:

.. code-block:: yaml

   mga:
     enabled:      true
     network_path: ''            # '' = current-config network; or a path to any *_OPT.nc
     dimensions:   [onwind, solar, electrolysis, methanolisation, battery]
     slack:        0.05          # cost budget = c_opt + slack·|c_opt|
     n_directions: 40            # 0 ⇒ Tier 1 only (no hull)
     direction_sampling: halton  # halton | evenly_spaced | random
     seed:         42
     robustness:
       enabled: false
       years:                    # mirrors stochastic.scenarios structure
         '2023': {CO2_cost: 100, CO2_cost_ref_year: 0}
         '2024': {CO2_cost: 100, CO2_cost_ref_year: 0}
       cost_bound: max

``slack`` is the cost relaxation :math:`\varepsilon`: the budget bounds the optimised
objective as ``c·x ≤ c_opt + slack·|c_opt|`` (see *Mathematical background* above).
The ``|c_opt|`` makes it correct for demand (``c_opt > 0`` → ``(1+slack)·c_opt``),
price (``c_opt < 0`` → ``(1−slack)·c_opt``), and stochastic (``c_opt`` = expected cost)
runs alike.

See :ref:`config-mga` for the full parameter reference.

---

Running
-------

With ``mga.enabled: true``, the NOS step is part of the default workflow::

   snakemake -n                          # NOS rule appears after solve_network
   snakemake --cores 4                   # solve + explore
   snakemake --cores 4 explore_near_optimal   # only the NOS step (needs *_OPT.nc)

**Choosing which solved network to explore**

- ``network_path: ''`` (default) — NOS explores the network described by the
  *current config* (the same ``{network}`` as ``solve_network`` / ``plot_results``),
  solving it in the same run if it is not already present. Outputs go to
  ``{output}/{network}/nos/``.
- ``network_path: <path/to/..._OPT.nc>`` — NOS loads that **pre-solved** network
  directly (no re-solve), like ``rolling_horizon.network_path``. Stochastic vs
  deterministic is **auto-detected** from the file, so you do not need a flag.
  Outputs go to a ``nos/`` folder *next to* the network.

  When pointing at a network solved under a *different* config, keep the relevant
  config keys consistent with how it was built — in particular ``n_flags`` and
  ``max_RE_to_grid``, which drive the custom RE-to-grid constraint re-applied
  during exploration (it skips gracefully if the buses don't match).

Outputs land in ``{output}/{network}/nos/`` (or next to ``network_path``):

- ``ranges.csv`` — Tier 1 min/max/optimal per dimension + must-have/avoid flags
- ``points.csv`` — Tier 2 extreme points (one row per solved direction)
- ``summary.json`` — slack, c_opt, hull volume, robustness centre, …
- ``plots/`` — ``nos_ranges.png``, ``nos_hull.png``, ``nos_robustness.png``

---

How it works
------------

Each tier solves the original capacity-expansion model repeatedly with an
alternative objective (minimise / maximise a technology, or maximise the
projection along a search direction) subject to the near-optimal cost budget.

.. note::

   PyPSA's built-in ``optimize_mga*`` helpers build their own model and do **not**
   run any ``extra_functionality`` hook, so GreenBubble's custom constraints (e.g.
   the maximum RE-to-grid constraint) would be silently dropped — yielding a
   *looser*, incorrect near-optimal space. GreenBubble therefore reproduces the
   MGA step manually and re-injects every custom constraint via
   :func:`scripts.helpers.apply_custom_constraints`, keeping NOS results
   consistent with the normal solve.

---

Limitations
-----------

- **NOS is time-demanding — Gurobi is the preferred solver.** Every tier works by
  re-solving the *entire* model, once per direction explored: Tier 1 solves twice per
  dimension, Tier 2 adds one solve per sampled direction (default 40, plus axis
  corners), and Tier 3 repeats both per robustness year. Even at a coarsened temporal
  resolution this is dozens to well over a hundred full solves for one NOS run, so
  solver speed matters far more here than for a single ordinary solve.
- **Solver choice also affects correctness, not just speed.** MGA objectives place
  cost on only one or two capacity variables, so the LP is highly *degenerate* (many
  equally-optimal vertices). In practice this has shown up as more than an accuracy
  tradeoff: ``highs-fast`` (barrier without crossover) has been observed to return
  an outright **infeasible** status with a large primal-dual objective error on a
  real GreenBubble network — silently failing a direction rather than just
  under-reporting its extent. ``gurobi-barrier-fast`` did **not** show this failure
  across 100+ direction solves in testing and remained fast, which is why it is
  **the recommended default for NOS** (used throughout this guide's examples and in
  :ref:`tutorial-5-near-optimal`). Without a Gurobi licence, fall back to a profile
  that ends at a vertex — ``highs-simplex`` or ``highs-default`` — which avoids the
  failure mode at the cost of noticeably slower solves.
- The hull is **approximated** from a finite set of search directions; more
  ``n_directions`` gives a better approximation at the cost of more solves
  (each direction is a full model solve, run serially).
- The number of dimensions is best kept small (4–5): the hull's complexity and
  the number of directions needed grow quickly with dimensionality.
- Robustness rebuilds and solves one network per year, then explores each — it
  is the most expensive tier.

---

See also
--------

- `PyPSA MGA example <https://docs.pypsa.org/latest/examples/mga/>`_
- `PyPSA near-optimal space example <https://docs.pypsa.org/latest/examples/near-opt-space/>`_
- :ref:`config-mga` — full configuration reference
- :mod:`scripts.near_optimal` — API reference
