.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _tutorial-5-near-optimal:

Tutorial 5 — Near-Optimal Space Exploration (MGA)
===================================================

Every previous tutorial reports **one** cost-optimal design. But real
investment decisions are made under cost and forecast uncertainty, so a
design that is only 1-5 % more expensive is, in practice, just as good a
candidate. **Modelling to Generate Alternatives (MGA)** explores that whole
neighbourhood of "almost as good" designs instead of stopping at the single
cheapest one — see :ref:`methods-near-optimal` for the underlying theory and
the original papers.

We reuse the **greenfield, demand-driven** plant from :ref:`tutorial-1-greenfield`
and add an ``mga`` block on top.

.. contents:: On this page
   :local:
   :depth: 1

---

1 · The idea, in one picture
-----------------------------

A cost-optimal solve gives one point. Allow the total cost to rise by a
small slack :math:`\varepsilon` above that optimum,

.. math::

   c^\top x \;\le\; c_\mathrm{opt} + \varepsilon\,\lvert c_\mathrm{opt}\rvert,

and every design satisfying this — together with all the model's normal
constraints — is *near-optimal*. GreenBubble maps the shape of that region
in three tiers of increasing detail: **ranges** (min/max per technology),
**hull** (the approximate shape of the full region), and **robustness**
(a design that stays near-optimal across several years at once).

.. admonition:: Time budget
   :class: caution

   NOS re-solves the *entire* model once per direction explored — dozens to
   well over a hundred full solves for one run. Use ``gurobi-barrier-fast``
   (as this tutorial does): it was the only profile that solved every
   direction reliably in testing, and it is fast enough to make this
   tractable. Without a Gurobi licence, use ``highs-simplex`` instead and
   expect it to run noticeably slower — see :ref:`guide-near-optimal` →
   *Limitations*.

---

2 · Run it
----------

.. code-block:: bash

   cp tutorials/5_near_optimal/config.yaml   config/config.yaml
   cp tutorials/5_near_optimal/n_config.yaml config/n_config.yaml
   snakemake --cores 4

.. code-block:: yaml

   mga:
     enabled:      true
     dimensions:   [onwind, solar, electrolysis, methanolisation, battery]
     slack:        0.05          # near-optimal budget: c·x <= c_opt + slack·|c_opt|
     n_directions: 20            # Tier 2 hull; 0 would run Tier 1 only
     robustness:
       enabled: true             # Tier 3, across 2023/2024/2025
       cost_bound: max

This solves the cost-optimal network exactly as in Tutorial 1, then runs all
three NOS tiers on it automatically — no separate command needed.

---

3 · Interpret the results
--------------------------

**Tier 1 — ranges.** For each of the five selected technologies, GreenBubble
reports the minimum and maximum installed capacity compatible with staying
within 5 % of the optimal cost:

.. figure:: /_static/tutorials/tut5_nos_ranges.png
   :width: 95%

   Near-optimal capacity band per technology (red dot = cost-optimal
   capacity). **Onwind, electrolysis and methanolisation never reach zero**
   even at the far end of their range — they are *must-have* technologies
   for this design. **Solar and battery** can shrink all the way to
   (near-)zero for a 5 % cost penalty — they are fully *substitutable*.

**Tier 2 — hull.** Sampling 40 combined search directions (plus the ±unit-axis
corners) traces the approximate shape of the full 5-dimensional near-optimal
region, shown here as pairwise 2-D projections:

.. figure:: /_static/tutorials/tut5_nos_hull.png
   :width: 95%

   Pairwise projections of the 30 extreme points found (hull volume ≈ 8.4M in
   the 5-dimensional capacity space), with the cost optimum marked (★). The
   spread confirms Tier 1's picture: wide, elongated hulls along the
   solar/battery axes (substitutable), tighter along onwind/electrolysis
   (must-have, less room to move).

**Tier 3 — robustness.** Repeating Tier 1-2 for 2023, 2024 and 2025
individually (each against the shared, most expensive year's budget) and
intersecting the three near-optimal regions finds a design that is
simultaneously near-optimal in *all three* years — its Chebyshev radius
(0.58, in the same capacity units as the axes) confirms the intersection
is non-trivially feasible, not just a single boundary point:

.. figure:: /_static/tutorials/tut5_nos_robustness.png
   :width: 95%

   Per-year near-optimal hulls (colour-coded by year) with the robust design
   (Chebyshev centre, ✕) marked in each projection. Where all three
   colours overlap is where a single design works well regardless of which
   year's weather and prices actually materialise.

.. admonition:: Key results
   :class: important

   - **Must-have**: onwind, electrolysis, methanolisation — present at every
     near-optimal design found, deterministic or robust.
   - **Substitutable**: solar and battery can both shrink to ~0 MW for a 5 %
     cost penalty.
   - **2024 is the tightest year** (optimal cost €66.9M, the most expensive
     of the three, so it sets the shared robustness budget); 2025 is the
     cheapest (€61.8M).
   - **A robust design exists**: onwind 147.6 MW, solar 5.2 MW,
     electrolysis 53.2 MW, methanolisation 0.85 MW, battery 17.1 MW stays
     within 5 % of optimal cost in 2023, 2024 *and* 2025 at once — larger
     onwind and electrolysis than the single-year optimum, trading some
     single-year cost for cross-year reliability.

---

What you learned
-----------------

- Why a single cost-optimal design understates real investment flexibility,
  and what "near-optimal" means precisely (the :math:`\varepsilon` budget).
- How to read Tier 1 ranges (must-have vs substitutable) and a Tier 2 hull
  plot (the shape of the near-optimal region).
- How Tier 3 turns several single-year near-optimal spaces into one
  cross-year robust design via their intersection.
- That solver choice affects more than speed for NOS — see the guide's
  *Limitations* section before running this on your own model.

.. seealso::

   :ref:`methods-near-optimal` · :ref:`guide-near-optimal` · :ref:`config-mga` ·
   :ref:`tutorial-1-greenfield` · :ref:`references`
