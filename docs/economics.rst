.. _economics:

Economic Assumptions
=====================

This page documents how GreenBubble translates technology investment data into
annual capital charges, how the discount rate is applied, and how brownfield
initial conditions are parameterised.

---

.. _economics-technology-data:

Technology-data source
----------------------

Investment costs, fixed O&M rates, efficiencies, and technical lifetimes come
from the `technology-data <https://github.com/BertoGBG/technology-data>`_
repository (branch ``pypsa-eur_AA``), which is a fork of the
`PyPSA-Eur technology-data catalogue <https://github.com/PyPSA/technology-data>`_.

The repository provides separate CSV files for each 5-year planning horizon:
2020, 2025, 2030, 2035, 2040, 2045, 2050.  All cost projections are in
**constant real EUR** of a fixed base year (i.e. they represent technology
learning curves, not nominal price inflation — see :ref:`economics-real-costs`).

The CSV files are downloaded automatically by the ``retrieve_tech_data`` rule
(all years at once) and stored in ``data/technology-data/outputs/``.  A git
blob SHA is cached alongside each file so that Snakemake detects upstream
changes without a manual ``--forcerun``.

Project-specific overrides (compressor sizing, GLS-specific equipment) are
defined as ``("technology", "parameter")`` tuples in
``scripts/technology_inputs.py`` and merged into the cost table via
``helpers.merge_into_costs()`` before the annuity is computed.

---

.. _economics-annuity:

Capital cost annualisation
--------------------------

The annual capital charge for a technology with investment cost *I*
(EUR/MW), fixed O&M rate *f* (% of investment per year), and technical
lifetime *L* (years) is:

.. math::

   \text{capital\_cost} = I \cdot \left[ \text{annuity}(r, L) + \frac{f}{100} \right]

where the annuity factor is:

.. math::

   \text{annuity}(r, L) = \frac{r}{1 - (1+r)^{-L}}

*r* is the project discount rate (see :ref:`economics-discount-rate`).

This is computed in ``helpers.read_costs()`` and stored in the ``fixed``
column of the cost DataFrame.  PyPSA then multiplies ``fixed`` by the
optimised capacity (``p_nom_opt``) to obtain the annual capital expenditure
in ``n.statistics.capex()``.

**Amortization period**

By default the annuity denominator equals the technical lifetime *L*.
Setting ``amortization_period`` in ``config.yaml`` to a positive number
(e.g. ``15``) overrides *L* for all **new** expandable capacity:

.. code-block:: yaml

   amortization_period: 15   # recover new investments over 15 years

A shorter amortization period → higher annual charge → harder to invest.
``null`` (default) restores the technical-lifetime behaviour.

For existing capacity the effective period is always ``amortization_period``
(warn if it exceeds the asset's remaining technical lifetime — reinvestment
may be implied; see :ref:`economics-brownfield`).

---

.. _economics-discount-rate:

Discount rate
-------------

``discount_rate`` in ``config.yaml`` is a **real** rate — it excludes
inflation.  All cost data are expressed in constant real EUR of a fixed
base year, so comparing costs across planning years (2020 vs 2030) is
valid without any price-level adjustment.

.. note::

   Do **not** use a nominal rate (which includes expected inflation).
   Mixing real costs with a nominal discount rate would systematically
   over-penalise future costs.

Typical real discount rates for energy projects range from 5 % (public
finance) to 10 % (private equity).  The default is 7 %.

---

.. _economics-year-investment:

year_investment
---------------

``year_investment`` selects which year's cost CSV is used for **new**
expandable capacity.  For example, ``year_investment: 2030`` loads
``data/technology-data/outputs/costs_2030.csv``.

This is independent of the energy/weather year (``En_price_year``) used
for electricity prices and capacity factors.

Available values: 2020, 2025, 2030, 2035, 2040, 2045, 2050.

---

.. _economics-brownfield:

Brownfield initial conditions
------------------------------

When ``initial capacity > 0`` in ``n_config.yaml``, an existing (``EXI_``)
component is added to the network.  Three parameters control its annual
capital charge:

``construction_year``
   The year the asset was built.  Used to look up the investment cost at
   the actual build year: ``I(construction_year)``.  Technology costs
   change between years (learning curves), so an older plant typically cost
   more than a plant built today.

   If not set (``null``), defaults to ``year_investment - 10`` (i.e. 10 years
   before the current planning year), capped at 2020 (the earliest available
   cost data).

``remaining_investment_fraction``
   The fraction of ``I(construction_year)`` that is **financially still
   outstanding**.  This is independent of the technical remaining lifetime:
   a fully paid-off asset has ``remaining_investment_fraction = 0`` even
   if it still has many years of useful life ahead.

   ``0`` (default) = sunk cost; the EXI_ component carries no annual CAPEX.
   ``1`` = the full original investment is still outstanding.

**Annual charge formula**

.. math::

   \text{EXI\_capital\_cost} =
       \text{rif} \times I(\text{construction\_year})
       \times \text{annuity}(r,\, \text{amortization\_period})

where *rif* = ``remaining_investment_fraction`` and the effective
amortization period comes from ``amortization_period`` in ``config.yaml``
(or the technical lifetime if ``null``).

.. note::

   If the asset's remaining technical lifetime
   (:math:`\text{construction\_year} + L - \text{year\_investment}`)
   is shorter than the amortization period, a ``UserWarning`` is issued:
   re-investment within the planning horizon may be implied.

**Example**

Existing electrolysis unit, commissioned in 2020, 60 % of original
investment still outstanding, no new capacity to be built on top:

.. code-block:: yaml

   # config/n_config.yaml
   electrolysis:
     initial capacity: 5       # MW_el
     expansion: false
     construction_year: 2020
     remaining_investment_fraction: 0.6

With ``year_investment: 2030``, ``amortization_period: null``,
``discount_rate: 0.07``, and a 25-year technical lifetime:

- ``remaining_lifetime = 2020 + 25 − 2030 = 15 years``
- ``I(2020)`` is read from ``costs_2020.csv``
- ``annuity(15, 0.07) ≈ 0.110``  (Python call: ``annuity(n, r)``)
- Annual charge = ``0.6 × I(2020) × 0.110`` EUR/MW/year

**Relationship to amortization_period and remaining lifetime**

+-----------------------+---------------------+----------------------------+
| remaining_lifetime    | amortization_period | Outcome                    |
+=======================+=====================+============================+
| > amortization_period | set                 | Recovery accelerated; asset|
|                       |                     | financially clear before   |
|                       |                     | technical end-of-life.     |
+-----------------------+---------------------+----------------------------+
| < amortization_period | set                 | Warning issued; implies    |
|                       |                     | re-investment before full  |
|                       |                     | recovery.                  |
+-----------------------+---------------------+----------------------------+
| = amortization_period | null (default)      | Standard case; annuity uses|
|                       |                     | remaining technical life.  |
+-----------------------+---------------------+----------------------------+

---

.. _economics-real-costs:

Real costs and currency
-----------------------

All monetary values in GreenBubble are expressed in **real EUR** of a fixed
base year (the base year is inherited from the technology-data repository).
The cost trajectories from 2020 to 2050 represent technology learning
(e.g. falling solar costs) — not changes in the general price level.

Consequence: there is **no need to inflate or deflate** investment costs
between ``construction_year`` and ``year_investment``.  ``I(2022)``
and ``I(2030)`` are already in the same real EUR, so comparing or dividing
them is financially consistent.

USD-denominated technologies are converted at the ``USD_to_EUR`` exchange
rate set in ``config.yaml``.

---

.. _economics-lcop:

Levelized Cost of Product (LCOP) and shadow prices
---------------------------------------------------

GreenBubble computes a per-technology Levelized Cost of Product (LCOP) for
every plant injecting into a tagged product collection bus (bioCH4, H2,
Methanol, ...). The method is the same one used for Levelized Cost of CDR
(LCCDR) in the `pypsa-eur fork's CDR checks
<https://github.com/BertoGBG/pypsa-eur/blob/pypsa-eur_AA/scripts/check_CDRs_pipeline.py>`_:
annualised CAPEX, plus snapshot-weighted VOM, plus the snapshot-weighted net
cost of every other input/output flow (feedstocks costed positive,
by-products/credits costed negative) priced at each bus's own nodal shadow
price, all divided by annual main-product output. Only the natural output
unit differs (EUR/tCO2 there vs. EUR/MWh here).

**Cost-based LCOP** (``compute_lcop_by_technology`` in ``scripts/plots.py``),
for a link *s* with main product at ``bus1`` (efficiency :math:`\eta_1`) and
any number of other buses *k* (feedstocks, by-products, electricity, ...):

.. math::

   \text{indirect OPEX}_s = -\sum_{k \neq 1} \eta_k
       \sum_t \left( p_{0,t} \cdot \lambda_{\text{bus}_k, t} \cdot w_t \right)

.. math::

   \text{LCOP}_s = \frac{\text{CAPEX}_s + \text{OPEX}_s + \text{indirect OPEX}_s}
                        {\sum_t \left( p_{0,t} \cdot \eta_1 \cdot w_t \right)}

where :math:`\lambda_{\text{bus}_k,t}` is the nodal shadow price
(``n.buses_t.marginal_price``) at bus *k*, snapshot *t*, and :math:`w_t` is
the objective snapshot weighting. CAPEX/OPEX come from ``n.statistics``.

**KKT-based cross-check** (``compute_lcop_kkt_by_technology``): the
production-weighted average shadow price the plant itself receives at its
own product bus,

.. math::

   \text{LCOP}_s^{\text{kkt}} = \frac{\sum_t \left( w_t \cdot \eta_1 \cdot
       p_{0,t} \cdot \pi_{\text{bus}_1, t} \right)}
       {\sum_t \left( w_t \cdot \eta_1 \cdot p_{0,t} \right)}

By LP complementary slackness, at optimum :math:`\text{LCOP}_s =
\text{LCOP}_s^{\text{kkt}}` **for the marginal (price-setting) technology**
— it earns zero economic profit. For an infra-marginal technology (e.g. a
sunk-cost brownfield asset with CAPEX already written off),
:math:`\text{LCOP}_s < \text{LCOP}_s^{\text{kkt}}`, and the gap is exactly
its profit margin. Both are saved, alongside the difference, in
``lcop_kkt_by_technology.csv`` for every run.

**Demand mode vs. price mode (targets.driver)**

- **demand mode**: the product's exogenous demand is a fixed physical
  target (an equality/lower-bound constraint). Its dual value — read
  directly from ``n.buses_t.marginal_price`` at the delivery bus — *is*
  already the LCOx of the marginal technology; no separate calculation is
  needed.
- **price mode**: the delivery bus's price is exogenously pinned to the
  assumed market price (the sale/purchase link's ``marginal_cost``), so it
  reveals nothing about any individual producer's own cost. The cost-based
  LCOP above is what's actually informative here — and it is computed the
  same way regardless of mode, since it never assumes the bus price means
  anything.

**Worked example** (price mode, from a real run): ``biogas upgrading`` came
out as the marginal supplier of bioCH4 — LCOP_cost ≈ LCOP_kkt ≈ 164.6
EUR/MWh (the flat market price), profit ≈ 0. ``EXI_electrolysis`` is a
sunk-cost brownfield asset (CAPEX = 0) — LCOP_cost (109.5 EUR/MWh) sits well
below the H2 price it actually receives (LCOP_kkt = 113.1 EUR/MWh), earning
real profit. Exactly the pattern the theory above predicts.

Short-run marginal cost (SRMC) per technology and snapshot — the
instantaneous cost of producing one more MWh right now, driving the merit
order — is computed the same way but without amortised CAPEX
(``compute_srmc_by_technology``, saved to ``srmc_by_technology.csv``):

.. math::

   \text{SRMC}_{s,t} = \frac{\lambda_{\text{bus}_0,t}
       - \sum_{k \geq 2} \eta_k \cdot \lambda_{\text{bus}_k,t}
       + \text{VOM}_{s,t}}{\eta_1}

.. note::

   pypsa-eur's CDR script sums the physical flow (tCO2 sequestered) using
   ``snapshot_weightings["stores"]`` and cost terms using
   ``snapshot_weightings["objective"]``, since these can differ under
   representative-period temporal clustering. GreenBubble uses
   ``["objective"]`` uniformly for both; this is equivalent today because
   ``clustering.temporal.resolution`` only does simple uniform downsampling
   (all weighting columns equal), but would need revisiting if a
   representative-period clustering method were ever adopted.
