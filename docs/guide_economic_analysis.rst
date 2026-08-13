.. _guide-economic-analysis:

Economic Analysis of Results
=============================

This guide describes the economic interpretation framework built into
GreenBubble's post-processing pipeline (``scripts/plots.py``).  It covers:

- how the network represents markets via buses and shadow prices,
- the product topology (collection bus + delivery bus),
- the Levelised Cost of Production (LCOP) formula and its components,
- the KKT-based LCOP verification (zero-profit condition),
- the short-run marginal cost (SRMC) and merit order,
- the full KKT revenue and annual profit,
- how to interpret LCOP versus the shadow price at the collection bus, and
- payback time and capital cost coverage aggregated by agent.

---

Network topology as a market
------------------------------

GreenBubble is a linear PyPSA model.  Every **bus** is a commodity market
at a specific location and quality level.  The **KKT multiplier** at a bus
(``n.buses_t.marginal_price``) is the shadow price of the energy balance
constraint there — the marginal value of one additional unit of that
commodity at that time step.

**Links** are technologies that convert commodities.  A multilink with
ports ``bus0 … busN`` satisfies:

.. code-block:: text

   flow at bus_k = p0 × efficiency_k   (k ≥ 1)
   flow at bus0  = −p0                  (convention: bus0 always consumes)

Positive ``efficiency_k`` → the link *produces* at bus_k (output or by-product).
Negative ``efficiency_k`` → the link *consumes* from bus_k (additional input).

---

Product topology
-----------------

Every product (bioCH4, H2, Methanol) uses a two-bus structure:

.. code-block:: text

   [technology A] ──┐
   [technology B] ──┤──► bioCH4 collection ──► bioCH4 delivery ──► demand / store
   [technology C] ──┘              ▲
                          is_product_bus = True

``bioCH4 collection``
   The **collection bus**.  All technologies producing bioCH4 inject here
   directly via ``bus1`` of their multilink.  Tagged ``is_product_bus=True``
   in the network so post-processing can discover it automatically.

``bioCH4 delivery``
   The **delivery bus**.  The exogenous demand load (demand mode) or the
   price-setting sale link (price mode) is attached here.

   - **Demand mode**: a zero-cost ``{product}_collection_to_demand`` link
     transfers product to the delivery bus, where a ``Load`` sets the
     time profile and an optional cyclic ``Store`` provides intra-period
     flexibility.
   - **Price mode**: a ``{product}_collection_to_delivery`` link carries a
     time-varying ``marginal_cost`` equal to the exogenous selling price
     series.  A non-cyclic annual ``Store`` (``e_nom_max = total_demand``)
     caps cumulative annual production.

The shadow price at the collection bus (``λ_collection``) is determined at
equilibrium by the marginal producer — the highest-cost technology that is
still dispatched.  Technologies with lower cost earn an *intra-marginal rent*.

---

Shared components
------------------

Compressors, HP storage, and other auxiliary equipment often serve multiple
technologies and are modelled as a single shared component (separate carrier).
They are **not** included in the LCOP computation for the production
technologies themselves.

Their capital and operational costs appear implicitly via the shadow price at
the buses they connect to (e.g. ``H2 distribution``, ``CO2 distribution``).
When a production technology draws H2 from ``H2 distribution``, the KKT
``λ_{H2 distribution}`` already reflects the full cost of H2 at that
pressure level, including compression.

---

LCOP — Levelised Cost of Production
-------------------------------------

For a multilink ``lk`` whose ``bus1`` is a collection bus (``is_product_bus=True``),
first define the **indirect OPEX**:

.. math::

   \text{indirect OPEX} = -\sum_{k \neq 1} \eta_k \cdot \overline{(\lambda_{k} \cdot p_0)}

where :math:`\eta_0 = -1` for bus0 (primary feedstock, always consumed), and
:math:`\overline{(\lambda_k \cdot p_0)} = \sum_t p_{0,t} \cdot \lambda_{k,t} \cdot w_t`
with snapshot weights :math:`w_t` from ``snapshot_weightings["objective"]``.

The sign convention makes indirect OPEX **positive when input costs dominate** (the
typical case) and negative only if by-product credits exceed all feedstock costs.

.. list-table:: Port contributions to indirect OPEX
   :header-rows: 1
   :widths: 28 12 60

   * - Port
     - :math:`\eta_k`
     - Contribution to indirect OPEX
   * - bus0 — primary feedstock
     - −1
     - :math:`+p_0 \cdot \lambda_{\text{bus0}} \cdot w` — adds feedstock cost
   * - bus2..N — additional input
     - < 0
     - :math:`+|\eta_k| \cdot p_0 \cdot \lambda_k \cdot w` — adds feedstock cost
   * - bus2..N — by-product output
     - > 0
     - :math:`-\eta_k \cdot p_0 \cdot \lambda_k \cdot w` — subtracts by-product credit

Then:

.. math::

   \text{LCOP} = \frac{\text{CAPEX} + \text{OPEX} + \text{indirect OPEX}}{Q_\text{main}}

where:

- :math:`\text{CAPEX}` = ``n.statistics.capex()`` for the link (annualised capital cost)
- :math:`\text{OPEX}` = ``n.statistics.opex()`` for the link (explicit ``marginal_cost`` × dispatch — variable O&M such as enzyme or maintenance rates, **not** feedstock costs)
- :math:`Q_\text{main} = \sum_t p_{0,t} \cdot \eta_1 \cdot w_t` — annual production at bus1

**No double-counting:** ``OPEX`` is the explicit variable O&M declared in
``tech_costs`` (VOM).  Indirect OPEX captures the *market value* of feedstocks
consumed and by-products produced via KKT shadow prices.  These measure
different things and do not overlap.

**Existing (EXI\_) assets:** whether CAPEX appears in the LCOP depends on
``remaining_investment_fraction`` (``rif``) in ``n_config.yaml``:

- ``rif = 0`` (default — sunk cost): ``capital_cost = 0`` in the network,
  so ``CAPEX = 0`` from statistics.  LCOP reflects short-run cost only
  (OPEX + indirect OPEX).
- ``rif > 0`` (residual obligation outstanding): ``capital_cost = rif ×
  I(construction_year) × annuity(amortization_period, discount_rate)``.
  CAPEX is non-zero and the LCOP includes this residual capital charge.

See :ref:`economics-brownfield` for the full parameter description.

---

Revenue, net market value, and annual profit
---------------------------------------------

.. math::

   \text{revenue main product} = \eta_1 \cdot \sum_t p_{0,t} \cdot \lambda_{\text{bus1},t} \cdot w_t

This is the market value of the main product at the collection bus — what the
market would pay for everything the technology produces there.

.. math::

   \text{net market value} = \text{revenue main product} - \text{indirect OPEX}

.. math::

   \text{annual profit} = \text{net market value} - \text{CAPEX} - \text{OPEX}
                        = \text{revenue main product} - \text{indirect OPEX}
                          - \text{CAPEX} - \text{OPEX}

Annual profit is the economic rent: what the market pays the technology for its
product, minus every cost it incurs (capital, O&M, feedstocks net of by-products).

- **annual profit ≈ 0** — the technology is the marginal (price-setting) producer.
- **annual profit > 0** — intra-marginal rent; technology has lower cost than the price-setter.
- **annual profit < 0** — technology is loss-making under current market conditions (possible for EXI\_ assets if market prices are low).

.. note::

   The zero-profit condition holds for optimally expanded technologies in a
   fully endogenous problem.  When exogenous selling prices are set (price mode),
   the complementary slackness argument breaks down and all dispatched
   technologies can earn positive profit if the price exceeds the LCOP of the
   marginal producer.

**Intra-marginal rent:**

.. code-block:: text

   rent_per_MWh ≈ λ̄_collection − LCOP

where ``λ̄_collection`` is the energy-weighted mean shadow price at the
collection bus (from ``shadow_prices_mean.csv``).  The technology with
``LCOP ≈ λ̄_collection`` is the price-setter; all others earn rent proportional
to the cost gap.

---

Output files
-------------

The following output files are generated by ``scripts/plots.py``:

``csv/lcop_by_technology.csv``
   One row per product link (index column: ``link``).

   .. list-table::
      :header-rows: 1
      :widths: 38 62

      * - Column
        - Description
      * - ``carrier``
        - PyPSA carrier name of the link
      * - ``product``
        - Product name (bioCH4, H2, Methanol …)
      * - ``CAPEX (EUR)``
        - Annualised capital cost from ``n.statistics.capex()``
      * - ``OPEX (EUR)``
        - Variable O&M from ``n.statistics.opex()`` (explicit marginal cost × dispatch)
      * - ``indirect OPEX (EUR)``
        - Feedstock costs minus by-product credits via KKT shadow prices;
          positive = net cost (typical), negative = by-products exceed feedstock cost
      * - ``revenue main product (EUR)``
        - Market value of annual main-product output at the collection bus
      * - ``net market value (EUR)``
        - ``revenue main product`` − ``indirect OPEX``
      * - ``annual production (MWh)``
        - Annual energy output at bus1 = :math:`\sum_t p_{0,t} \cdot \eta_1 \cdot w_t`
      * - ``LCOP (EUR/MWh)``
        - ``(CAPEX + OPEX + indirect OPEX) / annual production``
      * - ``annual profit (EUR)``
        - ``net market value − CAPEX − OPEX``; economic rent earned by the technology

``plots/lcop_by_technology.png``
   Two-panel bar chart: LCOP [€/MWh] (top) and annual profit [k€] (bottom).

``csv/lcop_kkt_by_technology.csv``
   Verification table produced by ``compute_lcop_kkt_by_technology()``.
   For each product link, LCOP is independently computed as the
   production-weighted average KKT shadow price at the collection bus:

   .. math::

      \text{LCOP}_\text{kkt} =
        \frac{\displaystyle\sum_t w_t \cdot \eta_1 \cdot p_{0,t} \cdot \lambda_{\text{bus1},t}}
             {\displaystyle\sum_t w_t \cdot \eta_1 \cdot p_{0,t}}

   At optimum this equals the cost-based LCOP (zero-profit condition).
   The ``diff cost−kkt`` column confirms agreement to < 0.01 €/MWh.
   Columns: ``carrier``, ``product``, ``annual_production_MWh``,
   ``LCOP_cost (EUR/MWh)``, ``LCOP_kkt (EUR/MWh)``, ``diff cost−kkt (EUR/MWh)``,
   ``π_bus1_mean``, ``π_bus1_std``, ``π_bus1_prod_weighted``.

``csv/srmc_by_technology.csv`` / ``plots/srmc_by_technology.png``
   Short-run marginal cost (SRMC) computed by ``compute_srmc_by_technology()``.
   For each product technology at every snapshot:

   .. math::

      \text{SRMC}_{s,t} =
        \frac{\lambda_{\text{bus0},t}
              - \displaystyle\sum_{k \geq 2} \eta_k \cdot \lambda_{\text{bus}_k,t}
              + \text{VOM}_{s}}{\eta_1}

   This is the instantaneous cost of producing one more MWh of main product
   at time *t*, given current input market prices.  It is distinct from
   the model input ``marginal_cost`` on links (which is the VOM, one term
   in the formula).

   The long-form CSV has columns: ``snapshot``, ``link``, ``product``,
   ``SRMC_EUR_per_MWh``, ``dispatch_MW``, ``π_product_bus``, ``in_merit``
   (whether SRMC ≤ product shadow price at that hour).

   The plot shows one subplot per product with SRMC time series per
   technology and the product bus shadow price as a dashed reference line.

---

Shadow price outputs
---------------------

``csv/shadow_prices_mean.csv``
   Energy-weighted mean KKT for every bus in ``plots_config.yaml → bus_list_mp``,
   including collection buses.  Computed regardless of whether any technology
   actively dispatches to the bus.

   Formula:

   .. math::

      \bar{\lambda}_\text{bus} = \frac{\sum_t \lambda_{t} \cdot q_t \cdot w_t}{\sum_t q_t \cdot w_t}

   where :math:`q_t` is the net injection at the bus (generators + positive-efficiency
   link outputs + storage discharge), and :math:`w_t` is the snapshot weight.
   Falls back to duration-weighted mean when a bus has no measurable injection.

   Columns:

   .. list-table::
      :header-rows: 1
      :widths: 40 60

      * - Column
        - Description
      * - ``bus`` (index)
        - Bus name from ``bus_list_mp``
      * - ``energy weighted mean (EUR/MWh)``
        - Energy-weighted mean shadow price at that bus

``plots/shd_prices_mean_bar.png``
   Bar chart of energy-weighted mean shadow prices.  Collection buses are
   excluded — delivery buses are sufficient to read the product market price.
   Buses with no active injection are also dropped.

``plots/shd_prices_violin.png``, ``plots/shd_prices_ldc.png``
   **Snapshot distribution** of shadow prices (one observation per time step),
   with the scenario-weighted mean marked.  These show *when* prices are high
   or low, not an energy-weighted average.  Restricted to the same delivery-bus
   subset as the bar chart, further filtered to buses with at least one injecting
   link above the ``LINK_TH`` capacity threshold.

---

Comparing LCOP to shadow prices
---------------------------------

A practical diagnostic workflow:

1. Look at ``shadow_prices_mean.csv`` — read the ``energy weighted mean (EUR/MWh)``
   for the delivery bus of each product (e.g. ``bioCH4 delivery``) — this is
   ``λ̄_delivery``, which equals ``λ̄_collection`` when the collection-to-delivery
   link is zero-cost.
2. Look at ``lcop_by_technology.csv`` — compare each technology's ``LCOP (EUR/MWh)``
   to ``λ̄_delivery`` of its product.
3. Technologies with ``annual_profit_EUR ≈ 0`` are price-setters.
4. Technologies with ``annual_profit_EUR > 0`` earn intra-marginal rent —
   check whether they are EXI\_ (legacy) assets or optimally expanded.
5. In price mode, all technologies that run may earn positive profit if the
   exogenous selling price exceeds the LCOP of the marginal producer.
6. Use ``srmc_by_technology.csv`` to see *when* each technology is in-merit
   (``in_merit = True``).  Hours with SRMC near the shadow price reveal the
   marginal technology.  Large SRMC variance indicates strong sensitivity to
   electricity or H2 price fluctuations.

---

Payback and capital cost coverage by agent
--------------------------------------------

While LCOP asks "what does this one technology cost to run", payback asks a
plant-level question: "does everything a given agent owns — digester,
upgrading, storage, shared infrastructure — collectively earn back what was
put into it, and how fast." See :ref:`economics-payback` for the full
theory (formulas, the KKT cash-flow computation, the brownfield investment
scaling, and the "priced at own margin" derivation). This section covers
how to read the output.

``csv/payback_by_agent.csv``
   One row per agent (the same ``n_flags``-based groups as
   ``TSC_by_agent.csv``), plus a ``TOTAL`` row. Only generated in price mode.

   .. list-table::
      :header-rows: 1
      :widths: 38 62

      * - Column
        - Description
      * - ``investment (EUR)``
        - Raw upfront investment (not annualised), summed across the
          agent's components; brownfield (``EXI_``) components scaled by
          their own ``remaining_investment_fraction``
      * - ``annual FOM (EUR/y)``
        - Fixed O&M, summed across the agent's components
      * - ``annual revenue (EUR/y)`` / ``annual running cost (EUR/y)``
        - ``n.statistics.revenue()`` / ``.opex()`` summed by agent (KKT
          shadow-price value net across every port, and explicit
          ``marginal_cost`` × dispatch, respectively)
      * - ``annual cash flow (EUR/y)``
        - ``revenue − running cost − FOM``
      * - ``capital cost coverage (%)``
        - ``cash flow ÷`` the agent's own pure capital-recovery annuity
          (no FOM). 100 % = exactly recovering the annualised capital
          charge; see :ref:`economics-payback` for the four-way reading
          (surplus / priced-at-margin / shortfall / loss)
      * - ``simple payback (years)`` / ``discounted payback (years)``
        - ``investment ÷ cash flow``, and the annuity-inverted discounted
          equivalent. ``nan`` = no investment; ``inf`` = never recovers
          (or, for discounted, recovers only past a coverage shortfall
          outside the margin tolerance)
      * - ``priced at own margin``
        - ``True`` when coverage is within ``MARGIN_TOLERANCE`` (default
          3 %) of 100 % — discounted payback is then reported as exactly
          the effective amortization period, not the numerically unstable
          raw value
      * - ``technical lifetime, investment-weighted (years)``
        - Investment-weighted average of each component's own **true**
          catalogue lifetime (independent of ``amortization_period``) —
          "does this pay back within its physical life"
      * - ``investment coverage (%)`` (``TOTAL`` row only)
        - Share of total annualised capex the investment column could
          actually resolve (composite technologies with no matching
          catalogue row are logged and excluded — see the console output
          for the list)

``plots/payback_by_agent.png``
   Two-panel figure, one shared legend:

   - **Left — discounted payback (years)**: one bar per agent, black tick
     marking each agent's own investment-weighted technical lifetime. Bars
     that never pay back (loss, or a shortfall beyond the margin tolerance)
     are capped and hatched rather than dropped — silently omitting them
     made a chart with several loss-making agents look like most agents
     had no data at all.
   - **Right — capital cost coverage (%)**: the same agents, colour-coded
     surplus (green) / priced-at-own-margin (amber) / shortfall (red) /
     loss (grey), with a 100 % reference line. Reads calmly exactly where
     the left panel is least stable (right around 100 % coverage).

   .. figure:: /_static/tutorials/tut2_payback_by_agent.png
      :width: 95%

      Tutorial 2 (brownfield, ``amortization_period: 10``). Brownfield-heavy
      agents (``biogas``, ``symbiosis``, ``meoh``) clear their small residual
      annuity easily — coverage well above 100 %, payback under two years.
      ``electrolysis`` sits at only 26 % coverage with an infinite payback:
      not mis-sized, but cross-subsidising biomethanation's extra profit
      with hydrogen it doesn't itself get paid full value for — see the
      reading below.

**How to read a low-coverage agent that isn't a brownfield asset.** If an
agent shows low coverage despite being freely, continuously sized by the
optimiser (no forced brownfield floor, no hard capacity minimum), check
whether it's a genuine underperformer or a **cross-subsidy**: does its
product feed another agent that shows unusually *high* coverage? If so,
the low-coverage agent is paying for value that shows up on someone else's
books — a legitimate system-level trade-off, not a modelling error. The
``electrolysis`` case in Tutorial 2 above is exactly this: its hydrogen
makes additional biomethanation capacity worthwhile, so part of its true
value is invisible if you only look at its own payback. Confirm the
suspected link by checking the product-flow topology (:ref:`guide-outputs`)
between the two agents.

**How to read a "priced at own margin" agent.** Coverage ≈ 100 % is the
*expected*, healthy signature of a continuously-sized (extendable)
technology at its optimum — the LP builds exactly until marginal revenue
equals marginal annualised cost. It is not a red flag, and the payback
plot marks it distinctly (amber, ``*``, pinned to the effective
amortization period) precisely so it doesn't read as "broken" the way an
unmarked near-infinite discounted payback would.
