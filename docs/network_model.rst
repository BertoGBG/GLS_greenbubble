.. _network-model:

Network Model
=============

GreenBubble represents an industrial cluster as a multi-energy network built
with `PyPSA <https://pypsa.readthedocs.io>`_ and solved as a linear programme
via `Linopy <https://linopy.readthedocs.io>`_.  This page describes the
network structure, the optimisation problem, and how the two fit together.

---

Multi-energy representation
-----------------------------

The cluster is modelled as a set of **buses** — one per energy carrier.
Components (generators, conversion units, storage) connect to these buses
via **links**, **stores**, and **loads**.

.. list-table:: Energy and material carriers
   :widths: 20 20 60
   :header-rows: 1

   * - Carrier
     - Unit
     - Role in the network
   * - Electricity
     - MW\ :sub:`el`
     - Grid connection, wind, solar, electrolysis input, internal distribution
   * - Hydrogen
     - MW\ :sub:`H₂`
     - Electrolysis output, methanation / methanol feed, internal distribution
   * - CO₂
     - t\ :sub:`CO₂`/h
     - Captured from biogas, consumed by methanation and methanol synthesis
   * - Biomethane (CH₄)
     - MW\ :sub:`CH₄`
     - Biogas upgrading output, methanation output, product delivery
   * - Methanol
     - t\ :sub:`MeOH`/h
     - Methanol synthesis output, product delivery
   * - Heat — medium temperature
     - MW\ :sub:`th`
     - Waste heat from conversion processes, biomass drying
   * - Heat — district heating (DH)
     - MW\ :sub:`th`
     - Optional sale to external DH grid
   * - Heat — low temperature
     - MW\ :sub:`th`
     - Low-grade heat from cooling loops
   * - Biomass (moist / pellets)
     - MW\ :sub:`bio`
     - Biogas feedstock

Each carrier may have **multiple buses** representing distinct physical
locations or pressure levels (e.g. H₂ at the electrolyser vs. H₂ at the
methanation inlet).  Internal pipes and compressors are explicit links with
their own efficiency losses and, optionally, expandable capacity.


PyPSA component types
----------------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Component
     - Use in GreenBubble
   * - ``Generator``
     - Renewable electricity sources (wind, solar) and biogas input; may be
       extendable (greenfield) or capacity-fixed (brownfield)
   * - ``Link``
     - All conversion processes: electrolysis, methanation, methanol synthesis,
       biogas engine, belt dryer, compressors.  A link can have up to five
       outputs (bus0 … bus4), enabling combined heat-and-power units
   * - ``Store``
     - Energy buffers where carrier is stored without direct dispatch control:
       CO₂ tanks, product delivery buffers
   * - ``StorageUnit``
     - Storage with explicit charge/discharge dispatch: batteries, H₂ vessels,
       thermal stores
   * - ``Load``
     - Demand time series for each product and internal energy consumer

Technology activation is controlled by the ``n_flags`` dict in
``config/config.default.yaml``.  Setting a flag to ``false`` removes **all**
components of that technology group from the network before the solve — no
parameters need to change elsewhere.


Optimisation problem
---------------------

GreenBubble solves a **single linear programme** that simultaneously determines
optimal **installed capacities** and **hourly dispatch** over a full year
(8 760 time steps at 1-hour resolution).

**Objective function**

Minimise total annualised system cost (or maximise revenue, depending on the
``targets.driver`` setting):

- *Cost mode* (``driver: 'demand'``): minimise capital cost + operating cost
  subject to fixed annual production requirements.
- *Revenue mode* (``driver: 'price'``): maximise net revenue from product sales
  minus system costs; production becomes an upper bound rather than a constraint.

Capital costs are annualised using the discount rate and component lifetime
from the technology-cost database.  Operating costs include variable O&M and
fuel purchases.

**Key constraints**

- **Energy balance** at every bus and every hour (enforced by PyPSA/Linopy)
- **Capacity bounds**: ``p_nom_min ≤ p_nom ≤ p_nom_max`` per component
- **Ramp limits**: ``|p(t) - p(t-1)| ≤ ramp_limit × p_nom`` for selected
  technologies (electrolysis, methanation, methanol synthesis)
- **Minimum load**: ``p(t) ≥ min_load × p_nom`` for online units
- **Annual product targets**: total production ≥ demand (demand mode) or
  ≤ capacity × 8760 (price mode)
- **RFNBO constraints**: optional additionality/temporal correlation requirements
  on electrolytic hydrogen (see :ref:`methods-rfnbo`)
- **CO₂ balance**: captured CO₂ = consumed CO₂ (closed internal loop with
  optional liquefaction and sequestration)
- **Renewable export cap**: ``RE export ≤ max_RE_to_grid × (internal use + export)``
- **Shared grid-connection capacity**: import capacity == export capacity
  (see :ref:`grid-connection-capex` below)

**Solver**

The LP is passed to HiGHS (open-source) or Gurobi via Linopy.
Solver profiles with pre-tuned tolerance and thread settings are defined in
``scripts/solver_profiles.py``.


.. _grid-connection-capex:

Modelling the electrical grid interface
------------------------------------------------------

The site connects to the external DK1 grid in two directions: import (one
``DK1_to_El_{agent}`` link per on-site agent, e.g. ``DK1_to_El_electrolysis``)
and export (``El3_to_DK1``, priced at real spot prices). Physically these
share one connection asset, so the model prices it as one shared capacity
rather than paying for import and export capacity separately:

- Every agent's import branch draws from a single bus, ``"ElDK1 buy bus"``,
  fed by one extendable link, ``"DK1_to_ElDK1_buy"`` (built in
  ``add_local_el_connections``, ``scripts/prepare_network.py``). This link
  alone carries the "electricity grid connection" capital cost; every branch
  from ``"ElDK1 buy bus"`` to an agent's local bus has ``capital_cost = 0``
  and keeps its own ``marginal_cost`` (purchase price), so per-agent
  operating-cost reporting is unaffected — only capital cost moved.
- When both the shared import link and ``El3_to_DK1`` exist for a given
  ``n_flags`` configuration, ``build_network()`` zeros the export link's own
  capital cost and flags the network (``n.meta["consolidated_grid_connection"]
  = True``) so a custom constraint,
  ``add_grid_connection_shared_capacity_constraint`` (``scripts/helpers.py``),
  ties the two links' ``p_nom`` together — one equality constraint, not one
  per snapshot, since each link's own native PyPSA bound already caps its
  hourly dispatch once the two ``p_nom`` variables are equal. Only one
  physical capacity is ever paid for.
- If a configuration has only one side (e.g. a pure-import site with no
  export capability, or a fully self-sufficient renewables site that never
  buys from the grid), there is nothing to share — that link's own capital
  cost is left untouched, and the constraint silently skips (same defensive,
  never-raise style as :func:`add_max_RE_sales_constraint`).
- For reporting, ``reallocate_grid_connection_capex`` (``scripts/helpers.py``,
  called from ``snakemake_plot.py`` after the network is solved) splits the
  shared link's total capex back onto the individual import/export links, in
  proportion to each link's share of flow at the year's peak-usage hour(s) —
  see :ref:`payback-cost-allocation` in :doc:`economics` for the method. This
  is a reporting-only, in-memory step; it never touches the optimisation.


Brownfield vs. greenfield
--------------------------

Each technology can be configured independently in ``n_config.default.yaml``:

- ``initial capacity > 0`` + ``expansion: false`` → brownfield fixed asset
  (costs zeroed, capacity locked)
- ``initial capacity > 0`` + ``expansion: true`` → brownfield with expansion
  allowed above the initial installed size
- ``initial capacity: 0`` + ``expansion: true`` → greenfield (capacity is a
  free variable from zero)

This allows modelling real plants at GreenLab Skive (or any other site) by
setting existing capacities for known assets while leaving greenfield
expansion open for new investments.


Output files
-------------

After the solve, the optimised PyPSA network is exported to:

.. code-block:: text

   outputs/<run>/<network_name>/
     networks/
       <network>_PRE.nc   ← pre-optimisation network (built by build_network)
       <network>_OPT.nc   ← post-optimisation network with p_nom_opt, dispatch TS
     plots/
       ...                ← dispatch figures, capacity charts, shadow prices
