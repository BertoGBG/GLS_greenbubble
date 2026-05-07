.. _guide-new-technology:

Guide for Adding New Technologies
==================================

This guide explains how to add a new technology inside an existing main
``add_`` function in ``scripts/prepare_network.py``.  A later section will
cover adding an entirely new ``add_`` function.

Overview
--------

Each sector of the GreenBubble network is built by a top-level function:

.. code-block:: text

   add_biogas()          add_electrolysis()     add_meoh()
   add_methanation()     add_central_heat_MT()  add_symbiosis()

These are called in sequence by ``build_network()``.  Each function must work
independently — the optimisation problem must be feasible whether or not the
other functions run.

---

Step 1 — Register the technology in ``n_config``
-------------------------------------------------

``n_config`` is loaded from a CSV file and indexed by technology name.
Every technology that can be switched on/off or sized needs the following
columns:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``initial capacity``
     - MW installed from day one (0 = none)
   * - ``expansion``
     - ``True`` / ``False`` — allow the optimiser to expand
   * - ``cost factor``
     - Multiplier applied to the base capital cost from ``tech_costs``

Add a row for each variant of the new technology
(e.g. ``"biomethanation CO2"``).

---

Step 2 — Declare the tech name in the ``techs`` list
-----------------------------------------------------

Inside the relevant ``add_`` function, append the technology name to the
``techs`` list exactly as it appears in ``n_config``:

.. code-block:: python

   techs = ["methanolisation", "my new tech"]   # existing + new
   cap_to_add, exp_to_add = tech_to_add(techs, n0_dict)

``tech_to_add()`` compares ``n_config`` with the current network state and
returns:

- ``cap_to_add`` — techs that need an initial-capacity component (``EXI_`` prefix)
- ``exp_to_add`` — techs that need an extendable expansion component

---

Step 3 — Call ``add_targets()`` before building the plant
----------------------------------------------------------

If the technology produces a product sold or delivered to an external bus
(H2, bioCH4, Methanol), call ``add_targets`` **before** adding the multilink.

.. code-block:: python

   for t in techs:
       if t in cap_to_add or t in exp_to_add:   # correct form — see Key Rules
           n.add("Carrier", t)
           n, product_bus = add_targets(
               n, plant=t,
               inputs_dict=inputs_dict,
               tech_costs=tech_costs,
               n_options=n_options,
               targets_dict=targets_dict,
           )
           my_buses.at["product bus", t] = product_bus

``add_targets`` creates the following network elements:

- ``"{product} {t}"`` — per-tech intermediate bus (returned as ``product_bus``)
- ``"{product} collection"`` — shared collection bus (tagged for discovery by other modules)
- ``"{t}_to_collection"`` — zero-cost link from per-tech bus to collection bus
- Collection-to-delivery/demand link and Store/Load — created once and shared across all plants

.. important::

   Do **not** copy ``product bus`` from another column of ``my_buses`` inside
   your ``add_*_cap_exp`` function.  ``add_targets`` has already set it
   correctly for both ``price`` and ``demand`` drivers.  Overwriting it breaks
   price mode.

---

Step 4 — Write the ``add_*_cap_exp`` inner function
----------------------------------------------------

This inner function receives ``prefix``, ``capital_cost``, ``capacity``,
``expansion``, ``carrier``, and the buses DataFrame.  It should:

1. Update the buses DataFrame for the new tech by copying shared buses from
   the base column — **never** the ``product bus`` row.
2. Add multilinks or generators using those buses.
3. Return ``(n, my_buses)``.

.. code-block:: python

   def add_my_tech_cap_exp(n, prefix, capital_cost, capacity, expansion, carrier, my_buses):
       # Copy shared buses from base column — never overwrite 'product bus'
       my_buses.at["H2 in bus",    t] = my_buses.at["H2 in bus",    "base"]
       my_buses.at["CO2 in bus",   t] = my_buses.at["CO2 in bus",   "base"]
       my_buses.at["local EL bus", t] = my_buses.at["local EL bus", "base"]
       # product bus is already set by add_targets — do NOT copy it here

       name = f"{prefix}{t}"
       n.add(
           "Link", name,
           carrier=carrier,
           bus0=my_buses.at["H2 in bus",    t],
           bus1=my_buses.at["product bus",  t],   # per-tech intermediate bus
           efficiency=...,
           p_nom=capacity,
           p_nom_extendable=expansion,
           capital_cost=capital_cost if expansion else 0,
           marginal_cost=...,
       )
       return n, my_buses

---

Step 5 — Call the inner function for capacity and expansion
-----------------------------------------------------------

.. code-block:: python

       if t in cap_to_add:
           cap = n_config.at[t, "initial capacity"]
           n, my_buses = add_my_tech_cap_exp(
               n, "EXI_", 0, cap, False, carrier=t, my_buses=my_buses)

       if t in exp_to_add:
           cost = tech_costs.at["my base tech", "fixed"] * n_config.at[t, "cost factor"]
           n, my_buses = add_my_tech_cap_exp(
               n, "", cost, 0, True, carrier=t, my_buses=my_buses)

The ``"EXI_"`` prefix marks initial-capacity (fixed) components.  The
expansion component uses an empty prefix and carries the capital cost.

---

Step 6 — Log new components and return
---------------------------------------

.. code-block:: python

   new_components = log_new_components(n, n0_dict)
   return n, new_components

``log_new_components`` diffs the network before and after to return a dict of
newly added links, buses, stores, etc., used for reporting and stochastic setup.

---

Step 7 — Handle the case where nothing is added
------------------------------------------------

If neither list has entries for this sector, return an empty dict immediately
so the function is a clean no-op:

.. code-block:: python

   if not (cap_to_add or exp_to_add):
       empty = {k: [] for k in
                ["links", "generators", "loads", "stores", "buses", "storage_units"]}
       return n, empty

---

Key Rules
---------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Rule
     - Why
   * - Always write ``if t in cap_to_add or t in exp_to_add:``
     - ``if t in (list_a or list_b):`` only checks the first non-empty list — silent bug
   * - Call ``add_targets()`` before building the multilink
     - Sets ``product_bus`` (per-tech bus) before the link wires to it
   * - Never overwrite ``product bus`` inside ``add_*_cap_exp``
     - ``add_targets`` sets it correctly for both ``price`` and ``demand`` drivers
   * - Use ``add_requirements_buses()`` for any bus that may already exist
     - Idempotent — safe even if another tech already created the bus
   * - Tag any shared bus via ``n.buses.loc[bus, "my_tag"]``
     - Lets ``add_symbiosis`` and other modules discover it without hardcoded names
   * - Return ``(n, new_components)`` or ``(n, empty)`` — never raise
     - Every ``add_`` function must leave the network in a feasible state

---

Product keyword matching in ``add_targets``
--------------------------------------------

``add_targets`` infers which product bus topology to use from the plant name:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Keywords in plant name (case-insensitive)
     - Product
     - Collection bus
   * - ``"biogas"``, ``"methanation"``
     - bioCH4
     - ``"bioCH4 collection"``
   * - ``"electrolysis"``
     - H2
     - ``"H2 collection"``
   * - ``"methanol"``, ``"methanolisation"``, ``"meoh"``
     - Methanol
     - ``"Methanol collection"``

Name the new technology so it contains the right keyword, or extend
``build_bus_list_demand_or_price`` inside ``add_targets`` for a new product
type.
