.. _guide-pypsa-eur-link:

PyPSA-Eur Soft-Link
====================

The PyPSA-Eur soft-link replaces GreenBubble's usual historical
``data/Inputs_{year}/`` market data (from Energinet/ENTSO-E/Renewables.ninja)
with exogenous prices, capacity factors, and a CO₂ cost extracted from a
**solved** `PyPSA-Eur <https://pypsa-eur.readthedocs.io/>`_ (sector-coupled)
network, at the node geographically containing GreenBubble's own site.

This lets a GreenBubble scenario sit consistently inside a wider European
energy-system pathway (e.g. a 2050 net-zero scenario) instead of relying on
today's historical prices, while requiring **zero changes** to
``prepare_network.py`` — the extracted data is written into the same CSV/JSON
schema the historical pipeline already produces.

---

Minimal footprint by design
----------------------------

Only two files are needed per linked run:

- a solved PyPSA-Eur(-sec) network (``.nc``), and
- its onshore-regions GeoJSON (``resources/regions_onshore_base_s_{n}.geojson``
  from the same PyPSA-Eur run).

Everything else is read directly off the solved network — nothing is pulled
from PyPSA-Eur's broader ``resources/`` tree, even where a more granular
pre-solve source exists (e.g. ENSPRESO-derived biomass potentials in
``resources/biomass_potentials_s_*.csv``). If a data point isn't on the solved
network, the soft-link does not extract it.

---

How it works
------------

1. **Node matching** — GreenBubble's ``latitude``/``longitude`` (config.yaml)
   is matched to a PyPSA-Eur cluster region by point-in-polygon lookup against
   the regions GeoJSON (not nearest-centroid/haversine distance, which can
   pick the wrong region near a boundary).
2. **Extraction** — at the matched node, ``scripts/pypsa_eur_link.py`` reads:

   .. list-table::
      :header-rows: 1
      :widths: 30 35 35

      * - GreenBubble series
        - PyPSA-Eur source
        - Notes
      * - Wind capacity factor
        - ``n.generators_t.p_max_pu`` for the node's ``onwind`` generator
        - onshore wind only (not offwind-ac/dc/float)
      * - Solar capacity factor
        - ``n.generators_t.p_max_pu`` for the node's ``solar`` generator
        - utility-scale only (not rooftop/-hsat)
      * - Electricity price
        - ``n.buses_t.marginal_price`` at the node's electricity bus
        -
      * - Natural gas price
        - ``n.buses_t.marginal_price`` at the node's ``gas`` bus
        -
      * - H2 price
        - ``n.buses_t.marginal_price`` at the node's ``H2`` bus
        -
      * - Methanol price
        - ``n.buses_t.marginal_price`` at the ``EU methanol`` bus, **plus**
          ``tech_costs.at["methanol", "CO2 intensity"] * co2_cost``
        - methanol is a single EU-wide bus in PyPSA-Eur-sec, not per-node.
          The CO2 markup mirrors the natural-gas treatment in
          ``helpers.en_market_prices_w_CO2`` (full ``co2_cost``, no
          reference-year netting — the commodity price carries no
          combustion-carbon tax) — applied here rather than downstream
          because, unlike gas/electricity, methanol has no later central
          markup step (``price_meoh`` is consumed directly).
      * - District heating price
        - ``n.buses_t.marginal_price`` at the node's ``rural heat`` bus
        -
      * - Solid biomass price
        - energy-weighted mean of ``n.buses_t.marginal_price`` at the node's
          ``solid biomass`` bus
        - written as a flat scalar — confirmed flat (std≈0) across the year
          in practice, i.e. a resource-value constant, not an hourly signal
      * - CO₂ cost
        - ``abs(n.global_constraints.at["CO2Limit", "mu"])``
        - sign inverted: a negative shadow price on a binding ``≤`` constraint
          in PyPSA-Eur's minimisation becomes a positive cost in GreenBubble
      * - Biogas / solid biomass potential
        - ``n.generators.at[..., "e_sum_max"]`` (annual MWh cap, **not**
          ``p_nom``)
        - extracted but not yet wired into ``n_config`` capacity limits (see
          :ref:`pypsa-eur-link-out-of-scope`)

3. **CSV/JSON write-out** — the extracted series are repeated across each
   snapshot's native resolution (e.g. 4× for a 4h-resolution network) into a
   full 8760-row calendar year, using GreenBubble's exact existing CSV column
   names/format. GreenBubble's own :ref:`temporal resampling
   <guide-temporal-resolution>` (``clustering.temporal.resolution``) then
   downsamples it straight back to the original values — this round-trip is
   exact as long as ``clustering.temporal.resolution`` matches the linked
   network's own resolution (enforced, see below).
4. Two scalars with no CSV mechanism of their own (CO₂ cost, solid-biomass
   price) are written to a small JSON sidecar,
   ``pypsa_eur_link_scalars.json``, alongside the CSVs — read back cheaply by
   ``scripts/config.py`` without reloading the (potentially large)
   PyPSA-Eur network.

---

Configuration
-------------

All settings live under ``pypsa_eur_link`` in ``config/config.yaml`` — see
:ref:`config-pypsa-eur-link` for the quick-reference version.

.. code-block:: yaml

   pypsa_eur_link:
     enabled:               true
     network_path:          pypsa-eur/networks/base_s_90__4h_2050.nc
     regions_path:          pypsa-eur/resources/regions_onshore_base_s_90.geojson
     id:                    ''       # optional; see "Folder naming" below
     co2_stored_price_mode: average  # "average" | "timeseries"
     override_co2_cost:            true
     override_solid_biomass_price: true
     override_DH_price:            true
     override_H2_price:            true
     override_methanol_price:      true
     override_bioCH4_price:        true

.. list-table:: Parameters
   :header-rows: 1
   :widths: 25 12 63

   * - Parameter
     - Type
     - Description
   * - ``enabled``
     - bool
     - Activates the soft-link. Off by default — nothing changes for a normal
       historical-data run.
   * - ``network_path``
     - string
     - Path to the solved PyPSA-Eur(-sec) ``.nc`` network. **Required** when
       ``enabled: true``.
   * - ``regions_path``
     - string
     - Path to the matching onshore-regions GeoJSON from the same PyPSA-Eur
       run. **Required** when ``enabled: true``.
   * - ``id``
     - string
     - Optional run identifier. See :ref:`pypsa-eur-link-folder-naming`.
   * - ``co2_stored_price_mode``
     - ``average`` \| ``timeseries``
     - How the ``co2 stored`` bus price (captured CO₂ sale price) is written:
       a single energy-weighted scalar, or the full hourly series.
   * - ``override_co2_cost``
     - bool
     - ``true`` (default): ``CO2_cost``/``CO2_cost_ref_year`` are set from the
       linked network's ``CO2Limit`` shadow price. ``false``: keep
       ``CO2_cost``/``CO2_cost_ref_year`` as configured in ``config.yaml``.
   * - ``override_solid_biomass_price``
     - bool
     - ``true`` (default): the ``pellets market``/``moist biomass market``
       price (``n_config.yaml``) is set from the linked network. ``false``:
       keep the configured ``n_config.yaml`` price.
   * - ``override_DH_price``
     - bool
     - ``true`` (default): points ``n_options.DH["price profile"]`` at the
       soft-link's extracted rural-heat price *only if* you haven't already
       set your own profile in ``n_config.yaml``. ``false``: never touches it.
   * - ``override_H2_price``
     - bool
     - ``true`` (default, price mode only): ``targets.price_H2`` is replaced
       by the linked network's H2 price. ``false``: keep the configured flat
       ``targets.price_H2``.
   * - ``override_methanol_price``
     - bool
     - Same as above for ``targets.price_meoh`` / the EU methanol bus price.
   * - ``override_bioCH4_price``
     - bool
     - ``true`` (default, price mode only): forces
       ``targets.price_bioCH4: 'NG_based'`` in code automatically — bioCH4
       shares PyPSA-Eur-sec's blended fossil+biomethane gas pool, so its
       price is derived from the (also soft-linked) NG price rather than
       extracted separately. ``false``: keep ``targets.price_bioCH4`` as
       configured (a flat number, or ``'NG_based'`` if you set it yourself).

Each ``override_*`` flag lets exactly one series fall back to your own
``config.yaml``/``n_config.yaml`` value while everything else stays
soft-linked — useful for sensitivity runs (e.g. "everything from PyPSA-Eur
except keep our own DH contract price").

**Forced automatically when ``enabled: true``:**

- ``amortization_period`` is set to ``null`` (each technology's own technical
  lifetime). A soft-linked run takes its prices as exogenous and fixed for
  the linked network's own planning horizon, so a separately shortened
  amortization window doesn't have a well-posed meaning here.
- ``clustering.temporal.resolution`` must be explicitly set to match the
  linked network's own snapshot spacing (e.g. ``'4h'``) — this is validated
  when the network is loaded during ``preprocess_inputs``, and the run fails
  fast with a clear error if it doesn't match.

.. _pypsa-eur-link-folder-naming:

Folder naming
-------------

Soft-linked runs write to ``data/Inputs_{En_price_year}_pypsa-eur[_id]/``
instead of the usual ``data/Inputs_{En_price_year}/``:

- ``En_price_year`` stays meaningful as the PyPSA-Eur planning-horizon year
  (e.g. ``2050``) — useful for future multi-year transition studies chaining
  several PyPSA-Eur networks — without colliding with a real historical
  ``data/Inputs_{year}/`` folder for the same calendar year.
- ``id`` lets more than one soft-linked scenario for the same year (different
  PyPSA-Eur network or config) coexist without overwriting each other.

The Snakemake marker file (``data/Inputs_{year}/.preprocessed``) intentionally
stays at the plain numeric path regardless — only the actual data moves,
keeping the wildcard/DAG structure identical to a normal run.

---

Running
-------

.. code-block:: bash

   snakemake --cores 4

Same DAG as a normal run — ``preprocess_inputs`` branches internally to
``scripts.pypsa_eur_link.write_softlink_inputs`` instead of the usual
API-download path, and everything downstream (``prepare_inputs``,
``build_network``, ``solve_network``, ``plot_results``) is unaffected.

.. note::

   On the very first ``preprocess_inputs`` run for a given
   ``En_price_year``/``id`` combination, ``config.py`` (imported by
   ``preprocess_inputs`` itself before the sidecar exists) falls back to
   ``config.yaml``'s own ``CO2_cost``/solid-biomass price with a warning.
   Re-run once the sidecar has been written if this matters for a
   ``preprocess_inputs``-only invocation; ``prepare_inputs`` onward always
   sees the correct soft-linked values.

---

.. _pypsa-eur-link-out-of-scope:

Deliberately out of scope (for now)
------------------------------------

- **Biogas / solid biomass potential** is extracted (``e_sum_max`` at the
  matched node) and written to the JSON sidecar, but not yet wired into
  ``n_config.yaml``'s ``max capacity`` fields for biogas/biomass technologies.
- **H2 demand sizing** in price mode is left exactly as GreenBubble's own
  manual ``targets.demand_H2``/H2-profile configuration — an unbounded H2
  sale link with no demand can make the problem infeasible, so this stays a
  user-set value rather than something derived from the linked network.
- **PyPSA-Eur is never re-run.** The soft-link only reads an already-solved
  network; it does not trigger or manage a PyPSA-Eur solve.
- **Regional/national CO₂ constraints** in PyPSA-Eur are skipped — only the
  system-wide ``CO2Limit`` global constraint's shadow price is used.

---

See also
--------

- :ref:`config-pypsa-eur-link` — configuration reference (same section, above)
- :ref:`guide-temporal-resolution` — how snapshot resampling works, relied on
  for the soft-link's 4h-repeat-then-downsample round-trip
- :ref:`guide-rolling-horizon` — a comparable "swap the exogenous inputs"
  pipeline, for cross-year dispatch rather than a different price source
- `PyPSA-Eur documentation <https://pypsa-eur.readthedocs.io/>`_
