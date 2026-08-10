"""Soft-link to a solved PyPSA-Eur(-sec) network.

Identifies the PyPSA-Eur cluster node geographically closest to GreenBubble's
own (``latitude``, ``longitude``) config, and extracts the exogenous
economic/weather inputs GreenBubble needs from that node: renewable capacity
factors, electricity/NG/H2/CO2-grid prices, the system-wide CO2 cost, and
biogas/solid-biomass resource potentials.

Deliberately minimal file footprint: only a solved PyPSA-Eur network (``.nc``)
and its onshore-regions GeoJSON are required. Everything is read directly off
the solved network — nothing is read from PyPSA-Eur's broader ``resources/``
tree, even where a more granular pre-solve source exists (e.g. biomass
potentials), so the soft-link only ever depends on two files per linked run.

See ``docs/guide_pypsa_eur_link.rst`` for the extraction table and design rationale
(node-matching method, CO2 cost sign convention, why capacity potential comes
from ``e_sum_max`` and not ``p_nom``, the onshore-wind/solar-only carrier
choice, and the resolution-matching requirement on the GreenBubble side).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

# Onshore wind and solar only — GreenBubble models a single representative
# wind/solar resource each, not the full offwind-ac/dc/float/-hsat/rooftop
# variant set PyPSA-Eur carries per node.
_CF_CARRIERS = ("onwind", "solar")
_BIOMASS_CARRIERS = ("biogas", "solid biomass")


def inputs_folder(year: int, enabled: bool, run_id: str = "") -> str:
    """Where soft-linked (or regular) input data for one year lands.

    Regular runs (``enabled=False``): unchanged, ``data/Inputs_{year}``.

    Soft-linked runs: ``data/Inputs_{year}_pypsa-eur[_run_id]`` — kept
    distinct from the plain ``data/Inputs_{year}`` naming for two reasons:
    (1) En_price_year is deliberately still meaningful for a soft-linked run
    (it's the PyPSA-Eur planning-horizon year, e.g. relevant for future
    multi-year transition studies chaining several PyPSA-Eur networks), so
    the same year value needs to be able to coexist with a *real* historical
    ``data/Inputs_{year}`` folder for that same calendar year without
    collision or ambiguity about which is which; (2) ``run_id`` lets more
    than one soft-linked scenario for the same year (different PyPSA-Eur
    network/config) coexist without overwriting each other.
    """
    if not enabled:
        return f"data/Inputs_{year}"
    suffix = "_pypsa-eur" + (f"_{run_id}" if run_id else "")
    return f"data/Inputs_{year}{suffix}"


def match_node(latitude: float, longitude: float, regions_path: str) -> str:
    """Return the PyPSA-Eur cluster region name (bus prefix) containing (lat, lon).

    Point-in-polygon against the clustered onshore-regions GeoJSON — the
    geometrically correct match. A nearest-centroid-by-distance approach can
    pick the wrong neighbour near irregular cluster boundaries; this doesn't,
    by construction.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    regions = gpd.read_file(regions_path)
    point = Point(longitude, latitude)  # GeoJSON coordinate order is (lon, lat)

    match = regions[regions.contains(point)]
    if match.empty:
        # Point sits exactly on an edge or just outside every polygon
        # (coastal rounding, etc.) — fall back to the nearest region boundary
        # rather than failing outright.
        regions = regions.copy()
        regions["_dist"] = regions.geometry.distance(point)
        match = regions.sort_values("_dist").head(1)

    return str(match.iloc[0]["name"])


def _bus(prefix: str, carrier: str | None = None) -> str:
    """Bus name for a carrier at a matched node: PyPSA-Eur's "{prefix} {carrier}"
    naming convention. ``carrier=None`` returns the bare AC (electricity) bus."""
    return prefix if carrier is None else f"{prefix} {carrier}"


def get_capacity_factor(n: pypsa.Network, prefix: str, carrier: str) -> pd.Series:
    """Onshore wind or solar capacity factor time series at the matched node.

    Parameters
    ----------
    carrier : "onwind" or "solar"
    """
    if carrier not in _CF_CARRIERS:
        raise ValueError(f"carrier must be one of {_CF_CARRIERS}, got {carrier!r}")
    gen = f"{prefix} 0 {carrier}"
    if gen not in n.generators_t.p_max_pu.columns:
        raise KeyError(
            f"Generator {gen!r} not found in n.generators_t.p_max_pu — check the "
            f"node prefix and that this network models {carrier!r} at this node."
        )
    return n.generators_t.p_max_pu[gen].copy()


def get_price(n: pypsa.Network, prefix: str, carrier: str | None = None) -> pd.Series:
    """Marginal price time series (shadow price of the energy-balance
    constraint) at a bus for the matched node.

    Parameters
    ----------
    carrier : None for electricity (bare AC bus), else e.g. "gas", "H2",
        "co2 stored" for the corresponding commodity bus at the same node.
    """
    bus = _bus(prefix, carrier)
    if bus not in n.buses_t.marginal_price.columns:
        raise KeyError(f"Bus {bus!r} not found in n.buses_t.marginal_price.")
    return n.buses_t.marginal_price[bus].copy()


def get_energy_weighted_mean_price(n: pypsa.Network, prefix: str, carrier: str | None = None) -> float:
    """Energy-weighted mean price at a bus for the matched node: a single
    scalar summarising ``get_price``'s time series.

    Reuses ``scripts.plots._energy_weighted_mean`` — the same weighting
    already established project-wide for ``shadow_prices_mean.csv``
    (weight = net injection into the bus at each snapshot, i.e. how much was
    actually flowing when each price applied), rather than a plain
    unweighted time average. Falls back to a duration-weighted mean if the
    bus has no measurable injection in this network.

    Useful where a flat scalar is the more natural fit than a full time
    series — e.g. GB's ``options.DH.price``-style config fields are single
    numbers, not time series, and a bus like "co2 stored" tends to have a
    narrow enough price range that little information is lost by averaging.
    """
    from scripts.plots import _energy_weighted_mean

    bus = _bus(prefix, carrier)
    means, _ = _energy_weighted_mean(n, [bus])
    if bus not in means:
        raise KeyError(f"Bus {bus!r} not found in n.buses_t.marginal_price.")
    return means[bus]


def get_co2_cost(n: pypsa.Network, constraint_name: str = "CO2Limit") -> float:
    """System-wide CO2 cost (EUR/tCO2): the absolute value of the CO2 limit
    global constraint's shadow price.

    Sign note: for a binding "<=" constraint in a minimisation LP, the dual
    (``mu``) is <= 0 by convention (confirmed on a real solved network:
    sense='<=', mu=-616.65) — relaxing the cap by one more unit would lower
    the objective. GreenBubble's ``CO2_cost`` is a positive cost, hence the
    sign inversion here, not a reinterpretation of the underlying economics.
    """
    if constraint_name not in n.global_constraints.index:
        raise KeyError(f"Global constraint {constraint_name!r} not found.")
    mu = float(n.global_constraints.at[constraint_name, "mu"])
    return abs(mu)


def get_biomass_potential(n: pypsa.Network, prefix: str, carrier: str) -> float:
    """Annual biomass resource potential (MWh/y) at the matched node.

    Parameters
    ----------
    carrier : "biogas" or "solid biomass"

    Reads ``e_sum_max`` — deliberately **not** ``p_nom``. On a real solved
    network, ``p_nom`` for these generators is set to the same large numeric
    value as ``e_sum_max`` (MWh, not MW) but does not itself bind; the actual
    annual resource constraint is ``e_sum_max`` (confirmed: dispatch sums to
    exactly ``e_sum_max``, with a nonzero ``mu_e_sum_max`` proving it's the
    binding constraint). Reading ``p_nom`` instead would silently give a
    potential ~8760x too large.
    """
    if carrier not in _BIOMASS_CARRIERS:
        raise ValueError(f"carrier must be one of {_BIOMASS_CARRIERS}, got {carrier!r}")
    gen = f"{prefix} {carrier}"
    if gen not in n.generators.index:
        raise KeyError(f"Generator {gen!r} not found in n.generators.")
    e_sum_max = n.generators.at[gen, "e_sum_max"]
    if not np.isfinite(e_sum_max):
        raise ValueError(f"{gen!r} has no finite e_sum_max — cannot use as a resource potential.")
    return float(e_sum_max)


def snapshot_resolution_hours(n: pypsa.Network) -> float:
    """Native temporal resolution of the linked network, in hours (e.g. 4.0).

    GreenBubble's own ``clustering.temporal.resolution`` must be forced to
    match this when the soft-link is active — resampling already-resampled
    data is lossy in a way that's hard to reason about (see module docstring).
    """
    diffs = n.snapshots.to_series().diff().dropna()
    if diffs.empty:
        raise ValueError("Could not determine snapshot resolution — fewer than 2 snapshots.")
    mode = diffs.mode()
    if len(mode) > 1:
        raise ValueError(
            f"Irregular snapshot spacing (multiple distinct gaps: "
            f"{sorted(diffs.unique())}) — cannot determine a single resolution."
        )
    return mode.iloc[0].total_seconds() / 3600.0


def extract_all(n: pypsa.Network, prefix: str) -> dict:
    """Bundle every soft-link extraction for one matched node into a single dict.

    Returns
    -------
    dict with keys:
        "CF_onwind", "CF_solar" : pd.Series (0-1 fraction)
        "price_electricity", "price_gas", "price_H2", "price_co2_stored",
        "price_methanol", "price_rural_heat" : pd.Series (EUR/MWh or EUR/t)
        "price_solid_biomass" : float (EUR/MWh) — energy-weighted mean, not a
            series: confirmed on a real solved network to be exactly flat
            (std=0.00) across the whole year, a resource-value constant with
            no genuine hourly signal, not an approximation of one
        "co2_cost" : float (EUR/tCO2)
        "biogas_potential", "solid_biomass_potential" : float (MWh/y)
        "resolution_hours" : float

    Notes
    -----
    "price_methanol" is read from the single **EU-wide** ``"EU methanol"``
    bus, not a per-node one — PyPSA-Eur-sec models methanol as a globally
    tradeable commodity, unlike gas/H2/heat which are spatially resolved.
    """
    return {
        "CF_onwind":              get_capacity_factor(n, prefix, "onwind"),
        "CF_solar":               get_capacity_factor(n, prefix, "solar"),
        "price_electricity":      get_price(n, prefix, None),
        "price_gas":              get_price(n, prefix, "gas"),
        "price_H2":               get_price(n, prefix, "H2"),
        "price_co2_stored":       get_price(n, prefix, "co2 stored"),
        "price_methanol":         get_price(n, "EU", "methanol"),
        "price_rural_heat":       get_price(n, prefix, "rural heat"),
        "price_solid_biomass":    get_energy_weighted_mean_price(n, prefix, "solid biomass"),
        "co2_cost":               get_co2_cost(n),
        "biogas_potential":       get_biomass_potential(n, prefix, "biogas"),
        "solid_biomass_potential": get_biomass_potential(n, prefix, "solid biomass"),
        "resolution_hours":       snapshot_resolution_hours(n),
    }


def load_and_extract(network_path: str, regions_path: str, latitude: float, longitude: float) -> dict:
    """Convenience entry point: load the network, match the node, extract everything.

    Adds a ``"node"`` key (the matched cluster prefix) to the ``extract_all``
    result dict.
    """
    n = pypsa.Network(network_path)
    prefix = match_node(latitude, longitude, regions_path)
    result = extract_all(n, prefix)
    result["node"] = prefix
    return result


def _repeat_to_hourly(series: pd.Series, resolution_hours: float, target_hours: pd.DatetimeIndex) -> pd.Series:
    """Upsample a native-resolution series (e.g. 4h) to a full 8760-row hourly
    series by repeating each value across the hours it spans.

    This is a deliberate trick, not a real upsampling: GreenBubble's existing
    preprocessing pipeline always writes/reads full 8760-row hourly CSVs and
    only downsamples *later*, inside prepare_network.py, based on
    ``clustering.temporal.resolution``. Repeating each native-resolution value
    ``resolution_hours`` times, then relying on that *same* existing
    downsampling step (forced to match, see scripts/config.py), reconstructs
    exactly the original native-resolution values — so this requires zero
    changes to the CSV schema or the downstream resampling logic. Block-
    constant repetition is exact for a mean-based downsampler; if
    prepare_network.py's resampler ever changes to something non-linear this
    would need revisiting.
    """
    step = int(round(resolution_hours))
    if abs(resolution_hours - step) > 1e-6:
        raise ValueError(f"resolution_hours={resolution_hours} is not an integer number of hours.")
    values = np.repeat(series.to_numpy(), step)
    if len(values) != len(target_hours):
        raise ValueError(
            f"Repeated series has {len(values)} rows but the target calendar year "
            f"needs {len(target_hours)} (8760, or 8784 for a leap year before Feb-29 "
            f"removal) — the linked network's snapshot count x resolution doesn't "
            f"divide evenly into a full year. See module docstring caveat."
        )
    return pd.Series(values, index=target_hours)


def write_softlink_inputs(year: int, network_path: str, regions_path: str,
                            latitude: float, longitude: float,
                            co2_stored_price_mode: str = "average",
                            out_folder: str | None = None,
                            run_id: str = "",
                            expected_resolution_hours: float | None = None) -> dict:
    """Write GreenBubble's standard ``data/Inputs_{year}/`` CSVs from a
    soft-linked PyPSA-Eur network, as a drop-in alternative to
    ``scripts.preprocessing.pre_processing_energy_data``.

    Writes ``Elspotprices_input.csv``, ``CF_wind.csv``, ``CF_solar.csv`` and
    ``NG_price_year_input.csv`` in the exact existing schema (same column
    names, same semicolon-delimited 8760-row format) so nothing downstream
    (``load_input_data``, ``prepare_network.py``) needs to change. Also
    writes ``H2_price_input.csv`` and ``CO2_stored_price_input.csv`` — new
    files with no existing consumer yet (wiring them into the network build
    is separate, follow-up scope).

    Writes ``CO2emis_input.csv`` (grid CO2 intensity) as all zeros —
    ``load_input_data()`` reads it unconditionally regardless of
    ``rfnbos_dict.limit``, and zero is the semantically correct value here
    (not just a safe filler): it makes helpers.py's CO2-cost markup on the
    electricity price vanish, avoiding double-counting a carbon price that's
    already fully embedded in the soft-link's own (PyPSA-Eur-derived)
    electricity price. See the inline comment at the write site for detail;
    revisit only if a genuine per-hour grid-mix CO2 intensity is ever needed
    for something else (e.g. ``rfnbos_dict.limit == "emissions"``).

    Returns
    -------
    dict : the same ``extract_all`` result (plus ``"node"``), for logging/
        the caller's own use — e.g. printing the matched node and CO2 cost
        so it's visible in the Snakemake log.
    """
    from scripts.helpers import build_snapshots

    n = pypsa.Network(network_path)
    prefix = match_node(latitude, longitude, regions_path)
    resolution_hours = snapshot_resolution_hours(n)

    if expected_resolution_hours is not None and abs(expected_resolution_hours - resolution_hours) > 1e-6:
        raise ValueError(
            f"clustering.temporal.resolution implies {expected_resolution_hours}h, but the "
            f"linked PyPSA-Eur network's own resolution is {resolution_hours}h. Set "
            f"clustering.temporal.resolution: '{resolution_hours:g}h' in config.yaml to match."
        )

    hours_in_period, _, _ = build_snapshots(year)
    if len(hours_in_period) % int(round(resolution_hours)) != 0:
        raise ValueError(
            f"{year} has {len(hours_in_period)} hours after leap-day removal, not evenly "
            f"divisible by the linked network's {resolution_hours}h resolution."
        )

    folder = out_folder or inputs_folder(year, True, run_id)
    import os
    os.makedirs(folder, exist_ok=True)

    def _write(series: pd.Series, filename: str, column: str) -> None:
        hourly = _repeat_to_hourly(series, resolution_hours, hours_in_period)
        df = pd.DataFrame({column: hourly.values}, index=hours_in_period)
        df.index.name = None
        df.to_csv(f"{folder}/{filename}", sep=";")

    # CO2emis_input.csv (grid CO2 intensity, t/MWh): load_input_data() reads
    # this unconditionally, so it must exist regardless of its values.
    # Its actual purpose (see helpers.py's mk_el_grid_price = el_grid_price +
    # CO2_emiss_El * (CO2_cost - CO2_cost_ref_year)) is correcting a
    # *historical* electricity price that already embeds a specific CO2 tax,
    # when modelling a different assumed tax than that reference year's.
    # That concept doesn't apply to a soft-linked price — it comes straight
    # out of PyPSA-Eur's own CO2-constrained solve, so there's no separate
    # "reference year" to correct against. scripts/config.py sets
    # CO2_cost_ref_year = CO2_cost when the soft-link is active specifically
    # so this correction term is zero regardless of what's written here;
    # zero is simply the natural, honest value in that case (no separate
    # grid-mix CO2 intensity is derived here — see get_price's per-node
    # extraction functions for what *is* pulled from the linked network).
    _write(pd.Series(0.0, index=n.snapshots), "CO2emis_input.csv", "CO2PerMWh")

    _write(get_capacity_factor(n, prefix, "onwind"), "CF_wind.csv", "CF wind")
    _write(get_capacity_factor(n, prefix, "solar"), "CF_solar.csv", "CF solar")
    _write(get_price(n, prefix, None), "Elspotprices_input.csv", "SpotPrice")
    _write(get_price(n, prefix, "gas"), "NG_price_year_input.csv", "THE_NG_pricesEUR_MWh")
    _write(get_price(n, prefix, "H2"), "H2_price_input.csv", "H2_price_EUR_MWh")
    _write(get_price(n, "EU", "methanol"), "Methanol_price_input.csv", "Methanol_price_EUR_MWh")

    # DH price profile: written comma-separated (not ';') with a genuine
    # calendar-matched datetime index — unlike every other file here, GB's
    # own add_symbiosis reindexes this one by *actual timestamp*
    # (n_options["DH"]["price profile"] -> pd.read_csv(..., parse_dates=True)
    # -> .reindex(n.snapshots)), not by row position. hours_in_period is
    # already real build_snapshots(year) dates, so this "just works" as long
    # as the separator/parse-dates expectation is respected.
    dh_hourly = _repeat_to_hourly(get_price(n, prefix, "rural heat"), resolution_hours, hours_in_period)
    pd.Series(dh_hourly.values, index=hours_in_period, name="DH_price_EUR_MWh").to_csv(
        f"{folder}/DH_price_input.csv"
    )

    if co2_stored_price_mode == "timeseries":
        _write(get_price(n, prefix, "co2 stored"), "CO2_stored_price_input.csv", "CO2_stored_price_EUR_t")
    elif co2_stored_price_mode == "average":
        mean_price = get_energy_weighted_mean_price(n, prefix, "co2 stored")
        flat = pd.Series(mean_price, index=n.snapshots)
        _write(flat, "CO2_stored_price_input.csv", "CO2_stored_price_EUR_t")
    else:
        raise ValueError(f"co2_stored_price_mode must be 'average' or 'timeseries', got {co2_stored_price_mode!r}")

    result = extract_all(n, prefix)
    result["node"] = prefix

    # Scalars sidecar: co2_cost/price_solid_biomass etc. need to reach
    # scripts/config.py, but Snakemake runs each rule in its own process --
    # an in-memory override made here (during preprocess_inputs) would not
    # persist to build_network's later, separate, fresh import of
    # scripts.config. Persisting to this small JSON file (read cheaply, no
    # PyPSA network load, unlike the .nc extraction itself) is what actually
    # carries these values across the process boundary; the CSV series
    # (CF/prices) already get this "for free" via the files they're written
    # to, since prepare_network.py's consumers already read from disk.
    scalars = {k: v for k, v in result.items() if isinstance(v, (int, float, str))}
    with open(scalars_path(folder), "w") as f:
        json.dump(scalars, f, indent=2)

    print(f"[pypsa_eur_link] node={prefix!r}  CO2_cost={result['co2_cost']:.2f} EUR/t  "
          f"biogas_potential={result['biogas_potential']:,.0f} MWh/y  "
          f"solid_biomass_potential={result['solid_biomass_potential']:,.0f} MWh/y  "
          f"-> wrote CSVs to {folder}/")
    return result


def scalars_path(folder: str) -> str:
    """Path to the scalars sidecar JSON written by ``write_softlink_inputs``
    and read back by ``scripts.config``. Factored out so both sides agree on
    the filename without duplicating the literal."""
    return f"{folder}/pypsa_eur_link_scalars.json"


def read_scalars(folder: str) -> dict | None:
    """Read the scalars sidecar written by ``write_softlink_inputs``, or
    ``None`` if it doesn't exist yet (expected on a fresh checkout before
    ``preprocess_inputs`` has run once — callers should fall back to
    config.yaml's own defaults in that case, not error)."""
    path = scalars_path(folder)
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)
