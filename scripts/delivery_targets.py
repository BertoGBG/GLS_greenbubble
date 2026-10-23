# SPDX-License-Identifier: MIT
"""Stock targets that let a rolling horizon meet binned product deliveries.

A delivery bin puts its whole demand on one snapshot. A rolling-horizon window
sees only the deliveries inside it, so a window far from the next delivery has
no demand and produces nothing; the window that finally sees the delivery may
then have too few hours left to make it.

The demand is set, not forecast, so the whole delivery schedule is known before
the first window is solved. From it this module derives, for each delivery
store:

* a lower stock target at the end of every window's kept part (``e_min_pu``),
  so that every later delivery can still be made;
* an upper bound equal to the deliveries still to come (``e_max_pu``), so that
  production can run ahead when it is cheap but never beyond what can be
  delivered.

The year starts and ends with the same stock ``e_carry``, as the cyclic store
of the capacity-expansion run does. It is the stock the first deliveries need
before the plant can make them, and is zero unless early bins are larger than
what the plant can produce from 1 January.

For binned deliveries the store size from the capacity-expansion run is lifted:
the plant capacity already limits how far production can run ahead. For a
continuous demand (``flat``, ``profile``) the store is the flexibility the user
set, so it keeps its size and only gains the remaining-demand cap.

See docs/guide_rolling_horizon.rst, "Demand bins in rolling horizon".
"""
import numpy as np
import pandas as pd

UNMET_CARRIER = "unmet delivery"


def window_ends(n_snapshots, horizon, overlap):
    """Last snapshot of the kept part of each window, as PyPSA builds them.

    ``optimize_with_rolling_horizon`` starts a window every ``horizon - overlap``
    snapshots and carries the stock at ``start - 1`` into the next window, so
    the snapshot before each start is where a window hands over its stock.
    """
    step = horizon - overlap
    return np.array([min(n_snapshots - 1, s + step - 1)
                     for s in range(0, n_snapshots, step)])


def minimum_stock(delivery, capacity, e_end=0.0):
    """Smallest stock after each snapshot that keeps every later delivery reachable.

    ``delivery`` is the energy due at each snapshot [MWh] and ``capacity`` the
    most that can reach the delivery bus in each snapshot [MWh]. Backward
    recursion::

        e_min(T-1) = e_end
        e_min(t-1) = max(0, e_min(t) + delivery(t) - capacity(t))

    Returns ``(e_min, e_start)``, where ``e_start`` is the stock needed before
    the first snapshot.
    """
    d = np.asarray(delivery, dtype=float)
    cap = np.asarray(capacity, dtype=float)
    e = np.zeros(len(d))
    e[-1] = e_end
    for t in range(len(d) - 1, 0, -1):
        e[t - 1] = max(0.0, e[t] + d[t] - cap[t])
    e_start = max(0.0, e[0] + d[0] - cap[0])
    return e, e_start


def pro_rata_stock(delivery, weights):
    """Stock after each snapshot when every bin is produced at a constant rate.

    A bin runs from the snapshot after the previous delivery to its own
    delivery. Inside it the stock grows in proportion to elapsed time. For a
    continuous demand every snapshot is its own bin and the result is zero.
    """
    d = np.asarray(delivery, dtype=float)
    w = np.asarray(weights, dtype=float)
    e = np.zeros(len(d))
    start = 0
    for t_k in np.flatnonzero(d > 0):
        span = np.cumsum(w[start:t_k + 1])
        e[start:t_k] = d[t_k] * span[:-1] / span[-1]
        start = t_k + 1
    return e


def delivery_stores(n):
    """Map each delivery store to its bus, for buses that carry a Load.

    In demand mode ``add_targets_per_product`` puts the Load and the delivery
    store on the same ``{product} delivery`` bus. Buses without a store have a
    rigid demand and need no target.
    """
    if n.loads.empty or n.stores.empty:
        return {}
    load_buses = set(n.loads.bus)
    return {s: b for s, b in n.stores.bus.items() if b in load_buses}


def _feeding_links(n, bus):
    """(link, port) pairs whose output port ``port`` is ``bus`` with a positive efficiency."""
    ports = sorted(int(c[3:]) for c in n.links.columns
                   if c.startswith("bus") and c[3:].isdigit() and int(c[3:]) >= 1)
    pairs = []
    for k in ports:
        col = "efficiency" if k == 1 else f"efficiency{k}"
        if col not in n.links.columns:
            continue
        mask = (n.links[f"bus{k}"] == bus) & (n.links[col] > 0)
        pairs += [(l, k) for l in n.links.index[mask]]
    return pairs


def _is_pass_through(n, bus):
    """A bus that only links feed: no store, generator, storage unit or load on it."""
    return not ((n.stores.bus == bus).any() or (n.generators.bus == bus).any()
                or (n.storage_units.bus == bus).any() or (n.loads.bus == bus).any())


def _inflow_mw(n, bus, depth):
    cap = pd.Series(0.0, index=n.snapshots)
    p_max = n.get_switchable_as_dense("Link", "p_max_pu")
    for link, k in _feeding_links(n, bus):
        eff = n.get_switchable_as_dense("Link", "efficiency" if k == 1 else f"efficiency{k}")[link]
        out = p_max[link] * n.links.at[link, "p_nom"] * eff
        bus0 = n.links.at[link, "bus0"]
        # a link cannot pass on more than reaches its input when only links feed it,
        # e.g. the collection link is limited by the plants behind it
        if depth > 0 and _is_pass_through(n, bus0) and _feeding_links(n, bus0):
            out = np.minimum(out, _inflow_mw(n, bus0, depth - 1) * eff)
        cap += out
    gens = n.generators.index[(n.generators.bus == bus)
                              & (n.generators.carrier != UNMET_CARRIER)]
    if len(gens):
        g_max = n.get_switchable_as_dense("Generator", "p_max_pu")[gens]
        cap += (g_max * n.generators.loc[gens, "p_nom"]).sum(axis=1)
    return cap


def inflow_capacity(n, bus, depth=1):
    """Most energy that can reach ``bus`` in each snapshot [MWh], fixed capacities.

    Each feeding link counts with ``p_nom * p_max_pu * efficiency``. When its
    input bus is fed only by other links, as the product collection bus is fed
    by the plants, it is also limited by their capacity, ``depth`` levels up.
    """
    return _inflow_mw(n, bus, depth) * n.snapshot_weightings.stores


def is_binned(delivery, max_share=0.5):
    """True when deliveries fall on fewer than ``max_share`` of the snapshots."""
    return (np.asarray(delivery) > 0).mean() < max_share


def add_unmet_generator(n, bus, cost, p_nom):
    """Add, or update, the ``unmet delivery`` generator on ``bus``.

    It supplies the bus at ``cost`` [EUR/MWh], the load-shedding price of an
    undelivered MWh, and keeps the problem feasible when a demand cannot be met.
    """
    if UNMET_CARRIER not in n.carriers.index:
        n.add("Carrier", UNMET_CARRIER)
    name = f"{bus} unmet"
    if name in n.generators.index:
        n.generators.loc[name, ["p_nom", "marginal_cost"]] = (p_nom, cost)
    else:
        n.add("Generator", name, bus=bus, carrier=UNMET_CARRIER,
              p_nom=p_nom, marginal_cost=cost)
    return name


def add_unmet_generators(n, cost):
    """Add an ``unmet delivery`` generator to every bus with a positive Load.

    Each is sized to the bus's peak load, so it can cover the whole demand in
    any snapshot. Returns the generator names.
    """
    if n.loads.empty:
        return []
    loads = n.get_switchable_as_dense("Load", "p_set")
    names = []
    for bus in n.loads.bus.unique():
        peak = loads.loc[:, n.loads.index[n.loads.bus == bus]].sum(axis=1).max()
        if peak > 0:
            names.append(add_unmet_generator(n, bus, cost, peak))
    return names


def set_delivery_targets(n, horizon, overlap, rule="pro_rata", unmet_cost=1e4):
    """Replace each delivery store's cap with window stock targets.

    rule:
        ``"e_min"``     lowest target that keeps every delivery reachable;
        ``"pro_rata"``  max(e_min, constant-rate production within each bin);
        ``"none"``      no lower target, only the remaining-demand cap.

    The store becomes non-cyclic. It starts and ends with ``e_carry``, the stock
    the first deliveries need (normally zero), and may never hold more than the
    deliveries still to come plus ``e_carry``. Binned deliveries lose the store size of the capacity-expansion run;
    continuous demands keep it. An ``unmet delivery`` generator on the bus, at ``unmet_cost``, keeps
    every window feasible and records any shortfall. One already carried over
    from the capacity-expansion network is resized and repriced.

    Call after capacities are fixed and after any cross-year patch of the loads.
    """
    if rule not in ("e_min", "pro_rata", "none"):
        raise ValueError(f"delivery_target must be e_min, pro_rata or none, not {rule!r}")
    stores = delivery_stores(n)
    if not stores:
        return {}
    w = n.snapshot_weightings.stores
    loads = n.get_switchable_as_dense("Load", "p_set")
    ends = window_ends(len(n.snapshots), horizon, overlap)

    report = {}
    for store, bus in stores.items():
        d = (loads.loc[:, n.loads.index[n.loads.bus == bus]].sum(axis=1) * w).to_numpy()
        total = d.sum()
        if total <= 0:
            continue
        cap = inflow_capacity(n, bus).to_numpy()
        # stock carried across the year end, as in the cyclic capacity-expansion
        # run; with it at both ends the recursion returns the same e_carry
        # whenever the plant can make the annual total
        _, e_carry = minimum_stock(d, cap)
        e_min, e_start = minimum_stock(d, cap, e_end=e_carry)
        floor = {"e_min": e_min,
                 "pro_rata": np.maximum(e_min, pro_rata_stock(d, w.to_numpy())),
                 "none": np.zeros(len(d))}[rule]
        remaining = (total - np.cumsum(d)).clip(min=0) + e_carry
        binned = is_binned(d)
        e_nom = total + e_carry if binned else max(n.stores.at[store, "e_nom"], e_carry)
        cap_e = np.minimum(remaining, e_nom)
        target = np.zeros(len(d))
        target[ends] = floor[ends]
        target[-1] = e_carry                        # close the year as it opened
        target = np.minimum(target, cap_e)

        n.stores.loc[store, ["e_nom", "e_nom_extendable", "e_cyclic", "e_initial"]] = (
            e_nom, False, False, e_carry)
        n.stores_t.e_min_pu[store] = pd.Series(target / e_nom, index=n.snapshots)
        n.stores_t.e_max_pu[store] = pd.Series(cap_e / e_nom, index=n.snapshots)

        add_unmet_generator(n, bus, unmet_cost, total / w.min())

        n_bins = int((d > 0).sum())
        report[store] = dict(bus=bus, total=total, n_bins=n_bins, binned=binned, e_nom=e_nom,
                             e_min_above_store=bool((e_min[ends] > e_nom + 1e-6).any()),
                             capacity=float(np.median(cap / w.to_numpy())),
                             largest_bin=float(d.max()), e_carry=e_carry,
                             feasible=bool(e_start <= e_carry + 1e-6),
                             max_e_min=float(e_min.max()),
                             windows_with_target=int((target[ends] > 1e-6).sum()),
                             n_windows=len(ends))
    return report


def unmet_deliveries(n):
    """Energy supplied by the ``unmet delivery`` generators, per bus [MWh]."""
    gens = n.generators.index[n.generators.carrier == UNMET_CARRIER]
    if not len(gens) or n.generators_t.p.empty:
        return pd.Series(dtype=float)
    w = n.snapshot_weightings.generators
    used = n.generators_t.p[gens].mul(w, axis=0).sum()
    # a stochastic network indexes generators by (scenario, name): keep the scenario
    used.index = [(g[0], n.generators.bus[g]) if isinstance(g, tuple) else n.generators.bus[g]
                  for g in used.index]
    return used


def drop_unused_unmet(n, tol=0.01):
    """Remove ``unmet delivery`` generators that were never used."""
    used = unmet_deliveries(n)
    gens = n.generators.index[n.generators.carrier == UNMET_CARRIER]
    idle = [g for g in gens if used.get(n.generators.bus[g], 0.0) <= tol]
    if idle:
        n.remove("Generator", idle)
    if not (n.generators.carrier == UNMET_CARRIER).any() and UNMET_CARRIER in n.carriers.index:
        n.remove("Carrier", UNMET_CARRIER)
