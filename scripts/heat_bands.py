# SPDX-License-Identifier: MIT
"""Assign a process heat stream to the model's heat circuits, by temperature.

One function answers "which heat bus does this stream connect to, and with how much
duty" for any plant -- compressor aftercooling, a reactor, a reboiler, a condenser.
Today that decision is hardcoded per plant: `compressor_calculation` splits the
aftercooling duty at a single point (Heat DH's floor) and has exactly two heat ports,
while every other plant has its heat direction hand-set in a `meth_heat_directions`
dict. Both are special cases of what is below.

MODEL
-----
Circuits come from config/p_config.default.yaml and form CONTIGUOUS temperature
intervals: each circuit's top is the floor of the circuit above it, and the highest
circuit uses its declared ``T_max``. With the current configuration:

    Heat MT   140 -> 180        (declared top)
    Heat DH    90 -> 140
    Heat LT    50 ->  90

A stream cooling from 160 to 50 C therefore spans three circuits, not the two the
compressor code can express -- the 160-140 slice is MT-grade heat currently handed to
the 90 C DH bus.

Pinch convention: allocation uses SHIFTED temperatures, hot streams down by dT_min/2
and cold streams up by dT_min/2, so a source and a sink at the same shifted temperature
are exactly pinched and neither can drive the other.

Sensible streams spread across the intervals they span; ISOTHERMAL (latent) streams --
a reboiler, a condenser -- put their whole duty in the one interval containing them.

NOT IN SCOPE: converting high-grade heat to power. A stream above the highest circuit
is clipped to that circuit (see ``assign_source``), not sent through a cycle.
"""
from __future__ import annotations


def circuit_bands(circuits: dict, ) -> list[tuple[str, float, float]]:
    """Contiguous (name, T_floor, T_top) intervals, hottest first.

    Each circuit's top is the floor of the circuit above; the hottest uses its declared
    ``T_max``, or is unbounded (``float('inf')``) if it declares none.
    """
    ordered = sorted(
        ((n, float(c["T_min"]), c.get("T_max")) for n, c in circuits.items()),
        key=lambda x: x[1], reverse=True,
    )
    bands = []
    for i, (name, floor, declared_top) in enumerate(ordered):
        if i == 0:
            top = float(declared_top) if isinstance(declared_top, (int, float)) else float("inf")
        else:
            top = ordered[i - 1][1]          # floor of the circuit above
        bands.append((name, floor, top))
    return bands


def assign_source(bands, T_hot, T_cold, *, duty=1.0, q_above=None, dT_min=10.0,
                  isothermal_tol=1.0):
    """Allocate heat REJECTED by a stream cooling from T_hot to T_cold.

    Parameters
    ----------
    bands : list of (name, floor, top), from :func:`circuit_bands`.
    T_hot, T_cold : float
        Stream inlet and outlet temperature [C]. ``T_hot == T_cold`` (within
        ``isothermal_tol``) marks a latent stream.
    duty : float
        Total duty. Returned values are in the same unit.
    q_above : callable, optional
        ``q_above(T) -> duty released above T``. Supply this for real fluids: cp varies
        strongly (CO2 near critical especially), so splitting by temperature fraction is
        wrong. Without it, constant cp is assumed and the split is linear in T.
    dT_min : float
        Minimum approach temperature [K]. A hot stream is shifted DOWN by dT_min/2.

    Returns
    -------
    dict {circuit_name: duty}, omitting circuits that receive nothing.

    THE VALUES DO NOT NECESSARILY SUM TO ``duty``. Heat below the lowest circuit floor
    has nowhere to go and is left out -- physically it is rejected to ambient. Callers
    must account for the remainder, ``duty - sum(result.values())``; use
    :func:`split_with_ambient` to get it back explicitly. A stream cooling 160 -> 50 C
    with dT_min = 10 loses its last 5 shifted K this way (~4.5% of the duty).

    A stream hotter than the top circuit is clipped to it -- the excess grade is lost,
    which is what happens physically when there is no higher-temperature sink and no
    power cycle. Heat below the lowest floor is dropped: it is not useful to any
    circuit (in the model it goes to ambient).
    """
    if T_hot < T_cold:
        T_hot, T_cold = T_cold, T_hot
    shift = dT_min / 2.0
    t_hi, t_lo = T_hot - shift, T_cold - shift

    if abs(T_hot - T_cold) <= isothermal_tol:          # latent: one band takes it all
        if bands and t_hi >= bands[0][2]:             # hotter than the top circuit
            return {bands[0][0]: duty}                # clipped: excess grade is lost
        for name, floor, top in bands:
            if floor <= t_hi < top:
                return {name: duty}
        return {}                                     # below every floor -> ambient

    if q_above is None:
        span = t_hi - t_lo
        frac = lambda a, b: max(0.0, (min(b, t_hi) - max(a, t_lo))) / span
    else:
        total = q_above(t_lo) - q_above(t_hi)
        def frac(a, b):
            a_, b_ = max(a, t_lo), min(b, t_hi)
            if b_ <= a_ or total <= 0:
                return 0.0
            return (q_above(a_) - q_above(b_)) / total

    out = {}
    for name, floor, top in bands:
        f = frac(floor, top)
        if f > 0:
            out[name] = duty * f
    return out


def assign_sink(bands, T_required, *, dT_min=10.0):
    """Which circuit can supply heat to a sink needing ``T_required`` [C]?

    A circuit can serve the sink if it can DELIVER at that temperature, which is a
    question about the circuit's TOP, not its floor: DH spans 90-140, so DH water can
    heat a 99.6 C reboiler even though the DH floor is below it.

    The sink is shifted UP by dT_min/2 and the circuit DOWN by dT_min/2, so the test is
    ``top >= T_required + dT_min``. Among circuits that qualify the COLDEST is chosen --
    drawing higher-grade heat than the process needs is a waste the model should not
    make for free. Returns ``None`` if no circuit is hot enough.
    """
    candidates = [(name, top) for name, _, top in bands if top >= T_required + dT_min]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[1])[0]


def split_with_ambient(bands, T_hot, T_cold, **kw):
    """:func:`assign_source`, plus the remainder that no circuit can take.

    Returns ``(assigned, to_ambient)``. ``sum(assigned.values()) + to_ambient == duty``
    by construction, so the caller cannot silently lose heat.
    """
    duty = kw.get("duty", 1.0)
    assigned = assign_source(bands, T_hot, T_cold, **kw)
    return assigned, duty - sum(assigned.values())
