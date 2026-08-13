# SPDX-License-Identifier: MIT
"""Near-optimal space (NOS) exploration via Modelling to Generate Alternatives (MGA).

Explores the set of designs whose total system cost stays within a small ``slack`` of
the cost optimum, to reveal *investment flexibility* and "must-have / must-avoid"
technologies, and (optionally) a *robust* design that remains near-optimal across
several weather/CO₂ years.

Three tiers, all driven by the ``mga`` config block:

* **Tier 1** :func:`mga_ranges` — per-technology min/max installed capacity at fixed
  slack. ``min > 0`` ⇒ must-have, ``max ≈ 0`` ⇒ must-avoid.
* **Tier 2** :func:`explore_hull` — sample many search directions in the selected
  capacity space, collect the extreme points, and build their convex hull (the
  approximated near-optimal polytope of Neumann & Brown 2021).
* **Tier 3** :func:`chebyshev_centre` + :func:`realise_design` — intersect the
  per-year near-optimal hulls and take the Chebyshev centre (deepest interior point)
  as a robust design, then map it back to a full network (Grochowicz et al. 2023).

Design notes
------------
* **Dimensions = installed capacity (MW)**, weight ``1`` per component. The set of
  selectable dimensions is auto-derived from the *extendable* components of the built
  network (see :func:`available_dimensions`) — nothing is hardcoded. A technology is
  offered iff it has at least one extendable, costed (``capital_cost > 0``) component
  whose grouping key is a known ``n_config`` technology.
* **Custom constraints**: PyPSA's own ``optimize_mga*`` helpers build their own model
  and never run an ``extra_functionality`` hook, so the project's custom constraints
  (e.g. ``add_max_RE_sales_constraint``) would be silently dropped and the near-optimal
  space computed against a *looser* feasible region. We therefore reproduce the MGA
  step manually (:func:`_solve_mga_in_direction`) and inject
  :func:`scripts.helpers.apply_custom_constraints` into every solve.
* **Budget constraint**: the ε-near-optimal definition of Neumann & Brown (2021) and
  Grochowicz et al. (2023) — bound the optimised objective ``c·x`` within a fraction
  ``slack`` of the optimum, ``c·x ≤ c_opt + slack·|c_opt|`` with ``c_opt = n.objective``
  (:func:`_add_budget_constraint`). The ``|c_opt|`` makes it sign-robust: for ``c_opt ≥ 0``
  (demand mode) it is the textbook ``(1+ε)·c_opt``; for ``c_opt < 0`` (price mode, net
  revenue) it becomes ``(1−ε)·c_opt`` so the optimum stays feasible. Identical in form
  for deterministic and stochastic networks (``n.objective`` is the expected cost), and
  needs no ``statistics`` (which mis-normalise across scenarios).
"""
from __future__ import annotations

import logging
import time as _time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Component class → (static attribute, capacity variable stem)
_COMP_SPECS = [
    ("Generator",   "generators",    "p_nom"),
    ("Link",        "links",         "p_nom"),
    ("StorageUnit", "storage_units", "p_nom"),
    ("Store",       "stores",        "e_nom"),
]
_CLS_TO_ATTR = {cls: attr for cls, attr, _ in _COMP_SPECS}


def load_run_config(network_path) -> dict:
    """Load the ``config_run.yaml`` written next to a solved network, if present.

    ``solve_network`` dumps the full config used for the optimisation into
    ``<network folder>/config_run.yaml``. When NOS explores a network via
    ``mga.network_path`` (possibly solved under a *different* config), reading this
    file lets the analysis use that network's own ``n_flags`` / ``max_RE_to_grid``
    for the re-applied custom constraint — instead of trusting the live config to
    match. Returns ``{}`` if the file is absent.
    """
    import yaml
    from pathlib import Path

    p = Path(network_path)
    for cand in (p.parent / "config_run.yaml", p.parent.parent / "networks" / "config_run.yaml"):
        if cand.exists():
            with cand.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
    return {}


def is_stochastic(n) -> bool:
    """True if the network is a PyPSA stochastic (multi-scenario) network.

    Auto-detected from the solved network — no input flag needed. Stochastic
    networks broadcast component tables to a ``(scenario, name)`` MultiIndex; the
    shared first-stage *investment* variables (``p_nom`` / ``e_nom``) are
    deduplicated to plain names by PyPSA, so the MGA solve itself is unaffected —
    only the static-reading helpers need to collapse the scenario level.
    """
    if getattr(n, "has_scenarios", False):
        return True
    # fallback for older PyPSA: look for a 'scenario' index level
    idx = getattr(getattr(n, "generators", None), "index", None)
    return isinstance(idx, pd.MultiIndex) and "scenario" in (idx.names or [])


def _flat_static(n, attr):
    """Return a component static table indexed by plain component name.

    For stochastic networks the per-scenario rows are collapsed (investment
    attributes — ``capital_cost``, ``*_extendable``, ``*_nom_opt``, ``carrier`` —
    are identical across scenarios for the shared first-stage variables). For
    deterministic networks the table is returned unchanged.
    """
    df = getattr(n, attr, None)
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex) and "name" in (df.index.names or []):
        return df.groupby(level="name").first()
    return df


def build_comp_tech_map(n, tech_index) -> dict[str, str]:
    """Component-name → tech key mapping, stochastic-safe.

    Wraps ``scripts.prepare_network._build_comp_tech_map`` (the same logic used at
    build time) but feeds it scenario-collapsed static tables, so it works on both
    deterministic and stochastic solved networks. ``comp_tech_map`` does not survive
    a netcdf round-trip, so the wrapper rebuilds it via this function.
    """
    from types import SimpleNamespace
    from scripts.prepare_network import _build_comp_tech_map

    flat = SimpleNamespace(**{attr: _flat_static(n, attr) for _, attr, _ in _COMP_SPECS})
    return _build_comp_tech_map(flat, set(tech_index))


def project_static(n, dimensions: dict) -> pd.Series:
    """Project a *solved static* network onto the selected dimensions (MW per dim).

    Reads optimal capacities from ``p_nom_opt`` / ``e_nom_opt`` directly, so it works
    on a network loaded from netcdf without an in-memory linopy model — unlike PyPSA's
    ``project_solved`` which requires a built model. Used for the cost-optimum reference.
    Scenario-collapsed for stochastic networks (investment is shared across scenarios).
    """
    out = {}
    for d, weights in dimensions.items():
        total = 0.0
        for cls, attrs in weights.items():
            df = _flat_static(n, _CLS_TO_ATTR[cls])
            if df is None:
                continue
            for nom, comps in attrs.items():
                col = f"{nom}_opt" if f"{nom}_opt" in df.columns else nom
                for name, w in comps.items():
                    if name in df.index:
                        total += float(w) * float(df.at[name, col])
        out[d] = total
    return pd.Series(out)


# --------------------------------------------------------------------------- #
# Dimension registry (the σ aggregation map of the papers)
# --------------------------------------------------------------------------- #
def available_dimensions(
    n,
    comp_tech_map: dict[str, str] | None = None,
    n_config_index=None,
    weight_by: str = "capacity",
) -> dict[str, dict]:
    """Auto-derive the selectable NOS dimensions from a built/solved network.

    A *dimension* is a technology; its value is a weighted sum over all extendable,
    costed components belonging to it. The returned structure is the nested
    ``weights`` dict that PyPSA's MGA API expects::

        {tech_key: {"Generator": {"p_nom": {comp_name: weight, ...}},
                    "Store":     {"e_nom": {comp_name: weight, ...}}, ...}}

    Grouping key per component: ``comp_tech_map[name]`` if available, else the
    component ``carrier`` (recovers techs that ``comp_tech_map`` drops, e.g. battery).
    Only components with ``capital_cost > 0`` are considered (skips zero-cost helper
    links such as the grid-sell store).

    Parameters
    ----------
    comp_tech_map
        ``component-name → tech key`` mapping (rebuild via :func:`build_comp_tech_map`
        — it does not survive a netcdf round-trip). If ``None``, grouping falls back to
        carrier only.
    n_config_index
        Iterable of valid ``n_config`` technology keys. If given, only dimensions
        whose key is in this set are kept (the "core technology" gate that removes
        auxiliary/balance-of-plant groupings). If ``None``, all groups are kept.
    weight_by : "capacity" | "investment"
        ``"capacity"`` (default): weight 1.0 per component — a dimension is the
        summed installed capacity in MW, GreenBubble's original convention.
        ``"investment"``: weight = the component's own ``capital_cost``
        (EUR/MW/year, already annualised — GreenBubble's ``capital_cost`` is
        computed via ``annuity()`` at build time, so no extra "scale by years"
        step is needed here, unlike the reference implementation working from
        multi-year raw investment figures) — a dimension becomes that
        technology's summed *annual investment cost* (EUR/year), matching the
        Grochowicz et al. (2023) reference implementation's own
        ``projection:`` config (every entry there sets
        ``weight: "capital_cost"``; see ``config/config-default.yaml`` in
        https://github.com/aleks-g/intersecting-near-opt-spaces). The
        near-optimal-space mathematics is agnostic to this choice — it's a
        modelling decision about what "a dimension" physically represents,
        not part of the theory (see docs/meetings/grochowicz_nos_summary).

    Stochastic networks are handled transparently: the ``(scenario, name)`` index is
    collapsed to plain names (the shared first-stage investment), so dimensions are
    keyed by plain component names — exactly what PyPSA's MGA expects.
    """
    if weight_by not in ("capacity", "investment"):
        raise ValueError(f"weight_by must be 'capacity' or 'investment', got {weight_by!r}.")
    comp_tech_map = comp_tech_map or {}
    core = set(n_config_index) if n_config_index is not None else None

    dims: dict[str, dict] = {}
    for cls, attr, nom in _COMP_SPECS:
        df = _flat_static(n, attr)
        ext_col = f"{nom}_extendable"
        if df is None or df.empty or ext_col not in df.columns:
            continue
        cc = df["capital_cost"].fillna(0.0) if "capital_cost" in df.columns else 0.0
        mask = df[ext_col].astype(bool) & (cc > 0)
        for name, row in df[mask].iterrows():
            tech_key = comp_tech_map.get(name)
            carrier_key = str(row.get("carrier", "")) or None
            if core is not None:
                # Prefer whichever grouping key is an actual n_config technology.
                # comp_tech_map is resolved for tech_costs lookups (e.g. the
                # size-suffixed "AEC large"/"AEC small" electrolysis rows,
                # scripts.prepare_network._build_comp_tech_map), which is right
                # for a cost report but not necessarily an n_config key; carrier
                # is often the bare technology name instead (e.g. "AEC", set
                # directly as the Link's carrier in add_electrolysis) and should
                # win when it's the one that actually matches n_config.
                if tech_key in core:
                    key = tech_key
                elif carrier_key in core:
                    key = carrier_key
                else:
                    key = tech_key or carrier_key
            else:
                key = tech_key or carrier_key
            if not key:
                continue
            if core is not None and key not in core:
                continue
            weight = float(row["capital_cost"]) if weight_by == "investment" else 1.0
            dims.setdefault(key, {}).setdefault(cls, {}).setdefault(nom, {})[name] = weight
    return dims


def resolve_dimensions(
    n,
    selected,
    comp_tech_map: dict[str, str] | None = None,
    n_config_index=None,
    weight_by: str = "capacity",
) -> dict[str, dict]:
    """Validate the user-selected dimension names and return the restricted weights dict.

    Empty / falsy ``selected`` ⇒ use *all* available dimensions. Unknown names raise
    a ``ValueError`` that lists what is available. ``weight_by`` — see
    :func:`available_dimensions`.
    """
    avail = available_dimensions(n, comp_tech_map, n_config_index, weight_by=weight_by)
    if not avail:
        raise ValueError(
            "No extendable, costed technologies found in the network — nothing to "
            "explore. Check that at least one technology has `expansion: true` in n_config."
        )
    selected = list(selected or [])
    if not selected:
        return dict(sorted(avail.items()))

    missing = [d for d in selected if d not in avail]
    if missing:
        raise ValueError(
            f"MGA dimension(s) {missing} are not available (not extendable, not costed, "
            f"or not a known technology in this network setup).\n"
            f"Available dimensions: {sorted(avail)}"
        )
    return {d: avail[d] for d in selected}


# --------------------------------------------------------------------------- #
# Constraint-correct MGA primitives
# --------------------------------------------------------------------------- #
def optimal_objective(n) -> float:
    """Return the cost-optimal objective value ``c_opt = n.objective``.

    This is the quantity the LP minimises (the ``c·x`` of Neumann & Brown 2021 /
    Grochowicz et al. 2023). For a stochastic network it is the *probability-weighted
    expected* cost — read directly off the solved network, so no scenario-aware
    statistics handling is needed. Must be captured *before* any MGA solve, which
    overwrites the network solution.
    """
    if not getattr(n, "is_solved", False):
        raise ValueError(
            "Network must be solved (cost-optimal) before NOS exploration — "
            "load the *_OPT.nc network."
        )
    return float(n.objective)


def _add_budget_constraint(m, cost_objective, c_opt: float, slack: float) -> None:
    """Add the ε-near-optimal cost budget to model ``m``.

    Implements the near-optimal feasible space of Neumann & Brown (2021) and
    Grochowicz et al. (2023): a solution is near-optimal if its objective stays
    within a fraction ``slack`` (= ε) of the cost optimum::

        c·x  ≤  c_opt + slack·|c_opt|

    where ``c·x`` is the model objective (``cost_objective``, captured before it is
    overwritten with the MGA direction objective) and ``c_opt = n.objective``. The
    ``|c_opt|`` makes the band sign-robust: for ``c_opt ≥ 0`` (demand mode) this is
    the textbook ``(1 + ε)·c_opt``; for ``c_opt < 0`` (price mode, where net revenue
    makes the objective negative) it becomes ``(1 − ε)·c_opt``, the correct band that
    still *contains* the optimum. It is identical in form for deterministic and
    stochastic networks (``n.objective`` already being the expected cost), so the
    near-optimal space is defined consistently across demand, price and stochastic.
    """
    from linopy.expressions import LinearExpression, QuadraticExpression

    expr = cost_objective
    if not isinstance(expr, (LinearExpression, QuadraticExpression)):
        expr = expr.expression
    bound = c_opt + slack * abs(c_opt)
    m.add_constraints(expr <= bound, name="budget")


def _solve_mga_in_direction(
    n,
    direction: dict,
    dimensions: dict,
    c_opt: float,
    slack: float,
    n_flags: dict | None = None,
    re_alpha: float | None = None,
    solver_name: str = "highs",
    solver_options: dict | None = None,
) -> tuple[str, str, pd.Series | None]:
    """Constraint-correct replacement for ``n.optimize.optimize_mga_in_direction``.

    Builds the model, injects the project's custom constraints, adds the budget, sets
    the objective to ``-Σ direction[k]·capacity_k`` (so minimising it *maximises* the
    projection along ``direction``), solves, and returns the projected coordinates.
    """
    from scripts.helpers import apply_custom_constraints

    m = n.optimize.create_model()
    cost_objective = m.objective  # capture before overwriting
    apply_custom_constraints(n, m, n_flags=n_flags, re_alpha=re_alpha)
    _add_budget_constraint(m, cost_objective, c_opt, slack)

    m.objective = -sum(
        float(direction[k]) * n.optimize.build_linexpr_from_weights(dimensions[k], model=m)
        for k in dimensions
        if direction.get(k, 0)
    )

    opts = dict(solver_options or {})
    status, condition = n.optimize.solve_model(solver_name=solver_name, **opts)
    coords = n.optimize.project_solved(dimensions) if status == "ok" else None
    return status, condition, coords


# --------------------------------------------------------------------------- #
# Tier 1 — per-technology capacity ranges
# --------------------------------------------------------------------------- #
def mga_ranges(
    n,
    dimensions: dict,
    slack: float = 0.05,
    n_flags: dict | None = None,
    re_alpha: float | None = None,
    solver_name: str = "highs",
    solver_options: dict | None = None,
    return_points: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Tier 1: min and max installed capacity per dimension within the cost budget.

    Returns a DataFrame indexed by dimension with columns ``[optimal, min, max]`` (MW)
    and the boolean flags ``must_have`` (min > tol) and ``must_avoid`` (max ≤ tol).

    ``return_points=True`` additionally returns the full coordinate vector (all
    dimensions, not just the one being minimised/maximised) from each of the
    ``2 * len(dimensions)`` cardinal-direction solves, as a second DataFrame —
    the real extreme points those solves actually landed on, not points
    reconstructed from the ``[min, max]`` table (which only keeps the projected
    scalar per dimension, not the other dimensions' values at that solve). This
    is the seed point set :func:`explore_hull_adaptive` (or the staged
    ``nos_seed`` -> ``nos_hull_adaptive`` pipeline) builds on, so it solves
    exactly once per cardinal direction rather than duplicating this loop.
    """
    c_opt = optimal_objective(n)
    keys = list(dimensions)

    # capacity at the cost optimum (read from the loaded static solution)
    optimal = project_static(n, dimensions)

    rows = {}
    points = [] if return_points else None
    for d in keys:
        res = {"optimal": float(optimal.get(d, np.nan))}
        for sense, sign in (("min", -1.0), ("max", +1.0)):
            status, cond, coords = _solve_mga_in_direction(
                n, {d: sign}, dimensions, c_opt, slack,
                n_flags=n_flags, re_alpha=re_alpha,
                solver_name=solver_name, solver_options=solver_options,
            )
            if status != "ok" or coords is None:
                logger.warning("MGA %s of dimension %r failed: %s / %s", sense, d, status, cond)
                res[sense] = np.nan
            else:
                res[sense] = float(coords[d])
                if return_points:
                    points.append(coords.reindex(keys))
        rows[d] = res

    df = pd.DataFrame.from_dict(rows, orient="index")[["optimal", "min", "max"]]
    tol = 1e-3  # MW
    df["must_have"] = df["min"] > tol
    df["must_avoid"] = df["max"] <= tol
    if return_points:
        return df, pd.DataFrame(points, columns=keys).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Tier 2 — near-optimal hull
# --------------------------------------------------------------------------- #
def generate_directions(keys, n_directions: int, sampling: str = "halton", seed=None) -> pd.DataFrame:
    """Wrapper over PyPSA's direction generators (halton | evenly_spaced | random)."""
    import pypsa.optimization.mga as mga

    keys = list(keys)
    if sampling == "evenly_spaced":
        return mga.generate_directions_evenly_spaced(keys, n_directions)
    if sampling == "random":
        return mga.generate_directions_random(keys, n_directions, seed=seed)
    if sampling == "halton":
        return mga.generate_directions_halton(keys, n_directions, seed=seed)
    raise ValueError(f"Unknown direction_sampling {sampling!r}; use halton|evenly_spaced|random.")


def explore_hull(
    n,
    dimensions: dict,
    slack: float = 0.05,
    n_directions: int = 40,
    sampling: str = "halton",
    seed=None,
    n_flags: dict | None = None,
    re_alpha: float | None = None,
    solver_name: str = "highs",
    solver_options: dict | None = None,
    include_axis_corners: bool = True,
    c_opt: float | None = None,
) -> dict:
    """Tier 2: collect extreme points of the near-optimal polytope.

    Solves one MGA problem per sampled direction (serial — the constraint-correct
    path cannot use PyPSA's parallel helper). Optionally seeds the point set with the
    ±unit-axis corners (the min/max of each single dimension) for a better hull.

    ``c_opt`` overrides the cost-budget anchor (the optimal objective) — used by the
    robustness tier to impose a single shared bound c* across years (Grochowicz et al.
    eq. 9). If ``None`` the network's own optimum is used.

    Returns a dict with keys: ``points`` (DataFrame, rows=solves, cols=dimensions),
    ``directions`` (DataFrame), ``hull`` (scipy ConvexHull or ``None`` if degenerate),
    ``volume`` (float or NaN), and ``c_opt`` (float, the budget anchor used).
    """
    c_opt = optimal_objective(n) if c_opt is None else float(c_opt)
    keys = list(dimensions)

    directions: list[dict] = []
    if include_axis_corners:
        for d in keys:
            directions.append({k: (1.0 if k == d else 0.0) for k in keys})   # max d
            directions.append({k: (-1.0 if k == d else 0.0) for k in keys})  # min d
    if n_directions > 0:
        dirs_df = generate_directions(keys, n_directions, sampling, seed)
        directions.extend(dirs_df.to_dict("records"))

    pts, used = [], []
    for i, direction in enumerate(directions):
        status, cond, coords = _solve_mga_in_direction(
            n, direction, dimensions, c_opt, slack,
            n_flags=n_flags, re_alpha=re_alpha,
            solver_name=solver_name, solver_options=solver_options,
        )
        if status == "ok" and coords is not None:
            pts.append(coords.reindex(keys))
            used.append(direction)
        else:
            logger.warning("MGA direction %d/%d failed: %s / %s", i + 1, len(directions), status, cond)

    points = pd.DataFrame(pts, columns=keys).reset_index(drop=True)
    directions_df = pd.DataFrame(used, columns=keys).reset_index(drop=True)

    hull, volume = _safe_convex_hull(points.to_numpy())
    return {
        "points": points,
        "directions": directions_df,
        "hull": hull,
        "volume": float(volume) if volume is not None else np.nan,
        "c_opt": c_opt,
        "dimensions": keys,
    }


def _safe_convex_hull(arr: np.ndarray):
    """ConvexHull with graceful handling of <k+1 points or degenerate (flat) sets."""
    from scipy.spatial import ConvexHull, QhullError

    if arr.shape[0] < arr.shape[1] + 1:
        return None, None
    try:
        h = ConvexHull(arr)
        return h, h.volume
    except QhullError as e:
        logger.warning("ConvexHull failed (degenerate near-optimal space): %s", e)
        return None, None


# --------------------------------------------------------------------------- #
# Tier 2 (adaptive) — Grochowicz et al.'s iterative facet/centre-ball sampler
# --------------------------------------------------------------------------- #
def explore_hull_adaptive(
    n,
    dimensions: dict,
    slack: float = 0.05,
    direction_method: str = "maximal-centre-then-facets",
    direction_angle_sep: float = 10.0,
    angle_tolerance: float = 0.1,
    conv_method: str = "volume",
    conv_eps: float = 1.0,
    conv_iter: int = 3,
    max_iter: int = 100,
    n_flags: dict | None = None,
    re_alpha: float | None = None,
    solver_name: str = "highs",
    solver_options: dict | None = None,
    seed=None,
    c_opt: float | None = None,
    seed_points: pd.DataFrame | None = None,
) -> dict:
    """Tier 2 (adaptive) — near-optimal hull via Grochowicz et al.'s iterative sampler.

    Alternative to :func:`explore_hull`'s fixed-count sampling: probes new search
    directions *adaptively*, chosen from the boundary of the hull built so far
    (facet normals, directions touching the current Chebyshev ball, or both), and
    stops on a volume- or Chebyshev-centre-shift convergence criterion rather than
    a fixed direction budget. Ported from the official Grochowicz et al. (2023)
    reference implementation's ``workflow/scripts/compute_near_opt.py``
    (https://github.com/aleks-g/intersecting-near-opt-spaces) — the direction
    generators and hull bookkeeping are the vendored
    :mod:`scripts.vendor.near_opt_geometry` (GPL-3.0-or-later, see that module's
    header), reused as-is; only the per-direction *solve* is GreenBubble's own
    constraint-correct :func:`_solve_mga_in_direction` (their code solves via the
    pre-Linopy ``pypsa.linopf`` API and in parallel worker processes — this ports
    the algorithm, run serially, onto GreenBubble's PyPSA 1.0.7 Linopy solve path;
    see the module docstring's provenance note).

    Parameters
    ----------
    direction_method : "facets" | "random-uniform" | "random-lhc" | "maximal-centre"
        | "maximal-centre-then-facets"
        Same five options as the reference implementation's ``directions`` config.
    conv_method : "volume" | "centre"
        Stop once the change between successive iterations (hull volume, or
        Euclidean shift of the Chebyshev centre as a percent of its norm) stays
        below ``conv_eps`` for ``conv_iter`` consecutive iterations.

    seed_points : pd.DataFrame | None
        Pre-solved cardinal-direction extreme points (columns = ``dimensions``
        keys), e.g. from ``mga_ranges(..., return_points=True)`` or the staged
        ``nos_seed`` Snakemake rule's output — skips the internal seeding
        solve loop below and reuses these instead. The direction-filter seed
        (``used_directions``) is set to the canonical ``±eye(k)`` cardinal set
        regardless of the exact per-row solve order in ``seed_points``: for
        filtering purposes only the *set* of already-probed directions matters
        (which cardinal axes to avoid re-suggesting), not which specific point
        value each one produced, so no direction↔point pairing is needed.

    Returns
    -------
    dict with keys ``points``, ``hull``, ``volume``, ``c_opt``, ``dimensions``,
    ``iterations`` (int, solves that actually advanced the hull), ``converged``
    (bool) — same shape as :func:`explore_hull`'s result plus the last two.
    """
    from scipy.spatial import ConvexHull

    from scripts.vendor import near_opt_geometry as geo

    if conv_method not in ("volume", "centre"):
        raise ValueError("conv_method must be 'volume' or 'centre'.")

    c_opt = optimal_objective(n) if c_opt is None else float(c_opt)
    keys = list(dimensions)
    k = len(keys)

    if seed_points is not None:
        used_directions: list[np.ndarray] = list(np.eye(k)) + list(-np.eye(k))
        points = seed_points[keys].to_numpy()
        logger.info("Adaptive Tier 2: reusing %d pre-solved seed points.", len(points))
    else:
        # Seed with the cardinal ±unit-axis directions (mirrors the reference
        # implementation's `mga.py` pass, and GreenBubble's own Tier-1 unit axes).
        probed_directions: list[np.ndarray] = list(np.eye(k)) + list(-np.eye(k))
        seed_pts, used_directions = [], []
        for i, d in enumerate(probed_directions):
            _t0 = _time.time()
            direction = {key: float(d[i2]) for i2, key in enumerate(keys)}
            status, cond, coords = _solve_mga_in_direction(
                n, direction, dimensions, c_opt, slack,
                n_flags=n_flags, re_alpha=re_alpha,
                solver_name=solver_name, solver_options=solver_options,
            )
            logger.info(
                "Adaptive Tier 2 seed %d/%d (%s): %s in %.1fs",
                i + 1, len(probed_directions), status, cond, _time.time() - _t0,
            )
            if status == "ok" and coords is not None:
                seed_pts.append(coords.reindex(keys).to_numpy())
                used_directions.append(d)
            else:
                logger.warning("Adaptive MGA seed direction failed: %s / %s", status, cond)

        if len(seed_pts) < k + 1:
            raise RuntimeError(
                f"Only {len(seed_pts)}/{2 * k} cardinal-direction seed solves "
                f"succeeded — need at least {k + 1} to form an initial hull."
            )
        points = np.array(seed_pts)

    # Rescale so each dimension has width 1 (reference implementation: makes the
    # hull closer to an orthoplex, so direction sampling explores each dimension
    # evenly and qhull is better conditioned numerically).
    scaling_ranges = points.max(axis=0) - points.min(axis=0)
    if not (scaling_ranges > 1e-9).all():
        raise RuntimeError(
            "Degenerate near-optimal space after seeding: at least one dimension "
            "has (near-)zero range."
        )
    scaled_points = points / scaling_ranges
    scaled_hull = ConvexHull(scaled_points, incremental=True)

    centre, radius, _ = geo.ch_centre(scaled_hull)
    history = [(centre, radius, scaled_hull.volume)]

    if seed is not None:
        np.random.seed(seed)  # the vendored samplers draw from the global RNG

    if direction_method == "random-uniform":
        sampler = geo.uniform_random_hypersphere_sampler(k)
        dir_gen = geo.filter_vectors_auto(
            sampler, init_angle=direction_angle_sep,
            initial_vectors=used_directions, min_angle_tolerance=angle_tolerance,
        )
    elif direction_method == "random-lhc":
        sampler = geo.lhc_random_hypersphere_sampler(k)
        dir_gen = geo.filter_vectors_auto(
            sampler, init_angle=direction_angle_sep,
            initial_vectors=used_directions, min_angle_tolerance=angle_tolerance,
        )
    elif direction_method == "facets":
        dir_gen = geo.large_facet_directions(
            scaled_hull, used_directions, direction_angle_sep,
            autodecrease=True, min_angle_tolerance=angle_tolerance,
        )
    elif direction_method == "maximal-centre":
        dir_gen = geo.touching_ball_directions(scaled_hull, used_directions, angle_tolerance)
    elif direction_method == "maximal-centre-then-facets":
        dir_gen = geo.maximal_centre_then_facets(
            scaled_hull, used_directions, direction_angle_sep, angle_tolerance,
        )
    else:
        raise ValueError(
            f"Unknown direction_method {direction_method!r}; use facets | "
            "random-uniform | random-lhc | maximal-centre | maximal-centre-then-facets."
        )

    num_iters = 0
    converged = False
    while num_iters < max_iter:
        _t_dir = _time.time()
        try:
            d = next(dir_gen)
        except StopIteration:
            break
        logger.info("Direction generation (%s) took %.2fs", direction_method, _time.time() - _t_dir)
        if d is None:
            logger.info("Adaptive sampler ran out of directions after %d iterations.", num_iters)
            break

        # Mark as probed before solving (matches reference: prevents the
        # generator re-offering the same direction even if the solve fails).
        used_directions.append(d)
        direction = {key: float(d[i]) for i, key in enumerate(keys)}
        _t_solve = _time.time()
        status, cond, coords = _solve_mga_in_direction(
            n, direction, dimensions, c_opt, slack,
            n_flags=n_flags, re_alpha=re_alpha,
            solver_name=solver_name, solver_options=solver_options,
        )
        logger.info(
            "Adaptive Tier 2 solve %d (%s): %s in %.1fs",
            num_iters + 1, status, cond, _time.time() - _t_solve,
        )
        if status != "ok" or coords is None:
            logger.warning("Adaptive MGA direction failed: %s / %s — trying another.", status, cond)
            continue

        num_iters += 1
        p = coords.reindex(keys).to_numpy()
        scaled_hull.add_points([p / scaling_ranges])

        centre, radius, _ = geo.ch_centre(scaled_hull)
        history.append((centre, radius, scaled_hull.volume))

        if len(history) - 1 >= conv_iter:
            if conv_method == "volume":
                vols = np.array([h[2] for h in history])
                deltas = 100 * (vols[1:] - vols[:-1]) / vols[:-1]
            else:  # "centre"
                centres = [h[0] for h in history]
                dists = np.array([
                    np.linalg.norm(c2 - c1) for c1, c2 in zip(centres[:-1], centres[1:])
                ])
                norms = np.array([np.linalg.norm(c) for c in centres[1:]])
                deltas = 100 * np.divide(
                    dists, norms, out=np.zeros_like(dists), where=norms > 0
                )
            if np.all(np.abs(deltas[-conv_iter:]) < conv_eps):
                logger.info(
                    "Adaptive hull converged after %d iterations (%s delta < %.2f%%).",
                    num_iters, conv_method, conv_eps,
                )
                converged = True
                break

    final_scaled = scaled_hull.points[scaled_hull.vertices]
    final_points = final_scaled * scaling_ranges
    final_hull, final_volume = _safe_convex_hull(final_points)

    return {
        "points": pd.DataFrame(final_points, columns=keys),
        "hull": final_hull,
        "volume": float(final_volume) if final_volume is not None else np.nan,
        "c_opt": c_opt,
        "dimensions": keys,
        "iterations": num_iters,
        "converged": converged,
    }


# --------------------------------------------------------------------------- #
# Tier 3 — robustness: intersection + Chebyshev centre
# --------------------------------------------------------------------------- #
def chebyshev_centre(hulls: list, keys=None) -> dict:
    """Chebyshev centre of the *intersection* of several near-optimal hulls.

    Each entry in ``hulls`` is either a scipy ``ConvexHull`` or an ``(N, k)`` point
    array. The intersection's half-spaces are the union of every hull's facet
    inequalities; the centre is the deepest interior point, found via the LP
    (Grochowicz et al. 2023, §2.4)::

        max r   s.t.   aᵢ·x + ‖aᵢ‖·r ≤ bᵢ  for every facet i,   r ≥ 0

    Returns ``{"centre": np.ndarray(k), "radius": float, "feasible": bool}``.
    A non-positive radius means the per-year near-optimal spaces do not overlap.

    Delegates to the vendored, solver-agnostic port of
    :func:`scripts.vendor.near_opt_geometry.ch_centre_from_constraints` (itself
    ported from the official Grochowicz et al. (2023) reference implementation,
    https://github.com/aleks-g/intersecting-near-opt-spaces) rather than a
    separate from-scratch LP — see ``scripts/vendor/near_opt_geometry.py`` for
    provenance and the GPL-3.0-or-later licence boundary this crosses.
    """
    from scipy.spatial import ConvexHull

    from scripts.vendor import near_opt_geometry as geo

    A_rows, eq_rows = [], []
    for h in hulls:
        if not isinstance(h, ConvexHull):
            h = ConvexHull(np.asarray(h))
        eq_rows.append(h.equations)  # rows [a_1..a_k, b] meaning a·x <= -b
    constraints = np.vstack(eq_rows)

    centre, radius, _tight = geo.ch_centre_from_constraints(constraints)
    if centre is None:
        return {"centre": None, "radius": float("nan"), "feasible": False}
    radius = float(radius)
    return {
        "centre": pd.Series(centre, index=list(keys)) if keys is not None else centre,
        "radius": radius,
        "feasible": radius > 0,
    }


def intersect_hulls(hulls: list) -> np.ndarray | None:
    """Vertices of the (approximate) geometric intersection of several hulls.

    Thin wrapper around the vendored
    :func:`scripts.vendor.near_opt_geometry.intersection` (qhull
    ``HalfspaceIntersection``-based, exact up to qhull's own approximation
    tolerance — see that function's docstring). Returns ``None`` if the
    intersection is empty. Each entry in ``hulls`` is a ``ConvexHull`` or an
    ``(N, k)`` point array, same as :func:`chebyshev_centre`.

    Unlike :func:`chebyshev_centre` (which only needs the *interior point*
    of the intersection), this recovers the intersection's full boundary —
    useful for plotting the actual intersected near-optimal region rather
    than only its centre.
    """
    from scipy.spatial import ConvexHull

    from scripts.vendor import near_opt_geometry as geo

    hulls = [h if isinstance(h, ConvexHull) else ConvexHull(np.asarray(h)) for h in hulls]
    return geo.intersection(hulls)


def realise_design(
    n,
    dimensions: dict,
    centre,
    slack: float = 0.05,
    n_flags: dict | None = None,
    re_alpha: float | None = None,
    solver_name: str = "highs",
    solver_options: dict | None = None,
) -> tuple[str, str]:
    """Map a low-dimensional point ``centre`` back to a full network design (the φ map).

    Fixes the aggregate capacity of each dimension to its ``centre`` value
    (``Σ capacity_d == centre[d]``) and re-solves the cost-minimising model subject to
    the budget, mutating ``n`` in place. Returns the solve ``(status, condition)``.
    """
    from scripts.helpers import apply_custom_constraints

    c_opt = optimal_objective(n)
    centre = pd.Series(centre) if not isinstance(centre, pd.Series) else centre

    m = n.optimize.create_model()
    cost_objective = m.objective
    apply_custom_constraints(n, m, n_flags=n_flags, re_alpha=re_alpha)
    _add_budget_constraint(m, cost_objective, c_opt, slack)
    for d, expr_weights in dimensions.items():
        if d not in centre.index:
            continue
        lhs = n.optimize.build_linexpr_from_weights(expr_weights, model=m)
        m.add_constraints(lhs == float(centre[d]), name=f"fix_{d}")

    opts = dict(solver_options or {})
    status, condition = n.optimize.solve_model(solver_name=solver_name, **opts)
    return status, condition


# --------------------------------------------------------------------------- #
# Tier 3 (continued) — simultaneous multi-year feasibility check
#
# Grochowicz et al. (2023)'s own validation step (their `validate_robust` /
# `solve_operations.py` + `summarise_feasibility.py`, see
# scripts/vendor/near_opt_feasibility.py for the ported primitives): rather
# than trusting that a Chebyshev-centre design realised against one reference
# network (:func:`realise_design`) is also feasible in every other year -- the
# hull it was checked against is only a *sampled approximation* of the true
# near-optimal region -- fix that design's capacities and re-solve pure
# operations (no more capacity expansion) against each year's own network,
# with a load-shedding safety valve so a shortfall shows up as a graded
# curtailment number instead of an opaque "infeasible".
# --------------------------------------------------------------------------- #
_NOMINAL_ATTR = {
    "generators": "p_nom",
    "links": "p_nom",
    "stores": "e_nom",
    "storage_units": "p_nom",
}


def apply_realised_capacities(n_target, n_source) -> None:
    """Copy a realised design's solved capacities from ``n_source`` onto ``n_target``.

    ``n_source`` is the network :func:`realise_design` was run against -- its
    ``_opt`` columns hold the per-component allocation consistent with the
    Chebyshev centre (the aggregate-dimension-to-individual-component mapping
    problem, solved once). ``n_target`` is a *different* year's network sharing
    the same component index (same model structure, different weather/price/
    demand time series); this transplants the allocation rather than
    re-deriving it. Modifies ``n_target`` in place.

    GreenBubble-specific: the reference implementation's ``set_nom_to_opt``
    (not vendored -- see ``near_opt_feasibility.py``'s docstring) copies
    ``_opt`` onto nominal *within one network*, since their multi-year
    robustness check runs on one combined multi-year network. GreenBubble
    instead has separate per-year network files, so the copy has to cross
    networks -- this function exists because of that architecture difference,
    not because the underlying idea differs.
    """
    for comp_attr, nom in _NOMINAL_ATTR.items():
        src_df = getattr(n_source, comp_attr)
        tgt_df = getattr(n_target, comp_attr)
        if src_df.empty or tgt_df.empty:
            continue
        opt_col = f"{nom}_opt"
        common = src_df.index.intersection(tgt_df.index)
        if len(common) == 0:
            continue
        vals = src_df.loc[common, opt_col] if opt_col in src_df.columns else src_df.loc[common, nom]
        tgt_df.loc[common, nom] = vals.astype(float).fillna(0.0)
        # Dirty fix mirroring solve_operations.py: replace any remaining NaN
        # nominal capacities (components untouched by the copy above) with 0.
        tgt_df[nom] = tgt_df[nom].fillna(0.0)


def add_load_shedding(n, marginal_cost: float = 7.3e3) -> None:
    """Add a slack generator at every bus carrying a Load, so a design that
    can't quite meet demand shows up as (expensive) load shedding instead of
    an outright infeasible solve.

    GreenBubble-specific adaptation of the reference implementation's
    ``solve_operations.add_load_shedding``: their version adds one at every
    AC-carrier bus, since their model is a country-level power network.
    GreenBubble has no generic "AC bus" concept -- demand instead sits on
    named product buses (H2/bioCH4/Methanol delivery, Heat DH, ...) -- so this
    targets every bus that actually carries a Load component instead.
    ``marginal_cost`` default (EUR/MWh-equivalent) is their same value (a
    highRES / Price & Zeyringer 2022 value-of-lost-load figure), applied
    uniformly across all Load buses regardless of commodity -- a
    simplification, same in spirit as their blanket per-AC-bus price.
    """
    if "load-shedding" not in n.carriers.index:
        n.add("Carrier", "load-shedding")
    for bus in n.loads["bus"].unique():
        name = f"{bus} load shedding"
        if name in n.generators.index:
            continue
        n.add(
            "Generator", name,
            bus=bus, carrier="load-shedding",
            marginal_cost=marginal_cost, p_nom=1e6,
        )


def validate_design_across_years(
    realised_network,
    year_network_paths: dict,
    n_flags: dict | None = None,
    re_alpha: float | None = None,
    solver_name: str = "highs",
    solver_options: dict | None = None,
) -> pd.DataFrame:
    """Simultaneous multi-year feasibility check (Grochowicz et al. 2023's
    ``validate_robust`` / ``solve_operations.py`` + ``summarise_feasibility.py``).

    ``realised_network`` is the output of :func:`realise_design` (its ``_opt``
    capacities define the design under test). For each entry in
    ``year_network_paths`` (``{year_label: path to that year's network}``),
    transplants those capacities (:func:`apply_realised_capacities`), fixes
    them non-extendable
    (:func:`scripts.vendor.near_opt_feasibility.set_extendable_false`), adds
    load-shedding slack at every Load bus (:func:`add_load_shedding`),
    re-applies GreenBubble's custom constraints (the design must still respect
    the RE-to-grid policy etc. -- same reasoning as
    :func:`_solve_mga_in_direction`), and re-solves *operations only* (no more
    capacity expansion).

    Returns one row per year
    (:func:`scripts.vendor.near_opt_feasibility.compute_feasibility_criteria`'s
    columns), plus the solve ``status``/``condition``.
    """
    import pypsa

    from scripts.helpers import apply_custom_constraints
    from scripts.vendor.near_opt_feasibility import (
        compute_feasibility_criteria,
        set_extendable_false,
    )

    rows = []
    for year, path in year_network_paths.items():
        n_year = pypsa.Network(str(path))
        apply_realised_capacities(n_year, realised_network)
        set_extendable_false(n_year)
        add_load_shedding(n_year)

        m = n_year.optimize.create_model()
        apply_custom_constraints(n_year, m, n_flags=n_flags, re_alpha=re_alpha)
        opts = dict(solver_options or {})
        status, condition = n_year.optimize.solve_model(solver_name=solver_name, **opts)

        if status == "ok":
            row = compute_feasibility_criteria(n_year, str(year))
        else:
            row = pd.DataFrame(
                [[float("nan"), float("nan")]],
                columns=["Total curtailment", "Relative curtailment"],
                index=[str(year)],
            )
        row["status"] = status
        row["condition"] = condition
        rows.append(row)

    return pd.concat(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _itertools_pairs(keys):
    from itertools import combinations
    return list(combinations(keys, 2))


def plot_ranges(ranges_df: pd.DataFrame, slack: float, outpath, title: str | None = None,
                 unit: str = "installed capacity [MW]"):
    """Tier 1 figure: horizontal bar of the [min, max] band per technology,
    with the cost-optimal value marked. ``unit`` labels the x-axis — override
    for ``mga.dimension_weight: investment`` dimensions (e.g. "annual investment
    [EUR/y]"), since the values are no longer MW in that mode."""
    import matplotlib.pyplot as plt

    df = ranges_df.sort_values("max", ascending=True)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(df) + 1)))
    ax.barh(y, df["max"] - df["min"], left=df["min"], height=0.55,
            color="#9ecae1", edgecolor="#3182bd", label=f"near-optimal band (slack {slack:.0%})")
    ax.scatter(df["optimal"], y, color="#de2d26", zorder=3, label="cost optimum")
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.set_xlabel(unit)
    ax.set_title(title or f"Near-optimal capacity ranges (slack {slack:.0%})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hull_projections(
    result: dict,
    optimal: pd.Series | None,
    outpath,
    scale: float = 1e3,
    centre: pd.Series | None = None,
    unit: str = "GW",
):
    """Tier 2 figure: pairwise 2-D projections of the near-optimal point cloud + convex
    hull, with the cost optimum marked. ``scale`` divides the raw dimension values
    before plotting (default 1e3, i.e. MW -> GW for capacity-weighted dimensions).

    ``unit`` labels the axes — override together with ``scale`` for
    ``mga.dimension_weight: investment`` dimensions, e.g. ``scale=1e6, unit="M EUR/y"``
    (the raw values are then EUR/year, not MW).

    ``centre``, if given, marks the hull's own Chebyshev centre (deepest interior
    point) — not a multi-year intersection centre like :func:`plot_robustness`'s,
    just this one hull's. Compute it with ``chebyshev_centre([result["hull"]],
    keys=result["dimensions"])["centre"]`` (or straight off ``result["points"]`` if
    ``result["hull"]`` is ``None``).
    """
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull, QhullError

    keys = result["dimensions"]
    pts = result["points"]
    pairs = _itertools_pairs(keys)
    if not pairs:
        return
    ncol = min(3, len(pairs))
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)

    for idx, (a, b) in enumerate(pairs):
        ax = axes[idx // ncol][idx % ncol]
        xy = pts[[a, b]].to_numpy() / scale
        ax.scatter(xy[:, 0], xy[:, 1], s=12, color="#3182bd", alpha=0.7)
        if xy.shape[0] >= 3:
            try:
                h = ConvexHull(xy)
                for s in h.simplices:
                    ax.plot(xy[s, 0], xy[s, 1], color="#08519c", lw=0.8)
            except QhullError:
                pass
        if optimal is not None and a in optimal.index and b in optimal.index:
            ax.scatter([optimal[a] / scale], [optimal[b] / scale],
                       color="#de2d26", marker="*", s=120, zorder=4, label="optimum")
        if centre is not None and a in centre.index and b in centre.index:
            ax.scatter([centre[a] / scale], [centre[b] / scale],
                       color="black", marker="X", s=110, zorder=5, label="Chebyshev centre")
        if (optimal is not None) or (centre is not None):
            ax.legend(fontsize=7, loc="best")
        ax.set_xlabel(f"{a} [{unit}]")
        ax.set_ylabel(f"{b} [{unit}]")
        ax.grid(alpha=0.3)
    for j in range(len(pairs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Near-optimal feasible space (slack {result.get('slack', '')})")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(per_year_points: dict, centre, keys, outpath, scale: float = 1e3, unit: str = "GW"):
    """Tier 3 figure: per-year near-optimal hulls overlaid on each 2-D projection, with
    the Chebyshev centre of their intersection marked. ``unit``/``scale`` — see
    :func:`plot_hull_projections` (override together for investment-weighted dimensions)."""
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull, QhullError

    keys = list(keys)
    pairs = _itertools_pairs(keys)
    if not pairs:
        return
    ncol = min(3, len(pairs))
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
    cmap = plt.get_cmap("tab10")

    for idx, (a, b) in enumerate(pairs):
        ax = axes[idx // ncol][idx % ncol]
        for yi, (year, pts) in enumerate(per_year_points.items()):
            xy = pts[[a, b]].to_numpy() / scale
            color = cmap(yi % 10)
            ax.scatter(xy[:, 0], xy[:, 1], s=8, color=color, alpha=0.4)
            if xy.shape[0] >= 3:
                try:
                    h = ConvexHull(xy)
                    for s in h.simplices:
                        ax.plot(xy[s, 0], xy[s, 1], color=color, lw=0.8)
                    ax.plot([], [], color=color, label=str(year))
                except QhullError:
                    pass
        if centre is not None and a in centre.index and b in centre.index:
            ax.scatter([centre[a] / scale], [centre[b] / scale],
                       color="black", marker="X", s=110, zorder=5, label="Chebyshev centre")
        ax.set_xlabel(f"{a} [{unit}]")
        ax.set_ylabel(f"{b} [{unit}]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="best")
    for j in range(len(pairs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Robust design: intersection of per-year near-optimal spaces")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
