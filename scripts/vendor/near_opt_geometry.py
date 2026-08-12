# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# ---------------------------------------------------------------------------
# GREENBUBBLE NOTE (not part of the original file)
# ---------------------------------------------------------------------------
# Vendored from https://github.com/aleks-g/intersecting-near-opt-spaces
# (`workflow/scripts/geometry.py`), the reference implementation for
#   Grochowicz, A., van Greevenbroek, K., Benth, F.E. & Zeyringer, M. (2023).
#   "Intersecting Near-Optimal Spaces: European Power Systems with More
#   Resilience to Weather Variability." Energy Economics.
#   https://doi.org/10.1016/j.eneco.2022.106496
# Source commit: 057ea463d24b5822d94fbc2b7629e3fd1e84403c (`main`, 2023-12-19),
# i.e. the version current as of the vendoring date below, which post-dates
# the paper's `v1.0.1` release tag (2022-10-19) that GreenBubble's NOS docs
# otherwise cite for the formulas. Vendored 2026-08-11.
#
# This file is deliberately kept under the ORIGINAL GPL-3.0-or-later licence
# and attribution -- it is not relicensed as part of GreenBubble (MIT). Only
# import from here where that licence is acceptable for your use of
# GreenBubble; do not merge this code into MIT-licensed modules.
#
# The only substantive change from the original: every Gurobi
# (`gurobipy`/`GRB`) linear-program call is replaced with an equivalent
# `scipy.optimize.linprog` call, so this module has no hard Gurobi licence
# dependency -- consistent with GreenBubble's own solver-agnostic (HiGHS or
# Gurobi) design. The LP formulations themselves (objective, constraints,
# bounds) are unchanged; only the solver backend differs. Everything else
# (docstrings, algorithm structure, function names/signatures) is preserved
# as closely as possible so this stays diffable against the upstream file.
# ---------------------------------------------------------------------------

"""Common geometric functions for use in the scripts and notebooks.

This includes methods for working with convex hulls (taking
intersections, finding the Chebyshev centre, checking containment and
non-emptyness, optimising over a convex hull) and methods for
generating and filtering directions in various ways (random sampling,
based on facet normals, etc.)

"""

import logging
import math
import time
from typing import Collection, Iterable

import numpy as np
import scipy.linalg as linalg
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.stats.qmc import LatinHypercube

logger = logging.getLogger(__name__)


def intersection(hulls: Collection[ConvexHull], return_centre=False):
    """Compute an approximate intersection of a collection of convex hulls.

    This function returns the vertices of (an approximation of) the
    intersection of the given set of ConvexHull objects. The method is
    to collect all linear constraints for all given convex hulls;
    together these constraints exactly define the intersection of the
    hulls. To compute the vertices of the intersection from the
    constraints, we use qhull through the
    scipy.spatial.HalfspaceIntersection interface. Internally, qhull
    dualises the constaints to points and computes the convex hull of
    these points, then dualises the facets back to points.

    In dimension d, the convex hull of n vertices may consist of
    O(n^floor(d/2)) facets; by duality n constraints can define a
    polytope with O(n^floor(d/2)) vertices. Given that the input
    convex hulls of this function may already have many facets, we
    cannot hope to compute all vertices of the intersection.
    Therefore, we make qhull compute a reasonable approximation. This
    is controlled by the C, A and W options for qhull, which merge
    nearby vertices, adjacent facets at a close-to-0 degree angle and
    vertices close to facets respectively. The thresholds for merging
    are set such that the approximation is accurate up to about
    1/100th of the maximum width of the intersection in any dimension.

    The intersection is returned as an array of vertices, one vertex
    per row. Optionally, the Chebyshev centre and radius of the
    intersection may be returned, since these are computed regardless
    in the process of finding the intersection. See the documentation
    of `ch_centre` for more details.

    If the intersection cannot be found (for example, if it is empty),
    None is returned.

    Parameters
    ----------
    hulls : Collection[ConvexHull]
        Convex hulls to be intersected.

    Returns
    -------
    points : np.array of shape (num_vertices, dims)
        Approximation of the intersection of the given hulls.
    centre : np.array of shape (dims,) (OPTIONAL)
    radius : float (OPTIONAL)

    """
    # Gather the defining constraints of all the hulls.
    constraints = np.concatenate([h.equations for h in hulls])

    # Now, the input hulls may be very large, meaning the last column
    # of the `constraints` matrix may be very large. For the sake of
    # numerical stability and controlling the fidelity of convex hull
    # approximation, we scale this colunm down to a managable
    # magnitude, effectively scaling the sizes of the input polytopes
    # down uniformly.
    b_range = max(constraints[:, -1]) - min(constraints[:, -1])
    scaled_constraints = np.copy(constraints)
    scaled_constraints[:, -1] = constraints[:, -1] / b_range

    # Find an interior point, needed to compute the intersection.
    c, radius, _ = ch_centre_from_constraints(scaled_constraints)

    # If such a point wasn't found, the intersection is empty (barring
    # numerical trouble in finding the point.)
    if c is None:
        logger.warning("The intersection is empty!")
        if return_centre:
            return None, None, None
        else:
            return None

    # Compute the intersection (using qhull in the background). The
    # option QJ "joggles" the input in order to circumvent precision
    # problems, and the C, A and W options ensure that qhull only
    # approximates the output, seeing as an exact answer may be too
    # large (have too many points).
    logger.info("Starting Qhull halfspace intersection...")
    t_start = time.time()
    hs = HalfspaceIntersection(scaled_constraints, c, qhull_options="QJ C0.001")
    t_stop = time.time()
    logger.info(f"Halfspace intersection took {t_stop - t_start:.2f} seconds.")

    # Extract the vertices of the intersection. Confusingly those
    # vertices are called "intersections" themselves, meaning
    # intersections of the given halfspaces (constraints).
    scaled_vertices = hs.intersections

    # Scale everything back.
    vertices = b_range * scaled_vertices
    c = b_range * c
    radius = b_range * radius

    # Return the vertices.
    if return_centre:
        return vertices, c, radius
    else:
        return vertices


def ch_centre(hull: ConvexHull) -> (np.array, float, np.array):
    r"""Compute the Chebyshev centre and its radius from a given convex hull.

    Writes a linear program that outputs the point with the maximal
    radius inside the convex hull. This corresponds to writing the
    problem as follows:

    max R s.t.
        a_i * x + R * np.linalg.norm(a_i) \leq b_i for i = 1,...,num_eqns
    (cf. Boyd and Vandenberghe, Ch. 8.5)

    where (a_i * x \leq b_i)_i are the equations defining the convex
    hull.

    Note that when using qhull we use `-b_i`, as normal points are
    defined to be pointing outward, i.e. the convex hull satisfies `Ax
    <= -b` (cf. http://www.qhull.org/html/qh-opto.htm#n). In our case
    the vectors a_i are already normalised, so we just have:

    max R s.t.
        a_i * x + R \leq -b_i for i = 1,...,num_eqns

    Using a matrix formulation:

    max (0,...,0, 1) \cdot (x, R) s.t.
        (a_i, 1) \cdot (x, R) \leq -b for i = 1,...,num_eqns.

    In addition to the actual Chebyshev centre and the radius of the
    Chebyshev ball, this function also returns the tight constraints
    of the above problem (in order of tightness) which are the facets
    touched by the Chebyshev ball.

    Under the hood, uses the auxiliary function
    `ch_centre_from_constraints`.

    Parameters
    ----------
    hull : scipy.spatial.ConvexHull

    Returns
    -------
    centre : np.array of shape (dims,),
    radius : float
    tight_constraints : np.array of shape (num_tight_constraints, dims)

    """
    return ch_centre_from_constraints(hull.equations)


def ch_centre_from_constraints(constraints: np.array) -> (np.array, float, np.array):
    r"""Compute the Chebyshev centre of a polytope gives by constraints.

    Each row R of `constraints` defines a linear equation of the form
        R[:-1] * x <= -R[-1].
    Note the minus sign on the right hand side: this follows qhulls
    convention for specifying linear constraints. Together, all the
    given constraints (rows of array `constraints`) may define a
    bounded polytope. In this case, this function returns the
    Chebyshev centre and radius of the polytope.

    The final object to be returned is an array of all the tight
    constraints on the Chebyshev ball, meaning the constraints
    (hyperplanes) which "touch" the Chebyshev ball of the polytope.
    They are sorted by their corresponding dual variables (non-zero
    since these are tight constraints). The constraint with the
    greatest dual variable (i.e. the "tightest" constraint) is the
    first row of the array, and so on.

    It is assumed that the normal vectors defined by the constraints
    (i.e. `constraints[i, :-1]`) are each normalised. This is the case
    with constraints obtained from qhull.

    If the given constraints do not define a bounded polytope (for
    instance, the polytope is empty or non-bounded), then None, None,
    None is returned.

    See the documentation of `ch_centre` for more details.

    Parameters
    ----------
    constraints : np.array of shape (num_eqs, dims+1)

    Returns
    -------
    centre : np.array of shape (dims),
    radius : float
    tight_constraints : np.array of shape (num_tight_constraints, dims)

    """
    num_eqn = constraints.shape[0]
    dims = constraints.shape[1] - 1

    # Prepare the objective function, which just has a single
    # coefficient for the radius. linprog minimises, so minimise -R to
    # maximise R (GREENBUBBLE: was `m.setObjective(objective @ x, GRB.MAXIMIZE)`).
    objective = np.array(([0] * dims) + [1])

    # Get the constraints of the form (a_i**T, norm(a_i)) * (x, R) <=
    # b_i. Note that we assume a_i to be normalised here.
    A = np.hstack((constraints[:, :-1], np.ones(shape=(num_eqn, 1))))
    b = -constraints[:, -1]  # note the sign coming from a qhull equation

    # Prepare variable bounds: coordinates are unbounded, radius is
    # nonnegative (GREENBUBBLE: was Gurobi `lb`/default `ub`).
    bounds = [(None, None)] * dims + [(0, None)]

    # Solve the linear program (GREENBUBBLE: was a `gurobipy` model;
    # ported to `scipy.optimize.linprog` -- same LP, no Gurobi dependency).
    res = linprog(-objective, A_ub=A, b_ub=b, bounds=bounds, method="highs")

    # Check of the optimisation was successful.
    if not res.success:
        logger.warning(
            "Could not find centre point. linprog failed at"
            f" optimisation with status {res.status}: {res.message}"
        )
        return None, None, None

    centre = res.x[:-1]
    radius = res.x[-1]

    # Extract the tight constraints, which are those whose
    # corresponding dual (marginal) values are non-zero (GREENBUBBLE:
    # was Gurobi constraint duals `c.pi`; HiGHS exposes the equivalent
    # via `res.ineqlin.marginals`).
    marginals = res.ineqlin.marginals
    duals = list(enumerate(marginals))
    duals.sort(reverse=True, key=lambda x: abs(x[1]))
    non_zero_dual_i = [i for i, d in duals if abs(d) > 1e-9]
    tight_constraints = A[non_zero_dual_i, :-1]

    # Return the results.
    return (centre, radius, tight_constraints)


def contains(hull: ConvexHull, point: np.array) -> bool:
    """Check if a convex hull contains a given point.

    Parameters
    ----------
    hull : scipy.spatial.ConvexHull
    point : np.array
        Must be of the same dimension as the points consituting `hull`.

    Returns
    -------
    Bool

    """
    num_eqn = hull.equations.shape[0]
    dims = hull.equations.shape[1] - 1
    if dims != len(point):
        raise ValueError("Dimension of hull and point do not match.")

    # In order to check if the point is in the convex hull, we simply
    # check that it satisfies every equation defining the hull.
    for i in range(num_eqn):
        eq = hull.equations[i, :-1]
        b = hull.equations[i, -1]
        if np.dot(eq, point) > -b:
            # The equation was violated!
            return False

    # If none of the equations were violated, then the point is
    # contained in the hull.
    return True


def is_nonempty(constraints: np.array) -> bool:
    """Check if a polytope is nonempty.

    Each row of `constraints` consists of the coefficients of an
    equation c_1 x_1 + c_2 x_2 + ... + c_n x_n <= -b. Return True if
    there exists a solution to all given constraints, or equivalently,
    if the polytope defined by the equations is non-empty.
    """
    A = constraints[:, :-1]
    b = -constraints[:, -1]
    c = np.array([1] * A.shape[1])  # The objective function is arbitrary.
    bounds = [(None, None)] * A.shape[1]
    # GREENBUBBLE: was a `gurobipy` feasibility solve; ported to
    # `scipy.optimize.linprog` (same LP, arbitrary objective `c`).
    res = linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    return bool(res.success)


def init_polytope(constraints: np.array) -> dict:
    """Return a solver-agnostic description of the polytope given by `constraints`.

    GREENBUBBLE: originally returned a `(gurobipy.Model, gurobipy.MVar)`
    pair to be reused across repeated `probe_polytope` calls (Gurobi keeps
    warm-start state on the model object). `scipy.optimize.linprog` has no
    equivalent persistent-model object, so this instead returns a plain
    dict of the constraint arrays that `probe_polytope` below re-solves
    from scratch each call -- same polytope, no incremental-solve speedup.
    """
    A = constraints[:, :-1]
    b = -constraints[:, -1]
    return {"A": A, "b": b, "dims": A.shape[1]}


def probe_polytope(m: dict, direction: np.array) -> np.array:
    """Return a point in `direction` inside the space defined by `constraints`."""
    bounds = [(None, None)] * m["dims"]
    # GREENBUBBLE: was `m.setMObjective(...); m.optimize()` on a Gurobi
    # model; ported to `scipy.optimize.linprog` (maximise direction @ x
    # == minimise -direction @ x, same polytope constraints).
    res = linprog(-np.asarray(direction), A_ub=m["A"], b_ub=m["b"], bounds=bounds, method="highs")
    if res.success:
        return res.x
    else:
        raise RuntimeError("linprog could not optimise over the given polytope.")


def facet_normals(convex_hull: ConvexHull) -> np.array:
    """Return the facet normals of a convex hull, sorted by facet size."""
    # Extract all facets of the convex hull by points and compute
    # their volume.
    facets = []
    for s, e in zip(convex_hull.simplices, convex_hull.equations[:, :-1]):
        # Get the points of the facet, and compute edge vectors
        # spanning the facet (which is a simplex).
        vertices = [convex_hull.points[p] for p in s]
        edges = [vertices[0] - v for v in vertices[1:]]
        # To compute the volume, we compute the QR decomposition of
        # the matrix whose column vectors are the simplex edges. This
        # gives those edges in an orthonormal basis for the simplex.
        # Then we take the product of the diagonal of these new
        # coordinates (R), which is the determinant since R is upper
        # triangular. This actually gives n factorial times the volume
        # (where n is the dimension), but we do not care since it is
        # just a uniform scaling factor.
        A = np.array(edges).T
        _, R = linalg.qr(A, mode="economic")
        volume = linalg.det(R)
        facets.append((e, volume))

    # Sort the facets by volume (negation to get decreasing order).
    facets.sort(key=lambda t: -t[1])
    return [f[0] for f in facets]


def uniform_random_hypersphere_sampler(n: int):
    """Generate points on the `n`-dimensional hypersphere at random.

    The points are normalised and following the uniform distribution
    on the hypersphere.
    """
    while True:
        # Transform from unit cube to cube around origin.
        p = 2 * np.random.random_sample((n,)) - 1
        if np.linalg.norm(p) <= 1:
            # Transform to lie on the unit hypersphere.
            yield p / np.linalg.norm(p)


def lhc_random_hypersphere_sampler(n: int):
    """Generate points on the `n`-dimensional hypersphere at random.

    The points are generated using Latin hypercube sampling and
    normalised. As in
    https://en.wikipedia.org/wiki/Latin_hypercube_sampling.

    The difference with `uniform_random_hypersphere_sampler` is that
    the points generated by this sampler do not follow the uniform
    distribution on the hypersphere. Instead, coordinates generated by
    LHS are more evenly distributed. This leads to a distribution on
    the hypersphere which is less dense around the axes.

    """
    sampler = LatinHypercube(d=n)
    while True:
        lhc = sampler.random(n)
        for p in lhc:
            q = 2 * p - 1  # Transform from unit cube to cube around origin.
            yield q / np.linalg.norm(q)  # Transform to lie on the unit hypersphere.


def angle_threshold(candidate: np.array, previous: np.array, angle: float):
    """Filter a candidate angle on previous angles.

    Return False if the vector `candidate` is within an angle of
    `angle` (in degrees) of any vector in `previous`, True otherwise.

    """
    for p in previous:
        p_norm = p / np.linalg.norm(p)
        c_norm = candidate / np.linalg.norm(candidate)
        dot = min(1, np.dot(p_norm, c_norm))  # Avoid getting 1.00000001
        t = np.degrees(np.arccos(dot))
        if t < angle:
            return False
    return True


def filter_vectors(
    vecs: Iterable,
    angle: float = 10,
    initial_vectors: Collection = None,
    max_retries: int = 1000,
):
    """Run a vector generator and filter similar vectors out.

    In particular, this generator keeps track of previously seen
    vectors and filters away any vector closer than an `angle`
    degrees to previous ones.

    Parameters
    ----------
    vecs : Iterable,
        The vectors to filter by angle.
    angle : float
        Initial threshold angle below which new vectors are discarded.
    initial_vectors : Collection
        Initial set of vectors with which new vectors from `vecs` are
        compared.
    max_retries : int
        Number of consecutive vectors from `vecs` that can be
        discarded for being too close to previously seen vectors,
        before the threshold angle is decreased or the generator
        terminates.

    """
    # Copy collection of previous vectors if any.
    if initial_vectors is not None:
        previous_vecs = initial_vectors[:]
    else:
        previous_vecs = []

    num_retries = 0
    for vec in vecs:
        # Check if we have reached the maximum number of retries, in
        # which case we give up.
        if num_retries >= max_retries:
            return
        # Check if the current vector is far enough away from all
        # previous ones.
        if not angle_threshold(vec, previous_vecs, angle):
            num_retries += 1
            continue
        previous_vecs = np.vstack((previous_vecs, vec))
        # Since we found a vector, reset the `num_retries` to 0.
        num_retries = 0
        yield vec


def filter_vectors_auto(
    vecs: Iterable,
    init_angle: float = 10.0,
    initial_vectors: Collection = None,
    max_retries: int = 100,
    min_angle_tolerance: float = 0.1,
):
    """Run a vector generator and filter similar vectors out.

    This generator yields vectors from `vecs` in order, but filters
    out any vectors too close to previously seen vectors. By
    "previously seen", we meet all the vectors previously yielded,
    together with the vectors in `initial_vectors`. By "too close", we
    mean within a certain angle theta. The angle theta is initially
    set to `init_angle`. However, every time `max_retries` consecutive
    vectors have been discarded, theta is decreased by 20%. This
    continues until theta drops below `min_angle_tolerance`, at
    which point the generator stops after `max_retries` consecutive
    discarded vectors.

    This method works best when `vecs` yields an indefinite number of
    vectors, such as by independent random generatation.

    Parameters
    ----------
    vecs : Iterable,
        The vectors to filter by angle.
    init_angle : float
        Initial threshold angle below which new vectors are discarded.
    initial_vectors : Collection
        Initial set of vectors with which new vectors from `vecs` are
        compared.
    max_retries : int
        Number of consecutive vectors from `vecs` that can be
        discarded for being too close to previously seen vectors,
        before the threshold angle is decreased or the generator
        terminates.
    min_angle_tolerance : float
        The minimum threshold angle allowed. If the threshold angle is
        decreased below this, the generated is terminated.

    """
    a = init_angle

    # Copy collection of previous vectors if any.
    if initial_vectors is not None:
        previous_vecs = initial_vectors[:]
    else:
        previous_vecs = []

    num_retries = 0
    for vec in vecs:
        # If we tried too many times to generate a new direction but
        # failed, decrease the threshold angle.
        if num_retries >= max_retries:
            a *= 0.8
            num_retries = 0
            if a < min_angle_tolerance:
                # At this point we have really run out of directions.
                return
            logger.info(f"Decreased angle threshold to {a}.")

        # Check if the current vector is far enough away from all
        # previous ones.
        if not angle_threshold(vec, previous_vecs, a):
            num_retries += 1
            continue
        previous_vecs = np.vstack((previous_vecs, vec))
        # Since we found a vector, reset the `num_retries` to 0.
        num_retries = 0
        yield vec


def hypersphere_packing_bound(dim: int, theta: float):
    """Lower bound on number of points `theta` degrees apart on unit hypersphere.

    Return a lower bound on the number of points which can be fitted
    on the (`dim`-1)-sphere (i.e. points in `dim`-dimensional
    Euclidean space at distance 1 from the origin) such that each pair
    of points is at least `theta` degrees apart.

    """
    # We only support dimensions `dim` for which the packing density
    # in dimension `dim`-1 is known. (Except dimension 24, which we
    # do not bother to support, for which the packing density is in
    # fact known.)
    if dim < 3:
        return ValueError(f"Dimension {dim} too low.")
    if dim > 9:
        return ValueError(f"Dimension {dim} too high.")

    # Hypersphere packing densities, see
    # https://mathworld.wolfram.com/HyperspherePacking.html.
    densities = {
        2: 0.90689968,
        3: 0.74048052,
        4: 0.61685029,
        5: 0.46525763,
        6: 0.37294756,
        7: 0.29529789,
        8: 0.25366952,
    }
    dim_const = (
        dim * math.gamma((dim + 1) / 2) * math.sqrt(math.pi) / math.gamma(dim / 2 + 1)
    )
    theta_rad = math.pi * theta / 180
    return densities[dim - 1] * dim_const / math.pow(theta_rad / 2, dim - 1)


# ---------------------------------------------------------------------------
# GREENBUBBLE NOTE: the three generators below are vendored from the SAME
# source repository/commit as the rest of this file, but originally lived in
# `workflow/scripts/compute_near_opt.py`, not `geometry.py` -- moved here
# unchanged because they only depend on the geometry functions above (no
# PyPSA/solver dependency), so they belong with the rest of the portable
# geometry rather than the PyPSA-specific solve orchestration. Still
# GPL-3.0-or-later, same attribution as the file header.
# ---------------------------------------------------------------------------


def large_facet_directions(
    hull: ConvexHull,
    probed_directions: Collection[np.array],
    init_min_angle: float = 10.0,
    autodecrease: bool = False,
    min_angle_tolerance: float = 0.1,
):
    """Generate directions based on facets.

    In particular, for each iteration this generator sorts the normal
    vectors of the facets of `hull` by facet volume, and returns the
    normal vector of the largest facet. Normals that are close to any
    vector in `probed_directions` are filtered out. If `autodecrease`
    is set to True (default: False), then we decrease the angles by 20% to
    try to get more vectors. Then it is necessary to add a minimial angle,
    `min_angle_tolerance`.

    The arguments `hull` and `probed_directions` are taken as
    references and may change between each iteration of this
    generator.

    This generator never terminates, and instead returns None when it
    runs out of directions. This is so that it can "try again" after
    it failed to find a new direction, but the hull was updated in the
    mean time.

    Parameters
    ----------
    hull : ConvexHull
        Based on this convex hull, generate directions by its facets.
    probed_directions : Collection[np.array]
        Filter the directions based on this collection.
    init_min_angle : float
        First angle threshold for filtering before possible reductions.
    autodecrease : bool
        If True, automatically decrease the angle threshold by 20% if
        no directions can be found otherwise, unless the angle goes
        below `min_angle_tolerance`.
    min_angle_tolerance : float
        Minimal threshold for direction filtering. Below it, the
        direction generation ends, as we have exhausted the possible
        vectors.

    """
    a = init_min_angle
    while True:
        # Get the normals for the current hull. (Note that `hull` is
        # updated for every iteration.)
        normals = facet_normals(hull)
        try:
            # Filter out directions close to ones we have seen before,
            # and return the first one.
            yield next(
                filter_vectors(normals, angle=a, initial_vectors=probed_directions)
            )
        except StopIteration:
            if autodecrease:
                # If we ran out of directions here, decrease the
                # minimum allowed angle between probed directions
                # (until we reach an absolute minimum).
                a *= 0.8
                if a < min_angle_tolerance:
                    # At this point we seem to really have run out of directions.
                    yield None
                    continue
                logger.info(f"Decreasing allowed angle between directions to {a}.")
                continue
            yield None


def touching_ball_directions(
    hull: ConvexHull,
    probed_directions: Collection[np.array],
    angle_tolerance: float,
):
    """Compute normals of planes touched by the Chebyshev ball at the centre.

    This generator computes for each iteration the normals of the
    hyperplanes touching the largest centre ball of `hull`, and
    returns one of them. Normals which are too close to any vector in
    `probed_directions` are filtered out (defined by the threshold
    `angle_tolerance`).

    The arguments `hull` and `probed_directions` are taken as
    references and may change between each iteration of this
    generator.

    Note that we do not decrease the angles here as we already start
    with a very low threshold to exploit the touching directions precisely
    from the beginning.

    This generator never terminates, and instead returns None when it
    runs out of directions. This is so that it can "try again" after
    it failed to find a new direction, but the hull was updated in the
    mean time.

    Parameters
    ----------
    hull : ConvexHull
        Based on this convex hull, generate directions by its facets.
    probed_directions : Collection[np.array]
        Filter the directions based on this collection.
    angle_tolerance : float
        Threshold for filtering directions.

    """
    while True:
        # Compute the Chebyshev centre of `hull` and get the
        # constraints which are tight in the resulting LP. These
        # constraints are exactly the normal vectors of the
        # hyperplanes of `hull` touching the centre ball.
        _, _, tight_constraints = ch_centre(hull)
        # Just yield any of the normals that is not too close to
        # something we have tried before.
        try:
            yield next(
                filter_vectors(
                    tight_constraints,
                    angle=angle_tolerance,
                    initial_vectors=probed_directions,
                )
            )
        except StopIteration:
            yield None


def maximal_centre_then_facets(
    hull: ConvexHull,
    probed_directions: Collection[np.array],
    init_min_facet_angle: float = 10.0,
    angle_tolerance: float = 0.1,
):
    """Generate directions first from centre ball, then facets.

    For each iteration of this generator, it is first checked if there
    are any normal vectors of facets touched by the centre ball of
    `hull` which have not been probed yet. If such a vector is found,
    it is yielded. If not, the normal vectors of the largest unchecked
    facets are returned instead. Once these are exhausted, we reduce
    the angle threshold automatically by 20%.

    Parameters
    ----------
    hull : ConvexHull
        Based on this convex hull, generate directions by its facets.
    probed_directions : Collection[np.array]
        Filter the directions based on this collection.
    init_min_angle : float
        First angle threshold for filtering before possible reductions.
    min_angle_tolerance : float
        Minimal threshold for direction filtering. Below it, the
        direction generation ends, as we have exhausted the possible
        vectors.

    """
    a = init_min_facet_angle
    while True:
        # First try generating a direction using the centre ball. This
        # generator yields None if nothing was found.
        d = next(touching_ball_directions(hull, probed_directions, angle_tolerance))
        if d is not None:
            logger.info("Generated direction based on maximal-centre.")
            yield d
            continue

        # In case we did not find any new directions from the facets
        # touching the centre ball, go with normal directions to large
        # facets.
        d = next(large_facet_directions(hull, probed_directions, a))
        if d is not None:
            logger.info("Generated direction based on largest facet.")
            yield d
        else:
            # If we ran out of directions here, decrease the minimum
            # allowed angle between probed directions (until we reach
            # an absolute minimum).
            a *= 0.8
            if a < angle_tolerance:
                # At this point we really seems to have run out of directions.
                yield None
                continue
            logger.info(f"Decreasing allowed angle between directions to {a}.")
