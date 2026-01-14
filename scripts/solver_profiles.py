# solver_profiles.py
import os
from copy import deepcopy

def _default_threads(fallback=8):
    try:
        return min(os.cpu_count() or fallback, fallback)
    except Exception:
        return fallback

GUROBI_PROFILES = {
    "gurobi-default": {
        "Threads": _default_threads(8),
        "Method": 2,                 # barrier
        "Crossover": 0,              # no crossover
        "BarConvTol": 1e-5,
        "Seed": 123,
        "AggFill": 0,
        "PreDual": 0,

    },
    "gurobi-numeric-focus": {
        "NumericFocus": 3,
        "Method": 2,
        "Crossover": 0,
        "BarHomogeneous": 1,
        "BarConvTol": 1e-5,
        "FeasibilityTol": 1e-4,
        "OptimalityTol": 1e-4,
        "ObjScale": -0.5,
        "Threads": _default_threads(8),
        "Seed": 123,
    },
    "gurobi-fallback": {
        "Crossover": 0,
        "Method": 2,
        "BarHomogeneous": 1,
        "BarConvTol": 1e-5,
        "FeasibilityTol": 1e-5,
        "OptimalityTol": 1e-5,
        "Seed": 123,
        "Threads": _default_threads(8),
    },

    "gurobi-barhom-diagnose": {
        "Threads": _default_threads(8),

        # Barrier + homogeneous algorithm for numeric trouble
        "Method": 2,
        "Crossover": 0,
        "BarHomogeneous": 1,
        "DualReductions": 0,         # Critical: avoid "infeasible or unbounded" ambiguity
        "InfUnbdInfo": 1,        # Ask gurobi for more info in inf/unbd cases
        "NumericFocus": 3,         # Make barrier a bit more robust
        "BarConvTol": 1e-6,
        "FeasibilityTol": 1e-6,         # Keep tolerances reasonable
        "OptimalityTol": 1e-6,
        "Seed": 123,
    },

    "gurobi-simplex-diagnose": {
        "Threads": _default_threads(8),
        "Method": 1,
        "DualReductions": 0,
        "InfUnbdInfo": 1,
        "Presolve": 2,
        "NumericFocus": 3,
        "Seed": 123,
    }

}

HIGHS_PROFILES = {
    "highs-default": {
        # https://ergo-code.github.io/HiGHS/dev/options/definitions/
        "threads": 1,
        "solver": "ipm",
        "run_crossover": "off",
        "small_matrix_value": 1e-6,
        "large_matrix_value": 1e9,
        "primal_feasibility_tolerance": 1e-5,
        "dual_feasibility_tolerance": 1e-5,
        "ipm_optimality_tolerance": 1e-4,
        "parallel": "on",
        "random_seed": 123,
    },
    "highs-simplex": {
        "solver": "simplex",
        "parallel": "on",
        "primal_feasibility_tolerance": 1e-5,
        "dual_feasibility_tolerance": 1e-5,
        "random_seed": 123,
    },
}

SOLVER_PROFILES = {"gurobi": GUROBI_PROFILES, "highs": HIGHS_PROFILES}
