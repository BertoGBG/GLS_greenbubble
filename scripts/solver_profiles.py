# solver_profiles.py
import os
from copy import deepcopy

def _default_threads(fallback=8):
    try:
        return min(os.cpu_count() or fallback, fallback)
    except Exception:
        return fallback

GUROBI_PROFILES = {
    # ✅ Default: simplex for stable duals
    "gurobi-default": {
        "Threads": _default_threads(8),
        "Method": 1,          # dual simplex :contentReference[oaicite:1]{index=1}
        "Presolve": 2,
        "Seed": 123,
    },

    # 🚀 Second choice: barrier (often faster on some big sparse LPs)
    "gurobi-barrier": {
        "Threads": _default_threads(8),
        "Method": 2,          # barrier :contentReference[oaicite:2]{index=2}
        "Crossover": -1,      # auto (recommended unless you have a reason) :contentReference[oaicite:3]{index=3}
        "BarConvTol": 1e-8,
        "Seed": 123,
    },

    # Keep your special profiles (they’re useful)
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

    "gurobi-barhom-diagnose": {
        "Threads": _default_threads(8),
        "Method": 2,
        "Crossover": 0,
        "BarHomogeneous": 1,
        "DualReductions": 0,
        "InfUnbdInfo": 1,
        "NumericFocus": 3,
        "BarConvTol": 1e-6,
        "FeasibilityTol": 1e-6,
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
    },

    "gurobi-barrier-fast": {
        "Threads": _default_threads(8),
        "Method": 2,                 # barrier
        "Crossover": 0,              # no crossover
        "BarConvTol": 1e-5,
        "Seed": 123,
        "AggFill": 0,
        "PreDual": 0,
    },

    "gurobi-stoch-diagnose": {
        "Threads": _default_threads(8),
        "Method": 1,          # simplex diagnose tends to give cleaner certs
        "DualReductions": 0,  # key: remove ambiguity
        "InfUnbdInfo": 1,
        "Seed": 123,
    }

}

HIGHS_PROFILES = {
    # ✅ Default: simplex (good duals, predictable)
    "highs-default": {
        "threads": _default_threads(8),
        "solver": "simplex",    # :contentReference[oaicite:5]{index=5}
        "parallel": "on",       # :contentReference[oaicite:6]{index=6}
        "presolve": "on",
        "primal_feasibility_tolerance": 1e-7,
        "dual_feasibility_tolerance": 1e-7,
        "random_seed": 123,
    },

    # 🚀 Second choice: IPM (barrier-like) + crossover ON
    "highs-ipm": {
        "threads": _default_threads(8),
        "solver": "ipm",         # :contentReference[oaicite:7]{index=7}
        "run_crossover": "on",   # default is on; keep it on for stability :contentReference[oaicite:8]{index=8}
        "parallel": "on",
        "presolve": "on",
        "primal_feasibility_tolerance": 1e-7,
        "dual_feasibility_tolerance": 1e-7,
        "ipm_optimality_tolerance": 1e-8,
        "random_seed": 123,
    },

    # Optional: keep a plain simplex profile if you like
    "highs-simplex": {
        "threads": _default_threads(8),
        "solver": "simplex",
        "parallel": "on",
        "presolve": "on",
        "primal_feasibility_tolerance": 1e-7,
        "dual_feasibility_tolerance": 1e-7,
        "random_seed": 123,
    },

    "highs-fast": {
        "threads": 8,                 # similar role to Gurobi Threads
        "solver": "ipm",             # interior-point (closest to barrier)
        "run_crossover": "off",      # like Crossover = 0
        "ipm_optimality_tolerance": 1e-5,
        "random_seed": 123,
        "presolve": "off",           # roughly similar spirit to AggFill/PreDual=0
    }
}

SOLVER_PROFILES = {"gurobi": GUROBI_PROFILES, "highs": HIGHS_PROFILES}
