# run_sweep.py
from __future__ import annotations
from itertools import combinations
from pathlib import Path
import scripts.config as c
import greenbubble_main as m
import pandas as pd
import re
import yaml
import math

# Agents
AGENTS = ['biogas','central_heat','renewables','electrolysis','meoh','methanation','storage', 'symbiosis']
BASE_FLAGS = dict(c.n_flags)  # keep non-agent flags unchanged

# Override the output base folder for all runs
BASE_OUT = Path("outputs/single_analysis/shapV")
BASE_OUT.mkdir(parents=True, exist_ok=True)
c.outputs_folder = str(BASE_OUT)  #


def run_subset(subset):
    flags = dict(BASE_FLAGS)
    # turn OFF all listed agents first
    for k in AGENTS:
        flags[k] = False
    # then enable only the chosen subset
    for k in subset:
        flags[k] = True

    c.n_flags = flags

    # set c. stochastic: false
    print(f"\n=== Running subset {subset} ===")
    m.main(n_flags=flags, outputs_folder=str(BASE_OUT))


def coalition_tuple(flags_or_subset, agents=AGENTS) -> tuple:
    if isinstance(flags_or_subset, dict):
        return tuple(a for a in agents if flags_or_subset.get(a, False))
    else:
        return tuple(flags_or_subset)  # e.g., the tuple from combinations


def read_flags_from_config(cfg_path: Path) -> dict[str, bool]:
    """
    Load config.yaml and return the 'n_flags' subsection as a dict.
    """
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    # Common patterns: either cfg['n_flags'] or cfg['config']['n_flags'] etc.
    for key in ("n_flags", ("config", "n_flags"), ("solve", "n_flags")):
        if isinstance(key, tuple):
            d = cfg
            ok = True
            for k in key:
                if isinstance(d, dict) and k in d:
                    d = d[k]
                else:
                    ok = False
                    break
            if ok and isinstance(d, dict):
                return d
        else:
            if isinstance(cfg, dict) and key in cfg and isinstance(cfg[key], dict):
                return cfg[key]
    # Fallback: try top-level booleans matching AGENTS
    return {a: bool(cfg.get(a, False)) for a in AGENTS}


def find_opt_nc(networks_dir: Path) -> Path | None:
    """
    Return the most recently modified *_OPT.nc file, if any.
    """
    candidates = list(networks_dir.glob("*_OPT.nc"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_objective_from_nc(nc_path: Path) -> float | None:
    """
    Try multiple strategies to extract the objective value from a solved PyPSA NetCDF.
    Works even without importing pypsa, by inspecting dataset attributes/variables.
    """

    # 4) Try pypsa if available (may be slower, but robust if the attribute is kept)
    try:
        import pypsa  # type: ignore
        n = pypsa.Network(str(nc_path))
        # Common places it might be kept:
        for cand in (
            getattr(n, "objective", None),
            getattr(getattr(n, "meta", {}), "get", lambda *_: None)("objective") if hasattr(n, "meta") else None,
        ):
            if cand is not None:
                try:
                    return float(cand)
                except Exception:
                    continue
    except Exception:
        pass

    try:
        import xarray as xr
        with xr.open_dataset(nc_path) as ds:
            # 1) Common attribute names
            for k in ("objective", "objective_value", "problem_objective", "opt_objective"):
                if k in ds.attrs:
                    try:
                        return float(ds.attrs[k])
                    except Exception:
                        pass
            # 2) Sometimes stored under a nested meta dict serialized as string
            #    e.g., attrs['meta'] = "{'objective': 123.45, ...}"
            meta = ds.attrs.get("meta") or ds.attrs.get("metadata")
            if isinstance(meta, str):
                # quick & safe parse for a number after 'objective'
                m = re.search(r"objective[\"'\s:]*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)", meta)
                if m:
                    return float(m.group(1))
            # 3) Rarely a data_var named 'objective'
            for k in ("objective", "objective_value"):
                if k in ds.data_vars:
                    v = ds[k]
                    try:
                        # scalar or 0-d array
                        return float(v.values.item() if hasattr(v.values, "item") else v.values)
                    except Exception:
                        pass
    except Exception:
        pass

    return None


def build_shapley_df(base_dir: Path = BASE_OUT, agents=AGENTS) -> pd.DataFrame:
    rows = []
    for sub in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        networks_dir = sub / "networks"
        cfg_path = networks_dir / "config_run.yaml"
        if not cfg_path.exists() or not networks_dir.exists():
            continue

        flags = read_flags_from_config(cfg_path)
        label = coalition_tuple(flags, agents)   # <- tuple index

        nc = find_opt_nc(networks_dir)
        objective = read_objective_from_nc(nc) if nc else None

        rows.append({
            "coalition": label,  # tuple
            "enabled_agents": list(label),  # optional, keep for readability
            "objective": objective,
            "nc_path": str(nc) if nc else None,
            "out_folder": str(sub),
        })

    if not rows:
        cols = ["objective", "enabled_agents", "nc_path", "out_folder"]
        return pd.DataFrame(columns=cols).set_index(pd.Index([], name="coalition"))

    df = pd.DataFrame(rows).set_index("coalition").sort_index()
    return df

#### split system cost with shapley values
def shapley_values(v):
    """
    Compute the Shapley values for a transferable-utility game using the
    combination-based formula.

    Parameters:
    -----------
    v : dict
        A dictionary mapping each coalition (as a frozenset of players) to its value.
        e.g., v = {
            frozenset(): 0,
            frozenset({'A'}): 1,
            frozenset({'B'}): 2,
            frozenset({'A', 'B'}): 4,
            ...
        }

    Returns:
    --------
    dict
        A dictionary mapping each player to their Shapley value.
    """

    players = list({p for coalition in v.keys() for p in coalition})
    n = len(players)

    phi = {player: 0.0 for player in players}

    for i in players:
        # All other players
        others = [j for j in players if j != i]
        # Iterate over coalition sizes k = 0, ..., n-1
        for k in range(len(others) + 1):
            # For each coalition S of size k
            for S in combinations(others, k):
                S = frozenset(S)
                # Weight: 1 / (n * C(n-1, k))
                weight = 1 / (n * math.comb(n - 1, k))
                # Marginal contribution of i to coalition S
                v_S = v.get(S, 0.0)
                v_Si = v.get(S | {i}, 0.0)
                phi[i] += weight * (v_Si - v_S)

    return phi

def is_game_convex(v):
    """
    Check whether a cooperative TU game is convex.

    Parameters:
    -----------
    v : dict
        Dictionary mapping each coalition (as a frozenset) to its value.

    Returns:
    --------
    is_convex: bool
        True if the game is convex, False otherwise.
    failures : list
        #List of counterexamples (tuples of S, T) where convexity fails.
    """
    coalitions = list(v.keys())
    failures = []
    v[frozenset()] = 0 # add the empty set

    for i, S in enumerate(coalitions):
        for j, T in enumerate(coalitions):
            S_union_T = S | T
            S_inter_T = S & T


            # All coalitions must be in v
            if S_union_T not in v or S_inter_T not in v:
                continue

            lhs = v[S] + v[T]
            rhs = v[S_union_T] + v[S_inter_T]

            if lhs < rhs: # note: for cost minimization (lhs < rhs) is a failure
                failures.append((S, T))

    is_convex = len(failures) == 0
    return is_convex, failures

def convexity_violations_marginal(c):
    vv = dict(c)
    vv.setdefault(frozenset(), 0.0)  # ensure empty coalition
    players = set().union(*vv.keys())
    viols = []
    subsets = list(vv.keys())
    for S in subsets:
        for T in subsets:
            if not S.issubset(T):
                continue
            for i in players - T:
                S_i, T_i = S | {i}, T | {i}
                if S_i in vv and T_i in vv:
                    left  = vv[S_i] - vv[S]
                    right = vv[T_i] - vv[T]
                    if left < right - 1e-9:  # tolerance
                        viols.append((i, S, T, left, right, right-left))
    return viols

def top_violations(c, k=5):
    viols = convexity_violations_marginal(c)
    viols.sort(key=lambda t: (t[5] if len(t)==6 else t[4]-t[3]), reverse=True)
    return viols[:k]



if __name__ == "__main__":
    # All coalitions (2^N).
    combos = []
    for r in range(1, len(AGENTS) + 1):
        for subset in combinations(AGENTS, r):
            run_subset(subset)
    # retunr DF for Shapley Values and convexity calculation
    df = build_shapley_df()
    # TOD check for NaN and rise an issue

    # Show a quick preview and save to CSV for later Shapley computation
    print(df.head(20))
    out_csv = BASE_OUT / "shapley_objectives.csv"
    df.to_csv(out_csv)

    # Convert to a dict keyed by frozensets
    v_dict = df['objective'].to_dict()  # {('biogas',): 2.93e8, ...}

    # Convert keys to frozenset
    v_frozen = {
        frozenset(k): float(v)
        for k, v in v_dict.items()
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    }

    # Compute Shapley values and convexity
    phi = shapley_values(v_frozen)
    is_convex, failures = is_game_convex(v_frozen)