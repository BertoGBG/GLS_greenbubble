# run_sweep.py (clean version)
from __future__ import annotations
from itertools import combinations, chain
from pathlib import Path
import scripts.config as c
import greenbubble_main as m
import pandas as pd
import yaml, re, math

AGENTS = ['biogas', 'central_heat','renewables','electrolysis','meoh','methanation','storage','symbiosis']
BASE_FLAGS = dict(c.n_flags)
BASE_OUT = Path("outputs/single_analysis/shapV")
BASE_OUT.mkdir(parents=True, exist_ok=True)
c.outputs_folder = str(BASE_OUT)


# --------------------------------------------------------------------------
#  Run subset
# --------------------------------------------------------------------------
def run_subset(subset):
    flags = {a: (a in subset) for a in AGENTS} | BASE_FLAGS
    print(f"\n=== Running subset {subset} ===")
    m.main(n_flags=flags, outputs_folder=str(BASE_OUT))


# --------------------------------------------------------------------------
#  Read configuration & objective
# --------------------------------------------------------------------------
def read_flags_from_config(cfg_path: Path) -> dict[str, bool]:
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    for key in ("n_flags", ("config", "n_flags"), ("solve", "n_flags")):
        d = cfg
        for k in (key if isinstance(key, tuple) else (key,)):
            d = d.get(k, {}) if isinstance(d, dict) else {}
        if isinstance(d, dict):
            return d
    return {a: bool(cfg.get(a, False)) for a in AGENTS}


def find_opt_nc(folder: Path) -> Path | None:
    nc_files = sorted(folder.glob("*_OPT.nc"), key=lambda p: p.stat().st_mtime)
    return nc_files[-1] if nc_files else None


def read_objective(nc_path: Path) -> float | None:
    import xarray as xr
    try:
        with xr.open_dataset(nc_path) as ds:
            for key in ("objective", "objective_value", "problem_objective", "opt_objective"):
                if key in ds.attrs:
                    return float(ds.attrs[key])
            meta = ds.attrs.get("meta", "")
            if isinstance(meta, str):
                m = re.search(r"objective[\"'\s:]*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", meta)
                if m:
                    return float(m.group(1))
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
#  Build DataFrame of all coalitions
# --------------------------------------------------------------------------
def build_shapley_df(base_dir: Path = BASE_OUT) -> pd.DataFrame:
    rows = []
    for sub in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        netdir = sub / "networks"
        cfg_path = netdir / "config_run.yaml"
        if not (cfg_path.exists() and netdir.exists()):
            continue
        flags = read_flags_from_config(cfg_path)
        coalition = tuple(a for a in AGENTS if flags.get(a, False))
        nc = find_opt_nc(netdir)
        objective = read_objective(nc) if nc else None
        rows.append(dict(coalition=coalition, objective=objective, out_folder=sub, nc_path=str(nc) if nc else None))
    df = pd.DataFrame(rows).set_index("coalition")
    return df.sort_index()


# --------------------------------------------------------------------------
#  Game theory functions
# --------------------------------------------------------------------------
def shapley_values(v: dict[frozenset, float]) -> dict[str, float]:
    players = list(set(chain.from_iterable(v.keys())))
    n = len(players)
    phi = {p: 0.0 for p in players}
    for i in players:
        others = [j for j in players if j != i]
        for k in range(len(others) + 1):
            for S in map(frozenset, combinations(others, k)):
                w = 1 / (n * math.comb(n - 1, k))
                phi[i] += w * (v.get(S | {i}, 0) - v.get(S, 0))
    return phi


def is_game_convex(v: dict[frozenset, float]):
    v = {**v, frozenset(): v.get(frozenset(), 0.0)}
    fails = []
    for S in v.keys():
        for T in v.keys():
            U, I = S | T, S & T
            if U not in v or I not in v:
                continue
            if v[S] + v[T] < v[U] + v[I]:
                fails.append((S, T))
    return (len(fails) == 0), fails


def convexity_violations_marginal(v):
    v = {**v, frozenset(): v.get(frozenset(), 0.0)}
    players = set().union(*v.keys())
    viols = []
    for S in v:
        for T in v:
            if not S.issubset(T):
                continue
            for i in players - T:
                S_i, T_i = S | {i}, T | {i}
                if S_i in v and T_i in v:
                    if (v[S_i] - v[S]) < (v[T_i] - v[T]) - 1e-9:
                        viols.append((i, S, T))
    return viols


# --------------------------------------------------------------------------
#  Main execution
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Run all coalitions
    for r in range(1, len(AGENTS) + 1):
        for subset in combinations(AGENTS, r):
            run_subset(subset)

    # Build DataFrame
    df = build_shapley_df().dropna(subset=["objective"])
    out_csv = BASE_OUT / "shapley_objectives.csv"
    df.to_csv(out_csv)
    print(f"\nSaved objectives → {out_csv}")

    # Prepare values dict
    v_frozen = {
        frozenset(k if isinstance(k, (list, tuple, set)) else [k]): float(v)
        for k, v in df["objective"].items()
        if pd.notna(v)
    }

    # Compute Shapley Values results
    phi = shapley_values(v_frozen)
    is_convex, fails = is_game_convex(v_frozen)

    # Display
    print("\n=== Shapley values ===")
    print(pd.Series(phi))
    print("\nConvex game:", is_convex)
    if not is_convex:
        print(f"Violations: {len(fails)} (showing first 5)")
        for S, T in fails[:5]:
            print("   ", S, "vs", T)
