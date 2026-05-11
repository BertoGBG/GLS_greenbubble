# SPDX-License-Identifier: MIT
"""Snakemake wrapper: rolling horizon dispatch optimisation.

Single-year mode (rh_year is null / same as En_price_year):
  Copies the OPT network, fixes all capacities, runs RH dispatch.

Cross-year mode (rh_year != En_price_year):
  Copies the OPT network, fixes capacities, then patches all year-dependent
  time series (CFs, prices) with rh_year data via patch_timeseries().
  No topology rebuild — the OPT network structure is reused exactly.

The solve_network (capacity expansion) rule is never triggered when this
script is the target — Snakemake's DAG guarantees it.
"""
from pathlib import Path
import sys
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

import pypsa
from scripts import config as c, parameters as p


rh        = c.rolling_horizon
horizon   = rh["horizon"]
overlap   = rh["overlap"]
rh_year   = rh["rh_year"]       # None → same year as En_price_year
main_year = c.En_price_year

print(f"[rolling_horizon] OPT network : {snakemake.input.network}")
print(f"[rolling_horizon] horizon={horizon} h  overlap={overlap} h")
print(f"[rolling_horizon] main year={main_year}  rh_year={rh_year or 'same'}")


# ── Disable cyclic storage constraints ───────────────────────────────────────
def disable_cyclic_constraints(n):
    """Remove end-of-period = start-of-period constraints from all storage.

    In a full-year perfect-foresight solve these constraints are meaningful.
    In rolling horizon, PyPSA carries the end-of-window state forward as the
    initial condition for the next window, so enforcing a cyclic constraint
    within each window would incorrectly force every window to return to its
    opening state of charge.
    """
    if not n.stores.empty and "e_cyclic" in n.stores.columns:
        n_cyclic = n.stores["e_cyclic"].sum()
        n.stores["e_cyclic"] = False
        print(f"[rolling_horizon] disabled e_cyclic on {n_cyclic} stores")

    if not n.storage_units.empty and "cyclic_state_of_charge" in n.storage_units.columns:
        n_cyclic = n.storage_units["cyclic_state_of_charge"].sum()
        n.storage_units["cyclic_state_of_charge"] = False
        print(f"[rolling_horizon] disabled cyclic_state_of_charge on {n_cyclic} storage_units")


# ── Fix all capacities (dispatch-only) ────────────────────────────────────────
def fix_capacities(n):
    """Copy p_nom_opt → p_nom and disable expansion for all extendable components."""
    for comp, nom_col, opt_col in [
        ("generators",    "p_nom", "p_nom_opt"),
        ("links",         "p_nom", "p_nom_opt"),
        ("storage_units", "p_nom", "p_nom_opt"),
        ("stores",        "e_nom", "e_nom_opt"),
    ]:
        df = getattr(n, comp)
        if df.empty:
            continue

        ext_col = "p_nom_extendable" if comp != "stores" else "e_nom_extendable"
        extendable = df.get(ext_col, None)
        if extendable is None:
            continue

        ext_mask = extendable.astype(bool)
        if not ext_mask.any():
            continue

        if opt_col in df.columns:
            df.loc[ext_mask, nom_col] = df.loc[ext_mask, opt_col]
            print(f"[rolling_horizon] fixed {ext_mask.sum()} {comp} from {opt_col}")
        else:
            print(f"[rolling_horizon] {comp}: no {opt_col} — using existing {nom_col}")

        df[ext_col] = False


# ── Build dispatch network ────────────────────────────────────────────────────
n_opt = pypsa.Network(snakemake.input.network)
n = n_opt.copy()
fix_capacities(n)
disable_cyclic_constraints(n)

if rh_year and rh_year != main_year:
    print(f"[rolling_horizon] patching time series to rh_year={rh_year} ...")

    from scripts.create_stoch_scenarios import set_input_paths, patch_timeseries
    from scripts.preprocessing import prepare_all_inputs

    with open(snakemake.input.tech_costs, "rb") as fh:
        tech_costs = pickle.load(fh)

    set_input_paths(p, rh_year)

    inputs_dict_rh = prepare_all_inputs(
        targets_dict      = c.targets_dict,
        CO2_cost          = c.CO2_cost,
        CO2_cost_ref_year = c.CO2_cost_ref_year,
        max_RE_to_grid    = c.max_RE_to_grid,
    )

    patch_timeseries(n, inputs_dict_rh, tech_costs, c.CO2_cost)
    print(f"[rolling_horizon] snapshots updated to {n.snapshots[0]} … {n.snapshots[-1]}")


# ── Rolling horizon solve ─────────────────────────────────────────────────────
solver      = c.optimization["solver"]
solver_opts = {}
if c.optimization["solver_profile"]:
    from scripts.solver_profiles import get_solver_options
    solver_opts = get_solver_options(c.optimization["solver_profile"])

print(f"[rolling_horizon] solving with {solver}, {len(n.snapshots)} snapshots ...")

n.optimize.optimize_with_rolling_horizon(
    solver_name=solver,
    horizon=horizon,
    overlap=overlap,
    solver_options=solver_opts,
)

print(f"[rolling_horizon] solve complete.")


# ── Export ────────────────────────────────────────────────────────────────────
out_path = snakemake.output.network
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
n.export_to_netcdf(out_path)
print(f"[rolling_horizon] saved to {out_path}")
