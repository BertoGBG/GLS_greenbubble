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
sys.path.insert(0, str(Path(__file__).parent.parent))

import pypsa
from scripts import config as c, parameters as p
from scripts.helpers import prepare_costs
from scripts.technology_inputs import tech_inputs


rh        = c.rolling_horizon
horizon   = rh["horizon"]
overlap   = rh["overlap"]
rh_year   = rh["rh_year"]       # None → same year as En_price_year
main_year = c.En_price_year

print(f"[rolling_horizon] OPT network : {snakemake.input.network}")
print(f"[rolling_horizon] horizon={horizon} h  overlap={overlap} h")
print(f"[rolling_horizon] main year={main_year}  rh_year={rh_year or 'same'}")


# ── Identify demand-buffer buses ──────────────────────────────────────────────
def find_demand_buffer_buses(n, concentration_threshold=0.95):
    """Return buses that carry a concentrated annual demand load.

    These buses have a virtual buffer store (e.g. 'bioCH4 delivery') that
    accumulates flexible production and discharges it at a point-in-time demand
    event.  Their e_cyclic constraint must be preserved in rolling horizon so
    that each window is self-balancing (net production = net discharge = demand
    per window) and the store does not accumulate across windows.
    """
    demand_buses = set()
    for name in n.loads_t.p_set.columns:
        ts = n.loads_t.p_set[name]
        total = ts.sum()
        if total <= 0:
            continue
        if ts.max() / total >= concentration_threshold:
            demand_buses.add(n.loads.loc[name, "bus"])
    return demand_buses


# ── Disable cyclic storage constraints ───────────────────────────────────────
def disable_cyclic_constraints(n, preserve_buses=None):
    """Remove end-of-period = start-of-period constraints from real storage.

    In a full-year perfect-foresight solve these constraints are meaningful.
    In rolling horizon, PyPSA carries the end-of-window state forward as the
    initial condition for the next window, so enforcing a cyclic constraint
    within each window would incorrectly force every window to return to its
    opening state of charge.

    Exception — demand buffer stores (preserve_buses): their e_cyclic is kept
    True so that each window is self-balancing.  Without it, the optimizer
    over-fills the buffer when RE is cheap and the year-end SOC drifts far from
    zero (seen as ~70 % over-production in practice).
    """
    preserve_buses = preserve_buses or set()

    if not n.stores.empty and "e_cyclic" in n.stores.columns:
        preserve_mask = n.stores["bus"].isin(preserve_buses)
        disable_mask  = n.stores["e_cyclic"] & ~preserve_mask
        n.stores.loc[disable_mask, "e_cyclic"] = False
        print(
            f"[rolling_horizon] disabled e_cyclic on {disable_mask.sum()} stores, "
            f"preserved on {preserve_mask.sum()} demand-buffer stores "
            f"({list(n.stores.index[preserve_mask])})"
        )

    if not n.storage_units.empty and "cyclic_state_of_charge" in n.storage_units.columns:
        n_cyclic = n.storage_units["cyclic_state_of_charge"].sum()
        n.storage_units["cyclic_state_of_charge"] = False
        print(f"[rolling_horizon] disabled cyclic_state_of_charge on {n_cyclic} storage_units")


# ── Distribute point demands ──────────────────────────────────────────────────
def distribute_point_demands(n, concentration_threshold=0.95):
    """Redistribute annual demands that are concentrated in a single time step.

    Annual demands are often modelled as a lump discharge at the last snapshot
    (backed by a virtual store that accumulates flexible production).  In
    rolling horizon every window except the last would see zero demand and
    therefore have no incentive to produce, making the problem infeasible or
    heavily sub-optimal.

    This function detects such loads (>= concentration_threshold share of total
    demand in a single timestep) and replaces them with a constant hourly
    demand so every window has a proportional production incentive.  The total
    annual demand is preserved exactly.
    """
    if n.loads_t.p_set.empty:
        return

    n_snapshots = len(n.snapshots)
    modified = []
    for name in n.loads_t.p_set.columns:
        ts = n.loads_t.p_set[name]
        total = ts.sum()
        if total <= 0:
            continue
        max_share = ts.max() / total
        if max_share >= concentration_threshold:
            n.loads_t.p_set[name] = total / n_snapshots
            modified.append(f"{name} ({total:.0f} MWh → {total/n_snapshots:.2f} MW constant)")

    if modified:
        print(f"[rolling_horizon] redistributed point demands to uniform rate:")
        for m in modified:
            print(f"  {m}")


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
demand_buses = find_demand_buffer_buses(n)
fix_capacities(n)
disable_cyclic_constraints(n, preserve_buses=demand_buses)
distribute_point_demands(n)

if rh_year and rh_year != main_year:
    print(f"[rolling_horizon] patching time series to rh_year={rh_year} ...")

    from scripts.create_stoch_scenarios import set_input_paths, patch_timeseries
    from scripts.preprocessing import prepare_all_inputs

    tech_costs = prepare_costs(
        latitude        = c.latitude,
        longitude       = c.longitude,
        tech_inputs     = tech_inputs,
        USD_to_EUR      = c.USD_to_EUR,
        discount_rate   = c.discount_rate,
        cost_path_EU    = snakemake.input.costs_eu,
        cost_path_US    = p.cost_path_US,
        dict_tech_US_EU = p.dict_tech_US_EU,
    )

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
    solver_opts = get_solver_options(solver, c.optimization["solver_profile"])

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
