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
    event.  Call this before distribute_point_demands() — after redistribution
    all loads are uniform and detection no longer works.
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
def disable_cyclic_constraints(n):
    """Remove end-of-period = start-of-period constraints from real storage.

    In a full-year perfect-foresight solve these constraints are meaningful.
    In rolling horizon, PyPSA carries the end-of-window state forward as the
    initial condition for the next window, so enforcing a cyclic constraint
    within each window would incorrectly force every window to return to its
    opening state of charge.

    Demand buffer stores are handled separately by cap_demand_buffer_stores():
    their e_cyclic is also set False here, but their e_nom is capped to ~2×
    per-window demand so the optimizer cannot massively over-produce early.
    Setting e_cyclic=True is NOT used — PyPSA treats it as a free initial SOC
    optimisation variable, causing systematic under-production.
    """
    if not n.stores.empty and "e_cyclic" in n.stores.columns:
        n_cyclic = n.stores["e_cyclic"].sum()
        n.stores["e_cyclic"] = False
        print(f"[rolling_horizon] disabled e_cyclic on {n_cyclic} stores")

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


# ── Cap demand-buffer store capacity to per-window scale ─────────────────────
def cap_demand_buffer_stores(n, demand_buses, horizon):
    """Limit e_nom of demand-buffer stores to 2× per-window demand.

    These stores are sized for full-year accumulation (e_nom ≈ annual demand).
    With e_cyclic=False and no capacity cap, the optimizer can massively
    over-produce early in the year (when RE is cheap) and coast afterwards,
    because there is nothing to stop the store from filling to annual capacity.

    Capping e_nom to 2× per-window demand (2 × horizon × avg_rate) limits the
    surplus carry-over to at most two windows and forces the optimizer to spread
    production over the full year.  e_initial is set to 0 so each fresh solve
    starts from an empty store.
    """
    if n.stores.empty or not demand_buses:
        return

    # Bus → total annual demand (loads_t.p_set already redistributed to uniform)
    bus_demand = {}
    weights = n.snapshot_weightings.generators
    for name in n.loads_t.p_set.columns:
        bus = n.loads.loc[name, "bus"]
        if bus in demand_buses:
            annual = (n.loads_t.p_set[name] * weights).sum()
            bus_demand[bus] = bus_demand.get(bus, 0.0) + annual

    n_snapshots = len(n.snapshots)
    store_mask = n.stores["bus"].isin(demand_buses)
    for store_name in n.stores.index[store_mask]:
        bus = n.stores.loc[store_name, "bus"]
        annual_demand = bus_demand.get(bus, 0.0)
        if annual_demand <= 0:
            continue
        per_window = annual_demand / n_snapshots * horizon
        cap = 2.0 * per_window
        old_e_nom = n.stores.loc[store_name, "e_nom"]
        n.stores.loc[store_name, "e_nom"]     = cap
        n.stores.loc[store_name, "e_initial"] = 0.0
        print(
            f"[rolling_horizon] capped '{store_name}' e_nom: "
            f"{old_e_nom:,.0f} → {cap:,.0f} MWh  "
            f"(2× {per_window:,.0f} MWh/window × {horizon} h)"
        )


# ── Post-solve annual balance diagnostic ─────────────────────────────────────
def check_annual_balance(n):
    """Print annual delivery and store SOC drift to verify RH energy closure."""
    weights = n.snapshot_weightings.generators
    print("[rolling_horizon] annual demand balance check:")
    for name in n.loads_t.p_set.columns:
        delivered = (n.loads_t.p_set[name] * weights).sum()
        print(f"  load '{name}': delivered = {delivered:,.0f} MWh")

    if not n.stores_t.e.empty:
        e_start = n.stores_t.e.iloc[0]
        e_end   = n.stores_t.e.iloc[-1]
        delta   = e_end - e_start
        significant = delta[delta.abs() > 1.0]
        if significant.empty:
            print("  ✓ no significant SOC drift — production ≈ demand")
        else:
            print("  store SOC drift (end − start):")
            for store, d in significant.items():
                sign = "over-produced" if d > 0 else "under-produced"
                print(f"    '{store}': Δ = {d:+,.0f} MWh  [{sign}]")


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


# ── Re-enable committable for RH dispatch ─────────────────────────────────────
def enable_committable_for_rh(n, n_config):
    """Re-enable committable=True for links whose tech has committable: true in n_config.

    GreenBubble's prepare_network.py sets committable=False whenever expansion=True,
    keeping the capacity expansion solve as a pure LP for speed and stochastic
    compatibility.  After fix_capacities() all p_nom are fixed, so unit commitment
    can safely be activated for the dispatch-only RH solve.

    Note: PyPSA itself does support committable + extendable simultaneously in
    deterministic capacity expansion via a big-M MILP formulation.  See
    https://docs.pypsa.org/latest/examples/committable-extendable/ .

    Matching is by link carrier == n_config index (e.g. 'electrolysis', 'biogas engine').
    Both EXI_ and new-build links with the same carrier are enabled.
    """
    if "committable" not in n_config.columns:
        print("[rolling_horizon] n_config has no 'committable' column — skipping")
        return

    committable_techs = n_config.index[n_config["committable"] == True].tolist()
    if not committable_techs:
        print("[rolling_horizon] no committable techs in n_config — RH is pure LP")
        return

    enabled = []
    for tech in committable_techs:
        mask = n.links["carrier"] == tech
        if not mask.any():
            continue
        n.links.loc[mask, "committable"] = True
        if "min load" in n_config.columns:
            min_load = n_config.at[tech, "min load"]
            if min_load == min_load:  # False for NaN
                n.links.loc[mask, "p_min_pu"] = float(min_load)
        enabled.extend(n.links.index[mask].tolist())

    if enabled:
        print(f"[rolling_horizon] committable enabled (MILP) for links: {enabled}")
    else:
        print(f"[rolling_horizon] committable techs {committable_techs} not found in network links")


# ── Build dispatch network ────────────────────────────────────────────────────
n_opt = pypsa.Network(snakemake.input.network)
n = n_opt.copy()
demand_buses = find_demand_buffer_buses(n)   # detect before redistribution
fix_capacities(n)
enable_committable_for_rh(n, c.n_config)
disable_cyclic_constraints(n)
distribute_point_demands(n)
cap_demand_buffer_stores(n, demand_buses, horizon)

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
check_annual_balance(n)

# ── Restore extendability flags so statistics.capex() works in post-processing ─
# fix_capacities() set p_nom_extendable=False on all components so the RH solver
# treats them as fixed.  Restoring the original flags (from n_opt) means
# n.statistics.capex() returns the correct per-carrier breakdown, and the same
# CSV/plot functions used for the PF result work unchanged on the RH network.
for _comp, _ext_col in [
    ("generators",    "p_nom_extendable"),
    ("links",         "p_nom_extendable"),
    ("storage_units", "p_nom_extendable"),
    ("stores",        "e_nom_extendable"),
]:
    _df     = getattr(n,     _comp)
    _df_opt = getattr(n_opt, _comp)
    if _df.empty or _ext_col not in _df_opt.columns:
        continue
    _df[_ext_col] = _df_opt[_ext_col]

# ── Cost summary ──────────────────────────────────────────────────────────────
# n.objective after RH solve contains only OPEX (all capacities were fixed).
# Use n.statistics.capex() + n.statistics.opex() for full comparable totals;
# statistics.capex() works here because extendability flags were restored above.
_pf_capex = n_opt.statistics.capex().sum()
_pf_opex  = n_opt.statistics.opex().sum()
_rh_capex = n.statistics.capex().sum()   # == _pf_capex (same capacities)
_rh_opex  = n.statistics.opex().sum()
_pf_total = _pf_capex + _pf_opex
_rh_total = _rh_capex + _rh_opex
print(f"[rolling_horizon] ── cost comparison ────────────────")
print(f"[rolling_horizon]   CAPEX (shared) : {_pf_capex/1e6:>10.3f} M€")
print(f"[rolling_horizon]   PF OPEX        : {_pf_opex/1e6:>10.3f} M€   → PF total: {_pf_total/1e6:.3f} M€")
print(f"[rolling_horizon]   RH OPEX        : {_rh_opex/1e6:>10.3f} M€   → RH total: {_rh_total/1e6:.3f} M€")
print(f"[rolling_horizon]   OPEX premium   : {(_rh_opex - _pf_opex)/1e6:>+10.3f} M€  ({(_rh_total/_pf_total - 1)*100:+.2f} %)")


# ── Export ────────────────────────────────────────────────────────────────────
out_path = snakemake.output.network
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
n.export_to_netcdf(out_path)
print(f"[rolling_horizon] saved to {out_path}")
