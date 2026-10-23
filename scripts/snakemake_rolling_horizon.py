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
import logging
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pypsa
from scripts import config as c, parameters as p
from scripts.helpers import prepare_costs
from scripts.technology_inputs import tech_inputs
from scripts.delivery_targets import set_delivery_targets, unmet_deliveries, drop_unused_unmet


rh        = c.rolling_horizon
horizon   = rh["horizon"]
overlap   = rh["overlap"]
rh_year   = rh["rh_year"]       # None → same year as En_price_year
main_year = c.En_price_year
delivery_target = rh.get("delivery_target", "pro_rata")
unmet_cost      = rh.get("unmet_cost", 1e4)

print(f"[rolling_horizon] OPT network : {snakemake.input.network}")
print(f"[rolling_horizon] horizon={horizon}  overlap={overlap} snapshots")
print(f"[rolling_horizon] main year={main_year}  rh_year={rh_year or 'same'}")


# ── Disable cyclic storage constraints ───────────────────────────────────────
def disable_cyclic_constraints(n):
    """Remove end-of-period = start-of-period constraints from real storage.

    In a full-year perfect-foresight solve these constraints are meaningful.
    In rolling horizon, PyPSA carries the end-of-window state forward as the
    initial condition for the next window, so enforcing a cyclic constraint
    within each window would incorrectly force every window to return to its
    opening state of charge.

    Product delivery stores get their own start, end and window targets in
    set_delivery_targets(). Setting e_cyclic=True is NOT used: PyPSA treats it
    as a free initial SOC optimisation variable, causing systematic
    under-production.
    """
    if not n.stores.empty and "e_cyclic" in n.stores.columns:
        n_cyclic = n.stores["e_cyclic"].sum()
        n.stores["e_cyclic"] = False
        print(f"[rolling_horizon] disabled e_cyclic on {n_cyclic} stores")

    if not n.storage_units.empty and "cyclic_state_of_charge" in n.storage_units.columns:
        n_cyclic = n.storage_units["cyclic_state_of_charge"].sum()
        n.storage_units["cyclic_state_of_charge"] = False
        print(f"[rolling_horizon] disabled cyclic_state_of_charge on {n_cyclic} storage_units")


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
            # Small headroom: a barrier solve without crossover can leave p_nom_opt
            # a hair below the flow it carries (e.g. a flat demand link short by
            # 7e-5 MW), which turns every RH window infeasible once it is fixed.
            opt = df.loc[ext_mask, opt_col]
            df.loc[ext_mask, nom_col] = (opt * (1 + 1e-6) + 1e-3).where(opt > 0, opt)
            if comp == "stores":
                # Free, non-cyclic stores are annual sink counters (CO2 vent,
                # ambient heat, DH sales, ...): e_nom_opt is just the PF year's
                # cumulative total. Capping RH at it makes the last windows
                # infeasible once RH dispatch drifts above that total.
                sink = ext_mask & (df["capital_cost"] == 0) & ~df["e_cyclic"].astype(bool)
                df.loc[sink, nom_col] = np.inf
                print(f"[rolling_horizon] left {sink.sum()} sink stores uncapped: {list(df.index[sink])}")
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
fix_capacities(n)
enable_committable_for_rh(n, c.n_config)
disable_cyclic_constraints(n)

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


# ── Delivery targets ──────────────────────────────────────────────────────────
# After the cross-year patch, so the targets follow the loads actually solved.
targets = set_delivery_targets(n, horizon, overlap, rule=delivery_target, unmet_cost=unmet_cost)
for store, r in targets.items():
    print(f"[rolling_horizon] delivery targets '{store}' ({delivery_target}): "
          f"{r['total']:,.0f} MWh in {r['n_bins']} deliveries, largest {r['largest_bin']:,.0f} MWh, "
          f"inflow capacity {r['capacity']:,.2f} MW, max e_min {r['max_e_min']:,.0f} MWh, "
          f"target set in {r['windows_with_target']}/{r['n_windows']} windows, "
          f"store {r['e_nom']:,.0f} MWh ({'binned: remaining demand' if r['binned'] else 'continuous: PF size'})")
    if r["e_min_above_store"]:
        print(f"[rolling_horizon]   WARNING: e_min exceeds the store size; the shortfall "
              f"goes to the unmet delivery generator.")
    if r["e_carry"] > 0:
        print(f"[rolling_horizon]   the first deliveries need {r['e_carry']:,.0f} MWh before "
              f"1 January; the store starts and ends the year with it, as the cyclic PF store does.")
    if not r["feasible"]:
        print(f"[rolling_horizon]   WARNING: the annual deliveries exceed what the inflow "
              f"capacity can make in a year; the shortfall goes to the unmet delivery generator.")


# ── Rolling horizon solve ─────────────────────────────────────────────────────
solver      = c.optimization["solver"]
solver_opts = {}
if c.optimization["solver_profile"]:
    from scripts.solver_profiles import get_solver_options
    solver_opts = get_solver_options(solver, c.optimization["solver_profile"])

print(f"[rolling_horizon] solving with {solver}, {len(n.snapshots)} snapshots ...")

# PyPSA only logs a warning when a window fails and carries on, so count the
# failures and stop instead of exporting an empty dispatch.
class _FailedWindows(logging.Handler):
    count = 0
    def emit(self, record):
        if record.getMessage().startswith("Optimization failed"):
            self.count += 1

_failed = _FailedWindows(level=logging.WARNING)
logging.getLogger("pypsa.optimization.abstract").addHandler(_failed)

n.optimize.optimize_with_rolling_horizon(
    solver_name=solver,
    horizon=horizon,
    overlap=overlap,
    solver_options=solver_opts,
    include_objective_constant=True,   # explicit; the default flips in PyPSA 2.0
)

if _failed.count:
    raise RuntimeError(
        f"[rolling_horizon] {_failed.count} window(s) failed to solve. "
        "Check the solver log above for the infeasible window."
    )
print(f"[rolling_horizon] solve complete.")
check_annual_balance(n)
unmet = unmet_deliveries(n)
for bus, mwh in unmet[unmet > 0.01].items():
    print(f"[rolling_horizon]   WARNING: {mwh:,.1f} MWh of '{bus}' deliveries unmet "
          f"(unmet delivery generator, {unmet_cost:g} EUR/MWh)")
drop_unused_unmet(n)

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
