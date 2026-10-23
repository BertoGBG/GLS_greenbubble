# SPDX-License-Identifier: MIT
"""Networks and summary for the value of information in stochastic runs.

All networks are taken from the stochastic network before it is solved, so
they share its scenario data exactly:

* wait-and-see (WS): one scenario, capacities free. E_WS is the
  probability-weighted mean of their objectives; EVPI = RP - E_WS.
* expected value (EV): one network whose time series are the
  probability-weighted mean over scenarios, capacities free.
* expected result of the EV solution (EEV): each scenario with capacities
  fixed to the EV solution, dispatch free. EEV is the probability-weighted
  mean of their objectives; VSS = EEV - RP.

RP is the stochastic objective. See docs/guide_stochastic.rst.
"""
import pandas as pd

# (list name, nominal attribute) of every component with a capacity variable
CAPACITIES = [("generators", "p_nom"), ("links", "p_nom"),
              ("stores", "e_nom"), ("storage_units", "p_nom")]
COMPONENTS = ["generators", "links", "loads", "stores", "storage_units", "buses"]


def scenario_probabilities(n):
    """Scenario weights of a stochastic network, as a Series of floats."""
    return n.scenario_weightings["weight"].astype(float)


def wait_and_see_networks(n):
    """One deterministic network per scenario of the unsolved stochastic network."""
    return {str(s): n.get_scenario(s) for s in n.scenarios}


def expected_value_network(n):
    """Deterministic network with every scenario-dependent value replaced by its
    probability-weighted mean (the expected-value problem)."""
    probs = scenario_probabilities(n)
    ev = n.get_scenario(n.scenarios[0])
    for comp in COMPONENTS:
        dyn, ev_dyn = getattr(n, f"{comp}_t"), getattr(ev, f"{comp}_t")
        for attr, df in dyn.items():
            if df.empty or not isinstance(df.columns, pd.MultiIndex):
                continue
            mean = sum(p * df.xs(s, axis=1, level="scenario") for s, p in probs.items())
            ev_dyn[attr] = mean.reindex(columns=ev_dyn[attr].columns.union(mean.columns))
        static, ev_static = getattr(n, comp), getattr(ev, comp)
        if not isinstance(static.index, pd.MultiIndex) or static.empty:
            continue
        num = static.select_dtypes("number")
        mean = sum(p * num.xs(s, level="scenario") for s, p in probs.items())
        differs = [col for col in mean.columns
                   if not num[col].groupby(level=-1).nunique().le(1).all()]
        if differs:
            ev_static.loc[mean.index, differs] = mean[differs]
    return ev


def fix_capacities(n, solved):
    """Fix every extendable capacity of ``n`` to its optimum in ``solved``.

    The component stays extendable with p_nom_min = p_nom_max, so its capital
    cost remains in the objective, as in the stochastic and EV problems.
    """
    for comp, nom in CAPACITIES:
        static, opt = getattr(n, comp), getattr(solved, comp)
        ext = static.index[static[f"{nom}_extendable"]].intersection(opt.index)
        if not len(ext):
            continue
        cap = opt.loc[ext, f"{nom}_opt"]
        static.loc[ext, [nom, f"{nom}_min", f"{nom}_max"]] = pd.concat([cap] * 3, axis=1).to_numpy()


def objective(n):
    """Total system cost of a solved network, including the objective constant."""
    return float(n.objective) + float(getattr(n, "objective_constant", 0.0) or 0.0)


def value_of_information(z_rp, probs, ws=None, ev=None, eev=None):
    """Table of WS and EEV objectives per scenario and RP, E_WS, EVPI, EV, EEV, VSS.

    ``ws`` and ``eev`` map scenario to solved network, ``ev`` is the solved EV
    network; each part is left out when not given.
    """
    probs = pd.Series({str(k): float(v) for k, v in probs.items()})
    df = pd.DataFrame({"probability": probs})
    rows = {"RP": z_rp}
    if ws:
        df["WS_objective"] = pd.Series({str(s): objective(m) for s, m in ws.items()})
        rows["E_WS"] = (df["WS_objective"] * df["probability"]).sum()
        rows["EVPI"] = z_rp - rows["E_WS"]
    if ev is not None:
        rows["EV"] = objective(ev)
    if eev:
        df["EEV_objective"] = pd.Series({str(s): objective(m) for s, m in eev.items()})
        rows["EEV"] = (df["EEV_objective"] * df["probability"]).sum()
        rows["VSS"] = rows["EEV"] - z_rp
    summary = pd.DataFrame({"value": pd.Series(rows)})
    return pd.concat([df.sort_index(), summary])
