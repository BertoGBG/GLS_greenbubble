"""EVPI and VSS from solved networks."""
from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest

from scripts.stochastic_values import expected_value_network, fix_capacities, value_of_information


def _solved(obj, const=0.0):
    return SimpleNamespace(objective=obj, objective_constant=const)


def test_evpi_and_vss_follow_their_definitions():
    probs = {"2022": 0.25, "2023": 0.75}
    ws = {"2022": _solved(80.0), "2023": _solved(100.0)}
    eev = {"2022": _solved(110.0), "2023": _solved(120.0, const=2.0)}
    df = value_of_information(100.0, probs, ws=ws, ev=_solved(95.0), eev=eev)
    v = df["value"]
    assert v["E_WS"] == pytest.approx(95.0)                       # 0.25*80 + 0.75*100
    assert v["EVPI"] == pytest.approx(5.0)
    assert v["EEV"] == pytest.approx(0.25 * 110 + 0.75 * 122)
    assert v["VSS"] == pytest.approx(v["EEV"] - 100.0)


def test_expected_value_network_averages_scenario_prices():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=3, freq="h"))
    n.add("Bus", "a")
    n.add("Generator", "g", bus="a", p_nom_extendable=True, marginal_cost=0.0)
    n.set_scenarios({"low": 0.25, "high": 0.75})
    n.generators_t.marginal_cost.loc[:, ("low", "g")] = 10.0
    n.generators_t.marginal_cost.loc[:, ("high", "g")] = 30.0
    ev = expected_value_network(n)
    assert not ev.has_scenarios
    assert (ev.generators_t.marginal_cost["g"] == 25.0).all()


def test_fix_capacities_keeps_capital_cost_in_the_objective():
    a, b = pypsa.Network(), pypsa.Network()
    for n in (a, b):
        n.add("Bus", "x")
        n.add("Generator", "g", bus="x", p_nom_extendable=True, capital_cost=1.0)
    b.generators.loc["g", "p_nom_opt"] = 7.0
    fix_capacities(a, b)
    assert a.generators.at["g", "p_nom_extendable"]
    assert a.generators.at["g", "p_nom_min"] == a.generators.at["g", "p_nom_max"] == 7.0
