"""Export of all constraint duals and the objective rebuilt from them."""
import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.helpers import (
    _assign_all_duals_to_network,
    dual_objective,
    dual_objective_from_file,
    export_all_duals,
    load_duals,
)

CUSTOM = "wind_cap__custom"


def _network(stochastic=False):
    """One bus, extendable wind and gas, and an existing extendable plant."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=24, freq="h"))
    n.snapshot_weightings.loc[:, :] = 2.0
    n.add("Bus", "el")
    n.add("Load", "demand", bus="el", p_set=10 + 5 * np.sin(np.arange(24) / 4))
    n.add("Generator", "wind", bus="el", p_nom_extendable=True, capital_cost=50,
          p_max_pu=0.5 + 0.4 * np.cos(np.arange(24) / 3))
    n.add("Generator", "gas", bus="el", p_nom_extendable=True, capital_cost=20, marginal_cost=60)
    # Extendable with existing p_nom: PyPSA subtracts the capex of the existing
    # part through the fixed variable objective_constant.
    n.add("Generator", "existing", bus="el", p_nom=4, p_nom_min=4, p_nom_extendable=True,
          capital_cost=30, marginal_cost=40)
    if stochastic:
        n.set_scenarios({"low": 0.3, "high": 0.7})
        n.generators_t.p_max_pu.loc[:, ("high", "wind")] *= 0.6
    return n


def _solve(n):
    def extra(n, sns):
        m = n.model
        m.add_constraints(m.variables["Generator-p_nom"].sum() <= 30, name=CUSTOM)

    status, _ = n.optimize(solver_name="highs", extra_functionality=extra)
    assert status == "ok"


@pytest.mark.parametrize("stochastic", [False, True], ids=["deterministic", "stochastic"])
def test_objective_is_rebuilt_from_the_duals(tmp_path, stochastic):
    n = _network(stochastic)
    _solve(n)

    summary = dual_objective(n)
    assert "objective_constant" in summary["bounded_variables"]
    assert summary["bounded_variables"]["objective_constant"] != 0
    assert summary["residual"] < 1e-8

    path = tmp_path / "duals.nc"
    export_all_duals(n, path)
    assert dual_objective_from_file(path) == pytest.approx(n.model.objective.value, rel=1e-8)


def test_every_constraint_is_saved_on_its_own_coordinates(tmp_path):
    n = _network()
    _solve(n)
    path = tmp_path / "duals.nc"
    export_all_duals(n, path)

    duals = load_duals(path)
    assert set(duals) == set(n.model.constraints)
    assert CUSTOM in duals
    # No outer join across constraints: the bus balance only holds the bus.
    assert list(duals["Bus-nodal_balance"].indexes["name"]) == ["el"]
    assert duals["Bus-nodal_balance"].attrs["sign"] == "="


def test_assigning_all_duals_keeps_the_scaled_prices():
    n = _network()
    _solve(n)
    prices = n.buses_t.marginal_price.copy()
    _assign_all_duals_to_network(n)

    pd.testing.assert_frame_equal(n.buses_t.marginal_price, prices)
    assert "mu_upper" in n.generators_t
    # Prices are the nodal duals divided by the snapshot weighting (2 h).
    raw = n.model.constraints["Bus-nodal_balance"].dual.to_series().unstack("name")
    np.testing.assert_allclose(prices["el"].values, raw["el"].values / 2.0)
