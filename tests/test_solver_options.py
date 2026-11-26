"""MILP stopping options and when they apply."""
import pandas as pd
import pypsa

from scripts.helpers import mip_options


def test_mip_options_use_each_solver_key():
    assert mip_options("gurobi", 0.005, 3600) == {"MIPGap": 0.005, "TimeLimit": 3600.0}
    assert mip_options("highs", 0.005, 3600) == {"mip_rel_gap": 0.005, "time_limit": 3600.0}
    assert mip_options("gurobi", None, None) == {}


def _network(committable):
    n = pypsa.Network()
    n.set_snapshots(pd.RangeIndex(4))
    n.add("Bus", "el")
    n.add("Load", "d", bus="el", p_set=[2, 8, 8, 2])
    n.add("Generator", "g", bus="el", p_nom=10, marginal_cost=10,
          committable=committable, p_min_pu=0.5 if committable else 0.0)
    n.add("Generator", "backup", bus="el", p_nom=10, marginal_cost=100)
    return n


def test_only_a_committable_model_is_a_milp():
    for committable, expected in [(False, False), (True, True)]:
        m = _network(committable).optimize.create_model()
        assert bool(len(m.binaries) or len(m.integers)) is expected
