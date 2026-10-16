"""Stock targets for product deliveries in rolling horizon."""
import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.delivery_targets import (UNMET_CARRIER, inflow_capacity, minimum_stock,
                                      pro_rata_stock, set_delivery_targets, window_ends)


def test_window_ends_follow_pypsa_starts():
    """Windows start every horizon - overlap snapshots; a window hands over at start - 1."""
    assert window_ends(10, horizon=4, overlap=1).tolist() == [2, 5, 8, 9]
    assert window_ends(8760, 168, 72)[:2].tolist() == [95, 191]


def test_minimum_stock_is_a_ramp_before_one_delivery():
    """10 MWh due at the last of 20 hours with 1 MWh/h: stock must ramp over the last 9 h."""
    d = np.zeros(20); d[-1] = 10.0
    e, e_start = minimum_stock(d, np.ones(20))
    assert e[9] == 0.0 and e[10] == 1.0 and e[18] == 9.0 and e[19] == 0.0
    assert e_start == 0.0


def test_minimum_stock_reaches_back_across_bins():
    """A large second delivery needs stock carried past the first one."""
    d = np.zeros(8); d[3] = 2.0; d[7] = 6.0
    e, _ = minimum_stock(d, np.ones(8))
    assert e[3] == 2.0          # after the first delivery, 2 MWh already held for the second
    assert e[2] == 3.0          # before it, the first delivery's 2 MWh on top of 1 MWh carried


def test_minimum_stock_start_when_first_delivery_is_too_early():
    d = np.zeros(4); d[0] = 5.0
    _, e_start = minimum_stock(d, np.ones(4))
    assert e_start == 4.0


def test_carried_stock_is_a_fixed_point_when_the_year_is_feasible():
    """Ending the year with e_carry does not raise the stock needed at the start."""
    d = np.zeros(10); d[1] = 4.0; d[9] = 2.0          # 6 MWh in 10 h at 1 MWh/h, early bin too big
    _, e_carry = minimum_stock(d, np.ones(10))
    e_min, e_start = minimum_stock(d, np.ones(10), e_end=e_carry)
    assert e_carry == 2.0 and e_start == e_carry and e_min[-1] == e_carry


def test_pro_rata_grows_linearly_within_each_bin():
    d = np.array([0, 0, 0, 4.0, 0, 2.0])
    e = pro_rata_stock(d, np.ones(6))
    assert e.tolist() == [1.0, 2.0, 3.0, 0.0, 1.0, 0.0]


def test_pro_rata_is_zero_for_continuous_demand():
    assert not pro_rata_stock(np.full(24, 3.0), np.ones(24)).any()


@pytest.fixture
def network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=48, freq="h"))
    n.add("Bus", "collection")
    n.add("Bus", "MeOH delivery")
    n.add("Link", "collection_to_demand", bus0="collection", bus1="MeOH delivery", p_nom=1.0)
    p_set = pd.Series(0.0, index=n.snapshots); p_set.iloc[-1] = 30.0
    n.add("Load", "MeOH delivery", bus="MeOH delivery", p_set=p_set)
    n.add("Store", "MeOH delivery store", bus="MeOH delivery", e_nom=5.0, e_cyclic=True)
    return n


def test_targets_only_at_window_ends_and_store_empties(network):
    n = network
    report = set_delivery_targets(n, horizon=12, overlap=4, rule="e_min")
    r = report["MeOH delivery store"]
    assert r["total"] == 30.0 and r["n_bins"] == 1
    e_min_pu = n.stores_t.e_min_pu["MeOH delivery store"]
    ends = window_ends(48, 12, 4)
    assert (e_min_pu.drop(n.snapshots[ends]) == 0).all()
    # 30 MWh at 1 MWh/h: hand-over after hour 39 needs 30 - 8 = 22 MWh
    assert e_min_pu.iloc[39] * 30.0 == pytest.approx(22.0)
    assert n.stores_t.e_max_pu["MeOH delivery store"].iloc[-1] == 0.0
    assert not n.stores.at["MeOH delivery store", "e_cyclic"]
    assert n.stores.at["MeOH delivery store", "e_nom"] == 30.0
    assert (n.generators.carrier == UNMET_CARRIER).sum() == 1


def test_pro_rata_target_is_at_least_e_min(network):
    n = network
    set_delivery_targets(n, horizon=12, overlap=4, rule="pro_rata")
    target = n.stores_t.e_min_pu["MeOH delivery store"] * 30.0
    # hand-over after hour 7: pro rata 30 * 8/48 = 5, e_min 0
    assert target.iloc[7] == pytest.approx(5.0)
    assert target.iloc[39] == pytest.approx(25.0)   # pro rata 30*40/48 = 25 > e_min 22


def test_continuous_demand_keeps_its_store_size():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=48, freq="h"))
    n.add("Bus", "CH4 delivery")
    n.add("Generator", "plant", bus="CH4 delivery", p_nom=3.0)
    n.add("Load", "CH4 delivery", bus="CH4 delivery", p_set=2.0)
    n.add("Store", "CH4 delivery store", bus="CH4 delivery", e_nom=5.0, e_cyclic=True)
    r = set_delivery_targets(n, horizon=12, overlap=4)["CH4 delivery store"]
    assert not r["binned"] and n.stores.at["CH4 delivery store", "e_nom"] == 5.0
    assert not n.stores_t.e_min_pu["CH4 delivery store"].any()
    assert n.stores_t.e_max_pu["CH4 delivery store"].iloc[-1] == 0.0


def test_inflow_capacity_is_limited_by_the_plants_behind_a_collection_bus(network):
    """The collection link is 1 MW, but the plant behind it makes only 0.8 MW of product."""
    n = network
    n.add("Bus", "H2")
    n.add("Link", "plant", bus0="H2", bus1="collection", p_nom=1.0, efficiency=0.8)
    assert inflow_capacity(n, "MeOH delivery").iloc[0] == pytest.approx(0.8)
