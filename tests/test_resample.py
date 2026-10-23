"""Temporal resampling of time-dependent network data."""
import pandas as pd
import pypsa
import pytest

from scripts.helpers import resample_network


def _network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2024-01-01", periods=48, freq="h"))
    n.add("Bus", "a")
    n.add("Bus", "b")
    n.add("Link", "slow", bus0="a", bus1="b", p_nom=10, ramp_limit_up=0.08, ramp_limit_down=0.1)
    n.add("Link", "fast", bus0="a", bus1="b", p_nom=10, ramp_limit_up=0.2)
    n.add("Link", "free", bus0="a", bus1="b", p_nom=10)
    return n


def test_ramp_limits_are_scaled_to_the_snapshot_length():
    """Per-hour limits become per-snapshot limits: 0.08 /h is 0.64 per 8 h."""
    n = resample_network(_network(), "8h")
    assert n.links.at["slow", "ramp_limit_up"] == pytest.approx(0.64)
    assert n.links.at["slow", "ramp_limit_down"] == pytest.approx(0.8)


def test_ramp_limits_that_cannot_bind_are_dropped():
    """0.2 /h over 8 h is 1.6 of p_nom: no constraint, and unset limits stay unset."""
    n = resample_network(_network(), "8h")
    assert pd.isna(n.links.at["fast", "ramp_limit_up"])
    assert pd.isna(n.links.at["free", "ramp_limit_up"])


def test_hourly_network_keeps_its_ramp_limits():
    n = resample_network(_network(), False)
    assert n.links.at["slow", "ramp_limit_up"] == pytest.approx(0.08)
