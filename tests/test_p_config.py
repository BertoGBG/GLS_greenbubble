"""p_config regression tests.

The point of these is narrow and important: p_config replaced a hand-written
Python dict with YAML, and the refactor is only safe for as long as the frame it
produces stays field-for-field identical in *shape and semantics* to what the
model expects. These tests pin that contract.

Run:  pytest tests/test_p_config.py
"""
import pandas as pd
import pytest

from scripts import config as c
from scripts.technology_inputs import symbiosis_n, lhv_biogas, biogas_mix, T_max_comp


EXPECTED_COLUMNS = {"fluid", "T", "P", "LHV", "carrier", "buses", "moisture"}
EXPECTED_N_STREAMS = 27


def test_streams_shape():
    assert len(c.p_streams) == EXPECTED_N_STREAMS
    assert set(c.p_streams.columns) == EXPECTED_COLUMNS


def test_symbiosis_n_is_p_streams():
    """technology_inputs must expose exactly what p_config built."""
    pd.testing.assert_frame_equal(symbiosis_n, c.p_streams, check_like=True)


def test_references_resolved():
    """No ${...} may survive into the frame."""
    for col in c.p_streams.columns:
        for v in c.p_streams[col]:
            assert not (isinstance(v, str) and v.startswith("${")), f"unresolved ref {v!r}"


def test_compressor_discharge_temperatures_use_the_global():
    """Streams downstream of a compressor sit at T_max_comp, not a stray literal."""
    for s in ["H2 to methanolisation", "CO2 to methanolisation", "biogas to methanation"]:
        assert c.p_streams.at[s, "T"] == T_max_comp


def test_lhv_biogas_is_derived_not_configured():
    """lhv_biogas must follow the composition, never be typed in."""
    assert "biogas" not in c.p_globals["lhv"], "lhv.biogas must stay derived"
    expected = c.p_globals["lhv"]["ch4"] * (
        biogas_mix["Methane"] * 16.04246e-3
        / (biogas_mix["Methane"] * 16.04246e-3 + biogas_mix["CarbonDioxide"] * 44.0098e-3)
    )
    assert lhv_biogas == pytest.approx(expected, rel=1e-4)


def test_heat_tiers_are_ordered():
    """MT above DH above LT -- the cascade the heat buses assume."""
    mt = c.p_streams.at["Heat MT min", "T"]
    dh = c.p_streams.at["Heat DH min", "T"]
    lt = c.p_streams.at["Heat LT min", "T"]
    assert mt > dh > lt


def test_process_streams_hooks_are_wellformed():
    """Whatever is declared must carry a duty_ref and a temperature pair."""
    for proc, streams in (c.p_process_streams or {}).items():
        for name, spec in streams.items():
            assert "duty_ref" in spec, f"{proc}.{name} missing duty_ref"
            assert ":" in spec["duty_ref"], f"{proc}.{name} duty_ref must be 'tech:param'"
            assert {"T_supply", "T_target"} <= set(spec), f"{proc}.{name} missing temperatures"
            assert spec.get("type") in {"source", "sink"}, f"{proc}.{name} bad type"
