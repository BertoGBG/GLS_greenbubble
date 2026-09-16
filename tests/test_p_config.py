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


EXPECTED_COLUMNS = {"fluid", "T", "P", "LHV", "carrier", "buses", "moisture", "bus_suffix"}
EXPECTED_N_STREAMS = 28   # +1: "H2 SOEC outlet", SOEC's low-pressure H2 bus


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


def test_every_stream_has_a_bus_carrier():
    """p_config's `carrier` is the BUS carrier (add_requirements_buses -> n.add("Bus", carrier=...)).

    It is never the component carrier: the stores on the HP-storage buses all share
    'HP gas storage' from n_config. Every stream needs one; 'H2 HP storage' was missing
    it historically, which went unnoticed only because another code path created that
    bus with the right carrier first.
    """
    missing = [i for i in c.p_streams.index if not isinstance(c.p_streams.at[i, "carrier"], str)]
    assert not missing, f"streams without a bus carrier: {missing}"


def test_hp_storage_carriers_match_their_fluid():
    assert c.p_streams.at["H2 HP storage", "carrier"] == "H2"
    assert c.p_streams.at["CO2 HP storage", "carrier"] == "CO2"


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


# --- grouped schema (shared: + processes:) -------------------------------------

def _raw():
    import yaml
    from pathlib import Path
    return yaml.safe_load(open(Path(c.__file__).parent.parent / "config" / "p_config.default.yaml"))


def test_no_pressure_references_a_temperature_global():
    """P must never resolve through a temperature.

    This is not hypothetical: the first generated p_config bound three methanation
    feed pressures to ${T_ambient}, purely because T_ambient and the pressure were
    both 20. Numerically identical, semantically wrong -- changing the ambient
    temperature would have moved three pressures.
    """
    import re
    from pathlib import Path
    text = (Path(c.__file__).parent.parent / "config" / "p_config.default.yaml").read_text()
    bad = re.findall(r'P:\s*"\$\{T_[A-Za-z_]+\}"', text)
    assert not bad, f"pressure bound to a temperature global: {bad}"


def test_ports_inherit_only_state_not_buses():
    """`from` must not drag a parent's bus list into a port."""
    raw = _raw()
    shared = raw["shared"]
    for proc, ports in (raw.get("processes") or {}).items():
        for role, spec in ports.items():
            parent = spec.get("from")
            if not parent:
                continue
            pbuses = (shared[parent.split(":", 1)[1]].get("model") or {}).get("buses", [])
            own = (spec.get("model") or {}).get("buses", [])
            assert not (set(pbuses) & set(own)), f"{proc}.{role} shares a bus with its parent"


def test_every_port_declares_its_own_buses():
    raw = _raw()
    for proc, ports in (raw.get("processes") or {}).items():
        for role, spec in ports.items():
            assert (spec.get("model") or {}).get("buses"), f"{proc}.{role} declares no buses"


def test_one_state_per_bus_validator_catches_a_conflict():
    """The validator must reject two states on one bus, not silently pick one."""
    import pandas as pd
    frame = pd.DataFrame(
        {"fluid": ["H2", "H2"], "T": [20, 160], "P": [30, 80], "carrier": ["H2", "H2"],
         "buses": [["shared bus"], ["shared bus"]]},
        index=["low", "high"],
    )
    with pytest.raises(ValueError, match="claimed by"):
        c._p_check_one_state_per_bus({}, frame)


def test_one_state_per_bus_allows_agreement():
    """Two ports on one bus are fine when they agree -- the inheritance case."""
    import pandas as pd
    frame = pd.DataFrame(
        {"fluid": ["CH4", "CH4"], "T": [50, 50], "P": [1, 1], "carrier": ["gas", "gas"],
         "buses": [["bioCH4 collection"], ["bioCH4 collection"]]},
        index=["from biomethanation", "from upgrading"],
    )
    c._p_check_one_state_per_bus({}, frame)   # must not raise


def test_bus_suffix_declared_not_hardcoded():
    """Plant-prefixed buses resolve via a declared suffix, not an if/elif in the code.

    'meoh H2 HP storage' and 'methanation H2 HP storage' are created with a runtime
    prefix, so p_config cannot enumerate them. It declares the suffix instead; the
    resolver in add_requirements_buses reads that column rather than naming the two
    streams in source.
    """
    suffixes = c.p_streams["bus_suffix"].dropna().to_dict()
    assert suffixes == {"H2 HP storage": "H2 HP storage", "CO2 HP storage": "CO2 HP storage"}

    import re
    from pathlib import Path
    src = (Path(c.__file__).parent / "prepare_network.py").read_text()
    assert 'bus_name.endswith("CO2 HP storage")' not in src, "suffix rule is hardcoded again"
    assert "bus_suffix" in src, "resolver no longer reads the declared suffix"


def test_stream_for_bus_resolves_exact_then_suffix():
    from scripts.prepare_network import stream_for_bus
    assert stream_for_bus("Heat MT") == "Heat MT min"          # exact, from buses:
    assert stream_for_bus("meoh H2 HP storage") == "H2 HP storage"   # via bus_suffix
    assert stream_for_bus("not a bus") is None


def test_dT_min_present_and_sane():
    """Minimum approach temperature, for the pinch work. Declared, not yet consumed."""
    assert c.p_dT_min == 10.0
    assert 0 < c.p_dT_min < 60, "dT_min outside any sensible range for gas/liquid HX"


def test_declared_but_unconsumed_fields_are_documented():
    """Every parameter no code reads must be listed in the guide's 'Declared but not consumed'.

    'Heat MT max' sat unread in p_config and was twice misread as a binding constraint
    while reasoning about what could feed the MT bus. Unread fields are not free -- they
    have to announce themselves.
    """
    from pathlib import Path
    doc = (Path(c.__file__).parent.parent / "docs" / "guide_process_streams.rst").read_text()
    for field in ["Heat MT max", "dT_min", "process_streams"]:
        assert field in doc, f"{field!r} is unconsumed but not documented as such"
    assert "Declared but not consumed" in doc


# --- heat circuits: pressure in, temperatures out ------------------------------

def test_circuits_stay_liquid():
    """Every water circuit must be below saturation at its top temperature.

    Before circuits existed, MT declared 180 C at 10 bar (P_sat = 10.03) and 140 C at
    3 bar (P_sat = 3.62) -- both at or below saturation, i.e. flashing.
    """
    import CoolProp.CoolProp as CP
    for name in c.p_streams.index:
        if not name.startswith("Heat") or c.p_streams.at[name, "fluid"] != "Water":
            continue
        T, P = float(c.p_streams.at[name, "T"]), float(c.p_streams.at[name, "P"])
        tsat = CP.PropsSI("T", "P", P * 1e5, "Q", 0, "Water") - 273.15
        assert T < tsat, f"{name}: {T} C at {P} bar is above saturation ({tsat:.1f} C)"


def test_circuit_floors_respect_dT_min():
    floors = [float(c.p_streams.at[f"{n} min", "T"]) for n in ["Heat MT", "Heat DH", "Heat LT"]]
    for hi, lo in zip(floors, floors[1:]):
        assert hi - lo >= c.p_dT_min, f"circuits {hi} and {lo} closer than dT_min"


def test_circuit_temperatures_are_unchanged_by_the_refactor():
    """The values the compressor split reads must not have moved."""
    assert float(c.p_streams.at["Heat MT max", "T"]) == 180
    assert float(c.p_streams.at["Heat MT min", "T"]) == 140
    assert float(c.p_streams.at["Heat DH min", "T"]) == 90   # read as T_split_C
    assert float(c.p_streams.at["Heat LT min", "T"]) == 50   # read as T_cool_C


def test_flashing_circuit_is_rejected():
    """A circuit whose top sits at saturation must fail the build, not pass quietly."""
    g = {"dT_min": 10, "liquid_margin_K": 5}
    raw = {"circuits": {"Bad": {"fluid": "Water", "P": 10, "T_min": 140, "T_max": 180}}}
    with pytest.raises(ValueError, match="would flash"):
        c._p_expand_circuits(raw, g)


def test_saturation_keyword_derives_the_top():
    """`T_max: saturation` is how a steam circuit will declare itself."""
    g = {"dT_min": 10, "liquid_margin_K": 5}
    raw = {"circuits": {"Steam": {"fluid": "Water", "P": 10, "T_min": 150, "T_max": "saturation"}}}
    out = c._p_expand_circuits(raw, g)
    assert abs(out["Steam max"]["T"] - 179.9) < 0.2
