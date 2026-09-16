"""Temperature-based assignment of process heat streams to heat circuits."""
import pytest

from scripts import config as c
from scripts.heat_bands import circuit_bands, assign_source, assign_sink

BANDS = circuit_bands(c._p_raw["circuits"])
DT = c.p_dT_min


def test_bands_are_contiguous_and_ordered():
    """Each circuit's top is the floor of the one above -- no gaps, no overlaps."""
    assert [n for n, _, _ in BANDS] == ["Heat MT", "Heat DH", "Heat LT"]
    for (n_hi, f_hi, _), (n_lo, _, t_lo) in zip(BANDS, BANDS[1:]):
        assert t_lo == f_hi, f"gap between {n_lo} and {n_hi}"
    assert BANDS[0][2] == 180.0        # declared T_max of the hottest circuit


def test_source_duty_is_conserved_only_together_with_ambient():
    """Heat below the lowest floor has no circuit and is rejected to ambient.

    A 160 -> 50 C stream shifts to 155 -> 45; the circuits cover down to 50, so the last
    5 shifted K (~4.5% of the duty) cannot be placed. The assigned values therefore do
    NOT sum to the duty, which is why split_with_ambient exists.
    """
    from scripts.heat_bands import split_with_ambient
    got = assign_source(BANDS, 160, 50, duty=1.0, dT_min=DT)
    assert sum(got.values()) < 1.0
    assigned, ambient = split_with_ambient(BANDS, 160, 50, duty=1.0, dT_min=DT)
    assert sum(assigned.values()) + ambient == pytest.approx(1.0)
    assert ambient == pytest.approx(0.0455, abs=1e-3)


def test_source_spans_three_circuits_not_two():
    """The point of the module: a compressor stream reaches MT.

    compressor_calculation splits at a single point and has only DH and LT ports, so
    the 160-140 C slice is currently handed to the 90 C DH bus.
    """
    got = assign_source(BANDS, 160, 50, duty=1.0, dT_min=DT)
    assert set(got) == {"Heat MT", "Heat DH", "Heat LT"}
    assert got["Heat MT"] > 0.1


def test_isothermal_stream_lands_in_one_band():
    assert assign_source(BANDS, 120, 120, duty=1.0, dT_min=DT) == {"Heat DH": 1.0}


def test_stream_hotter_than_every_circuit_is_clipped_not_lost():
    """A 247.5 C methanol reactor has no circuit that hot; it charges the hottest.

    The excess grade is lost. Converting it to power (de Oliveira's Rankine cycle) is
    explicitly out of scope.
    """
    assert assign_source(BANDS, 247.5, 247.5, duty=1.0, dT_min=DT) == {"Heat MT": 1.0}


def test_stream_below_every_floor_goes_nowhere():
    """A 53 C condenser cannot charge a 50 C circuit across a 10 K approach."""
    assert assign_source(BANDS, 53, 53, duty=1.0, dT_min=DT) == {}


def test_sink_is_served_on_circuit_top_not_floor():
    """DH spans 90-140, so DH water can heat a 99.6 C reboiler.

    Testing the floor instead would send it to MT and buy needlessly high-grade heat.
    """
    assert assign_sink(BANDS, 99.6, dT_min=DT) == "Heat DH"


def test_sink_takes_the_coldest_adequate_circuit():
    assert assign_sink(BANDS, 130, dT_min=DT) == "Heat DH"    # 140 >= 130+10
    assert assign_sink(BANDS, 135, dT_min=DT) == "Heat MT"    # 140 < 145, so MT
    assert assign_sink(BANDS, 40, dT_min=DT) == "Heat LT"


def test_sink_hotter_than_every_circuit_returns_none():
    assert assign_sink(BANDS, 250, dT_min=DT) is None


def test_dT_min_is_respected_at_the_boundary():
    """Exactly pinched is served; a hair hotter is not."""
    assert assign_sink(BANDS, 130.0, dT_min=DT) == "Heat DH"
    assert assign_sink(BANDS, 130.1, dT_min=DT) == "Heat MT"


def test_variable_cp_is_honoured_when_an_enthalpy_function_is_given():
    """Splitting by temperature fraction is wrong for a real fluid.

    A q_above() that releases most of its duty at the hot end must put most of the duty
    in the hot circuit, unlike the constant-cp split.
    """
    flat = assign_source(BANDS, 180, 50, duty=1.0, dT_min=DT)
    # all the duty released above 150 C
    q_above = lambda T: 1.0 if T < 150 else 0.0
    skewed = assign_source(BANDS, 180, 50, duty=1.0, q_above=q_above, dT_min=DT)
    assert skewed["Heat MT"] > flat["Heat MT"]
    assert sum(skewed.values()) == pytest.approx(1.0)
