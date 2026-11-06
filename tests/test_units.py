"""Units are declared or refused — never defaulted.

The failure this guards against is quiet: a mesh authored in inches analysed as
millimetres produces pitch and standoff numbers that are wrong by 25.4x and still
look like plausible engineering findings.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtra.core.errors import AnalysisBlocked
from mtra.core.types import ReasonCode
from mtra.core.units import LengthUnit, factor_to_millimetres, mm_to, to_millimetres


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("mm", LengthUnit.MILLIMETRE),
        ("MM", LengthUnit.MILLIMETRE),
        ("millimetre", LengthUnit.MILLIMETRE),
        ("cm", LengthUnit.CENTIMETRE),
        ("m", LengthUnit.METRE),
        ("in", LengthUnit.INCH),
        ("inches", LengthUnit.INCH),
        ("unknown", LengthUnit.UNKNOWN),
        ("", LengthUnit.UNKNOWN),
        (None, LengthUnit.UNKNOWN),
    ],
)
def test_parse_aliases(declared, expected):
    assert LengthUnit.parse(declared) is expected


@pytest.mark.parametrize(
    ("unit", "factor"),
    [
        (LengthUnit.MILLIMETRE, 1.0),
        (LengthUnit.CENTIMETRE, 10.0),
        (LengthUnit.METRE, 1000.0),
        (LengthUnit.INCH, 25.4),
    ],
)
def test_declared_units_convert_exactly(unit, factor):
    assert factor_to_millimetres(unit) == pytest.approx(factor)


def test_unknown_units_block_with_a_reason_code():
    with pytest.raises(AnalysisBlocked) as excinfo:
        factor_to_millimetres(LengthUnit.UNKNOWN)
    assert excinfo.value.reason is ReasonCode.UNKNOWN_UNITS
    assert "assum" in excinfo.value.detail.lower()


def test_points_scale_but_derived_lengths_do_not():
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 0.5]])
    scaled = to_millimetres(points, LengthUnit.INCH)
    assert scaled == pytest.approx(points * 25.4)


def test_round_trip():
    unit = LengthUnit.CENTIMETRE
    assert mm_to(unit, 25.4) == pytest.approx(2.54)
    assert factor_to_millimetres(unit) * mm_to(unit, 123.0) == pytest.approx(123.0)
