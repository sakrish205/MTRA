"""Units: declared, never assumed.

PRD §5.1 and §19 are blunt about this — "never silently assume STL/OBJ
coordinates are millimetres", because a scale error is indistinguishable from a
genuine spacing rule once it reaches the report. So every length entering the
pipeline passes through an explicit unit, and ``unknown`` is a value that blocks
analysis rather than a prompt that gets defaulted.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from mtra.core.errors import AnalysisBlocked
from mtra.core.types import ReasonCode

CANONICAL_UNIT = "mm"


class LengthUnit(StrEnum):
    MILLIMETRE = "mm"
    CENTIMETRE = "cm"
    METRE = "m"
    INCH = "inch"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, value: str | None) -> LengthUnit:
        if value is None:
            return cls.UNKNOWN
        normalised = value.strip().lower()
        aliases = {
            "mm": cls.MILLIMETRE,
            "millimetre": cls.MILLIMETRE,
            "millimeter": cls.MILLIMETRE,
            "cm": cls.CENTIMETRE,
            "centimetre": cls.CENTIMETRE,
            "centimeter": cls.CENTIMETRE,
            "m": cls.METRE,
            "metre": cls.METRE,
            "meter": cls.METRE,
            "in": cls.INCH,
            "inch": cls.INCH,
            "inches": cls.INCH,
            "unknown": cls.UNKNOWN,
            "": cls.UNKNOWN,
        }
        return aliases.get(normalised, cls.UNKNOWN)


_FACTORS_TO_MM: dict[LengthUnit, float] = {
    LengthUnit.MILLIMETRE: 1.0,
    LengthUnit.CENTIMETRE: 10.0,
    LengthUnit.METRE: 1000.0,
    LengthUnit.INCH: 25.4,
}


def factor_to_millimetres(unit: LengthUnit) -> float:
    if unit is LengthUnit.UNKNOWN:
        raise AnalysisBlocked(
            ReasonCode.UNKNOWN_UNITS,
            "source units are undeclared; declare them in the project manifest "
            "rather than assuming millimetres",
        )
    return _FACTORS_TO_MM[unit]


def to_millimetres(points: np.ndarray, unit: LengthUnit) -> np.ndarray:
    """Scale coordinates into the canonical unit.

    Applied to geometry only — never to angles, curvatures or already-derived
    lengths, which is why this takes points rather than a bare scalar.
    """
    return np.asarray(points, dtype=float) * factor_to_millimetres(unit)


def mm_to(unit: LengthUnit, value_mm: float) -> float:
    return float(value_mm) / factor_to_millimetres(unit)
