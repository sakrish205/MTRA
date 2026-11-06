"""Measurement layer: one module per catalogue group, one honest accounting pass.

``measure_pair`` runs every phase-1 group against an analysed pair and then lets
the catalogue sweep fill the rest. Group modules never write to each other; they
receive the same ``PairView`` and a shared ``MeasurementSet``, which is what makes
a group testable in isolation.
"""

from __future__ import annotations

from mtra.config import Catalogue, load_catalogue
from mtra.core.types import EvidenceStatus, MeasurementRecord, ReasonCode
from mtra.measurements import (
    alignment,
    continuity,
    direction,
    orientation,
    regions,
    robot_motion,
    spacing,
    stroke_shape,
    surface,
)
from mtra.measurements.catalogue import MeasurementSet, numeric_stats
from mtra.measurements.common import PairView

__all__ = ["MeasurementSet", "PairView", "measure_pair", "measure_groups", "numeric_stats"]

_GROUPS = (
    alignment.measure_alignment,
    direction.measure_direction,
    spacing.measure_spacing,
    stroke_shape.measure_stroke_shape,
    surface.measure_surface,
    regions.measure_regions,
    orientation.measure_orientation,
    continuity.measure_continuity,
    robot_motion.measure_robot_motion,
)


def measure_groups(view: PairView, measurements: MeasurementSet) -> None:
    """Run every phase-1 group. One group failing must not erase the others."""
    for group in _GROUPS:
        try:
            group(view, measurements)
        except Exception as exc:  # each group is independent by design
            measurements.unavailable(
                _group_label(group),
                ReasonCode.COMPUTATION_FAILED,
                f"{type(exc).__name__}: {exc}",
                status=EvidenceStatus.UNRESOLVED,
            )


def _group_label(group) -> str:
    """Best-effort group id when a module raises, so the gap is attributed."""
    name = group.__module__.rsplit(".", 1)[-1]
    return {
        "alignment": "A6",
        "direction": "B3",
        "spacing": "C4",
        "stroke_shape": "D1",
        "surface": "E9",
        "regions": "H1",
        "orientation": "I7",
        "continuity": "J6",
        "robot_motion": "K8",
    }.get(name, "A6")


def measure_pair(view: PairView, catalogue: Catalogue | None = None) -> list[MeasurementRecord]:
    catalogue = catalogue or load_catalogue()
    set_ = MeasurementSet(catalogue, view.pair_id)
    measure_groups(view, set_)
    return set_.finalise()
