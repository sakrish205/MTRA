"""Group D — stroke shape vs mesh geometry (PRD §7, phase 1 items D1–D7).

A "stroke" is a maximal paint-on run, resampled at a fixed arc-length step. Every
shape number below is computed on that resampled polyline, and the step is
reported with it — because smoothness and curvature on a polyline are only
comparable at equal sampling.

D8/D9 relate stroke shape to surface curvature and mesh resolution; those are
hypothesis tests and are reported out of scope.
"""

from __future__ import annotations

import numpy as np

from mtra.core.geometry import (
    discrete_curvature,
    pca_direction,
    perpendicular_distance_rms,
    smoothness_rms,
)
from mtra.core.types import ReasonCode, SegmentClass
from mtra.measurements.catalogue import MeasurementSet, numeric_stats
from mtra.measurements.common import PairView


def stroke_table(view: PairView) -> list[dict[str, object]]:
    """Per-stroke detail, emitted as a CSV artefact alongside the summary record.

    The report carries one row per parameter; this table is what makes the
    distribution in that row checkable stroke by stroke.
    """
    rows: list[dict[str, object]] = []
    for stroke in view.painting_strokes:
        points = view.stroke_points_mesh(stroke)
        if len(points) < 2:
            continue
        curvature = discrete_curvature(points)
        direction = pca_direction(points)
        straightness = perpendicular_distance_rms(points, points.mean(axis=0), direction)
        rows.append(
            {
                "pair_id": view.pair_id,
                "stroke_id": stroke.stroke_id,
                "segment_id": stroke.segment_id,
                "instruction_count": len(stroke.instruction_ids),
                "first_instruction": stroke.instruction_ids[0] if stroke.instruction_ids else None,
                "last_instruction": stroke.instruction_ids[-1] if stroke.instruction_ids else None,
                "length_mm": round(stroke.length_mm, 4),
                "sampled_points": int(len(points)),
                "straightness_rms_mm": round(float(straightness), 6),
                "curvature_mean_per_mm": round(float(np.mean(curvature)), 8),
                "curvature_std_per_mm": round(float(np.std(curvature)), 8),
                "smoothness_rms_mm": round(float(smoothness_rms(points)), 6),
                "start": [round(float(v), 4) for v in points[0]],
                "end": [round(float(v), 4) for v in points[-1]],
            }
        )
    return rows


def measure_stroke_shape(view: PairView, measurements: MeasurementSet) -> None:
    strokes = view.painting_strokes
    if not strokes:
        for parameter_id in ("D1", "D2", "D3", "D4", "D5", "D7"):
            measurements.unavailable(
                parameter_id,
                ReasonCode.NO_PAINTING_SEGMENTS,
                "no painting strokes were classified",
            )
        measurements.unavailable(
            "D6",
            ReasonCode.NO_PAINTING_SEGMENTS,
            "fragmentation is the count of paint-off interruptions between passes; "
            "there are no passes to interrupt",
        )
        return

    step = float(view.settings.get("trajectory.resample_step_mm", 2.0))
    lengths: list[float] = []
    straightness: list[float] = []
    curvature_means: list[float] = []
    curvature_spreads: list[float] = []
    smoothness: list[float] = []

    for stroke in strokes:
        points = view.stroke_points_mesh(stroke)
        if len(points) < 2:
            continue
        lengths.append(stroke.length_mm)
        direction = pca_direction(points)
        straightness.append(
            perpendicular_distance_rms(points, points.mean(axis=0), direction)
        )
        if len(points) >= 3:
            curvature = discrete_curvature(points)
            curvature_means.append(float(np.mean(curvature)))
            curvature_spreads.append(float(np.std(curvature)))
            smoothness.append(float(smoothness_rms(points)))

    if not lengths:
        for parameter_id in ("D1", "D2", "D3", "D4", "D5", "D7"):
            measurements.unavailable(
                parameter_id,
                ReasonCode.DEGENERATE_GEOMETRY,
                "every painting stroke had fewer than two usable samples",
            )
    else:
        length_stats = numeric_stats(np.array(lengths), unit_label="mm")
        measurements.add(
            "D1",
            length_stats,
            method=(
                "polyline length of each paint-on stroke after arc-length resampling "
                f"at {step} mm; distribution across {len(lengths)} stroke(s)"
            ),
            uncertainty=length_stats["std"],
            source_refs=["traj.sampling"],
        )

        straightness_stats = numeric_stats(np.array(straightness), unit_label="mm")
        measurements.add(
            "D2",
            straightness_stats,
            method="RMS perpendicular distance of stroke samples from the stroke's best-fit line",
            uncertainty=straightness_stats["std"],
            source_refs=["traj.sampling"],
        )

        if curvature_means:
            curvature_stats = numeric_stats(np.array(curvature_means), unit_label="1/mm")
            measurements.add(
                "D3",
                curvature_stats,
                method=(
                    "mean discrete curvature per stroke (circumscribed-circle estimate "
                    f"on the {step} mm resampled polyline)"
                ),
                uncertainty=curvature_stats["std"],
                source_refs=["traj.sampling"],
            )
            spread_stats = numeric_stats(np.array(curvature_spreads), unit_label="1/mm")
            measurements.add(
                "D4",
                spread_stats,
                method="within-stroke standard deviation of discrete curvature",
                uncertainty=spread_stats["std"],
                source_refs=["traj.sampling"],
            )
            smoothness_stats = numeric_stats(np.array(smoothness), unit_label="mm")
            measurements.add(
                "D5",
                smoothness_stats,
                method=(
                    "RMS second difference of the stroke samples; valid only at the "
                    f"declared {step} mm resampling step"
                ),
                uncertainty=smoothness_stats["std"],
                source_refs=["traj.sampling"],
            )
        else:
            for parameter_id, name in (
                ("D3", "curvature"),
                ("D4", "curvature variation"),
                ("D5", "smoothness"),
            ):
                measurements.unavailable(
                    parameter_id,
                    ReasonCode.DEGENERATE_GEOMETRY,
                    f"{name} needs at least three samples per stroke; every stroke was "
                    "shorter than that after resampling",
                )

        fragment_stats = numeric_stats(np.array(lengths), unit_label="mm")
        measurements.add(
            "D7",
            {
                "fragments": int(len(lengths)),
                "min_mm": fragment_stats["min"],
                "mean_mm": fragment_stats["mean"],
                "max_mm": fragment_stats["max"],
            },
            method="length of each paint-on fragment; one fragment per uninterrupted pass",
            source_refs=["traj.sampling"],
        )

    # Fragmentation: paint switching off between two passes inside the same footprint.
    painting_ids = [
        index
        for index, segment in enumerate(view.segments)
        if segment.segment_class is SegmentClass.PAINTING
    ]
    interruptions = 0
    for previous, current in zip(painting_ids[:-1], painting_ids[1:], strict=True):
        between = view.segments[previous + 1 : current]
        if between and all(
            segment.segment_class in {SegmentClass.TRANSFER, SegmentClass.UNKNOWN}
            for segment in between
        ):
            interruptions += 1

    measurements.add(
        "D6",
        {
            "paint_fragments": int(len(painting_ids)),
            "interruptions_in_footprint": interruptions,
        },
        method=(
            "count of paint-on runs and of paint-off interruptions that sit between two "
            "passes in program order, i.e. the trajectory was split without leaving the footprint"
        ),
        source_refs=["traj.segments"],
    )
