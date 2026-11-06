"""Group I — tool orientation and standoff vs mesh (PRD §7, phase 1 items I1–I7).

The distinction this group must not blur:

* **I5 standoff** is the distance from the TCP to the point where the *tool axis*
  meets the surface. It is a ray distance, not a nearest-point distance. On a
  curved part the two differ by the surface's obliquity, and using the nearest
  point would report a standoff that varies with curvature for a perfectly
  controlled process — inventing a rule where none exists.
* **A3** is that nearest-point distance, reported separately.

Lead and lag need a travel direction, so they are computed only on strokes that
have an unambiguous tangent. They are evaluated in the surface's own frame (axis
against normal and tangent), which makes them independent of where the part sits.

I8/I9 relate orientation to curvature and travel; those are hypothesis tests and
are reported out of scope.
"""

from __future__ import annotations

import numpy as np

from mtra.core.geometry import rotation_angle_deg
from mtra.core.types import EvidenceStatus, ReasonCode
from mtra.measurements.catalogue import MeasurementSet, numeric_stats
from mtra.measurements.common import (
    PairView,
    angular_deviation_deg,
    axis_normal_angle_deg,
    mean_vector,
    tangents,
    unit_rows,
)


def measure_orientation(view: PairView, measurements: MeasurementSet) -> None:
    strokes = view.painting_strokes
    if not strokes:
        for parameter_id in ("I1", "I2", "I3", "I4", "I5", "I6", "I7"):
            measurements.unavailable(
                parameter_id,
                ReasonCode.NO_PAINTING_SEGMENTS,
                "no painting strokes were classified, so there are no tool poses to analyse",
            )
        return

    world_axes = np.vstack([view.stroke_axes_world(stroke) for stroke in strokes])
    nonzero = world_axes[np.linalg.norm(world_axes, axis=1) > 1e-9]
    if nonzero.size == 0:
        for parameter_id in ("I1", "I2", "I3", "I4", "I7"):
            measurements.unavailable(
                parameter_id,
                ReasonCode.UNKNOWN_TOOL_AXIS,
                "every sampled tool axis is a zero vector; the pose rotation or the "
                "declared tool axis is unusable",
            )
    else:
        mean_axis = mean_vector(nonzero)
        deviation = angular_deviation_deg(nonzero, mean_axis)
        measurements.add(
            "I1",
            {
                "mean_axis": [float(v) for v in mean_axis],
                "samples": int(len(nonzero)),
                "dispersion_deg": float(np.std(deviation)),
                "max_deviation_deg": float(np.max(deviation)),
            },
            method=(
                "mean of the per-sample tool +Z axes in the robot world frame; the "
                "declared manifest tool axis is used when present, otherwise the axis is "
                "taken from the controller W/P/R rotation (the weaker claim)"
            ),
            status=(
                EvidenceStatus.OBSERVED
                if view.tool_axis_world is not None
                else EvidenceStatus.INFERRED
            ),
            uncertainty=float(np.std(deviation)),
            source_refs=["robot.parse", "traj.sampling"],
        )

    normals_all: list[np.ndarray] = []
    axes_mesh_all: list[np.ndarray] = []
    leads: list[float] = []

    for stroke in strokes:
        points = view.stroke_points_mesh(stroke)
        if len(points) < 3:
            continue
        axes = view.stroke_axes_mesh(stroke)
        sample = view.surface.sample(points)
        stroke_tangents = tangents(points)

        normals_all.append(sample.normals)
        axes_mesh_all.append(axes)

        for index in range(1, len(points) - 1):
            normal = unit_rows(sample.normals[index : index + 1])[0]
            axis = unit_rows(axes[index : index + 1])[0]
            if not np.any(normal) or not np.any(axis):
                continue
            tangent = stroke_tangents[index]
            in_plane = tangent - np.dot(tangent, normal) * normal
            norm = float(np.linalg.norm(in_plane))
            if norm < 1e-9:
                continue
            in_plane /= norm
            # Use |axis·normal| so a tool pointing into the surface — the normal
            # painting attitude — measures as zero lead rather than 180°.
            lead = float(
                np.degrees(
                    np.arctan2(np.dot(axis, in_plane), abs(np.dot(axis, normal)))
                )
            )
            leads.append(lead)

    if normals_all:
        normals = np.vstack(normals_all)
        axes_mesh = np.vstack(axes_mesh_all)
        angles = axis_normal_angle_deg(normals, axes_mesh)
        if angles.size:
            # Fold to [0, 90]: the *line* of the surface normal defines
            # perpendicularity, so a tool pointing into the surface reads as 0°,
            # not 180°. Which side the tool is on is reported with I1.
            folded = np.where(angles > 90.0, 180.0 - angles, angles)
            stats = numeric_stats(folded, unit_label="degrees")
            measurements.add(
                "I2",
                stats,
                method=(
                    "angle between the tool +Z axis and the surface normal line at the same "
                    "sample, folded to [0°, 90°] so a normal painting attitude reads 0°; the "
                    "signed angle to the outward normal is available from I1 and E1"
                ),
                uncertainty=stats["std"],
                source_refs=["mesh.features", "traj.sampling"],
            )
        else:
            measurements.unavailable(
                "I2",
                ReasonCode.NO_SURFACE_CORRESPONDENCE,
                "no sample produced both a tool axis and a surface normal",
            )
    else:
        measurements.unavailable(
            "I2",
            ReasonCode.INSUFFICIENT_STROKES,
            "no stroke had the three samples needed for a surface normal and a tangent",
        )

    lead_array = np.array(leads, dtype=float)
    if lead_array.size == 0:
        for parameter_id in ("I3", "I4"):
            measurements.unavailable(
                parameter_id,
                ReasonCode.INSUFFICIENT_STROKES,
                "no sample produced both a travel tangent and a surface normal, so lead "
                "and lag have no reference frame",
            )
    else:
        measurements.add(
            "I3",
            numeric_stats(lead_array, unit_label="degrees"),
            method=(
                "signed atan2(axis·travel_tangent, axis·surface_normal) per sample; "
                "positive means the tool tilts ahead of the direction of travel"
            ),
            uncertainty=float(np.std(lead_array)),
            source_refs=["traj.sampling", "mesh.features"],
        )
        lag = lead_array[lead_array < 0.0]
        measurements.add(
            "I4",
            {
                "lag_samples": int(lag.size),
                "lag_fraction": float(lag.size / lead_array.size),
                "mean_abs_lag_deg": float(np.mean(np.abs(lag))) if lag.size else 0.0,
                "max_abs_lag_deg": float(np.max(np.abs(lag))) if lag.size else 0.0,
            },
            method=(
                "samples with a negative lead angle; the mean magnitude is reported only "
                "over those samples, and their fraction shows whether lag is systematic or "
                "incidental"
            ),
            source_refs=["traj.sampling", "mesh.features"],
        )

    standoffs = np.array(
        [
            record.impact_distance_mm
            for record in view.correspondence.records
            if record.impact_distance_mm is not None
        ],
        dtype=float,
    )
    if standoffs.size == 0:
        detail = (
            "no tool axis intersects the mesh, so no ray distance exists"
            if all(record.ray_hit_count == 0 for record in view.correspondence.records)
            else "no sample carried a usable tool axis"
        )
        measurements.unavailable("I5", ReasonCode.NO_RAY_INTERSECTION, detail)
        measurements.unavailable(
            "I6",
            ReasonCode.NO_RAY_INTERSECTION,
            "standoff variation needs standoff samples",
        )
    else:
        stats = numeric_stats(standoffs, unit_label="mm")
        measurements.add(
            "I5",
            stats,
            method=(
                "distance along the tool +Z axis from the TCP to its first forward "
                "intersection with the mesh — a ray distance, not a nearest-point distance"
            ),
            uncertainty=stats["std"],
            source_refs=["correspondence.spray_axis"],
        )
        measurements.add(
            "I6",
            {
                "std_mm": float(np.std(standoffs)),
                "range_mm": float(np.max(standoffs) - np.min(standoffs)),
                "p95_minus_p05_mm": float(
                    np.percentile(standoffs, 95) - np.percentile(standoffs, 5)
                ),
                "missing_samples": int(
                    sum(
                        1
                        for record in view.correspondence.records
                        if record.impact_distance_mm is None
                    )
                ),
            },
            method=(
                "spread of the tool-axis standoff; samples without an intersection are "
                "counted separately rather than imputed"
            ),
            uncertainty=float(np.std(standoffs)),
            source_refs=["correspondence.spray_axis"],
        )

    steps: list[float] = []
    for stroke in strokes:
        axes = unit_rows(view.stroke_axes_world(stroke))
        if len(axes) < 2:
            continue
        for first, second in zip(axes[:-1], axes[1:], strict=True):
            if not np.any(first) or not np.any(second):
                continue
            steps.append(rotation_angle_deg(first, second))

    if steps:
        step_array = np.array(steps, dtype=float)
        mean_step = float(np.mean(step_array))
        measurements.add(
            "I7",
            {
                "score": round(float(np.clip(1.0 - mean_step / 90.0, 0.0, 1.0)), 4),
                "mean_step_deg": mean_step,
                "max_step_deg": float(np.max(step_array)),
                "samples": int(step_array.size),
            },
            method=(
                "1 - mean angular step between consecutive samples / 90°; 1 is a constant "
                "tool axis, 0 means the axis swings 90° per sample"
            ),
            uncertainty=float(np.std(step_array)),
            source_refs=["traj.sampling"],
        )
    else:
        measurements.unavailable(
            "I7",
            ReasonCode.INSUFFICIENT_STROKES,
            "no stroke carried two usable consecutive tool axes",
        )
