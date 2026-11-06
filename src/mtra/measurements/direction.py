"""Group B — path direction vs mesh geometry (PRD §7, phase 1 items B1–B6).

Direction is *axial*: a pass run start-to-end and the same pass run end-to-start
describe one bearing, so a bearing is stored modulo 180° and averaged as axial data.
A naive mean would report 90° for a set of identical strokes evaluated with mixed
sign, which is the classic way a raster pattern gets described as diagonal.

The primary quantity here is a 3D axis, not an angle in the XY plane. A vertical pass
on a cylinder wall has a perfectly definite direction and *no* bearing: its XY
projection is a point, and any angle read off that point is rounding noise. So every
direction record carries the 3D axis and ``bearing_deg`` is nullable — absent when
the stroke does not travel far enough in XY to have one (threshold:
``measurement.min_bearing_extent_mm``). ``elevation_deg`` is reported next to it, so
a missing bearing is explained (90° = the pass is normal to XY) rather than
mysterious.

When a bearing does exist it is expressed in mesh-frame XY. That is exact for a part
lying in the XY plane and a foreshortened projection otherwise, so the fitted surface
plane normal and its tilt are reported alongside B4 — a reader can then see whether
the projection was significant instead of guessing.

B7/B8 compare direction against curvature and region shape; those are hypothesis
tests (phase 2) and are reported as out of scope by the catalogue sweep.
"""

from __future__ import annotations

import numpy as np

from mtra.core.geometry import circular_dispersion_deg, circular_mean_deg
from mtra.core.types import ReasonCode
from mtra.measurements.catalogue import MeasurementSet, numeric_stats
from mtra.measurements.common import (
    PairView,
    angle_between_axial_deg,
    axial_angle_between,
    axial_deviation_deg,
    axial_mean_axis,
    bearing_xy,
    best_fit_plane,
    convex_hull_2d,
    elevation_deg,
    principal_axis_xy,
    stroke_axis,
    tangents,
)

# Above this elevation a mean axis is too close to the XY normal for its projection
# to carry information: the planar component is dominated by lateral wobble at that
# foreshortening. 84° = within 6° of vertical, which is the point where the reported
# bearing would start describing noise rather than the pass.
_MAX_BEARING_ELEVATION_DEG = 84.0


def _project_xy(points: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    return np.column_stack([pts[:, 0], pts[:, 1]])


def measure_direction(view: PairView, measurements: MeasurementSet) -> None:
    strokes = view.painting_strokes
    surface_points = view.surface_points()
    plane = best_fit_plane(surface_points) if len(surface_points) >= 3 else None
    tilt_deg = (
        float(np.degrees(np.arccos(np.clip(abs(plane.normal[2]), 0.0, 1.0))))
        if plane is not None
        else None
    )
    projection_note = (
        "bearings are 3D travel directions projected onto the mesh-frame XY plane"
        + (
            f"; the fitted surface plane is tilted {tilt_deg:.1f}° from XY, so the "
            "projection is foreshortened by that factor"
            if tilt_deg is not None and tilt_deg > 5.0
            else "; the fitted surface plane is within 5° of XY, so the projection is "
            "effectively undistorted"
            if tilt_deg is not None
            else ""
        )
    )

    if not strokes:
        for parameter_id in ("B1", "B2", "B3", "B5", "B6"):
            measurements.unavailable(
                parameter_id,
                ReasonCode.NO_PAINTING_SEGMENTS,
                "no painting strokes were classified, so no path direction exists to compare",
            )
        measurements.unavailable(
            "B4",
            ReasonCode.NO_SURFACE_CORRESPONDENCE,
            "no surface points were corresponded, so no mesh direction exists",
        )
        return

    min_extent = float(view.settings.get("measurement.min_bearing_extent_mm", 1.0))
    stroke_points = [view.stroke_points_mesh(stroke) for stroke in strokes]
    per_stroke_axes = [stroke_axis(points) for points in stroke_points]
    per_stroke_bearings = [
        bearing_xy(points, min_extent_mm=min_extent) for points in stroke_points
    ]
    defined_axes = np.array(
        [axis for axis in per_stroke_axes if axis is not None], dtype=float
    )

    mean_axis, stability = axial_mean_axis(defined_axes)
    dominant_bearing: float | None = None
    mean_elevation: float | None = None
    dispersion: float | None = None
    if mean_axis is not None:
        mean_elevation = elevation_deg(mean_axis)
        dispersion = float(
            np.sqrt(np.mean(axial_deviation_deg(defined_axes, mean_axis) ** 2))
        )
        if mean_elevation <= _MAX_BEARING_ELEVATION_DEG:
            dominant_bearing = float(
                np.mod(np.degrees(np.arctan2(mean_axis[1], mean_axis[0])), 180.0)
            )

    if mean_axis is None or dispersion is None or mean_elevation is None:
        for parameter_id in ("B1", "B2", "B3"):
            measurements.unavailable(
                parameter_id,
                ReasonCode.DEGENERATE_GEOMETRY,
                "no painting stroke travelled far enough to define a direction, so there "
                "is no path direction to describe",
            )
    else:
        measurements.add(
            "B1",
            {
                "mean_axis": [float(v) for v in mean_axis],
                "bearing_deg": dominant_bearing,
                "elevation_deg": mean_elevation,
                "axial_stability": round(float(stability), 4),
                "dispersion_deg": round(dispersion, 4),
                "stroke_count": int(len(defined_axes)),
            },
            method=(
                "axial mean of the per-stroke 3D travel directions — the leading "
                "eigenvector of the summed outer products, so driving a pass backwards "
                "does not change it. bearing_deg is that axis projected into mesh XY, and "
                "is null when the axis lies within 6° of the XY normal because a nearly "
                "vertical pass has no meaningful projection; elevation_deg states the "
                "axis's angle from XY so a null bearing is explained rather than missing. "
                "axial_stability is (λ1-λ2)/λ1: near 0 the strokes do not share a "
                f"direction. {projection_note}"
            ),
            uncertainty=float(dispersion),
            source_refs=["traj.sampling"],
        )

        stroke_records: list[dict[str, object]] = []
        for stroke, axis, bearing in zip(
            strokes, per_stroke_axes, per_stroke_bearings, strict=True
        ):
            stroke_records.append(
                {
                    "stroke_id": stroke.stroke_id,
                    "axis_xyz": None if axis is None else [float(v) for v in axis],
                    "bearing_deg": (
                        None if not np.isfinite(bearing) else round(float(bearing), 3)
                    ),
                    "elevation_deg": (
                        None if axis is None else round(elevation_deg(axis), 3)
                    ),
                }
            )
        with_bearing = [r["bearing_deg"] for r in stroke_records if r["bearing_deg"] is not None]
        measurements.add(
            "B2",
            {
                "stroke_count": len(stroke_records),
                "strokes_with_bearing": len(with_bearing),
                "bearing_stats": (
                    numeric_stats(np.array(with_bearing, dtype=float), unit_label="degrees")
                    if with_bearing
                    else None
                ),
                "strokes": stroke_records,
            },
            method=(
                "per-stroke 3D travel direction (start→end) with its axial XY bearing and "
                f"elevation; a stroke with less than {min_extent:.2f} mm of lateral travel "
                "has bearing null — that is an absence, not a zero, and bearing_stats "
                "summarises only the strokes that do have one"
            ),
            source_refs=["traj.sampling"],
        )

        pair_angles = [
            axial_angle_between(first, second)
            for first, second in zip(per_stroke_axes[:-1], per_stroke_axes[1:], strict=True)
            if first is not None and second is not None
        ]
        if not pair_angles:
            measurements.unavailable(
                "B3",
                ReasonCode.INSUFFICIENT_STROKES,
                "fewer than two painting strokes had a defined direction, so no "
                "neighbouring pair could be compared",
            )
        else:
            mean_pair = float(np.mean(pair_angles))
            consistency = float(np.clip(1.0 - mean_pair / 90.0, 0.0, 1.0))
            measurements.add(
                "B3",
                {
                    "score": round(consistency, 4),
                    "mean_pair_angle_deg": round(mean_pair, 4),
                    "max_pair_angle_deg": round(float(np.max(pair_angles)), 4),
                    "pairs": len(pair_angles),
                },
                method=(
                    "1 - mean undirected angle between neighbouring stroke directions / 90°, "
                    "where 90° is the largest possible axial angle (crossing passes, and the "
                    "same bound as orthogonal raster); 0 = neighbouring passes share no "
                    "direction, 1 = every neighbouring pair is parallel"
                ),
                uncertainty=float(np.std(pair_angles)),
                source_refs=["traj.sampling"],
            )

    if plane is None:
        measurements.unavailable(
            "B4",
            ReasonCode.DEGENERATE_GEOMETRY,
            "corresponded surface points do not span a plane, so no principal surface "
            "direction can be defined",
        )
    else:
        footprint = _project_xy(surface_points)
        principal = principal_axis_xy(footprint)
        if principal is None:
            measurements.unavailable(
                "B4",
                ReasonCode.DEGENERATE_GEOMETRY,
                "the corresponded surface footprint collapses to a point or a line in "
                "mesh XY, so it has no principal direction",
            )
        else:
            heading, anisotropy = principal
            measurements.add(
                "B4",
                {
                    "mean_deg": heading,
                    "anisotropy": round(anisotropy, 4) if np.isfinite(anisotropy) else None,
                    "plane_normal": [float(v) for v in plane.normal],
                    "plane_tilt_deg": tilt_deg,
                    "in_plane_extent_mm": [float(v) for v in np.ptp(footprint, axis=0)],
                },
                method=(
                    "dominant direction of the corresponded surface footprint from the "
                    "first singular vector of its 2D scatter in mesh XY; the anisotropy "
                    "ratio says whether that direction is actually meaningful (near 1 = "
                    "square footprint, direction is not distinguishable). plane_normal and "
                    "plane_tilt_deg are reported because this is an XY projection: when "
                    "the footprint's plane is far from XY, one axis of the footprint has "
                    "collapsed and the bearing describes the projection, not the surface"
                ),
                uncertainty=(
                    None
                    if not np.isfinite(anisotropy) or anisotropy >= 1.2
                    else 45.0
                ),
                source_refs=["mesh.features"],
            )

    tangent_deviations: list[float] = []
    for stroke in strokes:
        points = view.stroke_points_mesh(stroke)
        if len(points) < 3:
            continue
        sample = view.surface.sample(points)
        stroke_tangents = tangents(points)
        for tangent, normal in zip(stroke_tangents[1:-1], sample.normals[1:-1], strict=True):
            if not np.any(tangent) or not np.any(normal):
                continue
            # A surface-following tangent is perpendicular to the normal (90°).
            cosine = float(np.clip(abs(np.dot(tangent, normal)), 0.0, 1.0))
            tangent_deviations.append(abs(90.0 - float(np.degrees(np.arccos(cosine)))))

    if tangent_deviations:
        deviations = np.array(tangent_deviations, dtype=float)
        measurements.add(
            "B5",
            numeric_stats(deviations, unit_label="degrees"),
            method=(
                "angle between the local stroke tangent and the surface tangent plane "
                "at the same sample; 0° means the path runs along the surface, 90° means "
                "it drives into it"
            ),
            uncertainty=float(np.std(deviations)),
            source_refs=["mesh.features", "traj.sampling"],
        )
    else:
        measurements.unavailable(
            "B5",
            ReasonCode.NO_SURFACE_CORRESPONDENCE,
            "no sample produced both a stroke tangent and a surface normal",
        )

    hull = convex_hull_2d(_project_xy(surface_points)) if len(surface_points) >= 3 else np.zeros((0, 2))
    if dominant_bearing is None:
        measurements.unavailable(
            "B6",
            ReasonCode.DEGENERATE_GEOMETRY,
            "the dominant path direction has no in-plane bearing — it lies within 6° of "
            "the XY normal — so there is no planar path angle to compare against a "
            "boundary that is itself measured in XY",
        )
    elif len(hull) < 3:
        measurements.unavailable(
            "B6",
            ReasonCode.DEGENERATE_GEOMETRY,
            "the painted footprint has no polygon boundary to compare against",
        )
    else:
        edges = np.roll(hull, -1, axis=0) - hull
        lengths = np.linalg.norm(edges, axis=1)
        longest = int(np.argmax(lengths))
        boundary_heading = float(
            np.mod(np.degrees(np.arctan2(edges[longest, 1], edges[longest, 0])), 180.0)
        )
        measurements.add(
            "B6",
            {
                "path_deg": dominant_bearing,
                "path_elevation_deg": mean_elevation,
                "boundary_deg": boundary_heading,
                "difference_deg": angle_between_axial_deg(
                    dominant_bearing, boundary_heading
                ),
                "boundary_vertices": int(len(hull)),
            },
            method=(
                "axial angle between the dominant path bearing and the longest edge of "
                "the convex hull of the painted footprint — the boundary is the region "
                "the trajectory actually covers, not a mesh edge loop. Both angles are "
                "XY projections, so path_elevation_deg says how much of the path the "
                "projection discarded"
            ),
            source_refs=["traj.sampling", "mesh.features"],
        )
