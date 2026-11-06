"""Trajectory-to-surface correspondence.

For every resampled trajectory sample this produces:

* the closest point on the mesh, its triangle and barycentric coordinates;
* the distance from TCP to that surface point (parameter A3);
* the tool-axis intersection with the surface (A4), when a tool axis is known;
* the disagreement between those two surface locations (A5) — which is the honest
  way to show that "where the surface is" depends on how you project onto it.

All stored locations are expressed in the **mesh frame**, and the transform used
is recorded on the result, because a report that mixes frames silently is a
report whose numbers cannot be checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mtra.config import Settings
from mtra.core.types import CorrespondenceRecord, Diagnostic, Severity
from mtra.correspondence import frames as frame_ops
from mtra.correspondence.spray_axis import intersect_axes
from mtra.mesh.features import SurfaceField
from mtra.mesh.loader import as_trimesh
from mtra.traj.sampling import Stroke


@dataclass
class PairCorrespondence:
    pair_id: str
    world_from_mesh: np.ndarray
    mesh_from_world: np.ndarray
    transform_source: str
    records: list[CorrespondenceRecord] = field(default_factory=list)
    stroke_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def range_for(self, stroke_id: str) -> tuple[int, int]:
        return self.stroke_ranges[stroke_id]

    def arrays(self, stroke_id: str) -> dict[str, np.ndarray]:
        start, end = self.stroke_ranges[stroke_id]
        subset = self.records[start:end]
        return {
            "tcp": np.array([r.tcp_position for r in subset], dtype=float),
            "surface": np.array([r.closest_point for r in subset], dtype=float),
            "distance": np.array([r.distance_mm for r in subset], dtype=float),
            "impact": np.array(
                [r.impact_point if r.impact_point is not None else (np.nan, np.nan, np.nan)
                 for r in subset],
                dtype=float,
            ),
            "impact_distance": np.array(
                [r.impact_distance_mm if r.impact_distance_mm is not None else np.nan
                 for r in subset],
                dtype=float,
            ),
        }

    @property
    def sample_count(self) -> int:
        return len(self.records)

    def distance_statistics(self) -> dict[str, float]:
        distances = np.array([r.distance_mm for r in self.records], dtype=float)
        if distances.size == 0:
            return {}
        return {
            "count": float(distances.size),
            "mean_mm": float(np.mean(distances)),
            "median_mm": float(np.median(distances)),
            "std_mm": float(np.std(distances)),
            "p95_mm": float(np.percentile(distances, 95)),
            "max_mm": float(np.max(distances)),
        }


def build_correspondence(
    pair_id: str,
    strokes: list[Stroke],
    surface: SurfaceField,
    world_from_mesh: np.ndarray,
    *,
    transform_source: str,
    settings: Settings,
    tool_axis_world: np.ndarray | None = None,
) -> PairCorrespondence:
    """Project every stroke sample onto the mesh.

    ``tool_axis_world``, when supplied, overrides the per-sample pose axis. The
    manifest's declared tool axis is a fixed, calibrated direction; the pose-derived
    axis comes from controller Euler angles and is the weaker of the two claims.
    """
    mesh_from_world = frame_ops.invert(world_from_mesh)
    mesh = as_trimesh(surface.mesh) if isinstance(surface.mesh, object) else None
    if mesh is None:  # pragma: no cover - defensive
        raise ValueError("surface field does not expose a mesh")

    result = PairCorrespondence(
        pair_id=pair_id,
        world_from_mesh=world_from_mesh,
        mesh_from_world=mesh_from_world,
        transform_source=transform_source,
    )

    missing_axis = 0
    no_hit = 0
    multi_hit = 0

    for stroke in strokes:
        start_index = len(result.records)
        tcp_world = stroke.points
        tcp_mesh = frame_ops.apply(mesh_from_world, tcp_world)

        if tool_axis_world is not None:
            axes_mesh = np.tile(
                frame_ops.apply_direction(mesh_from_world, tool_axis_world)[0], (len(tcp_mesh), 1)
            )
        else:
            axes_mesh = frame_ops.apply_direction(mesh_from_world, stroke.tool_axes)
            missing_axis += int(np.count_nonzero(~np.any(axes_mesh, axis=1)))

        sample = surface.sample(tcp_mesh)
        hits = intersect_axes(mesh, tcp_mesh, axes_mesh)

        for index in range(len(tcp_mesh)):
            hit = hits[index]
            if hit.ok:
                impact = np.asarray(hit.point, dtype=float)
                impact_distance = float(hit.distance) if hit.distance is not None else None
            else:
                impact = None
                impact_distance = None
                if hit.status == "no_intersection":
                    no_hit += 1
                elif hit.hit_count > 1:
                    multi_hit += 1

            result.records.append(
                CorrespondenceRecord(
                    sample_id=f"{stroke.stroke_id}:P{index:04d}",
                    pair_id=pair_id,
                    segment_id=stroke.segment_id,
                    tcp_position=tuple(float(v) for v in tcp_mesh[index]),
                    closest_point=tuple(float(v) for v in sample.closest_points[index]),
                    mesh_face_id=int(sample.face_ids[index]),
                    barycentric=tuple(float(v) for v in sample.barycentric[index]),
                    distance_mm=float(sample.distances[index]),
                    impact_point=None if impact is None else tuple(float(v) for v in impact),
                    impact_distance_mm=impact_distance,
                    ray_hit_count=hit.hit_count,
                    method="closest_point_on_mesh + tool_axis_ray",
                    confidence=1.0 if hit.ok else 0.75,
                )
            )
        result.stroke_ranges[stroke.stroke_id] = (start_index, len(result.records))

    total = len(result.records)
    if total == 0:
        result.diagnostics.append(
            Diagnostic(
                severity=Severity.ERROR,
                code="correspondence.empty",
                message="no trajectory samples available for correspondence",
                stage="correspondence.projection",
            )
        )
        return result

    if missing_axis:
        result.diagnostics.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="correspondence.tool_axis_missing",
                message=(
                    f"{missing_axis}/{total} samples carry a zero tool axis; their spray-axis "
                    "intersection is unavailable"
                ),
                stage="correspondence.projection",
            )
        )
    if no_hit:
        result.diagnostics.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="correspondence.axis_missed",
                message=(
                    f"{no_hit}/{total} tool axes do not intersect the mesh; standoff for those "
                    "samples is reported as unavailable rather than as a nearest-point distance"
                ),
                stage="correspondence.projection",
            )
        )
    if multi_hit:
        result.diagnostics.append(
            Diagnostic(
                severity=Severity.INFO,
                code="correspondence.multiple_hits",
                message=(
                    f"{multi_hit}/{total} axes intersect the mesh more than once (closed solid); "
                    "the nearest forward hit is used and the count is recorded per sample"
                ),
                stage="correspondence.projection",
            )
        )

    max_distance = float(settings.get("measurement.pitch_correspondence_max_mm", 40.0))
    distances = np.array([r.distance_mm for r in result.records], dtype=float)
    far = int(np.count_nonzero(distances > max_distance))
    if far:
        result.diagnostics.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="correspondence.far_samples",
                message=(
                    f"{far}/{total} samples are further than {max_distance} mm from the surface; "
                    "correspondence may be systematically offset (standoff, frame or tool axis)"
                ),
                stage="correspondence.projection",
            )
        )

    return result
