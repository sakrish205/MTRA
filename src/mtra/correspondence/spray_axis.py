"""Ray/spray-axis intersection with the mesh.

This module is where "TCP" and "impact point" are structurally separated. A tool
axis projected from the TCP is intersected with the surface, and the nearest
forward hit is the candidate impact point. Nothing here ever treats the TCP as if
it were on the surface (PRD §19: "TCP mistaken for spray target").

Uncertainties are returned, not smoothed away:

* no hit — the axis misses the mesh entirely (wrong tool axis, wrong frame, or a
  tool pointing away from the part);
* multiple hits — a closed solid produces a front and a back face; the nearest is
  chosen, and the count is reported so a reviewer can see the geometry was closed.

Real FANUC-style ``DO``-driven cells often paint with a *cone*, not a ray. A single
axis is the honest v1 abstraction: one well-defined line per pose, whose
relationship to the surface can be measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from mtra.core.geometry import unit


@dataclass(frozen=True)
class RayHit:
    point: np.ndarray | None
    distance: float | None
    hit_count: int
    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.point is not None


def intersect_axis(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    min_distance: float = 1e-6,
) -> RayHit:
    """Nearest forward intersection of a ray with the mesh."""
    origin_array = np.atleast_2d(np.asarray(origin, dtype=float))
    direction_unit = unit(np.asarray(direction, dtype=float))
    if not np.any(direction_unit):
        return RayHit(None, None, 0, "invalid_direction", "tool axis is a zero vector")

    try:
        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=origin_array,
            ray_directions=np.atleast_2d(direction_unit),
            multiple_hits=True,
        )
    except Exception as exc:  # pragma: no cover - native failure path
        return RayHit(None, None, 0, "query_failed", str(exc))

    if len(locations) == 0:
        return RayHit(None, None, 0, "no_intersection", "axis does not meet the surface")

    locations = np.asarray(locations, dtype=float)
    index_ray = np.asarray(index_ray, dtype=int)
    local = locations[index_ray == 0]
    if len(local) == 0:
        return RayHit(None, None, 0, "no_intersection", "axis does not meet the surface")

    offsets = local - origin_array[0]
    distances = offsets @ direction_unit
    forward = distances > min_distance
    if not np.any(forward):
        return RayHit(
            None,
            None,
            int(len(local)),
            "behind_origin",
            "all intersections lie behind the tool origin",
        )

    forward_distances = distances[forward]
    forward_points = local[forward]
    nearest = int(np.argmin(forward_distances))
    return RayHit(
        point=forward_points[nearest],
        distance=float(forward_distances[nearest]),
        hit_count=int(len(local)),
        status="ok",
        detail=f"{len(local)} intersection(s) along the axis",
    )


def intersect_axes(
    mesh: trimesh.Trimesh, origins: np.ndarray, directions: np.ndarray
) -> list[RayHit]:
    """Per-sample convenience wrapper.

    Deliberately a loop: ``intersects_location`` returns intersections grouped by
    ray index, and reconstructing that mapping per sample is more error-prone than
    paying for a bounded loop.
    """
    origins_array = np.atleast_2d(np.asarray(origins, dtype=float))
    directions_array = np.atleast_2d(np.asarray(directions, dtype=float))
    return [
        intersect_axis(mesh, origins_array[i], directions_array[i])
        for i in range(len(origins_array))
    ]
