"""Group H — boundary and region relationships (PRD §7, phase 1 items H1–H7).

"Region" is defined the way a paint engineer uses the word: the area the
trajectory actually covers. It is *not* a mesh connected component, because a
closed panel has one component and several painted areas. So the footprint is the
convex hull of the corresponded surface points, and the boundary is the edge of
that hull — unless the mesh itself has open boundary loops, in which case the
physical part edge is the more useful boundary and is used instead. Which one was
used is stated in every record, because the two answer different questions.

H8–H10 (clipping method, boundary-to-pitch rule, region-to-pattern rule) are
rules to be discovered, not measured, and are reported out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from mtra.core.types import ReasonCode
from mtra.measurements.catalogue import MeasurementSet, numeric_stats
from mtra.measurements.common import (
    PairView,
    convex_hull_2d,
    distance_to_polygon_edges,
    polygon_area,
)


@dataclass
class RegionFootprint:
    points_2d: np.ndarray
    hull_2d: np.ndarray
    boundary_source: str
    mesh_boundary_2d: list[np.ndarray]


def _project_xy(points: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    return np.column_stack([pts[:, 0], pts[:, 1]])


def default_cluster_threshold(view: PairView) -> float:
    """3x the median inter-stroke gap, floored at 5 mm.

    Derived from the data rather than configured, because "are these two passes the
    same region?" is a question about this pattern's density.
    """
    strokes = view.painting_strokes
    if len(strokes) < 2:
        return 5.0
    distances = []
    for index in range(1, len(strokes)):
        previous = view.stroke_points_mesh(strokes[index - 1])
        current = view.stroke_points_mesh(strokes[index])
        if len(previous) < 2 or len(current) < 2:
            continue
        tree = cKDTree(previous)
        distance, _ = tree.query(current)
        distances.append(float(np.min(distance)))
    if not distances:
        return 5.0
    return max(3.0 * float(np.median(distances)), 5.0)


def cluster_strokes(view: PairView, threshold_mm: float) -> list[list[int]]:
    """Group stroke indices whose surface footprints touch within ``threshold_mm``."""
    stroke_points = [
        view.stroke_points_mesh(stroke) for stroke in view.painting_strokes
    ]
    parents = list(range(len(stroke_points)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parents[root_b] = root_a

    trees = [cKDTree(points) for points in stroke_points if len(points) >= 2]
    valid = [index for index, points in enumerate(stroke_points) if len(points) >= 2]
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            distance, _ = trees[j].query(stroke_points[valid[i]])
            if float(np.min(distance)) <= threshold_mm:
                union(valid[i], valid[j])

    clusters: dict[int, list[int]] = {}
    for index in valid:
        clusters.setdefault(find(index), []).append(index)
    return sorted(clusters.values(), key=len, reverse=True)


def stroke_region_map(view: PairView, threshold_mm: float | None = None) -> dict[str, int]:
    """Stroke id to region index, in region-size order."""
    threshold = (
        default_cluster_threshold(view) if threshold_mm is None else float(threshold_mm)
    )
    strokes = view.painting_strokes
    mapping: dict[str, int] = {}
    for region_index, cluster in enumerate(cluster_strokes(view, threshold)):
        for stroke_index in cluster:
            mapping[strokes[stroke_index].stroke_id] = region_index
    return mapping


def footprint(view: PairView) -> RegionFootprint | None:
    surface_points = view.surface_points()
    if len(surface_points) < 3:
        return None
    points_2d = _project_xy(surface_points)
    hull = convex_hull_2d(points_2d)

    mesh_boundaries: list[np.ndarray] = []
    for loop in view.surface.mesh_record.boundary_loops:
        if len(loop) >= 3:
            mesh_boundaries.append(_project_xy(view.surface.mesh.vertices[loop]))

    if mesh_boundaries:
        # Concat for distance work; the loop identity is kept for reporting.
        boundary_source = (
            f"mesh boundary loops ({len(mesh_boundaries)} loop(s)) in mesh XY"
        )
    else:
        boundary_source = (
            "convex hull of the painted footprint in mesh XY (the mesh is closed, so it "
            "has no boundary loop)"
        )
    return RegionFootprint(
        points_2d=points_2d,
        hull_2d=hull,
        boundary_source=boundary_source,
        mesh_boundary_2d=mesh_boundaries,
    )


def _boundary_distance(points_2d: np.ndarray, region: RegionFootprint) -> np.ndarray:
    if region.mesh_boundary_2d:
        per_loop = [
            distance_to_polygon_edges(points_2d, loop) for loop in region.mesh_boundary_2d
        ]
        return np.nanmin(np.vstack(per_loop), axis=0)
    if len(region.hull_2d) >= 2:
        return distance_to_polygon_edges(points_2d, region.hull_2d)
    return np.full(len(points_2d), np.nan)


def measure_regions(view: PairView, measurements: MeasurementSet) -> None:
    region = footprint(view)
    if region is None:
        for parameter_id in ("H2", "H3", "H4", "H5", "H6", "H7"):
            measurements.unavailable(
                parameter_id,
                ReasonCode.NO_SURFACE_CORRESPONDENCE,
                "the painted footprint has fewer than three corresponded surface points",
            )
        measurements.unavailable(
            "H1",
            ReasonCode.NO_PAINTING_SEGMENTS,
            "no painting strokes exist, so there are no painted regions",
        )
        return

    strokes = view.painting_strokes
    threshold = default_cluster_threshold(view)
    clusters = cluster_strokes(view, threshold)

    measurements.add(
        "H1",
        {
            "region_count": len(clusters),
            "stroke_count": len(strokes),
            "cluster_threshold_mm": round(threshold, 4),
            "strokes_per_region": [len(cluster) for cluster in clusters],
        },
        method=(
            "painting strokes with no surface gap wider than 3x the median inter-stroke "
            "gap (minimum 5 mm) are treated as one region; the threshold is reported so "
            "the grouping can be challenged"
        ),
        source_refs=["traj.sampling"],
    )

    face_ids = np.array(
        [record.mesh_face_id for record in view.correspondence.records], dtype=np.int64
    )
    unique_faces = np.unique(face_ids)
    face_areas = np.asarray(view.surface.mesh.area_faces, dtype=float)
    covered_area = float(np.sum(face_areas[unique_faces])) if unique_faces.size else 0.0
    hull_area = polygon_area(region.hull_2d)
    measurements.add(
        "H2",
        {
            "covered_face_area_mm2": covered_area,
            "covered_faces": int(unique_faces.size),
            "footprint_hull_area_mm2": hull_area,
            "hull_fill_ratio": (covered_area / hull_area) if hull_area > 1e-9 else None,
        },
        method=(
            "sum of the areas of the unique mesh triangles the trajectory corresponds to; "
            "the hull area is the planar footprint, and their ratio shows how much of the "
            "footprint the trajectory actually touches"
        ),
        source_refs=["mesh.loader", "correspondence.projection"],
    )

    centred = region.points_2d - region.points_2d.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    if singular[1] <= 1e-9:
        measurements.unavailable(
            "H3",
            ReasonCode.DEGENERATE_GEOMETRY,
            "the footprint is collinear, so it has no aspect ratio",
        )
        measurements.unavailable(
            "H4",
            ReasonCode.DEGENERATE_GEOMETRY,
            "the footprint is collinear, so its orientation is well defined but its "
            "shape is not — aspect ratio is the precondition for this pair of parameters",
        )
    else:
        measurements.add(
            "H3",
            round(float(np.sqrt(singular[0] / singular[1])), 6),
            method="square root of the ratio of the first two singular values of the footprint",
            source_refs=["correspondence.projection"],
        )
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        orientation = float(
            np.mod(np.degrees(np.arctan2(vh[0][1], vh[0][0])), 180.0)
        )
        measurements.add(
            "H4",
            orientation,
            method="axial bearing of the footprint's first singular direction in mesh XY",
            source_refs=["correspondence.projection"],
        )

    boundary_distances = _boundary_distance(region.points_2d, region)
    finite = boundary_distances[np.isfinite(boundary_distances)]
    if finite.size == 0:
        measurements.unavailable(
            "H5",
            ReasonCode.DEGENERATE_GEOMETRY,
            "no boundary could be constructed for the footprint",
        )
    else:
        stats = numeric_stats(finite, unit_label="mm")
        measurements.add(
            "H5",
            stats,
            method=f"distance from each corresponded sample to the {region.boundary_source}",
            uncertainty=stats["std"],
            source_refs=["correspondence.projection"],
        )

    if len(strokes) >= 1:
        starts = np.array([view.stroke_points_mesh(s)[0] for s in strokes], dtype=float)
        ends = np.array([view.stroke_points_mesh(s)[-1] for s in strokes], dtype=float)
        start_distances = _boundary_distance(_project_xy(starts), region)
        end_distances = _boundary_distance(_project_xy(ends), region)
        first = int(np.nanargmin(start_distances))
        last = int(np.nanargmin(end_distances))
        measurements.add(
            "H6",
            {
                "point": [float(v) for v in starts[first]],
                "distance_to_boundary_mm": float(start_distances[first]),
                "stroke_id": strokes[first].stroke_id,
            },
            method=(
                "stroke start pose closest to the footprint boundary; the boundary side "
                "the pattern is judged to enter from"
            ),
            location=tuple(float(v) for v in starts[first]),
            source_refs=["traj.sampling"],
        )
        measurements.add(
            "H7",
            {
                "point": [float(v) for v in ends[last]],
                "distance_to_boundary_mm": float(end_distances[last]),
                "stroke_id": strokes[last].stroke_id,
            },
            method=(
                "stroke end pose closest to the footprint boundary; the boundary side the "
                "pattern is judged to leave by"
            ),
            location=tuple(float(v) for v in ends[last]),
            source_refs=["traj.sampling"],
        )
    else:
        for parameter_id in ("H6", "H7"):
            measurements.unavailable(
                parameter_id,
                ReasonCode.NO_PAINTING_SEGMENTS,
                "no stroke start or end pose exists",
            )
