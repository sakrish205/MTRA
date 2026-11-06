"""Group C — path spacing vs mesh geometry (PRD §7, phase 1 items C1–C6).

Pitch is measured between *consecutive painting strokes in execution order*, by
nearest-neighbour distance from each sample of the later stroke to the earlier
one. That estimator is chosen deliberately over "distance between stroke
centrelines": centrelines are a fitted abstraction, and fitting a line to a curved
stroke pulls the answer away from the geometry the robot actually followed.

Three measures are kept separate because they answer different questions and
disagree on curved surfaces:

* C1 XY projected pitch — what a drawing/coverage view shows;
* C2 3D Euclidean pitch — the straight-line gap through space;
* C3 geodesic pitch — the distance an ant would walk along the surface.

On a flat plate all three coincide, which is exactly why the synthetic fixture can
prove them: a 25 mm raster must measure 25 mm in all three.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from mtra.core.types import ReasonCode
from mtra.measurements.catalogue import MeasurementSet, numeric_stats
from mtra.measurements.common import PairView
from mtra.traj.sampling import Stroke


def _nearest_distances(
    source: np.ndarray, target: np.ndarray, *, xy_only: bool
) -> np.ndarray:
    src = np.atleast_2d(np.asarray(source, dtype=float))
    tgt = np.atleast_2d(np.asarray(target, dtype=float))
    if xy_only:
        src = src[:, :2]
        tgt = tgt[:, :2]
    tree = cKDTree(src)
    distances, _ = tree.query(tgt)
    return np.asarray(distances, dtype=float)


def _geodesic_pitch(
    view: PairView,
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    max_samples: int,
    max_vertices: int,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Geodesic distances for a bounded sample of nearest vertex pairs.

    Returns ``(distances, snap_offsets_mm, reason_if_unavailable)``.

    pygeodesic works vertex-to-vertex, so each trajectory sample is snapped to its
    nearest mesh vertex. That snapping has a *bias*, not just noise: if one stroke's
    samples happen to sit 3 mm off the vertex grid and the next stroke's sit on it,
    the vertex-to-vertex distance is inflated by 3 mm even on a perfectly flat plate.
    So each raw geodesic gets a first-order endpoint correction,

        d = d_geo(va, vb) + <p_b - vb, u> - <p_a - va, u>,

    where ``u`` is the in-surface direction from ``va`` to ``vb``. That is exact for a
    locally flat surface and second-order-wrong on curvature, which is a far better
    trade than a systematic offset. The snap offsets are returned so the size of the
    correction is visible in the report rather than trusted.
    """
    mesh = view.surface.mesh
    vertex_count = int(len(mesh.vertices))
    empty = np.zeros(0)
    if vertex_count > max_vertices:
        return empty, empty, (
            f"mesh has {vertex_count} vertices, above the configured geodesic limit of "
            f"{max_vertices}; exact geodesics would not finish in reasonable time"
        )
    try:
        from pygeodesic import geodesic as pygeodesic
    except ImportError:
        return empty, empty, "pygeodesic is not installed (optional 'geometry' extra)"

    indices_b = np.arange(len(points_b))
    if len(indices_b) > max_samples:
        indices_b = np.linspace(0, len(points_b) - 1, max_samples, dtype=int)

    samples_a = np.asarray(points_a, dtype=float)
    samples_b = np.asarray(points_b, dtype=float)[indices_b]
    # Geodesic distance is a surface quantity, so the TCP samples are projected onto
    # the mesh first. Snapping the raw poses would fold the standoff into the snap
    # offset and make the reported error meaningless.
    surface_a = view.surface.sample(samples_a).closest_points
    surface_b = view.surface.sample(samples_b).closest_points
    vertices_a = view.surface.nearest_vertex(surface_a)
    vertices_b = view.surface.nearest_vertex(surface_b)
    offsets_a = surface_a - mesh.vertices[vertices_a]
    vertices_a_points = mesh.vertices[vertices_a]
    tree = cKDTree(vertices_a_points)

    algorithm = pygeodesic.PyGeodesicAlgorithmExact(
        np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)
    )
    distances: list[float] = []
    snap_offsets: list[float] = []
    for vertex_b, offset_b in zip(
        vertices_b, surface_b - mesh.vertices[vertices_b], strict=True
    ):
        _, nearest_index = tree.query(mesh.vertices[int(vertex_b)])
        vertex_a = int(vertices_a[int(nearest_index)])
        raw, _ = algorithm.geodesicDistance(vertex_a, int(vertex_b))
        raw = float(raw)
        if not np.isfinite(raw) or raw < 0.0:
            continue
        direction = mesh.vertices[int(vertex_b)] - mesh.vertices[vertex_a]
        norm = float(np.linalg.norm(direction))
        correction = 0.0
        if norm > 1e-9:
            unit_direction = direction / norm
            correction = float(
                np.dot(offset_b, unit_direction)
                - np.dot(offsets_a[int(nearest_index)], unit_direction)
            )
        distances.append(max(raw + correction, 0.0))
        snap_offsets.append(
            float(
                max(
                    np.linalg.norm(offsets_a[int(nearest_index)]),
                    np.linalg.norm(offset_b),
                )
            )
        )
    return (
        np.array(distances, dtype=float),
        np.array(snap_offsets, dtype=float),
        None,
    )


def measure_spacing(view: PairView, measurements: MeasurementSet) -> None:
    strokes: list[Stroke] = view.painting_strokes
    if len(strokes) < 2:
        for parameter_id in ("C1", "C2", "C3", "C4", "C5", "C6"):
            measurements.unavailable(
                parameter_id,
                ReasonCode.INSUFFICIENT_STROKES,
                f"pitch needs at least two painting strokes; found {len(strokes)}",
            )
        return

    max_gap = float(view.settings.get("measurement.pitch_correspondence_max_mm", 40.0))
    max_samples = int(view.settings.get("measurement.geodesic_max_samples_per_stroke", 25))
    max_vertices = int(view.settings.get("measurement.geodesic_max_vertices", 400_000))

    xy_pitches: list[float] = []
    xyz_pitches: list[float] = []
    geodesic_pitches: list[float] = []
    geodesic_snaps: list[float] = []
    pair_table: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    geodesic_reason: str | None = None

    for first, second in zip(strokes[:-1], strokes[1:], strict=True):
        points_a = view.stroke_points_mesh(first)
        points_b = view.stroke_points_mesh(second)
        if len(points_a) < 2 or len(points_b) < 2:
            continue

        xy = _nearest_distances(points_a, points_b, xy_only=True)
        xyz = _nearest_distances(points_a, points_b, xy_only=False)
        xy_median = float(np.median(xy))
        xyz_median = float(np.median(xyz))

        if xy_median > max_gap:
            # Consecutive in program order but not neighbours on the surface (a jump
            # to another region, for instance). Excluding it keeps a real jump from
            # being averaged in as a wide pass.
            excluded.append(
                {
                    "from": first.stroke_id,
                    "to": second.stroke_id,
                    "xy_median_mm": xy_median,
                    "reason": f"beyond {max_gap} mm correspondence limit",
                }
            )
            continue

        xy_pitches.append(xy_median)
        xyz_pitches.append(xyz_median)

        geodesic_value: float | None = None
        snap_value: float | None = None
        if geodesic_reason is None:
            geodesic_sample, snap_sample, geodesic_reason = _geodesic_pitch(
                view,
                points_a,
                points_b,
                max_samples=max_samples,
                max_vertices=max_vertices,
            )
            if geodesic_reason is None and geodesic_sample.size:
                geodesic_value = float(np.median(geodesic_sample))
                snap_value = float(np.max(snap_sample)) if snap_sample.size else None
                geodesic_pitches.append(geodesic_value)
                if snap_value is not None:
                    geodesic_snaps.append(snap_value)

        pair_table.append(
            {
                "from": first.stroke_id,
                "to": second.stroke_id,
                "xy_mm": round(xy_median, 4),
                "xyz_mm": round(xyz_median, 4),
                "geodesic_mm": None if geodesic_value is None else round(geodesic_value, 4),
                "geodesic_snap_offset_mm": None if snap_value is None else round(snap_value, 4),
                "samples": int(len(points_b)),
            }
        )

    if not xyz_pitches:
        for parameter_id in ("C1", "C2", "C3", "C4", "C5", "C6"):
            measurements.unavailable(
                parameter_id,
                ReasonCode.NO_SURFACE_CORRESPONDENCE,
                "no consecutive stroke pair was within the correspondence limit; pitch "
                "would have to compare unrelated passes",
            )
        return

    xy_stats = numeric_stats(np.array(xy_pitches), unit_label="mm")
    xyz_stats = numeric_stats(np.array(xyz_pitches), unit_label="mm")

    measurements.add(
        "C1",
        xy_stats,
        method=(
            "median nearest-neighbour distance per stroke pair, projected into mesh XY; "
            f"{len(excluded)} pair(s) excluded beyond {max_gap} mm"
        ),
        uncertainty=xy_stats["std"],
        source_refs=["traj.sampling", "correspondence.projection"],
    )

    measurements.add(
        "C2",
        xyz_stats,
        method="median 3D nearest-neighbour distance per consecutive stroke pair",
        uncertainty=xyz_stats["std"],
        source_refs=["traj.sampling", "correspondence.projection"],
    )

    if geodesic_pitches:
        geodesic_stats = numeric_stats(np.array(geodesic_pitches), unit_label="mm")
        geodesic_stats["max_snap_offset_mm"] = (
            round(float(np.max(geodesic_snaps)), 4) if geodesic_snaps else None
        )
        measurements.add(
            "C3",
            geodesic_stats,
            method=(
                "exact geodesic distance on the mesh between nearest vertices of "
                f"corresponding samples (≤{max_samples} samples per stroke pair), with a "
                "first-order correction for the distance from each sample to its snapped "
                "vertex along the geodesic direction; the largest remaining snap offset is "
                "reported as max_snap_offset_mm and bounds how much the vertex grid can "
                "still bias this value"
            ),
            uncertainty=geodesic_stats["std"],
            source_refs=["mesh.features", "pygeodesic"],
        )
    else:
        measurements.unavailable(
            "C3",
            ReasonCode.OPTIONAL_DEPENDENCY_MISSING
            if geodesic_reason and "not installed" in geodesic_reason
            else ReasonCode.COMPUTATION_FAILED,
            geodesic_reason or "geodesic pitch could not be computed for any stroke pair",
        )

    mean_pitch = float(np.mean(xyz_pitches))
    measurements.add(
        "C4",
        {
            "mean_mm": mean_pitch,
            "median_mm": float(np.median(xyz_pitches)),
            "stroke_pairs": int(len(xyz_pitches)),
            "excluded_pairs": int(len(excluded)),
        },
        method="mean of the per-pair 3D Euclidean pitches",
        uncertainty=xyz_stats["std"],
        source_refs=["traj.sampling"],
    )

    measurements.add(
        "C5",
        {"pairs": pair_table, "excluded": excluded},
        method="per-stroke-pair pitch table, in execution order",
        source_refs=["traj.sampling"],
    )

    cv = (
        float(np.std(xyz_pitches) / mean_pitch) if mean_pitch > 1e-9 else 0.0
    )
    measurements.add(
        "C6",
        round(cv, 6),
        method="coefficient of variation of the per-pair 3D pitches (std/mean)",
        uncertainty=None,
        source_refs=["traj.sampling"],
    )
