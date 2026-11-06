"""Shared view over one analysed pair, plus the few primitives every measurement module needs.

The point of ``PairView`` is frame discipline. Trajectory poses arrive in the robot
frame; the mesh lives in the mesh frame; every surface comparison must happen in
exactly one of them. Rather than each module remembering which array is which,
the view exposes explicitly-named accessors (``stroke_points_mesh``,
``stroke_axes_mesh``) and the transform is stored once, on the correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mtra.config import Settings
from mtra.core.geometry import unit
from mtra.core.types import SegmentClass, TrajectorySegment
from mtra.correspondence import frames as frame_ops
from mtra.correspondence.projection import PairCorrespondence
from mtra.mesh.features import SurfaceField
from mtra.robot.common_schema import ParsedProgram
from mtra.traj.sampling import Stroke


@dataclass
class PairView:
    pair_id: str
    program: ParsedProgram
    segments: list[TrajectorySegment]
    strokes: list[Stroke]
    surface: SurfaceField
    correspondence: PairCorrespondence
    world_from_mesh: np.ndarray
    mesh_from_world: np.ndarray
    settings: Settings
    tool_axis_world: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    # -- frames ------------------------------------------------------------- #

    def stroke_points_mesh(self, stroke: Stroke) -> np.ndarray:
        return frame_ops.apply(self.mesh_from_world, stroke.points)

    def stroke_axes_mesh(self, stroke: Stroke) -> np.ndarray:
        axes = (
            np.tile(unit(self.tool_axis_world), (len(stroke.points), 1))
            if self.tool_axis_world is not None
            else np.asarray(stroke.tool_axes, dtype=float)
        )
        return frame_ops.apply_direction(self.mesh_from_world, axes)

    def stroke_axes_world(self, stroke: Stroke) -> np.ndarray:
        if self.tool_axis_world is not None:
            return np.tile(unit(self.tool_axis_world), (len(stroke.points), 1))
        return np.asarray(stroke.tool_axes, dtype=float)

    # -- convenience -------------------------------------------------------- #

    @property
    def painting_strokes(self) -> list[Stroke]:
        return [s for s in self.strokes if len(s.points) >= 2]

    def surface_points(self) -> np.ndarray:
        if not self.correspondence.records:
            return np.zeros((0, 3))
        return np.array([r.closest_point for r in self.correspondence.records], dtype=float)

    def surface_normals(self) -> np.ndarray:
        points = self.surface_points()
        if len(points) == 0:
            return np.zeros((0, 3))
        return self.surface.sample(points).normals

    def correspondence_ranges(self) -> dict[str, tuple[int, int]]:
        return self.correspondence.stroke_ranges

    def transfer_segments(self) -> list[TrajectorySegment]:
        return [s for s in self.segments if s.segment_class is SegmentClass.TRANSFER]


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def tangents(points: np.ndarray) -> np.ndarray:
    """Unit tangent at every sample, by central difference with endpoint padding."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return np.zeros((len(pts), 3))
    forward = np.gradient(pts, axis=0)
    return np.array([unit(v) for v in forward])


def bearing_xy(points: np.ndarray, *, min_extent_mm: float = 1.0) -> float:
    """Axial bearing of a stroke in mesh-frame XY, in [0, 180), or NaN.

    NaN is the honest answer for a stroke whose travel is normal to the XY plane: it
    has a direction, it just has no *bearing*. Projecting such a stroke and reporting
    the result invents one — the principal direction of a numerically vertical point
    set in XY is whatever rounding noise happens to point at, which is how a set of
    vertical passes gets described as 90° strokes. ``min_extent_mm`` is the lateral
    travel below which no bearing is claimed, and it is configuration because it is
    a judgement about what counts as "in-plane" for a given cell.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return float("nan")
    extent = float(np.max(np.ptp(pts[:, :2], axis=0)))
    if extent < min_extent_mm:
        return float("nan")
    delta = pts[-1, :2] - pts[0, :2]
    if np.linalg.norm(delta) < 0.5 * extent:
        # A bowed or returning stroke: the chord is a poor descriptor, the scatter is
        # not. The extent test above still guards this branch.
        centred = pts[:, :2] - pts[:, :2].mean(axis=0)
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        delta = vh[0]
    angle = np.degrees(np.arctan2(delta[1], delta[0]))
    return float(np.mod(angle, 180.0))


def stroke_axis(points: np.ndarray) -> np.ndarray | None:
    """Unit 3D travel direction of a stroke (start→end), or None if it does not travel.

    The sign is kept rather than folded: J2 asks whether consecutive passes were
    driven in *opposite* directions, which is a question about the signed travel
    vector. The B-group folds the sign away itself.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return None
    delta = pts[-1] - pts[0]
    length = float(np.linalg.norm(delta))
    if length < 1e-9 or float(np.max(np.ptp(pts, axis=0))) < 1e-9:
        return None
    return delta / length


def axial_mean_axis(axes: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Dominant undirected 3D axis of a direction set, and how well it describes it.

    Undirected because a pass run backwards is the same pass: each axis contributes
    ``n nᵀ``, which is sign-invariant, so the leading eigenvector of the sum is the
    axial mean. ``stability`` is ``(λ1 - λ2) / λ1`` — near 1 the set really has a
    dominant axis, near 0 the axes are spread out and quoting a single "dominant
    direction" would be a fiction. Reporting it is the 3D counterpart of B4's
    anisotropy ratio.
    """
    array = np.atleast_2d(np.asarray(axes, dtype=float))
    if array.size == 0 or len(array) == 0:
        return None, 0.0
    matrix = array.T @ array
    try:
        values, vectors = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return None, 0.0
    order = np.argsort(values)[::-1]
    top = float(values[order[0]])
    if top <= 1e-12:
        return None, 0.0
    second = float(values[order[1]]) if len(order) > 1 else 0.0
    axis = unit(vectors[:, order[0]])
    if not np.any(axis):  # pragma: no cover - defensive
        return None, 0.0
    return axis, float(np.clip((top - second) / top, 0.0, 1.0))


def axial_angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two undirected 3D directions, in [0, 90]."""
    first = unit(np.asarray(a, dtype=float))
    second = unit(np.asarray(b, dtype=float))
    if not np.any(first) or not np.any(second):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(first, second))), 0.0, 1.0))))


def signed_turn_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two *signed* travel directions, in [0, 180].

    0° = same direction, 180° = reversed, 90° = crossed at right angles. Unlike
    ``axial_angle_between`` this keeps the sign, which is what J2 classifies on.
    """
    first = unit(np.asarray(a, dtype=float))
    second = unit(np.asarray(b, dtype=float))
    if not np.any(first) or not np.any(second):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(np.dot(first, second)), -1.0, 1.0))))


def axial_deviation_deg(axes: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-axis undirected angle to a reference axis, in [0, 90]."""
    array = np.atleast_2d(np.asarray(axes, dtype=float))
    ref = unit(np.asarray(reference, dtype=float))
    if array.size == 0 or len(array) == 0 or not np.any(ref):
        return np.zeros(0)
    cosines = np.clip(np.abs(array @ ref), 0.0, 1.0)
    return np.degrees(np.arccos(cosines))


def elevation_deg(axis: np.ndarray) -> float:
    """Angle between an axis and the XY plane, in [0, 90].

    The companion to a nullable bearing: a vertical pass has no bearing but it does
    have an elevation, and 90° says exactly why the bearing is absent.
    """
    vector = unit(np.asarray(axis, dtype=float))
    if not np.any(vector):
        return float("nan")
    return float(np.degrees(np.arcsin(np.clip(abs(vector[2]), 0.0, 1.0))))


def mean_vector(vectors: np.ndarray) -> np.ndarray:
    """Normalised mean of a direction set; zero when the set cancels out."""
    array = np.asarray(vectors, dtype=float)
    if array.size == 0:
        return np.zeros(3)
    return unit(array.mean(axis=0))


def angle_between_axial_deg(a_deg: float, b_deg: float) -> float:
    """Difference between two axial headings, folded into [0, 90].

    0° means the two bearings agree; 90° means they are orthogonal.
    """
    difference = float(np.mod(a_deg - b_deg, 180.0))
    return float(min(difference, 180.0 - difference))


def principal_axis_xy(points: np.ndarray) -> tuple[float, float] | None:
    """Dominant in-plane bearing of a 2D point set, plus its anisotropy.

    Returns ``(heading_deg, anisotropy)`` where anisotropy is the square root of the
    ratio of the two principal spreads. A footprint that is nearly square has
    anisotropy ≈ 1 and its "principal direction" is not meaningful — reporting the
    ratio alongside the angle is what lets a reader see that, instead of trusting a
    bearing that is really numerical noise.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.shape[1] < 2 and pts.shape[0] >= 2:
        pts = pts.reshape(-1, 2)
    if len(pts) < 3:
        return None
    centred = pts[:, :2] - pts[:, :2].mean(axis=0)
    try:
        _, singular, vh = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return None
    if singular[0] <= 1e-12:
        return None
    anisotropy = float(singular[0] / singular[1]) if singular[1] > 1e-12 else float("inf")
    heading = float(np.mod(np.degrees(np.arctan2(vh[0][1], vh[0][0])), 180.0))
    return heading, anisotropy


def angular_deviation_deg(vectors: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-vector angle to a reference direction, in degrees, in [0, 180]."""
    array = np.asarray(vectors, dtype=float)
    ref = unit(np.asarray(reference, dtype=float))
    if array.size == 0:
        return np.zeros(0)
    cosines = np.clip(array @ ref, -1.0, 1.0)
    return np.degrees(np.arccos(cosines))


def unit_step_interval(points: np.ndarray) -> np.ndarray:
    """Consecutive sample spacing, in the points' own units."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return np.zeros(0)
    return np.linalg.norm(np.diff(pts, axis=0), axis=1)


def axis_normal_angle_deg(normals: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Angle between tool axis and surface normal at each corresponded sample."""
    normal_array = unit_rows(normals)
    axis_array = unit_rows(axes)
    if normal_array.shape != axis_array.shape or len(normal_array) == 0:
        return np.zeros(0)
    cosines = np.clip(np.sum(axis_array * normal_array, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosines))


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    array = np.atleast_2d(np.asarray(vectors, dtype=float))
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return array / norms


# --------------------------------------------------------------------------- #
# Planar footprint (used by region, boundary and pitch measurements)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlaneBasis:
    """A best-fit plane: origin, two in-plane axes and the normal."""

    origin: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    normal: np.ndarray

    def to_2d(self, points: np.ndarray) -> np.ndarray:
        offset = np.atleast_2d(np.asarray(points, dtype=float)) - self.origin
        return np.column_stack([offset @ self.axis_u, offset @ self.axis_v])

    def to_3d(self, points_2d: np.ndarray) -> np.ndarray:
        array = np.atleast_2d(np.asarray(points_2d, dtype=float))
        return self.origin + array[:, :1] * self.axis_u + array[:, 1:2] * self.axis_v


def best_fit_plane(points: np.ndarray) -> PlaneBasis | None:
    """Least-squares plane through a point set, or None if it is degenerate.

    Used instead of assuming the XY plane: a painted panel may be mounted at any
    attitude, and projecting onto world XY would then report a foreshortened
    footprint as if it were the part's true extent.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return None
    origin = pts.mean(axis=0)
    centred = pts - origin
    try:
        _, singular, vh = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return None
    if singular[0] <= 1e-9 or singular[1] / singular[0] < 1e-6:
        return None
    return PlaneBasis(
        origin=origin,
        axis_u=unit(vh[0]),
        axis_v=unit(vh[1]),
        normal=unit(vh[2]),
    )


def convex_hull_2d(points_2d: np.ndarray) -> np.ndarray:
    """Ordered hull vertices of a 2D point set; empty when it is degenerate."""
    from scipy.spatial import ConvexHull, QhullError

    array = np.asarray(points_2d, dtype=float)
    if len(array) < 3:
        return np.zeros((0, 2))
    try:
        hull = ConvexHull(array)
    except (QhullError, ValueError):
        return np.zeros((0, 2))
    return array[hull.vertices]


def polygon_edge_lengths(polygon_2d: np.ndarray) -> np.ndarray:
    """Closed-polygon edge lengths."""
    poly = np.asarray(polygon_2d, dtype=float)
    if len(poly) < 2:
        return np.zeros(0)
    return np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)


def distance_to_polygon_edges(points_2d: np.ndarray, polygon_2d: np.ndarray) -> np.ndarray:
    """Shortest distance from each point to the closed polygon boundary."""
    pts = np.atleast_2d(np.asarray(points_2d, dtype=float))
    poly = np.asarray(polygon_2d, dtype=float)
    if len(poly) < 2 or len(pts) == 0:
        return np.full(len(pts), np.nan)
    starts = poly
    ends = np.roll(poly, -1, axis=0)
    deltas = ends - starts
    lengths_sq = np.sum(deltas**2, axis=1)
    lengths_sq[lengths_sq < 1e-18] = 1e-18
    out = np.empty(len(pts), dtype=float)
    for index, point in enumerate(pts):
        projections = np.clip(
            np.sum((point - starts) * deltas, axis=1) / lengths_sq, 0.0, 1.0
        )
        closest = starts + projections[:, None] * deltas
        out[index] = float(np.min(np.linalg.norm(point - closest, axis=1)))
    return out


def polygon_area(polygon_2d: np.ndarray) -> float:
    """Shoelace area of a closed polygon."""
    poly = np.asarray(polygon_2d, dtype=float)
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)
