"""Small geometric primitives shared by parsing, correspondence and measurement.

Deliberately dependency-light: numpy and scipy only. Higher-level geometry work
(curvature, geodesics, proximity) lives in ``mtra.mesh`` and uses libigl,
pygeodesic and trimesh — but the primitives below are the ones whose definitions
appear in the report, so they stay inspectable in one file.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Rotations
# --------------------------------------------------------------------------- #


def rotation_from_wpr(
    w_deg: float, p_deg: float, r_deg: float, convention: str = "rz_ry_rx"
) -> np.ndarray:
    """Controller W/P/R triple to a rotation matrix.

    FANUC writes (W, P, R) as rotations about Z, Y and X. SciPy's uppercase
    intrinsic sequence 'ZYX' composes them as ``Rz @ Ry @ Rx``, which is the
    convention recorded in ``config/defaults.yaml`` so it can be audited rather
    than assumed.
    """
    if convention != "rz_ry_rx":
        raise ValueError(f"unsupported Euler convention: {convention!r}")
    return Rotation.from_euler("ZYX", [w_deg, p_deg, r_deg], degrees=True).as_matrix()


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two directions, in degrees, robust at both extremes."""
    a_unit = unit(a)
    b_unit = unit(b)
    cross = np.linalg.norm(np.cross(a_unit, b_unit))
    dot = float(np.clip(np.dot(a_unit, b_unit), -1.0, 1.0))
    return float(np.degrees(np.arctan2(cross, dot)))


def axial_angle_deg(vector: np.ndarray, reference: np.ndarray) -> float:
    """Orientation of an undirected axis in a reference plane, in [0, 180).

    Stroke direction is axial — a pass run in reverse describes the same bearing —
    so comparing signed angles would report a 180° difference for identical
    geometry.
    """
    v = np.asarray(vector, dtype=float).copy()
    ref = np.asarray(reference, dtype=float).copy()
    v[2] = 0.0
    ref[2] = 0.0
    if np.linalg.norm(v) < EPS or np.linalg.norm(ref) < EPS:
        return float("nan")
    angle = np.degrees(np.arctan2(v[1], v[0]) - np.arctan2(ref[1], ref[0]))
    return float(np.mod(angle, 180.0))


# --------------------------------------------------------------------------- #
# Vectors
# --------------------------------------------------------------------------- #


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < EPS:
        return np.zeros(3, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def project_to_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = unit(normal)
    v = np.asarray(vector, dtype=float)
    return v - np.dot(v, n) * n


# --------------------------------------------------------------------------- #
# Polylines
# --------------------------------------------------------------------------- #


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def resample_polyline(points: np.ndarray, step_mm: float) -> np.ndarray:
    """Arc-length resample, preserving both endpoints.

    Measurement operates on resampled strokes so that controller sampling density
    cannot masquerade as spacing structure. Returned point count is
    ``max(2, ceil(length / step) + 1)``; a zero-length polyline is returned as-is.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2 or step_mm <= 0:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cumulative[-1])
    if total < EPS:
        return pts
    count = max(2, int(np.ceil(total / step_mm)) + 1)
    targets = np.linspace(0.0, total, count)
    out = np.empty((count, 3), dtype=float)
    for axis in range(3):
        out[:, axis] = np.interp(targets, cumulative, pts[:, axis])
    return out


def pca_direction(points: np.ndarray) -> np.ndarray:
    """Dominant direction of a point set, as a unit vector.

    Points are centred before the SVD; orientation is arbitrary, so callers that
    need a consistent sign align it themselves (``align_direction``).
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return np.zeros(3, dtype=float)
    centred = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    return unit(vh[0])


def align_direction(direction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip ``direction`` so it has a non-negative dot with ``reference``."""
    return -direction if np.dot(direction, reference) < 0 else direction


def perpendicular_distance_rms(points: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    """RMS perpendicular distance from points to a line — straightness error."""
    pts = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    d = unit(direction)
    along = np.outer(pts @ d, d)
    perpendicular = pts - along
    return float(np.sqrt(np.mean(np.sum(perpendicular**2, axis=1))))


def discrete_curvature(points: np.ndarray) -> np.ndarray:
    """Curvature magnitude at interior points via the circumscribed-circle formula.

    ``kappa_i = 2 |sin(theta)| / |chord|`` for the angle between the two adjacent
    segment directions. Endpoints receive the nearest interior value so arrays
    stay aligned with samples.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return np.zeros(len(pts), dtype=float)
    v1 = pts[1:-1] - pts[:-2]
    v2 = pts[2:] - pts[1:-1]
    chord = np.linalg.norm(v2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_theta = np.linalg.norm(np.cross(v1, v2), axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        )
        kappa = 2.0 * sin_theta / chord
    kappa = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([[kappa[0]], kappa, [kappa[-1]]])


def smoothness_rms(points: np.ndarray) -> float:
    """RMS second difference — local deviation not explained by a straight line.

    Reported in millimetres (PRD D5) and therefore only comparable between strokes
    sampled at the same step; the measurement layer records the sampling step
    alongside the value.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0
    second = pts[:-2] - 2.0 * pts[1:-1] + pts[2:]
    return float(np.sqrt(np.mean(np.sum(second**2, axis=1))))


# --------------------------------------------------------------------------- #
# Triangles and barycentrics
# --------------------------------------------------------------------------- #


def barycentric_coordinates(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of the projection of ``point`` onto a triangle.

    Clamped to the triangle so numerical noise cannot produce negative weights in
    a report that claims an exact surface correspondence.
    """
    a, b, c = (np.asarray(v, dtype=float) for v in triangle)
    v0 = b - a
    v1 = c - a
    v2 = np.asarray(point, dtype=float) - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < EPS:
        return np.array([1.0, 0.0, 0.0])
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    bary = np.array([u, v, w])
    bary = np.clip(bary, 0.0, 1.0)
    total = bary.sum()
    return bary / total if total > EPS else np.array([1.0, 0.0, 0.0])


def interpolate_vertex_field(bary: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolate a per-vertex field using barycentric weights."""
    return np.asarray(bary, dtype=float) @ np.asarray(values, dtype=float)


# --------------------------------------------------------------------------- #
# Circular statistics (for axial directions)
# --------------------------------------------------------------------------- #


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Mean of axial angles, doubled-angle method.

    0° and 180° describe the same pass, so a naive mean would report 90° for a
    set of identical strokes evaluated with mixed sign.
    """
    angles = np.asarray(angles_deg, dtype=float)
    if angles.size == 0:
        return float("nan")
    doubled = np.radians(2.0 * angles)
    mean_angle = np.arctan2(np.mean(np.sin(doubled)), np.mean(np.cos(doubled))) / 2.0
    return float(np.mod(np.degrees(mean_angle), 180.0))


def circular_dispersion_deg(angles_deg: np.ndarray) -> float:
    """Dispersion of axial angles in degrees, in [0, 45].

    45° is the maximum possible dispersion for axial data (orthogonal strokes);
    returned instead of the circular standard deviation so B3 is readable as a
    score without a distributional assumption.
    """
    angles = np.asarray(angles_deg, dtype=float)
    if angles.size < 2:
        return 0.0
    doubled = np.radians(2.0 * angles)
    resultant = np.hypot(np.mean(np.sin(doubled)), np.mean(np.cos(doubled)))
    resultant = float(np.clip(resultant, 1e-9, 1.0))
    return float(np.degrees(np.sqrt(-2.0 * np.log(resultant)) / 2.0))
