"""Rigid registration: fitting, conditioning and residuals.

Phase 1 does not fit a transform into the pipeline — ``frames.resolve_pair_transform``
requires a declared transform (PRD §6 stage 5). What lives here is the arithmetic
Phase 2 needs, plus the conditioning check Phase 1 *does* use:

A point set that is nearly planar or collinear does not determine a rigid
transform. If the corresponded surface points are planar — which they usually
are, on a painted panel — then any future registration estimated from them is
ambiguous about rotation within that plane. Reporting that up front stops a later
phase from presenting an arbitrary in-plane orientation as a measurement.

The fit itself is Kabsch/SVD, verified by tests, including the degenerate cases
where it must refuse rather than return a number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RigidFit:
    rotation: np.ndarray
    translation: np.ndarray
    matrix: np.ndarray
    residuals: np.ndarray
    singular_values: np.ndarray
    conditioning_ratio: float
    is_degenerate: bool

    @property
    def rms_mm(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2))) if self.residuals.size else 0.0

    def summary(self) -> dict[str, float | bool]:
        return {
            "rms_mm": self.rms_mm,
            "median_mm": float(np.median(self.residuals)) if self.residuals.size else 0.0,
            "p95_mm": float(np.percentile(self.residuals, 95)) if self.residuals.size else 0.0,
            "max_mm": float(np.max(self.residuals)) if self.residuals.size else 0.0,
            "conditioning_ratio": self.conditioning_ratio,
            "is_degenerate": self.is_degenerate,
        }


def conditioning_ratio(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Singular values of the centred point set, and the smallest/largest ratio.

    A planar set has a near-zero third singular value; a collinear set has two.
    The ratio is therefore a direct statement about which rigid degrees of freedom
    the data can actually constrain.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return np.zeros(3), 0.0
    centred = pts - pts.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    largest = float(singular[0]) if singular.size else 0.0
    if largest <= 0:
        return np.asarray(singular, dtype=float), 0.0
    return np.asarray(singular, dtype=float), float(singular[-1] / largest)


def rigid_fit(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    min_conditioning_ratio: float = 0.02,
) -> RigidFit:
    """Least-squares rigid transform mapping ``source_points`` onto ``target_points``.

    Returns a fit flagged ``is_degenerate`` when the source set cannot constrain a
    unique rotation. Callers must treat a degenerate fit as unusable rather than
    as a poor fit.
    """
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.shape != target.shape:
        raise ValueError(f"point sets differ in shape: {source.shape} vs {target.shape}")
    if len(source) < 3:
        raise ValueError("rigid fit requires at least three correspondence pairs")

    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centred = source - source_centroid
    target_centred = target - target_centroid

    covariance = source_centred.T @ target_centred
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[2, 2] = -1.0
    rotation = vt.T @ correction @ u.T
    translation = target_centroid - rotation @ source_centroid

    residuals = np.linalg.norm(source_centred @ rotation.T + target_centroid - target, axis=1)
    _, ratio = conditioning_ratio(source)

    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation

    return RigidFit(
        rotation=rotation,
        translation=translation,
        matrix=matrix,
        residuals=residuals,
        singular_values=np.asarray(singular, dtype=float),
        conditioning_ratio=ratio,
        is_degenerate=ratio < min_conditioning_ratio,
    )


def residual_statistics(distances: np.ndarray) -> dict[str, float]:
    """Robust summary of a residual distribution (PRD §8.1: never a bare mean)."""
    values = np.asarray(distances, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        "count": float(values.size),
        "mean_mm": float(np.mean(values)),
        "median_mm": float(np.median(values)),
        "std_mm": float(np.std(values)),
        "p95_mm": float(np.percentile(values, 95)),
        "max_mm": float(np.max(values)),
        "cv": float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0.0,
    }
