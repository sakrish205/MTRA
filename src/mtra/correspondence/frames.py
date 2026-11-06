"""Frame algebra.

A 4x4 homogeneous matrix mapping mesh coordinates to robot/world coordinates.
The manifest declares these; nothing in the pipeline invents one.

Direction of the convention is fixed and documented once, here: a ``FrameSpec``
transform maps **from** its own frame **into** its parent (or world when parent is
None). Trajectory points are in the robot frame, mesh points in the mesh frame, so
correspondence converts trajectory points with the inverse.

Phase 1 uses declared transforms only. PRD §6 stage 5 is explicit about why: an
unconstrained fit can make unrelated geometry look aligned, and that failure is
undetectable from the report.
"""

from __future__ import annotations

import numpy as np

from mtra.core.errors import AnalysisBlocked
from mtra.core.types import ReasonCode
from mtra.project.manifest import Project, PairSpec

IDENTITY = np.eye(4, dtype=float)


def as_matrix(value: list[list[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected 4x4 transform, got shape {matrix.shape}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("last row of a rigid transform must be [0, 0, 0, 1]")
    return matrix


def invert(matrix: np.ndarray) -> np.ndarray:
    """Invert a rigid transform without a general matrix inverse."""
    m = as_matrix(matrix)
    rotation = m[:3, :3]
    translation = m[:3, 3]
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def compose(*matrices: np.ndarray) -> np.ndarray:
    result = IDENTITY.copy()
    for matrix in matrices:
        result = result @ as_matrix(matrix)
    return result


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a rigid transform to an (N, 3) point array."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    homogenised = np.hstack([pts, np.ones((len(pts), 1))])
    return (as_matrix(matrix) @ homogenised.T).T[:, :3]


def apply_direction(matrix: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate directions (no translation)."""
    return (as_matrix(matrix)[:3, :3] @ np.atleast_2d(np.asarray(vectors, dtype=float)).T).T


def rotation_difference_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angular disagreement between the rotational parts of two transforms.

    Uses the relative rotation's trace rather than comparing matrix entries: two
    matrices that differ element-wise can encode the same rotation.
    """
    rotation_a = as_matrix(a)[:3, :3]
    rotation_b = as_matrix(b)[:3, :3]
    relative = rotation_a.T @ rotation_b
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def resolve_pair_transform(
    project: Project, pair: PairSpec
) -> tuple[np.ndarray, str, bool, list[str]]:
    """Return ``(world_from_mesh, source_label, is_declared, notes)``.

    Raises ``AnalysisBlocked`` when neither a declared transform nor a frame
    identity can justify a mapping. That is the honest outcome: correspondence
    against an unknown frame would produce numbers with no meaning.
    """
    notes: list[str] = []
    mesh_frame = pair.mesh_frame
    robot_frame = pair.robot_frame

    if mesh_frame == robot_frame:
        notes.append(
            f"mesh frame and robot frame are both {mesh_frame!r}; identity transform "
            "assumed because no conversion is required by the manifest"
        )
        return IDENTITY.copy(), f"identity ({mesh_frame} == {robot_frame})", True, notes

    declared = project.spec.frame(mesh_frame)
    if declared is not None and declared.transform is not None:
        matrix = as_matrix(declared.transform)
        notes.append(f"transform for frame {mesh_frame!r} declared in manifest ({declared.source})")
        if robot_frame != "world":
            robot_declared = project.spec.frame(robot_frame)
            if robot_declared is not None and robot_declared.transform is not None:
                matrix = compose(as_matrix(robot_declared.transform), matrix)
                notes.append(f"composed with declared transform for frame {robot_frame!r}")
            else:
                notes.append(
                    f"robot frame {robot_frame!r} has no declared transform; assuming it is "
                    "coincident with world"
                )
        return matrix, f"declared:{mesh_frame}", True, notes

    raise AnalysisBlocked(
        ReasonCode.UNKNOWN_FRAME,
        f"no declared transform maps mesh frame {mesh_frame!r} to robot frame "
        f"{robot_frame!r}; add a frame transform to the manifest. MTRA will not estimate "
        "an unconstrained registration, because a wrong fit is indistinguishable from a "
        "genuine geometric relationship",
    )
