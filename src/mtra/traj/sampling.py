"""Stroke extraction and arc-length resampling.

A stroke is a maximal run of painting intervals, resampled at a fixed arc-length
step. Resampling exists so that pitch and direction describe the trajectory the
engineer produced, not the controller's point spacing — PRD K3/K4 measure sampling
separately, precisely so it is not mistaken for path structure.

Tool axes travel with the points by nearest raw sample, not by interpolating
rotation matrices: averaging two rotations is not a rotation, and a silently
non-orthonormal "axis" would corrupt every orientation measurement downstream.
The nearest-sample choice is recorded in the measurement method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mtra.core.geometry import unit
from mtra.core.types import SegmentClass, TrajectorySegment
from mtra.robot.common_schema import ParsedProgram


@dataclass
class Stroke:
    stroke_id: str
    segment_id: str
    points: np.ndarray
    raw_points: np.ndarray
    tool_axes: np.ndarray
    instruction_ids: list[int]
    tcp_positions: np.ndarray

    @property
    def length_mm(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=1)))

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]


def _resample_indices(cumulative: np.ndarray, targets: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(cumulative, targets, side="right") - 1
    return np.clip(indices, 0, len(cumulative) - 1)


def resample_with_axes(
    points: np.ndarray, tool_axes: np.ndarray, step_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Arc-length resample points, carrying the nearest raw tool axis."""
    pts = np.asarray(points, dtype=float)
    axes = np.asarray(tool_axes, dtype=float)
    if len(pts) < 2 or step_mm <= 0:
        return pts, axes

    segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(cumulative[-1])
    if total <= 0:
        return pts, axes

    count = max(2, int(np.ceil(total / step_mm)) + 1)
    targets = np.linspace(0.0, total, count)
    resampled = np.empty((count, 3), dtype=float)
    for axis in range(3):
        resampled[:, axis] = np.interp(targets, cumulative, pts[:, axis])

    raw_indices = _resample_indices(cumulative, targets)
    resampled_axes = np.array([unit(axes[i]) for i in raw_indices])
    return resampled, resampled_axes


def extract_strokes(
    program: ParsedProgram,
    segments: list[TrajectorySegment],
    *,
    step_mm: float,
) -> list[Stroke]:
    motions_by_index = {motion.instruction_index: motion for motion in program.motions}
    strokes: list[Stroke] = []
    for segment in segments:
        if segment.segment_class is not SegmentClass.PAINTING:
            continue
        motions = [
            motions_by_index[instruction]
            for instruction in segment.ordered_motion_ids
            if instruction in motions_by_index
        ]
        if len(motions) < 2:
            continue
        raw_points = np.vstack([motion.tcp for motion in motions])
        raw_axes = np.vstack([motion.tool_axis for motion in motions])
        points, axes = resample_with_axes(raw_points, raw_axes, step_mm)
        strokes.append(
            Stroke(
                stroke_id=f"{segment.pair_id}:K{len(strokes):03d}",
                segment_id=segment.segment_id,
                points=points,
                raw_points=raw_points,
                tool_axes=axes,
                instruction_ids=[motion.instruction_index for motion in motions],
                tcp_positions=raw_points,
            )
        )
    return strokes
