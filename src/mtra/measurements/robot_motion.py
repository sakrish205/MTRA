"""Group K — how the robot trajectory is represented (PRD §7, phase 1 items K1–K9).

These are properties of the *program*, not of the part: sampling interval, curve
representation, fitting error. They are measured separately from the geometric
groups because controller sampling is a confound — a 1 mm sample interval and a
25 mm sample interval describe the same raster, and any spacing "rule" derived
from sample density would be an artefact of the post-processor.

K7 is deliberately about representation loss, not about surface deviation: it asks
how far a straight-line interpolation between programmed poses departs from a
smooth curve through the same poses. On a raster's straight passes it is
numerically zero, which is the correct answer.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from mtra.core.types import MotionType, ReasonCode, SegmentClass
from mtra.measurements.catalogue import MeasurementSet, numeric_stats
from mtra.measurements.common import PairView, unit_step_interval


def _curve_fitting_error(points: np.ndarray) -> float:
    """Max chord-to-spline deviation, in millimetres."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    if cumulative[-1] <= 1e-9:
        return 0.0
    degree = min(3, len(pts) - 1)
    spline = CubicSpline(cumulative, pts, axis=0, bc_type="natural")
    midpoints = 0.5 * (cumulative[:-1] + cumulative[1:])
    smooth = spline(midpoints)
    straight = 0.5 * (pts[:-1] + pts[1:])
    return float(np.max(np.linalg.norm(smooth - straight, axis=1)))


def measure_robot_motion(view: PairView, measurements: MeasurementSet) -> None:
    motions = view.program.motions
    if not motions:
        for parameter_id in ("K1", "K2", "K3", "K4", "K5", "K6", "K7"):
            measurements.unavailable(
                parameter_id,
                ReasonCode.INSUFFICIENT_STROKES,
                "the program produced no motion instructions",
            )
    else:
        positions = np.vstack([motion.tcp for motion in motions])
        measurements.add(
            "K1",
            {
                "instruction_count": int(len(motions)),
                "bbox_min": [float(v) for v in positions.min(axis=0)],
                "bbox_max": [float(v) for v in positions.max(axis=0)],
                "centroid": [float(v) for v in positions.mean(axis=0)],
                "frame": view.program.user_frame or "unspecified",
            },
            method=(
                "TCP positions exactly as written in the program, in the program's user "
                "frame; no transform is applied here so the values match the controller"
            ),
            source_refs=["robot.parse"],
        )

        angles = np.array([motion.tcp_orientation for motion in motions], dtype=float)
        axes = np.vstack([motion.tool_axis for motion in motions])
        measurements.add(
            "K2",
            {
                "wpr_min": [float(v) for v in angles.min(axis=0)],
                "wpr_max": [float(v) for v in angles.max(axis=0)],
                "mean_tool_axis": [float(v) for v in axes.mean(axis=0)],
                "frame": view.program.tool_frame or "unspecified",
            },
            method=(
                "raw W/P/R triples as written, plus the mean of the derived tool +Z axis; "
                "the Euler convention (rz_ry_rx) is declared in config/defaults.yaml"
            ),
            source_refs=["robot.parse"],
        )

        intervals = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        if intervals.size:
            stats = numeric_stats(intervals, unit_label="mm")
            measurements.add(
                "K3",
                stats,
                method="distance between consecutive programmed TCP poses",
                uncertainty=stats["std"],
                source_refs=["robot.parse"],
            )

            mean_interval = float(np.mean(intervals))
            measurements.add(
                "K4",
                round(1.0 / mean_interval, 6) if mean_interval > 1e-12 else 0.0,
                method="1 / mean programmed pose interval (poses per millimetre)",
                source_refs=["robot.parse"],
            )

            low = float(np.percentile(intervals, 5))
            high = float(np.percentile(intervals, 95))
            ratio = high / low if low > 1e-9 else float("inf")
            threshold = float(view.settings.get("trajectory.adaptive_sampling_ratio", 2.0))
            measurements.add(
                "K5",
                {
                    "p95_over_p05": round(ratio, 4) if np.isfinite(ratio) else None,
                    "class": "adaptive" if ratio > threshold else "uniform",
                    "threshold": threshold,
                },
                method=(
                    "ratio of the 95th to 5th percentile pose interval; above the "
                    "configured ratio the sampling adapts to the path rather than stepping "
                    "at a fixed distance"
                ),
                source_refs=["robot.parse"],
            )
        else:
            for parameter_id in ("K3", "K4", "K5"):
                measurements.unavailable(
                    parameter_id,
                    ReasonCode.INSUFFICIENT_STROKES,
                    "a single pose has no interval to measure",
                )

        motion_types = {motion.motion_type for motion in motions}
        if motion_types == {MotionType.LINEAR}:
            representation = "linear_polyline"
        elif MotionType.CIRCULAR in motion_types:
            representation = "linear_and_circular"
        elif motion_types == {MotionType.JOINT}:
            representation = "joint_only"
        else:
            representation = "mixed"
        measurements.add(
            "K6",
            {
                "class": representation,
                "motion_type_counts": {
                    str(kind): sum(1 for m in motions if m.motion_type is kind)
                    for kind in sorted(motion_types, key=lambda value: str(value))
                },
            },
            method="classification of the motion instruction types present in the program",
            source_refs=["robot.parse"],
        )

        errors = [
            _curve_fitting_error(view.stroke_points_mesh(stroke))
            for stroke in view.painting_strokes
            if len(stroke.raw_points) >= 2
        ]
        if errors:
            stats = numeric_stats(np.array(errors), unit_label="mm")
            pose_counts = [len(stroke.raw_points) for stroke in view.painting_strokes]
            measurements.add(
                "K7",
                {
                    **stats,
                    "poses_per_stroke_max": int(max(pose_counts)) if pose_counts else 0,
                    "poses_per_stroke_mean": (
                        round(float(np.mean(pose_counts)), 3) if pose_counts else 0.0
                    ),
                },
                method=(
                    "max deviation between the straight-line chord of each programmed pose "
                    "segment and a cubic spline through the same poses, evaluated at the "
                    "segment midpoint; measures representation loss, not surface error. A "
                    "stroke written as a single linear segment between two poses is the "
                    "chord, so 0 here means there was nothing to lose — check "
                    "poses_per_stroke_max before reading it as a smooth path"
                ),
                uncertainty=stats["std"],
                source_refs=["traj.sampling"],
            )
        else:
            measurements.unavailable(
                "K7",
                ReasonCode.INSUFFICIENT_STROKES,
                "no painting stroke carried at least two programmed poses, so there is no "
                "chord to compare against a curve",
            )

    counts = {segment.segment_class.value: 0 for segment in view.segments}
    for segment in view.segments:
        counts[segment.segment_class.value] = counts.get(segment.segment_class.value, 0) + 1
    if view.segments:
        measurements.add(
            "K8",
            {
                "counts": counts,
                "painting_length_mm": round(
                    float(
                        sum(
                            segment.length_mm
                            for segment in view.segments
                            if segment.segment_class is SegmentClass.PAINTING
                        )
                    ),
                    4,
                ),
                "total_length_mm": round(
                    float(sum(segment.length_mm for segment in view.segments)), 4
                ),
            },
            method=(
                "segment classification from process-signal state at both interval "
                "endpoints; adjacent runs are merged before counting"
            ),
            source_refs=["traj.segments"],
        )
    else:
        measurements.unavailable(
            "K8",
            ReasonCode.INSUFFICIENT_STROKES,
            "no trajectory intervals could be formed",
        )

    states = view.program.process_states
    if not states:
        measurements.unavailable(
            "K9",
            ReasonCode.NO_PROCESS_STATE,
            "the parser produced no per-motion process state",
        )
    else:
        on_count = sum(1 for state in states if state.paint_on is True)
        off_count = sum(1 for state in states if state.paint_on is False)
        unknown_count = sum(1 for state in states if state.paint_on is None)
        signal = next((state.signal for state in states if state.signal), None)
        measurements.add(
            "K9",
            {
                "signal": signal,
                "on_count": on_count,
                "off_count": off_count,
                "unknown_count": unknown_count,
                "resolved_fraction": round((on_count + off_count) / len(states), 4),
            },
            method=(
                "per-motion process state resolved from the configured paint signal in "
                "instruction order; never inferred from motion geometry"
            ),
            source_refs=["robot.parse"],
        )
