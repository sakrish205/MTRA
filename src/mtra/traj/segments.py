"""Movement classification: painting, approach, departure, transfer, unknown.

The rule that matters most here is the one that says "don't guess". PRD §5.2: do
not infer paint on/off from geometric motion, and if process state is absent or
ambiguous, classify as uncertain and expose it.

Implementation consequence: an interval counts as painting only when the
configured process signal is on at *both* of its endpoints. Checking one endpoint
would sweep the traverse after switch-off into the last stroke, which inflates
stroke length and adds a phantom pass to the pitch statistics.
"""

from __future__ import annotations

import numpy as np

from mtra.config import Settings
from mtra.core.types import (
    Diagnostic,
    EvidenceStatus,
    MotionType,
    SegmentClass,
    Severity,
    TrajectorySegment,
)
from mtra.robot.common_schema import ParsedProgram

_LINEAR = {MotionType.LINEAR, MotionType.CIRCULAR}


def _length(points: list[np.ndarray]) -> float:
    if len(points) < 2:
        return 0.0
    return float(sum(np.linalg.norm(b - a) for a, b in zip(points[:-1], points[1:], strict=True)))


def _interval_class(
    program: ParsedProgram, index: int
) -> tuple[SegmentClass, float, list[str]]:
    """Classify the interval between motion ``index`` and ``index + 1``."""
    start_state = program.process_states[index]
    end_state = program.process_states[index + 1]
    motion = program.motions[index]

    if start_state.paint_on is None or end_state.paint_on is None:
        return (
            SegmentClass.UNKNOWN,
            0.0,
            ["process state unresolved; PRD §5.2 forbids inferring paint state from motion"],
        )

    if start_state.paint_on and end_state.paint_on:
        if motion.motion_type in _LINEAR:
            return (
                SegmentClass.PAINTING,
                1.0,
                [
                    f"{start_state.signal} on at instruction {motion.instruction_index} "
                    f"and at instruction {program.motions[index + 1].instruction_index}",
                    f"motion type {motion.motion_type}",
                ],
            )
        return (
            SegmentClass.UNKNOWN,
            0.3,
            [
                f"process is on but motion type is {motion.motion_type}; a joint move with "
                "the process on is not a surface pass and is left unresolved"
            ],
        )

    return (SegmentClass.TRANSFER, 1.0, [f"{start_state.signal} not on throughout the interval"])


def classify(
    program: ParsedProgram,
    settings: Settings,
    *,
    min_stroke_length_mm: float | None = None,
) -> tuple[list[TrajectorySegment], list[Diagnostic]]:
    """Segment a program into classified trajectory runs.

    Returns segments plus diagnostics; a program with no resolvable process state
    yields one UNKNOWN segment over the whole trajectory rather than a guess.
    """
    diagnostics: list[Diagnostic] = []
    if len(program.motions) < 2:
        diagnostics.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="segments.too_few_motions",
                message="fewer than two motions; no trajectory intervals exist",
                stage="traj.segments",
                source_path=program.source_path,
            )
        )
        return [], diagnostics

    if min_stroke_length_mm is None:
        min_stroke_length_mm = float(settings.get("trajectory.min_stroke_length_mm", 1.0))

    intervals: list[tuple[SegmentClass, float, list[str]]] = [
        _interval_class(program, index) for index in range(len(program.motions) - 1)
    ]

    # Group consecutive intervals of the same class into runs.
    runs: list[tuple[SegmentClass, int, int, float, list[str]]] = []
    start = 0
    for index in range(1, len(intervals) + 1):
        if index == len(intervals) or intervals[index][0] != intervals[start][0]:
            segment_class, confidence, evidence = intervals[start]
            runs.append((segment_class, start, index, confidence, evidence))
            start = index

    # Painting runs are strokes; off runs take their role from what they connect.
    painting_runs = [run for run in runs if run[0] is SegmentClass.PAINTING]

    segments: list[TrajectorySegment] = []
    for run_index, (segment_class, first, last, confidence, evidence) in enumerate(runs):
        motion_ids = list(range(first, last + 1))  # interval `last-1` ends at motion `last`
        points = [program.motions[i].tcp for i in motion_ids]
        length = _length(points)

        if segment_class is SegmentClass.TRANSFER:
            touches_previous_painting = any(
                run[0] is SegmentClass.PAINTING and run[2] == first for run in painting_runs
            )
            touches_next_painting = any(
                run[0] is SegmentClass.PAINTING and run[1] == last for run in painting_runs
            )
            if touches_previous_painting and touches_next_painting:
                segment_class = SegmentClass.TRANSFER
                evidence = [*evidence, "connectivity: joins two painting strokes"]
            elif touches_next_painting:
                segment_class = SegmentClass.APPROACH
                evidence = [*evidence, "connectivity: ends where a painting stroke begins"]
                confidence = 0.8
            elif touches_previous_painting:
                segment_class = SegmentClass.DEPARTURE
                evidence = [*evidence, "connectivity: begins where a painting stroke ends"]
                confidence = 0.8

        if segment_class is SegmentClass.PAINTING and length < min_stroke_length_mm:
            evidence = [
                *evidence,
                f"length {length:.3f} mm below minimum passing length "
                f"{min_stroke_length_mm} mm — treated as punctuation, not a pass",
            ]

        segments.append(
            TrajectorySegment(
                segment_id=f"{program.pair_id}:S{run_index:04d}",
                pair_id=program.pair_id,
                segment_class=segment_class,
                ordered_motion_ids=[program.motions[i].instruction_index for i in motion_ids],
                start_pose=tuple(float(v) for v in points[0]),
                end_pose=tuple(float(v) for v in points[-1]),
                length_mm=length,
                confidence=float(confidence),
                classification_evidence=evidence,
                status=(
                    EvidenceStatus.OBSERVED
                    if segment_class is not SegmentClass.UNKNOWN
                    else EvidenceStatus.UNRESOLVED
                ),
            )
        )

    unknown_count = sum(1 for segment in segments if segment.segment_class is SegmentClass.UNKNOWN)
    if unknown_count and not painting_runs:
        diagnostics.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="segments.no_painting",
                message=(
                    "no interval could be classified as painting; downstream painting "
                    "measurements will be reported as unavailable rather than approximated "
                    "from all motion"
                ),
                stage="traj.segments",
                source_path=program.source_path,
            )
        )
    return segments, diagnostics


def painting_segments(segments: list[TrajectorySegment]) -> list[TrajectorySegment]:
    return [s for s in segments if s.segment_class is SegmentClass.PAINTING]
