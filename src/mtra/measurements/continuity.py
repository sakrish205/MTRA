"""Group J — path continuity and sequencing (PRD §7, phase 1 items J1–J8).

"Connection method" is decided by what the tool did between passes, not by how far
it travelled: a transfer that stays at painting standoff is an *interpolated* link
(common on surfaces that must not be crossed), while one that lifts clear of the
part is a retract. Clearance is measured against the same surface the rest of the
report uses, so a "lifted" classification is a geometric fact rather than a
threshold applied to a distance.

Direction alternation uses the *signed* 3D travel direction. B-group measurements fold
angles to 0–180° because a pass is a line; alternation is exactly the question of
whether successive passes were driven in opposite directions, so the sign has to
survive — and it has to survive in 3D, because a vertical pass has no XY bearing but
does have a travel direction. Comparing XY bearings would call every pair of vertical
passes "same" no matter which way they went.
"""

from __future__ import annotations

import numpy as np

from mtra.core.geometry import discrete_curvature
from mtra.core.types import ReasonCode, SegmentClass
from mtra.correspondence import frames as frame_ops
from mtra.measurements.catalogue import MeasurementSet, numeric_stats
from mtra.measurements.common import PairView, signed_turn_deg, stroke_axis
from mtra.measurements.regions import stroke_region_map


def _transfer_path(view: PairView, stroke_order: int) -> np.ndarray:
    """Mesh-frame poses the robot occupied between stroke *k* and stroke *k+1*.

    The path is assembled from the segment list rather than from stroke endpoints
    alone, so a retract-then-reposition shows up as the extra travel it really is.
    Motion poses are transformed into the mesh frame here; mixing frames would make
    the clearance measurement meaningless.
    """
    motions = {motion.instruction_index: motion for motion in view.program.motions}
    painting_positions = [
        index
        for index, segment in enumerate(view.segments)
        if segment.segment_class is SegmentClass.PAINTING
    ]
    if stroke_order + 1 >= len(painting_positions):
        return np.zeros((0, 3))

    first_segment = painting_positions[stroke_order]
    second_segment = painting_positions[stroke_order + 1]

    points: list[np.ndarray] = [
        view.stroke_points_mesh(view.painting_strokes[stroke_order])[-1]
    ]
    for segment in view.segments[first_segment + 1 : second_segment]:
        for instruction in segment.ordered_motion_ids:
            motion = motions.get(instruction)
            if motion is not None:
                points.append(
                    frame_ops.apply(
                        view.mesh_from_world, np.asarray(motion.tcp, dtype=float)[None, :]
                    )[0]
                )
    points.append(view.stroke_points_mesh(view.painting_strokes[stroke_order + 1])[0])
    return np.vstack(points)


def measure_continuity(view: PairView, measurements: MeasurementSet) -> None:
    strokes = view.painting_strokes
    if not strokes:
        for parameter_id in ("J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8"):
            measurements.unavailable(
                parameter_id,
                ReasonCode.NO_PAINTING_SEGMENTS,
                "no painting strokes were classified, so there is no sequence to describe",
            )
        return

    mesh_points = [view.stroke_points_mesh(stroke) for stroke in strokes]

    measurements.add(
        "J1",
        {
            "order": [stroke.stroke_id for stroke in strokes],
            "count": int(len(strokes)),
            "first_instruction": int(strokes[0].instruction_ids[0]),
            "last_instruction": int(strokes[-1].instruction_ids[-1]),
        },
        method="stroke order as written in the robot program, with the instruction range covered",
        source_refs=["robot.parse", "traj.sampling"],
    )

    if len(strokes) < 2:
        for parameter_id in ("J2", "J3", "J4", "J5", "J6"):
            measurements.unavailable(
                parameter_id,
                ReasonCode.INSUFFICIENT_STROKES,
                "transition parameters need at least two painting strokes",
            )
    else:
        axes = [stroke_axis(points) for points in mesh_points]
        labels: list[str] = []
        turn_angles: list[float] = []
        for first, second in zip(axes[:-1], axes[1:], strict=True):
            if first is None or second is None:
                # A stroke that returns to its start has no travel direction, so the
                # pair cannot be called same/reversed/cross without inventing one.
                labels.append("undefined")
                continue
            turn = signed_turn_deg(first, second)
            turn_angles.append(turn)
            if turn >= 135.0:
                labels.append("reversed")
            elif turn <= 45.0:
                labels.append("same")
            else:
                labels.append("cross")

        defined = [label for label in labels if label != "undefined"]
        if not defined:
            measurements.unavailable(
                "J2",
                ReasonCode.INSUFFICIENT_STROKES,
                "no consecutive pair of painting strokes both had a defined travel "
                "direction, so neither alternation nor parallelism can be claimed",
            )
        else:
            reversed_fraction = defined.count("reversed") / len(defined)
            same_fraction = defined.count("same") / len(defined)
            if reversed_fraction >= 0.7:
                alternation = "alternating"
            elif same_fraction >= 0.7:
                alternation = "parallel"
            else:
                alternation = "mixed"

            measurements.add(
                "J2",
                {
                    "class": alternation,
                    "reversed_pairs": defined.count("reversed"),
                    "same_pairs": defined.count("same"),
                    "cross_pairs": defined.count("cross"),
                    "undefined_pairs": labels.count("undefined"),
                    "travel_turn_deg": [round(value, 3) for value in turn_angles],
                    "axes_xyz": [
                        None if axis is None else [float(v) for v in axis]
                        for axis in axes
                    ],
                },
                method=(
                    "angle between consecutive strokes' signed 3D travel directions "
                    "(start→end, so the same pass driven the other way reads 180°); "
                    "≥135° reversed, ≤45° same, otherwise crossing. Measured in 3D rather "
                    "than XY because a vertical pass has no bearing but does have a travel "
                    "direction. Class is alternating/parallel when ≥70% of the pairs that "
                    "could be classified agree, else mixed"
                ),
                source_refs=["traj.sampling"],
            )

        direct_limit = float(view.settings.get("continuity.direct_transition_mm", 5.0))
        lifted_limit = float(view.settings.get("continuity.lifted_transition_mm", 50.0))
        baseline = float(
            np.median([record.distance_mm for record in view.correspondence.records])
        ) if view.correspondence.records else 0.0

        gaps: list[float] = []
        path_lengths: list[float] = []
        curvatures: list[float] = []
        classes: list[str] = []
        transitions: list[dict[str, object]] = []

        for order in range(len(strokes) - 1):
            start = mesh_points[order][-1]
            end = mesh_points[order + 1][0]
            gap = float(np.linalg.norm(end - start))
            gaps.append(gap)

            path = _transfer_path(view, order)
            path_length = (
                float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
                if len(path) >= 2
                else gap
            )
            path_lengths.append(path_length)

            if len(path) >= 3:
                curvature = float(np.mean(discrete_curvature(path)))
            else:
                curvature = 0.0
            curvatures.append(curvature)

            peak_clearance = (
                float(np.max(view.surface.sample(path).distances))
                if len(path) >= 1
                else baseline
            )

            if path_length - gap > lifted_limit:
                connection = "lifted"
            elif gap <= direct_limit:
                connection = "direct"
            elif peak_clearance > baseline * 1.5 + direct_limit:
                connection = "lifted"
            else:
                connection = "interpolated"
            classes.append(connection)

            transitions.append(
                {
                    "from": strokes[order].stroke_id,
                    "to": strokes[order + 1].stroke_id,
                    "gap_mm": round(gap, 4),
                    "path_mm": round(path_length, 4),
                    "peak_clearance_mm": round(peak_clearance, 4),
                    "connection": connection,
                    "alternation": labels[order],
                }
            )

        gap_stats = numeric_stats(np.array(gaps), unit_label="mm")
        measurements.add(
            "J3",
            gap_stats,
            method=(
                "straight-line distance from the end of one painting stroke to the start of "
                "the next, in the mesh frame"
            ),
            uncertainty=gap_stats["std"],
            source_refs=["traj.sampling"],
        )

        measurements.add(
            "J4",
            {
                "counts": {
                    label: classes.count(label) for label in ("direct", "interpolated", "lifted")
                },
                "transitions": transitions,
            },
            method=(
                f"direct when the gap is ≤{direct_limit} mm; lifted when the path travels "
                f"more than {lifted_limit} mm further than the gap or clears the surface by "
                "1.5x the painting distance plus the direct limit; otherwise interpolated"
            ),
            source_refs=["traj.segments", "correspondence.projection"],
        )

        if curvatures:
            curvature_stats = numeric_stats(np.array(curvatures), unit_label="1/mm")
            measurements.add(
                "J5",
                curvature_stats,
                method=(
                    "mean discrete curvature along the pose path connecting two passes; "
                    "zero when the passes are directly adjacent"
                ),
                uncertainty=curvature_stats["std"],
                source_refs=["traj.segments"],
            )
        else:
            measurements.unavailable(
                "J5",
                ReasonCode.DEGENERATE_GEOMETRY,
                "no transition path had the three poses needed for a curvature estimate",
            )

        continuity = classes.count("direct") / len(classes) if classes else 0.0
        measurements.add(
            "J6",
            {
                "direct_fraction": round(float(continuity), 4),
                "transitions": int(len(classes)),
                "path_to_gap_ratio": (
                    round(float(np.sum(path_lengths) / np.sum(gaps)), 4)
                    if np.sum(gaps) > 1e-9
                    else None
                ),
            },
            method=(
                "fraction of inter-pass transitions classified direct; the path-to-gap ratio "
                "shows how much extra travel the connection method costs"
            ),
            uncertainty=None,
            source_refs=["traj.segments"],
        )

    region_map = stroke_region_map(view)
    sequence = [region_map.get(stroke.stroke_id, -1) for stroke in strokes]
    compressed: list[int] = []
    for region in sequence:
        if not compressed or compressed[-1] != region:
            compressed.append(region)
    measurements.add(
        "J7",
        {
            "region_sequence": compressed,
            "stroke_region_ids": sequence,
            "region_switch_count": max(len(compressed) - 1, 0),
        },
        method=(
            "region id per stroke using the same clustering threshold as H1, run-length "
            "compressed in execution order; more than one switch means the program "
            "interleaves regions"
        ),
        source_refs=["traj.sampling"],
    )

    first_point = mesh_points[0][0]
    last_point = mesh_points[-1][-1]
    measurements.add(
        "J8",
        {
            "start_point": [float(v) for v in first_point],
            "end_point": [float(v) for v in last_point],
            "start_stroke": strokes[0].stroke_id,
            "end_stroke": strokes[-1].stroke_id,
            "closure_distance_mm": float(np.linalg.norm(last_point - first_point)),
        },
        method="first pose of the first painting stroke and last pose of the last, in the mesh frame",
        location=tuple(float(v) for v in first_point),
        source_refs=["traj.sampling"],
    )
