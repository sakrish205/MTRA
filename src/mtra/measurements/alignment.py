"""Group A — coordinate alignment and correspondence (PRD §7).

These six parameters establish whether the rest of the report is even meaningful:
if the transform between mesh and robot frames is wrong, every angle and spacing
below it is wrong in a way that still looks plausible.

They are also where the TCP/impact-point separation is first enforced. A3 is a
nearest-surface distance from the TCP; A4 is a ray intersection along the tool
axis; A5 is the disagreement between them. Reporting A3 as "standoff" would be the
single most common way to invent a spacing rule that does not exist (PRD §19).
"""

from __future__ import annotations

import numpy as np

from mtra.core.types import EvidenceStatus, ReasonCode
from mtra.measurements.catalogue import MeasurementSet, numeric_stats
from mtra.measurements.common import PairView


def _surface_distance_stats(view: PairView) -> dict[str, float] | None:
    distances = np.array(
        [r.distance_mm for r in view.correspondence.records], dtype=float
    )
    if distances.size == 0:
        return None
    return numeric_stats(distances, unit_label="mm")


def _impact_offsets(view: PairView) -> np.ndarray:
    """Distance between the nearest surface point and the spray-axis impact point."""
    offsets: list[float] = []
    for record in view.correspondence.records:
        if record.impact_point is None:
            continue
        offsets.append(
            float(np.linalg.norm(np.asarray(record.impact_point) - np.asarray(record.closest_point)))
        )
    return np.array(offsets, dtype=float)


def measure_alignment(view: PairView, measurements: MeasurementSet) -> None:
    transform = np.asarray(view.world_from_mesh, dtype=float)
    measurements.add(
        "A1",
        transform.tolist(),
        method=(
            "world_from_mesh rigid transform from the manifest "
            f"(source: {view.correspondence.transform_source})"
        ),
        source_refs=[view.correspondence.transform_source],
    )

    measurements.unavailable(
        "A2",
        ReasonCode.NO_INDEPENDENT_MEASUREMENT,
        "the transform is declared in the manifest, so there is no fitted residual to "
        "report; a registration error requires an independent fiducial or fit, which "
        "Phase 1 deliberately does not perform (PRD §6 stage 5)",
        method="declared transform: no residual exists",
        status=EvidenceStatus.OBSERVED,
    )

    distance_stats = _surface_distance_stats(view)
    if distance_stats is None:
        measurements.unavailable(
            "A3",
            ReasonCode.NO_SURFACE_CORRESPONDENCE,
            "no trajectory samples were corresponded to the mesh",
        )
    elif not any(r.ray_hit_count for r in view.correspondence.records):
        # Still a real measurement — it is the nearest-surface distance, which is
        # exactly what the parameter names. The tool-axis warning belongs to I5.
        measurements.add(
            "A3",
            distance_stats,
            method=(
                "nearest point on mesh from each resampled TCP sample; this is NOT "
                "standoff — no tool-axis intersection was available for any sample "
                "(see I5)"
            ),
            uncertainty=distance_stats["std"],
            source_refs=["correspondence.projection"],
        )
    else:
        measurements.add(
            "A3",
            distance_stats,
            method="nearest point on mesh from each resampled TCP sample (standoff is I5)",
            uncertainty=distance_stats["std"],
            source_refs=["correspondence.projection"],
        )

    impacts = [r for r in view.correspondence.records if r.impact_point is not None]
    total = len(view.correspondence.records)
    if not impacts:
        measurements.unavailable(
            "A4",
            ReasonCode.NO_RAY_INTERSECTION,
            f"none of the {total} tool axes intersect the mesh; check tool axis, frame "
            "and whether the tool points at the part",
        )
    else:
        centroids = np.array([r.impact_point for r in impacts], dtype=float)
        measurements.add(
            "A4",
            {
                "samples": total,
                "intersections": len(impacts),
                "fraction": round(len(impacts) / total, 4),
                "multiple_hit_samples": int(
                    sum(1 for r in impacts if r.ray_hit_count > 1)
                ),
                "centroid": [float(v) for v in centroids.mean(axis=0)],
                "bbox_min": [float(v) for v in centroids.min(axis=0)],
                "bbox_max": [float(v) for v in centroids.max(axis=0)],
            },
            method="nearest forward intersection of the tool +Z axis with the mesh",
            status=EvidenceStatus.OBSERVED,
            source_refs=["correspondence.spray_axis"],
        )

    offsets = _impact_offsets(view)
    if offsets.size == 0:
        measurements.unavailable(
            "A5",
            ReasonCode.NO_RAY_INTERSECTION,
            "surface projection error needs both a nearest point and an axis impact; "
            "one of them is unavailable",
        )
    else:
        stats = numeric_stats(offsets, unit_label="mm")
        measurements.add(
            "A5",
            stats,
            method=(
                "distance between the nearest surface point and the tool-axis impact "
                "point for the same sample; nonzero where the axis is not normal to a "
                "curved surface"
            ),
            uncertainty=stats["std"],
            source_refs=["correspondence.projection"],
        )

    ranges = view.correspondence.stroke_ranges
    measurements.add(
        "A6",
        {
            "samples": len(view.correspondence.records),
            "strokes": len(ranges),
            "mesh_face_contacts": int(
                len({r.mesh_face_id for r in view.correspondence.records})
            ),
            "transform_source": view.correspondence.transform_source,
            "max_distance_mm": (
                float(max(r.distance_mm for r in view.correspondence.records))
                if view.correspondence.records
                else None
            ),
        },
        method="one-to-one mapping from arc-length resampled trajectory samples to mesh faces",
        source_refs=["correspondence.projection"],
    )
