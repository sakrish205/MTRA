"""Run-level metrics for the report header.

This is deliberately *not* a measurement module: catalogue values belong to
``mtra.measurements`` and must never be recomputed here. What lives here is the
small set of derived summaries a reader needs in order to judge how much of the
input the numbers actually cover — sample counts, pass counts, correspondence
spread — plus one robust summary used by several callers. No value in this module
is ever presented as a catalogue parameter.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from mtra.core.types import MeasurementRecord


def robust_summary(values: Iterable[float] | np.ndarray) -> dict[str, Any] | None:
    """Median-centred summary with spread, extremes and coefficient of variation.

    Returns ``None`` for an empty or entirely non-finite sample so callers can
    report an absence instead of inventing a zero.
    """
    array = np.asarray(values, dtype=float).ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    return {
        "count": int(finite.size),
        "mean": mean,
        "median": float(np.median(finite)),
        "std": std,
        "cv": (std / mean) if mean != 0.0 else None,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }


def coverage_ratio(records: Sequence[MeasurementRecord]) -> float:
    """Fraction of catalogue records that carry a value."""
    if not records:
        return 0.0
    return sum(1 for record in records if record.is_available) / len(records)


def reason_histogram(records: Sequence[MeasurementRecord]) -> dict[str, int]:
    """Unavailable records grouped by reason code — the report's absence table."""
    histogram: dict[str, int] = {}
    for record in records:
        if record.unavailable is None:
            continue
        key = str(record.unavailable.reason)
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: (-item[1], item[0])))


def pair_metrics(pair) -> dict[str, Any]:
    """Coverage and input-size facts for one :class:`pipeline.PairResult`.

    Duck-typed on purpose: importing ``pipeline.run`` here would close an import
    cycle, and this module has no business depending on the pipeline.
    """
    view = getattr(pair, "view", None)
    records = list(pair.measurements)
    metrics: dict[str, Any] = {
        "pair_id": pair.pair_id,
        "status": pair.status,
        "blocked_reason": pair.blocked_reason,
        "parameters": len(records),
        "with_value": sum(1 for record in records if record.is_available),
        "coverage_ratio": coverage_ratio(records),
        "unavailable_by_reason": reason_histogram(records),
    }
    if view is None:
        return metrics

    distances = [record.distance_mm for record in view.correspondence.records]
    metrics.update(
        {
            "mesh_vertices": int(view.surface.record.vertices.shape[0]),
            "mesh_faces": int(view.surface.record.faces.shape[0]),
            "samples": len(view.correspondence.records),
            "strokes": len(view.painting_strokes),
            "segments": len(view.segments),
            "transform_source": view.correspondence.transform_source,
            "curvature_available": bool(view.surface.curvature_available),
            "surface_distance_mm": robust_summary(distances),
        }
    )
    return metrics


def run_metrics(result) -> dict[str, Any]:
    """Aggregate metrics across every pair in one analysis."""
    pairs = [pair_metrics(pair) for pair in result.pairs]
    records = result.all_measurements()
    return {
        "project_id": result.project.project_id,
        "pairs": len(pairs),
        "pairs_analysed": sum(1 for pair in pairs if pair["status"] == "analysed"),
        "pairs_blocked": sum(1 for pair in pairs if pair["status"] == "blocked"),
        "parameters": len(records),
        "with_value": sum(1 for record in records if record.is_available),
        "coverage_ratio": coverage_ratio(records),
        "unavailable_by_reason": reason_histogram(records),
        "per_pair": pairs,
    }
