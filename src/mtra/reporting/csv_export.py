"""Flat exports: what a spreadsheet user needs without reading the model.

These files are conveniences, not the evidence. The authoritative artefacts are
``report.json`` (the exact model both documents were rendered from) and the HTML
report itself. Column names are stable; values are never rounded here.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

from mtra.core.types import MeasurementRecord

MEASUREMENT_COLUMNS = (
    "pair_id",
    "parameter_id",
    "group",
    "name",
    "status",
    "value",
    "unit",
    "reason",
    "detail",
    "method",
    "uncertainty",
    "stroke_id",
    "region_id",
    "source_refs",
)

STROKE_COLUMNS = (
    "pair_id",
    "stroke_id",
    "segment_id",
    "samples",
    "instruction_first",
    "instruction_last",
    "start_x",
    "start_y",
    "start_z",
    "end_x",
    "end_y",
    "end_z",
)


def write_measurements_csv(
    records: Sequence[MeasurementRecord], path: str | Path
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MEASUREMENT_COLUMNS))
        writer.writeheader()
        for record in records:
            row = record.as_row()
            row["pair_id"] = record.pair_id
            writer.writerow({column: row.get(column, "") for column in MEASUREMENT_COLUMNS})
    return target


def _stroke_row(pair_id: str, stroke) -> dict[str, Any]:
    points = stroke.points
    instructions: Iterable[int] = stroke.instruction_ids
    ordered = list(instructions)
    return {
        "pair_id": pair_id,
        "stroke_id": stroke.stroke_id,
        "segment_id": stroke.segment_id,
        "samples": len(points),
        "instruction_first": ordered[0] if ordered else "",
        "instruction_last": ordered[-1] if ordered else "",
        "start_x": float(points[0][0]) if len(points) else "",
        "start_y": float(points[0][1]) if len(points) else "",
        "start_z": float(points[0][2]) if len(points) else "",
        "end_x": float(points[-1][0]) if len(points) else "",
        "end_y": float(points[-1][1]) if len(points) else "",
        "end_z": float(points[-1][2]) if len(points) else "",
    }


def write_strokes_csv(result, path: str | Path, *, frame: str = "robot") -> Path | None:
    """Write one row per painted pass. Returns ``None`` when there is nothing to write."""
    rows: list[dict[str, Any]] = []
    for pair in result.pairs:
        view = getattr(pair, "view", None)
        if view is None:
            continue
        for stroke in view.painting_strokes:
            row = _stroke_row(pair.pair_id, stroke)
            if frame == "mesh":
                points = view.stroke_points_mesh(stroke)
                if len(points):
                    row.update(
                        {
                            "start_x": float(points[0][0]),
                            "start_y": float(points[0][1]),
                            "start_z": float(points[0][2]),
                            "end_x": float(points[-1][0]),
                            "end_y": float(points[-1][1]),
                            "end_z": float(points[-1][2]),
                        }
                    )
            rows.append(row)

    if not rows:
        return None

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(STROKE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return target
