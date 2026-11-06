"""The report model: measurements in, one renderer-ready structure out.

Two rules live here rather than in the templates:

1. **Accounting is proven before rendering.** ``build_report`` refuses to produce a
   model unless every catalogue id is present for every pair. A missing row is a
   bug, and a report is the wrong place to discover it.
2. **Values are formatted once.** HTML and PDF read the same strings, so the two
   artefacts cannot disagree about what was measured.

Values keep their machine-readable form alongside the rendered text: the text is
for the engineer, ``value_json`` is for whoever re-checks the number.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mtra.core.types import MeasurementRecord
from mtra.measurements.catalogue import jsonable
from mtra.project.inventory import display_path
from mtra.validation.metrics import pair_metrics, run_metrics

_STAT_KEYS = ("count", "mean", "median", "std", "min", "max", "p05", "p95")


# --------------------------------------------------------------------------- #
# Value rendering
# --------------------------------------------------------------------------- #


def _number(value: float) -> str:
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e-3:
        return f"{value:.6g}"
    return f"{value:.3e}"


def _is_stats(value: dict[str, Any]) -> bool:
    return "count" in value and any(key in value for key in ("median", "mean"))


def _stats_text(value: dict[str, Any], unit: str) -> str:
    count = value.get("count")
    headline_key = "median" if "median" in value else "mean"
    headline = value.get(headline_key)
    if not isinstance(headline, (int, float)):
        return json.dumps(jsonable(value), ensure_ascii=False)
    unit_text = f" {unit}" if unit else ""
    text = f"{_number(float(headline))}{unit_text} ({headline_key}, n={count})"
    p05, p95 = value.get("p05"), value.get("p95")
    if isinstance(p05, (int, float)) and isinstance(p95, (int, float)) and p05 != p95:
        text += f", p05-p95 {_number(float(p05))}-{_number(float(p95))}"
    return text


def _vector_text(value: list[Any]) -> str:
    if all(isinstance(item, (int, float)) for item in value):
        return "[" + ", ".join(_number(float(item)) for item in value) + "]"
    return f"list of {len(value)} values"


def format_value(value: Any, unit: str = "") -> str:
    """One human-readable line for a stored value. Never raises."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _number(float(value))
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if _is_stats(value):
            return _stats_text(value, unit)
        parts = []
        for key, item in value.items():
            if isinstance(item, (int, float, np.integer, np.floating)):
                parts.append(f"{key}={_number(float(item))}")
            elif isinstance(item, str):
                parts.append(f"{key}={item}")
            elif isinstance(item, list) and len(item) <= 3:
                parts.append(f"{key}={_vector_text(item)}")
            elif isinstance(item, dict) and _is_stats(item):
                parts.append(f"{key}=[{_stats_text(item, '')}]")
        return ", ".join(parts) if parts else f"{len(value)} fields"
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(row, (list, tuple)) for row in value):
            rows = len(value)
            columns = len(value[0])
            return f"{rows}x{columns} matrix"
        if len(value) <= 6:
            return _vector_text(list(value))
        return f"list of {len(value)} values"
    return str(value)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReportRow:
    parameter_id: str
    group: str
    name: str
    unit: str
    status: str
    value_text: str
    value_json: Any
    reason: str
    detail: str
    method: str
    uncertainty: float | None
    stroke_id: str | None
    region_id: str | None
    source_refs: list[str]

    @property
    def available(self) -> bool:
        return self.value_text != ""


@dataclass(frozen=True)
class GroupSection:
    group: str
    title: str
    rows: list[ReportRow]

    @property
    def available(self) -> int:
        return sum(1 for row in self.rows if row.available)


@dataclass(frozen=True)
class PairReport:
    pair_id: str
    status: str
    blocked_reason: str | None
    coverage: dict[str, int]
    sections: list[GroupSection]
    unavailable: list[ReportRow]
    diagnostics: list[dict[str, Any]]
    metrics: dict[str, Any]
    figures: list[str] = field(default_factory=list)

    @property
    def rows(self) -> list[ReportRow]:
        return [row for section in self.sections for row in section.rows]


@dataclass(frozen=True)
class ReportData:
    project_id: str
    generated_at_utc: str
    mtra_version: str
    settings_summary: dict[str, Any]
    metrics: dict[str, Any]
    pairs: list[PairReport]
    diagnostics: list[dict[str, Any]]
    provenance: dict[str, Any]

    @property
    def catalogue_total(self) -> int:
        return sum(len(pair.rows) for pair in self.pairs)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def _row(record: MeasurementRecord) -> ReportRow:
    available = record.is_available
    return ReportRow(
        parameter_id=record.parameter_id,
        group=record.group,
        name=record.name,
        unit=record.unit,
        status=str(record.status),
        value_text=format_value(record.value, record.unit) if available else "",
        value_json=record.value if available else None,
        reason=str(record.unavailable.reason) if record.unavailable else "",
        detail=record.unavailable.detail if record.unavailable else "",
        method=record.method,
        uncertainty=record.uncertainty,
        stroke_id=record.stroke_id,
        region_id=record.region_id,
        source_refs=list(record.source_refs),
    )


def _sections(catalogue, rows: list[ReportRow]) -> list[GroupSection]:
    by_group: dict[str, list[ReportRow]] = {}
    for row in rows:
        by_group.setdefault(row.group, []).append(row)
    sections: list[GroupSection] = []
    for group in catalogue.groups:
        members = by_group.get(group)
        if members:
            sections.append(
                GroupSection(
                    group=group,
                    title=catalogue.group_name(group),
                    rows=members,
                )
            )
    return sections


def _diagnostics(diagnostics, base: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for diagnostic in diagnostics:
        node = diagnostic.model_dump()
        if node.get("source_path"):
            node["source_path"] = display_path(node["source_path"], base)
        rows.append(node)
    return rows


def build_report(result, *, figures: dict[str, list[str]] | None = None) -> ReportData:
    """Turn an :class:`AnalysisResult` into a renderer-ready model.

    Raises ``ValueError`` when a pair is missing a catalogue id: that is an
    accounting bug, and a report that silently omits a row is worse than a failed
    run.
    """
    figures = figures or {}
    expected = {spec.id for spec in result.catalogue.parameters}
    base = result.project.paths.base_dir
    pairs: list[PairReport] = []

    for pair in result.pairs:
        rows = [_row(record) for record in pair.measurements]
        present = {row.parameter_id for row in rows}
        missing = expected - present
        if missing:
            raise ValueError(
                f"pair {pair.pair_id}: report is missing catalogue ids "
                f"{', '.join(sorted(missing))}"
            )
        duplicates = len(rows) - len(present)
        if duplicates:
            raise ValueError(f"pair {pair.pair_id}: {duplicates} duplicated catalogue rows")

        ordered = sorted(rows, key=lambda row: _catalogue_index(result.catalogue, row.parameter_id))
        pairs.append(
            PairReport(
                pair_id=pair.pair_id,
                status=pair.status,
                blocked_reason=pair.blocked_reason,
                coverage=pair.coverage(),
                sections=_sections(result.catalogue, ordered),
                unavailable=[row for row in ordered if not row.available],
                diagnostics=_diagnostics(pair.diagnostics, base),
                metrics=pair_metrics(pair),
                figures=list(figures.get(pair.pair_id, [])),
            )
        )

    return ReportData(
        project_id=result.project.project_id,
        generated_at_utc=result.provenance.generated_at_utc,
        mtra_version=result.provenance.mtra_version,
        settings_summary=result.settings.data,
        metrics=run_metrics(result),
        pairs=pairs,
        diagnostics=_diagnostics(result.diagnostics, base),
        provenance=jsonable(result.provenance.model_dump()),
    )


def _catalogue_index(catalogue, parameter_id: str) -> int:
    for index, spec in enumerate(catalogue.parameters):
        if spec.id == parameter_id:
            return index
    return len(catalogue.parameters)


# --------------------------------------------------------------------------- #
# Round trip (so `mtra report` can re-render without re-analysing)
# --------------------------------------------------------------------------- #


def write_report_json(report: ReportData, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def load_report_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
