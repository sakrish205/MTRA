"""Catalogue accounting: the mechanism behind the report's central promise.

PRD §7: every one of the 112 parameters must appear with a value *or* a reason
code. That is an accounting invariant, not a formatting rule, so it is enforced
here rather than in the template:

* a measurement module asks the catalogue for a parameter id; the id, group and
  name cannot drift from ``config/parameters.yaml``;
* ``finalise`` sweeps the catalogue and emits an explicit record for anything
  still missing — ``out_of_phase_scope`` for later phases, and a loud
  ``computation_failed`` for a phase-1 parameter no module produced, which turns
  an implementation gap into a visible finding instead of a blank cell.

The second case matters: silently omitting a parameter is exactly how a report
starts lying.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mtra.config import Catalogue, ParameterSpec
from mtra.core.types import (
    EvidenceStatus,
    MeasurementRecord,
    ReasonCode,
    Unavailable,
)


def jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays so artefacts serialise cleanly."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


class MeasurementSet:
    """Collects measurements for one pair and proves the catalogue is covered."""

    def __init__(self, catalogue: Catalogue, pair_id: str) -> None:
        self._catalogue = catalogue
        self._pair_id = pair_id
        self._records: list[MeasurementRecord] = []
        self._emitted: set[str] = set()

    # -- emission ---------------------------------------------------------- #

    def _spec(self, parameter_id: str) -> ParameterSpec:
        try:
            return self._catalogue.by_id(parameter_id)
        except KeyError as exc:  # pragma: no cover - config error path
            raise KeyError(f"parameter {parameter_id} is not in the catalogue") from exc

    def add(
        self,
        parameter_id: str,
        value: Any,
        *,
        method: str,
        unit: str | None = None,
        status: EvidenceStatus = EvidenceStatus.OBSERVED,
        uncertainty: float | None = None,
        region_id: str | None = None,
        stroke_id: str | None = None,
        location: tuple[float, float, float] | None = None,
        source_refs: list[str] | None = None,
    ) -> None:
        spec = self._spec(parameter_id)
        self._records.append(
            MeasurementRecord(
                parameter_id=spec.id,
                group=spec.group,
                name=spec.name,
                pair_id=self._pair_id,
                value=jsonable(value),
                unit=spec.unit if unit is None else unit,
                method=method,
                region_id=region_id,
                stroke_id=stroke_id,
                location=location,
                status=status,
                uncertainty=uncertainty,
                source_refs=source_refs or [],
            )
        )
        self._emitted.add(spec.id)

    def unavailable(
        self,
        parameter_id: str,
        reason: ReasonCode,
        detail: str,
        *,
        method: str = "",
        status: EvidenceStatus = EvidenceStatus.UNRESOLVED,
        source_refs: list[str] | None = None,
    ) -> None:
        spec = self._spec(parameter_id)
        self._records.append(
            MeasurementRecord(
                parameter_id=spec.id,
                group=spec.group,
                name=spec.name,
                pair_id=self._pair_id,
                unavailable=Unavailable(reason=reason, detail=detail),
                method=method,
                status=status,
                source_refs=source_refs or [],
            )
        )
        self._emitted.add(spec.id)

    def has(self, parameter_id: str) -> bool:
        return parameter_id in self._emitted

    # -- accounting -------------------------------------------------------- #

    def finalise(self) -> list[MeasurementRecord]:
        """Emit a record for every catalogue entry not already accounted for."""
        for spec in self._catalogue.parameters:
            if spec.id in self._emitted:
                continue
            if spec.phase > 1:
                self.unavailable(
                    spec.id,
                    ReasonCode.OUT_OF_PHASE_SCOPE,
                    f"assigned to phase {spec.phase}; this build reports it so the gap "
                    "stays visible rather than disappearing from the table",
                )
            else:
                self.unavailable(
                    spec.id,
                    ReasonCode.COMPUTATION_FAILED,
                    "no phase-1 measurement module produced this parameter for this input; "
                    "treated as an implementation gap, not as a valid absence",
                )
        return self._records

    # -- views -------------------------------------------------------------- #

    @property
    def records(self) -> list[MeasurementRecord]:
        return list(self._records)

    def by_group(self) -> dict[str, list[MeasurementRecord]]:
        grouped: dict[str, list[MeasurementRecord]] = {}
        for record in self._records:
            grouped.setdefault(record.group, []).append(record)
        return grouped

    def coverage(self) -> dict[str, int]:
        total = len(self._catalogue)
        available = sum(1 for record in self._records if record.is_available)
        return {
            "catalogue_total": total,
            "accounted": len(self._records),
            "with_value": available,
            "without_value": len(self._records) - available,
            "missing": total - len({r.parameter_id for r in self._records}),
        }


def numeric_stats(values: np.ndarray, *, unit_label: str = "") -> dict[str, Any]:
    """Distribution summary used wherever a parameter is a sample set.

    PRD §8.1 forbids reporting a bare mean for a distribution, so every
    distributional parameter carries spread and extremes.
    """
    array = np.asarray(values, dtype=float).ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0, "unit": unit_label}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
        "unit": unit_label,
    }
