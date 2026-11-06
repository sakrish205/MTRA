"""The analysis pipeline: stages 1–8 of the PRD, in one auditable order.

Every stage is a pure function of the previous stage's records. Nothing is cached
across runs, and nothing is re-read from disk after ingestion — which is what makes
"the report is a function of these file hashes" a true statement rather than an
aspiration.

Failure policy, deliberately uniform:

* a *pair-level* precondition failure (unknown units, missing frame, unsupported
  dialect) blocks that pair and emits the whole catalogue as unavailable with the
  blocking reason. The report still accounts for all 112 parameters, so a blocked
  run is informative instead of empty;
* a *stage-level* failure inside measurement is caught per group and attributed to
  that group, so one broken measurement cannot erase the others.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mtra import __version__
from mtra.config import Catalogue, Settings, load_catalogue, load_settings
from mtra.core.errors import AnalysisBlocked, InputError, MTRAError, UnsupportedInput
from mtra.core.types import (
    Diagnostic,
    EvidenceStatus,
    MeasurementRecord,
    Provenance,
    ReasonCode,
    Severity,
)
from mtra.correspondence import frames as frame_ops
from mtra.correspondence.projection import build_correspondence
from mtra.measurements import PairView, measure_pair
from mtra.measurements.catalogue import MeasurementSet, jsonable
from mtra.mesh.features import SurfaceField
from mtra.mesh.loader import load_mesh
from mtra.project.inventory import inventory_file
from mtra.project.manifest import PairSpec, Project, load_manifest
from mtra.project.validation import validate_project
from mtra.robot.registry import parse_program
from mtra.traj.sampling import extract_strokes
from mtra.traj.segments import classify

_PACKAGES = ("numpy", "scipy", "trimesh", "pandas", "pydantic", "pyyaml", "jinja2", "reportlab")


@dataclass
class PairResult:
    pair_id: str
    status: str  # "analysed" | "blocked"
    measurements: list[MeasurementRecord] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    view: PairView | None = None
    blocked_reason: str | None = None

    def coverage(self) -> dict[str, int]:
        available = sum(1 for record in self.measurements if record.is_available)
        return {
            "parameters": len(self.measurements),
            "with_value": available,
            "without_value": len(self.measurements) - available,
        }


@dataclass
class AnalysisResult:
    project: Project
    provenance: Provenance
    pairs: list[PairResult]
    diagnostics: list[Diagnostic]
    settings: Settings
    catalogue: Catalogue

    def all_measurements(self) -> list[MeasurementRecord]:
        rows: list[MeasurementRecord] = []
        for pair in self.pairs:
            rows.extend(pair.measurements)
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project.project_id,
            "provenance": jsonable(self.provenance.model_dump()),
            "diagnostics": [diagnostic.model_dump() for diagnostic in self.diagnostics],
            "pairs": [
                {
                    "pair_id": pair.pair_id,
                    "status": pair.status,
                    "blocked_reason": pair.blocked_reason,
                    "coverage": pair.coverage(),
                    "diagnostics": [d.model_dump() for d in pair.diagnostics],
                    "measurements": [
                        jsonable(record.as_row()) for record in pair.measurements
                    ],
                }
                for pair in self.pairs
            ],
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _blocked_pair(
    catalogue: Catalogue, pair_id: str, reason: ReasonCode, detail: str
) -> PairResult:
    """Emit the whole catalogue as unavailable, with one blocking reason."""
    set_ = MeasurementSet(catalogue, pair_id)
    for spec in catalogue.parameters:
        set_.unavailable(spec.id, reason, detail, status=EvidenceStatus.UNRESOLVED)
    diagnostics = [
        Diagnostic(
            severity=Severity.ERROR,
            code=f"analysis.blocked.{reason}",
            message=detail,
            stage="pipeline.run",
        )
    ]
    return PairResult(
        pair_id=pair_id,
        status="blocked",
        measurements=set_.records,
        diagnostics=diagnostics,
        blocked_reason=str(reason),
    )


def _package_versions() -> dict[str, str]:
    from importlib import metadata

    versions: dict[str, str] = {"mtra": __version__}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except Exception:  # pragma: no cover - not installed
            versions[name] = "unavailable"
    for optional in ("libigl", "pygeodesic", "pyvista", "vtk"):
        try:
            versions[optional] = metadata.version(optional)
        except Exception:  # pragma: no cover - optional
            versions[optional] = "not installed"
    return versions


def _tool_axis(pair: PairSpec) -> np.ndarray | None:
    if pair.tool is None or pair.tool.tool_axis is None:
        return None
    axis = np.asarray(pair.tool.tool_axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return None
    return axis / norm


def _analyse_pair(
    project: Project, pair: PairSpec, settings: Settings, catalogue: Catalogue
) -> PairResult:
    spec = project.spec
    unit = spec.unit_for(pair)
    diagnostics: list[Diagnostic] = []

    try:
        from mtra.core.units import factor_to_millimetres

        factor_to_millimetres(unit)  # raises AnalysisBlocked when undeclared

        mesh = load_mesh(
            project.paths.mesh_path(pair), pair.pair_id, unit, diagnostics=diagnostics
        )
        program = parse_program(
            str(project.paths.program_path(pair)),
            pair.pair_id,
            pair.robot_type,
            settings=settings,
        )
        diagnostics.extend(program.diagnostics)

        segments, segment_diagnostics = classify(program, settings)
        diagnostics.extend(segment_diagnostics)

        step = float(settings.get("trajectory.resample_step_mm", 2.0))
        strokes = extract_strokes(program, segments, step_mm=step)

        world_from_mesh, transform_source, _declared, notes = frame_ops.resolve_pair_transform(
            project, pair
        )
        for note in notes:
            diagnostics.append(
                Diagnostic(
                    severity=Severity.INFO,
                    code="frames.resolution",
                    message=note,
                    stage="correspondence.frames",
                )
            )

        surface = SurfaceField(mesh)
        if not surface.curvature_available and surface.curvature_error:
            diagnostics.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="mesh.curvature_unavailable",
                    message=surface.curvature_error,
                    stage="mesh.features",
                )
            )

        correspondence = build_correspondence(
            pair.pair_id,
            strokes,
            surface,
            world_from_mesh,
            transform_source=transform_source,
            settings=settings,
            tool_axis_world=_tool_axis(pair),
        )
        diagnostics.extend(correspondence.diagnostics)

        view = PairView(
            pair_id=pair.pair_id,
            program=program,
            segments=segments,
            strokes=strokes,
            surface=surface,
            correspondence=correspondence,
            world_from_mesh=world_from_mesh,
            mesh_from_world=correspondence.mesh_from_world,
            settings=settings,
            tool_axis_world=_tool_axis(pair),
        )

        measurements = measure_pair(view, catalogue)
    except AnalysisBlocked as exc:
        blocked = _blocked_pair(catalogue, pair.pair_id, exc.reason, exc.detail)
        blocked.diagnostics = diagnostics + blocked.diagnostics
        return blocked
    except (InputError, UnsupportedInput) as exc:
        blocked = _blocked_pair(
            catalogue, pair.pair_id, ReasonCode.COMPUTATION_FAILED, str(exc)
        )
        blocked.diagnostics = diagnostics + blocked.diagnostics
        return blocked
    except MTRAError as exc:  # pragma: no cover - defensive
        blocked = _blocked_pair(
            catalogue, pair.pair_id, ReasonCode.COMPUTATION_FAILED, str(exc)
        )
        blocked.diagnostics = diagnostics + blocked.diagnostics
        return blocked

    return PairResult(
        pair_id=pair.pair_id,
        status="analysed",
        measurements=measurements,
        diagnostics=diagnostics,
        view=view,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def analyse_project(
    manifest_path: str | Path,
    *,
    settings: Settings | None = None,
    catalogue: Catalogue | None = None,
) -> AnalysisResult:
    settings = settings or load_settings()
    catalogue = catalogue or load_catalogue()
    project = load_manifest(manifest_path)

    validation = validate_project(project, settings)
    diagnostics = list(validation.diagnostics)

    results: list[PairResult] = []
    for pair in project.spec.pairs:
        pair_errors = [
            diagnostic
            for diagnostic in validation.diagnostics
            if diagnostic.severity is Severity.ERROR
            and _diagnostic_mentions_pair(diagnostic, pair)
        ]
        if pair_errors:
            blocked = _blocked_pair(
                catalogue,
                pair.pair_id,
                ReasonCode.COMPUTATION_FAILED,
                "; ".join(d.message for d in pair_errors),
            )
            blocked.diagnostics = pair_errors + blocked.diagnostics
            results.append(blocked)
            continue
        results.append(_analyse_pair(project, pair, settings, catalogue))

    provenance = Provenance(
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        mtra_version=__version__,
        python_version=platform.python_version(),
        package_versions=_package_versions(),
        source_files=list(validation.source_files),
        configuration=jsonable(settings.data),
    )

    return AnalysisResult(
        project=project,
        provenance=provenance,
        pairs=results,
        diagnostics=diagnostics,
        settings=settings,
        catalogue=catalogue,
    )


def _diagnostic_mentions_pair(diagnostic: Diagnostic, pair: PairSpec) -> bool:
    text = f"{diagnostic.message} {diagnostic.source_path or ''}"
    if pair.pair_id in text:
        return True
    for candidate in (pair.mesh, pair.robot_program):
        if candidate and candidate.split("/")[-1] in text:
            return True
    return False


def inventory_sources(project: Project) -> list[Diagnostic]:
    """Re-hash inputs without analysing, for `mtra project validate`."""
    diagnostics: list[Diagnostic] = []
    for pair in project.spec.pairs:
        for path, role in (
            (project.paths.mesh_path(pair), "mesh"),
            (project.paths.program_path(pair), "robot_program"),
        ):
            try:
                inventory_file(path, role)
            except InputError as exc:
                diagnostics.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code=f"files.{role}_missing",
                        message=str(exc),
                        stage="project.inventory",
                        source_path=str(path),
                    )
                )
    return diagnostics
