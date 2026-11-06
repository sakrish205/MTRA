"""Manifest validation: the gate between "a folder of files" and "an analysis".

Design stance: this stage produces diagnostics rather than booleans, because the
report has to explain *why* a run was downgraded or refused. Only genuinely
unrecoverable preconditions (undeclared units, missing files, unsupported robot
format) are errors; everything else is a warning that travels with the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mtra.config import Settings, known_robot_formats, load_robot_format
from mtra.core.errors import InputError, UnsupportedInput
from mtra.core.types import Diagnostic, Severity, SourceFile
from mtra.project.inventory import (
    detect_mesh_format,
    detect_robot_format,
    display_path,
    inventory_file,
)
from mtra.project.manifest import Project

_SUPPORTED_MESH_FORMATS = {"stl-binary", "stl-ascii", "obj"}


@dataclass
class ValidationResult:
    source_files: list[SourceFile] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    units_declared: bool = False
    frame_declared: bool = False
    tool_data_complete: dict[str, bool] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(d.severity is Severity.ERROR for d in self.diagnostics)


def _diag(
    severity: Severity,
    code: str,
    message: str,
    *,
    stage: str = "project.validation",
    source_path: str | None = None,
    line: int | None = None,
) -> Diagnostic:
    return Diagnostic(
        severity=severity,
        code=code,
        message=message,
        stage=stage,
        source_path=source_path,
        line=line,
    )


def validate_project(project: Project, settings: Settings) -> ValidationResult:
    result = ValidationResult()
    spec = project.spec

    # --- units -------------------------------------------------------------- #
    unit = spec.declared_unit
    result.units_declared = str(unit) != "unknown"
    for pair in spec.pairs:
        if pair.units is None:
            continue
        pair_unit = spec.unit_for(pair)
        if str(pair_unit) != str(unit) and str(unit) != "unknown":
            result.diagnostics.append(
                _diag(
                    Severity.WARNING,
                    "units.pair_override",
                    f"pair {pair.pair_id} overrides project units "
                    f"({spec.units!r} -> {pair.units!r})",
                )
            )

    require_units = bool(settings.get("units.require_declared", True))
    if not result.units_declared and require_units:
        result.diagnostics.append(
            _diag(
                Severity.ERROR,
                "units.undeclared",
                "manifest does not declare units; analysis is blocked because a scale "
                "error is indistinguishable from a spacing rule in the report",
            )
        )
    elif not result.units_declared:
        result.diagnostics.append(
            _diag(
                Severity.WARNING,
                "units.undeclared",
                "units undeclared and refusal disabled: measurements carry an "
                "unverified scale",
            )
        )

    # --- coordinate convention ---------------------------------------------- #
    if spec.coordinate_convention == "unknown":
        result.diagnostics.append(
            _diag(
                Severity.WARNING,
                "frames.convention_unknown",
                "coordinate convention (handedness) is not declared; direction signs "
                "remain ambiguous",
            )
        )

    # --- frames -------------------------------------------------------------- #
    for pair in spec.pairs:
        mesh_frame = spec.frame(pair.mesh_frame)
        if mesh_frame is None:
            result.diagnostics.append(
                _diag(
                    Severity.WARNING,
                    "frames.undeclared",
                    f"pair {pair.pair_id}: mesh frame {pair.mesh_frame!r} is not declared; "
                    "registration must be estimated and is reported as ambiguous",
                )
            )
        elif mesh_frame.transform is not None:
            result.frame_declared = True

    # --- tool data ----------------------------------------------------------- #
    for pair in spec.pairs:
        tool = pair.tool
        complete = bool(tool and tool.tool_axis is not None)
        result.tool_data_complete[pair.pair_id] = complete
        if not complete:
            result.diagnostics.append(
                _diag(
                    Severity.WARNING,
                    "tool.incomplete",
                    f"pair {pair.pair_id}: no declared tool axis; standoff and spray-axis "
                    "intersection will be reported as unavailable rather than derived "
                    "from robot pose alone",
                )
            )

    # --- files --------------------------------------------------------------- #
    for pair in spec.pairs:
        mesh_path = project.paths.mesh_path(pair)
        try:
            inventory = inventory_file(
                mesh_path, "mesh", display_path=display_path(mesh_path, project.paths.base_dir)
            )
        except InputError as exc:
            result.diagnostics.append(
                _diag(Severity.ERROR, "files.mesh_missing", str(exc), source_path=str(mesh_path))
            )
        else:
            result.source_files.append(inventory)
            if inventory.detected_format not in _SUPPORTED_MESH_FORMATS:
                result.diagnostics.append(
                    _diag(
                        Severity.ERROR,
                        "files.mesh_format",
                        f"pair {pair.pair_id}: mesh format {inventory.detected_format!r} "
                        "is not supported (STL ascii/binary and OBJ only)",
                        source_path=str(mesh_path),
                    )
                )

        program_path = project.paths.program_path(pair)
        try:
            program_inventory = inventory_file(
                program_path,
                "robot_program",
                display_path=display_path(program_path, project.paths.base_dir),
            )
        except InputError as exc:
            result.diagnostics.append(
                _diag(
                    Severity.ERROR,
                    "files.program_missing",
                    str(exc),
                    source_path=str(program_path),
                )
            )
            continue
        result.source_files.append(program_inventory)

        _check_robot_format(project, pair, program_path, result)

    if not result.source_files:
        result.diagnostics.append(
            _diag(Severity.ERROR, "files.none", "no source files could be inventoried")
        )

    return result


def _check_robot_format(project: Project, pair, program_path: Path, result: ValidationResult) -> None:
    declared = pair.robot_type.strip().lower()
    try:
        grammar = load_robot_format(declared)
    except InputError as exc:
        result.diagnostics.append(
            _diag(Severity.ERROR, "parser.unknown_format", str(exc), source_path=str(program_path))
        )
        return

    if not grammar.supported:
        result.diagnostics.append(
            _diag(
                Severity.ERROR,
                "parser.unsupported_format",
                f"pair {pair.pair_id}: {grammar.display_name} adapter is not implemented — "
                f"{grammar.note}",
                source_path=str(program_path),
            )
        )
        return

    guessed = detect_robot_format(program_path)
    if guessed is not None and guessed != declared:
        result.diagnostics.append(
            _diag(
                Severity.ERROR,
                "parser.format_mismatch",
                f"pair {pair.pair_id}: manifest declares robot_type {declared!r} but the "
                f"file extension indicates {guessed!r}",
                source_path=str(program_path),
            )
        )

    if grammar.provisional:
        result.diagnostics.append(
            _diag(
                Severity.WARNING,
                "parser.provisional",
                f"pair {pair.pair_id}: {grammar.display_name} grammar is provisional — "
                "validated against synthetic fixtures only until a real program is supplied",
                source_path=str(program_path),
            )
        )


def require_supported_format(key: str) -> None:
    """Raise ``UnsupportedInput`` for a configured-but-unimplemented dialect."""
    grammar = load_robot_format(key)
    if not grammar.supported:
        raise UnsupportedInput(f"{grammar.display_name}: {grammar.note}")


def supported_formats() -> tuple[str, ...]:
    return tuple(k for k in known_robot_formats() if load_robot_format(k).supported)
