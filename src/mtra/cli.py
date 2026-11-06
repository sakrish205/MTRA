"""Command line: a thin shell over ``pipeline.run`` and ``reporting``.

Every command prints what it did and exits non-zero when the thing it was asked to
do could not be done. In particular ``analyse`` exits 1 when a pair is blocked: a
report full of reason codes is a correct artefact, but it is not a successful
analysis, and a shell script must be able to tell the difference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
import yaml

from mtra import __version__
from mtra.config import Settings, known_robot_formats, load_settings
from mtra.core.errors import MTRAError
from mtra.core.types import Severity
from mtra.core.units import LengthUnit
from mtra.mesh.loader import load_mesh
from mtra.pipeline.run import analyse_project
from mtra.project.manifest import load_manifest
from mtra.project.validation import validate_project
from mtra.reporting import (
    build_report,
    render_figures,
    write_all,
    write_measurements_csv,
    write_strokes_csv,
)
from mtra.reporting.report_data import load_report_json
from mtra.reporting.html_report import write_html
from mtra.reporting.pdf_report import write_pdf
from mtra.robot.registry import parse_program

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Mesh-Trajectory Relationship Analyser: measure, never generate.",
)
project_app = typer.Typer(no_args_is_help=True, help="Inspect and validate a project manifest.")
mesh_app = typer.Typer(no_args_is_help=True, help="Inspect a mesh without analysing a pair.")
robot_app = typer.Typer(no_args_is_help=True, help="Parse a robot program without pairing it.")
app.add_typer(project_app, name="project")
app.add_typer(mesh_app, name="mesh")
app.add_typer(robot_app, name="robot")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _fail(message: str, code: int = 2) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)


def _load_settings_override(path: Path | None) -> Settings:
    if path is None:
        return load_settings()
    if not path.is_file():
        _fail(f"settings file not found: {path}")
    try:
        override = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        _fail(f"{path} is not valid YAML: {exc}")
    if not isinstance(override, dict):
        _fail(f"{path} must contain a YAML mapping")
    return Settings(data=_deep_merge(load_settings().data, override))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _print_diagnostics(diagnostics: list[Any]) -> int:
    errors = 0
    for diagnostic in diagnostics:
        severity = str(getattr(diagnostic, "severity", "info"))
        if severity == str(Severity.ERROR):
            errors += 1
        location = getattr(diagnostic, "source_path", None) or ""
        line = getattr(diagnostic, "line", None)
        suffix = f" ({location}{':' + str(line) if line else ''})" if location else ""
        typer.echo(
            f"  [{severity}] {getattr(diagnostic, 'code', '')}: "
            f"{getattr(diagnostic, 'message', diagnostic)}{suffix}"
        )
    return errors


def _exit_code(result) -> int:
    return 0 if all(pair.status == "analysed" for pair in result.pairs) else 1


def _brief(value: Any, limit: int = 90) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --------------------------------------------------------------------------- #
# project
# --------------------------------------------------------------------------- #


@project_app.command("validate")
def project_validate(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False, help="project.json"),
) -> None:
    """Check a manifest's declarations without analysing geometry."""
    try:
        project = load_manifest(manifest)
    except MTRAError as exc:
        _fail(str(exc))

    validation = validate_project(project, load_settings())
    typer.echo(f"project: {project.project_id}")
    typer.echo(f"pairs:   {len(project.spec.pairs)}")
    for pair in project.spec.pairs:
        typer.echo(
            f"  - {pair.pair_id}: mesh={pair.mesh} program={pair.robot_program} "
            f"robot={pair.robot_type} units={project.spec.unit_for(pair)}"
        )
    typer.echo(
        f"units declared: {validation.units_declared}   "
        f"frame declared: {validation.frame_declared}"
    )
    typer.echo(f"source files hashed: {len(validation.source_files)}")

    errors = _print_diagnostics(validation.diagnostics)
    if errors:
        typer.echo(f"\n{errors} blocking problem(s): analysis would be refused.", err=True)
        raise typer.Exit(1)
    if not validation.diagnostics:
        typer.echo("no diagnostics: declarations are complete.")


# --------------------------------------------------------------------------- #
# mesh
# --------------------------------------------------------------------------- #


@mesh_app.command("inspect")
def mesh_inspect(
    path: Path = typer.Argument(..., exists=True, dir_okay=False),
    units: str = typer.Option("unknown", help="Declared units: mm, cm, m, inch, or unknown."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Load a mesh and report quality. Units are required, never assumed."""
    try:
        record = load_mesh(path, path.stem, LengthUnit.parse(units))
    except MTRAError as exc:
        _fail(str(exc))

    payload = record.summary()
    if as_json:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    typer.echo(f"mesh:    {record.source_path}")
    typer.echo(f"format:  {record.format}   declared units: {record.declared_units}")
    typer.echo(f"vertices: {payload['vertex_count']}   faces: {payload['face_count']}")
    quality = record.mesh_quality
    for key in (
        "watertight",
        "winding_consistent",
        "component_count",
        "boundary_edge_count",
        "non_manifold_edge_count",
        "degenerate_face_count",
        "duplicate_vertex_count",
        "surface_area_mm2",
        "volume_mm3",
        "extents_mm",
        "edge_length_mm",
    ):
        if key in quality:
            typer.echo(f"  {key}: {_brief(quality[key])}")
    for note in record.transform_history:
        typer.echo(f"  transform: {note}")


# --------------------------------------------------------------------------- #
# robot
# --------------------------------------------------------------------------- #


@robot_app.command("parse")
def robot_parse(
    path: Path = typer.Argument(..., exists=True, dir_okay=False),
    robot_type: str = typer.Option("fanuc", "--type", help=f"One of: {', '.join(known_robot_formats())}"),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    limit: int = typer.Option(0, help="Show at most N motions (0 = all)."),
) -> None:
    """Parse a robot program and preserve source-line traceability."""
    try:
        program = parse_program(str(path), path.stem, robot_type)
    except MTRAError as exc:
        _fail(str(exc))

    motions = program.motions
    if as_json:
        payload = {
            "source_path": program.source_path,
            "robot_type": program.robot_type,
            "program_name": program.program_name,
            "motion_count": len(motions),
            "unknown_line_count": len(program.unknown_lines),
            "unknown_lines": program.unknown_lines,
            "motions": [motion.model_dump() for motion in motions],
            "diagnostics": [d.model_dump() for d in program.diagnostics],
        }
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    typer.echo(f"program: {program.program_name}   type: {program.robot_type}")
    typer.echo(f"motions: {len(motions)}   unknown lines: {len(program.unknown_lines)}")
    shown = motions if limit <= 0 else motions[:limit]
    for motion in shown:
        position = ", ".join(f"{value:.3f}" for value in motion.tcp_position)
        typer.echo(
            f"  line {motion.source_line:>5}  #{motion.instruction_index:<3} "
            f"{str(motion.motion_type):<8} P[...]  tcp=({position})"
        )
    if len(shown) < len(motions):
        typer.echo(f"  ... {len(motions) - len(shown)} more")
    for line, text in program.unknown_lines[:10]:
        typer.echo(f"  [unknown] line {line}: {text.strip()[:80]}")
    _print_diagnostics(program.diagnostics)


# --------------------------------------------------------------------------- #
# analyse / report
# --------------------------------------------------------------------------- #


@app.command("analyse")
def analyse(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False, help="project.json"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory."),
    figures: bool = typer.Option(
        True, "--figures/--no-figures", help="Render mesh/pose figures when PyVista is available."
    ),
    settings: Path | None = typer.Option(None, "--settings", help="YAML overrides for defaults."),
    quiet: bool = typer.Option(False, "--quiet", help="Only print the artefact paths."),
) -> None:
    """Analyse every declared pair and write the report artefacts."""
    configured = _load_settings_override(settings)
    try:
        result = analyse_project(manifest, settings=configured)
    except MTRAError as exc:
        _fail(str(exc))

    output_dir = Path(output) if output else Path(manifest).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_map: dict[str, list[str]] = {}
    figure_warnings: list[str] = []
    wants_figures = figures and bool(configured.get("reporting.figures_enabled", True))
    if wants_figures:
        figure_map, figure_warnings = render_figures(
            result,
            output_dir,
            dpi=int(configured.get("reporting.figure_dpi", 110)),
            enabled=True,
        )

    report = build_report(result, figures=figure_map)
    artefacts = write_all(report, output_dir, figure_base=output_dir)

    measurements_json = output_dir / "measurements.json"
    measurements_json.write_text(
        json.dumps(result.to_dict(), indent=2, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    artefacts["measurements_json"] = str(measurements_json)
    artefacts["measurements_csv"] = str(
        write_measurements_csv(result.all_measurements(), output_dir / "measurements.csv")
    )
    strokes_csv = write_strokes_csv(result, output_dir / "strokes.csv")
    if strokes_csv is not None:
        artefacts["strokes_csv"] = str(strokes_csv)

    if not quiet:
        typer.echo(f"project: {result.project.project_id}")
        for pair in result.pairs:
            coverage = pair.coverage()
            typer.echo(
                f"  {pair.pair_id}: {pair.status} - {coverage['with_value']}/"
                f"{coverage['parameters']} parameters with a value"
            )
            if pair.blocked_reason:
                typer.echo(f"    blocked: {pair.blocked_reason}", err=True)
        _print_diagnostics(result.diagnostics)
        for warning in figure_warnings:
            typer.echo(f"  [warning] {warning}", err=True)
        typer.echo("\nartefacts:")
    for name, path in artefacts.items():
        typer.echo(f"  {name:<18} {path}")

    raise typer.Exit(_exit_code(result))


@app.command("report")
def report(
    report_json: Path = typer.Argument(..., exists=True, dir_okay=False, help="report.json"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory."),
) -> None:
    """Re-render HTML and PDF from a saved report model, without re-analysing."""
    payload = load_report_json(report_json)
    target_dir = Path(output) if output else report_json.parent

    from mtra.reporting.report_data import PairReport, ReportData, GroupSection, ReportRow

    pairs = []
    for pair_node in payload["pairs"]:
        sections = [
            GroupSection(
                group=section["group"],
                title=section["title"],
                rows=[ReportRow(**row) for row in section["rows"]],
            )
            for section in pair_node["sections"]
        ]
        pairs.append(
            PairReport(
                pair_id=pair_node["pair_id"],
                status=pair_node["status"],
                blocked_reason=pair_node["blocked_reason"],
                coverage=pair_node["coverage"],
                sections=sections,
                unavailable=[ReportRow(**row) for row in pair_node["unavailable"]],
                diagnostics=pair_node["diagnostics"],
                metrics=pair_node["metrics"],
                figures=pair_node["figures"],
            )
        )
    model = ReportData(
        project_id=payload["project_id"],
        generated_at_utc=payload["generated_at_utc"],
        mtra_version=payload["mtra_version"],
        settings_summary=payload["settings_summary"],
        metrics=payload["metrics"],
        pairs=pairs,
        diagnostics=payload["diagnostics"],
        provenance=payload["provenance"],
    )

    html_path = write_html(model, target_dir / "report.html")
    pdf_path = write_pdf(model, target_dir / "report.pdf", base_dir=target_dir)
    typer.echo(f"  html  {html_path}")
    typer.echo(f"  pdf   {pdf_path}")


@app.command("version")
def version() -> None:
    """Print the MTRA version."""
    typer.echo(__version__)


def main() -> None:  # pragma: no cover - console entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
