"""Scope guard: MTRA measures, it never generates or exports robot programs.

PRD §1 and §19 make this a product boundary, not a convention, so it is asserted
behaviourally: run the real command line over a fixture and check that nothing
resembling a robot program appears in the output, and that every input byte is
unchanged afterwards.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mtra.cli import app
from tests.fixtures.synthetic import build_project

ROBOT_PROGRAM_SUFFIXES = {".ls", ".lst", ".mod", ".prg", ".tid", ".tip"}
FANUC_PROGRAM_MARKER = b"/PROG"


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def analysed(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    manifest, truth = build_project(root / "project")
    runner = CliRunner()
    output = root / "output"
    result = runner.invoke(
        app,
        ["analyse", str(manifest), "--output", str(output), "--no-figures"],
    )
    assert result.exit_code == 0, result.output
    return manifest, output, truth


def test_analyse_writes_no_robot_program_files(analysed):
    _manifest, output, _truth = analysed
    offenders = [
        path for path in output.rglob("*") if path.suffix.lower() in ROBOT_PROGRAM_SUFFIXES
    ]
    assert offenders == []


def test_no_artefact_contains_an_exportable_program(analysed):
    _manifest, output, _truth = analysed
    for path in output.rglob("*"):
        if not path.is_file():
            continue
        assert FANUC_PROGRAM_MARKER not in path.read_bytes(), path


def test_analyse_leaves_every_input_byte_unchanged(analysed):
    manifest, _output, _truth = analysed
    root = manifest.parent
    before = _tree_hashes(root)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "analyse",
            str(manifest),
            "--output",
            str(root / "rerun"),
            "--no-figures",
        ],
    )
    assert result.exit_code == 0, result.output

    after = {key: value for key, value in _tree_hashes(root).items() if not key.startswith("rerun")}
    assert after == before


def test_blocked_pair_is_a_non_zero_exit(tmp_path):
    manifest, _truth = build_project(tmp_path / "nounits", declare_units=False)
    result = CliRunner().invoke(
        app,
        [
            "analyse",
            str(manifest),
            "--output",
            str(tmp_path / "out"),
            "--no-figures",
        ],
    )
    assert result.exit_code == 1
    assert "blocked" in result.output.lower()


def test_mesh_inspection_refuses_to_assume_millimetres(tmp_path):
    manifest, _truth = build_project(tmp_path / "project")
    mesh = manifest.parent / "input" / "meshes" / "plate.stl"

    result = CliRunner().invoke(app, ["mesh", "inspect", str(mesh)])
    assert result.exit_code == 2
    assert "unknown_units" in result.output


def test_project_validation_reports_declarations_without_geometry(tmp_path):
    manifest, _truth = build_project(tmp_path / "project")
    result = CliRunner().invoke(app, ["project", "validate", str(manifest)])
    assert result.exit_code == 0, result.output
    assert "units declared: True" in result.output
    assert "frame declared: True" in result.output
