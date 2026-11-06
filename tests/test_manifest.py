"""Manifest intake: explicit pairing, declared declarations, loud refusals.

PRD §5.3 makes pairing a reviewed act and §19 makes an undeclared unit or frame a
refusal rather than a default. These tests pin both behaviours at the intake
boundary, before any geometry is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtra.config import load_settings
from mtra.core.errors import InputError
from mtra.core.types import ReasonCode, Severity
from mtra.pipeline.run import analyse_project
from mtra.project.manifest import load_manifest
from mtra.project.validation import validate_project
from tests.fixtures.synthetic import build_project


def test_fixture_manifest_loads_with_resolved_paths(plate_project):
    manifest, truth = plate_project
    project = load_manifest(manifest)

    assert project.project_id == "synthetic_raster"
    assert len(project.spec.pairs) == 1
    pair = project.spec.pairs[0]
    assert pair.pair_id == "plate"
    assert pair.robot_type == "fanuc"
    assert str(project.spec.declared_unit) == "mm"

    assert project.paths.mesh_path(pair).is_file()
    assert project.paths.program_path(pair).is_file()
    assert project.paths.output_dir == Path(manifest).parent / "output"
    assert truth["pitch_mm"] == 25.0


def test_duplicate_pair_ids_are_rejected(tmp_path):
    manifest = {
        "project_id": "dupe",
        "units": "mm",
        "pairs": [
            {"pair_id": "a", "mesh": "m.stl", "robot_program": "p.ls", "robot_type": "fanuc"},
            {"pair_id": "a", "mesh": "m2.stl", "robot_program": "p2.ls", "robot_type": "fanuc"},
        ],
    }
    path = tmp_path / "project.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InputError, match="duplicate pair_id"):
        load_manifest(path)


def test_unknown_manifest_keys_are_rejected_not_ignored(tmp_path):
    manifest = {
        "project_id": "typo",
        "units": "mm",
        "pairs": [
            {
                "pair_id": "a",
                "mesh": "m.stl",
                "robot_program": "p.ls",
                "robot_type": "fanuc",
                "unit": "mm",
            }
        ],
    }
    path = tmp_path / "project.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InputError):
        load_manifest(path)


def test_undeclared_units_are_an_error_not_a_warning(tmp_path):
    manifest, _truth = build_project(tmp_path / "nounits", declare_units=False)
    project = load_manifest(manifest)
    validation = validate_project(project, load_settings())

    errors = [d for d in validation.diagnostics if d.severity is Severity.ERROR]
    assert any(d.code == "units.undeclared" for d in errors)
    assert not validation.ok


def test_undeclared_frame_transform_blocks_with_unknown_frame(tmp_path):
    manifest, _truth = build_project(tmp_path / "noframe", declare_frame=False)
    project = load_manifest(manifest)
    validation = validate_project(project, load_settings())

    assert validation.frame_declared is False
    assert validation.ok, "a missing transform is not a malformed manifest"

    result = analyse_project(manifest)
    pair = result.pairs[0]
    assert pair.status == "blocked"
    assert pair.blocked_reason == ReasonCode.UNKNOWN_FRAME
    assert all(record.unavailable is not None for record in pair.measurements)


def test_missing_input_files_are_reported_with_paths(tmp_path):
    manifest = {
        "project_id": "missing",
        "units": "mm",
        "pairs": [
            {
                "pair_id": "p",
                "mesh": "input/meshes/absent.stl",
                "robot_program": "input/robot_programs/absent.ls",
                "robot_type": "fanuc",
            }
        ],
    }
    path = tmp_path / "project.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    project = load_manifest(path)
    validation = validate_project(project, load_settings())
    codes = {d.code for d in validation.diagnostics}
    assert {"files.mesh_missing", "files.program_missing"} <= codes
    assert not validation.ok
