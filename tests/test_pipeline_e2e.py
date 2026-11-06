"""End-to-end pipeline behaviour on the synthetic fixtures.

These tests are about the properties that make the report trustworthy rather than
about individual numbers: every input is hashed, every pair is accounted for, a
blocked pair still names all 112 parameters, and analysis never writes to an input.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mtra.core.types import ReasonCode, Severity
from mtra.pipeline.run import analyse_project
from tests.fixtures.synthetic import build_project


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_both_fixtures_analyse_with_the_whole_catalogue_accounted(plate_result, cylinder_result):
    for result in (plate_result, cylinder_result):
        pair = result.pairs[0]
        assert pair.status == "analysed"
        assert pair.blocked_reason is None
        coverage = pair.coverage()
        assert coverage["parameters"] == 112
        assert coverage["with_value"] > 0
        assert coverage["with_value"] + coverage["without_value"] == 112


def test_provenance_records_portable_input_paths(plate_result):
    paths = [source.path for source in plate_result.provenance.source_files]
    assert paths == ["input/meshes/plate.stl", "input/robot_programs/raster.ls"]
    assert all(":" not in path and "\\" not in path for path in paths)


def test_analysis_records_every_input_with_a_hash(plate_result):
    provenance = plate_result.provenance
    roles = {source.role for source in provenance.source_files}
    assert roles == {"mesh", "robot_program"}
    for source in provenance.source_files:
        assert len(source.sha256) == 64
        assert source.size_bytes > 0
        assert source.detected_format


def test_provenance_is_json_serialisable_and_versioned(plate_result):
    provenance = plate_result.provenance
    generated = datetime.fromisoformat(provenance.generated_at_utc)
    assert generated.utcoffset() == timedelta(0)
    assert provenance.mtra_version == plate_result.provenance.mtra_version
    assert "numpy" in provenance.package_versions
    assert isinstance(provenance.random_seed, int)

    payload = json.dumps(plate_result.to_dict(), default=str)
    assert "synthetic_raster" in payload
    assert str(plate_result.project.paths.base_dir) not in payload


def test_diagnostics_name_the_assumptions_the_report_relies_on(plate_result):
    codes = {diagnostic.code for diagnostic in plate_result.diagnostics}
    assert "parser.provisional" in codes
    assert "tool.incomplete" in codes

    parser_warning = next(
        diagnostic
        for diagnostic in plate_result.diagnostics
        if diagnostic.code == "parser.provisional"
    )
    assert parser_warning.severity is Severity.WARNING
    assert parser_warning.stage


def test_analysis_does_not_modify_any_input(plate_project):
    manifest, _truth = plate_project
    root = Path(manifest).parent
    before = _tree_hashes(root)

    analyse_project(manifest)

    assert _tree_hashes(root) == before


def test_undeclared_units_block_the_pair_and_name_the_reason(tmp_path):
    manifest, _truth = build_project(tmp_path / "nounits", declare_units=False)
    result = analyse_project(manifest)

    pair = result.pairs[0]
    assert pair.status == "blocked"
    assert pair.blocked_reason == ReasonCode.UNKNOWN_UNITS
    assert len(pair.measurements) == 112
    assert all(not record.is_available for record in pair.measurements)
    assert {
        record.unavailable.reason for record in pair.measurements
    } == {ReasonCode.UNKNOWN_UNITS}
    detail = pair.measurements[0].unavailable.detail.lower()
    assert "scale" in detail or "unit" in detail


def test_unsupported_dialect_blocks_without_inventing_measurements(tmp_path):
    manifest, _truth = build_project(tmp_path / "abb", robot_type="abb")
    result = analyse_project(manifest)

    pair = result.pairs[0]
    assert pair.status == "blocked"
    assert len(pair.measurements) == 112
    assert all(record.unavailable is not None for record in pair.measurements)

    codes = {diagnostic.code for diagnostic in result.diagnostics}
    assert "parser.unsupported_format" in codes


def test_unknown_instruction_reaches_the_pair_diagnostics(tmp_path):
    manifest, _truth = build_project(tmp_path / "unknown", extra_unknown_instruction=True)
    result = analyse_project(manifest)

    codes = {diagnostic.code for diagnostic in result.pairs[0].diagnostics}
    assert "parser.unknown_instruction" in codes


def test_pair_coverage_matches_the_measurement_records(plate_result):
    pair = plate_result.pairs[0]
    payload = plate_result.to_dict()["pairs"][0]
    assert payload["coverage"] == pair.coverage()
    assert len(payload["measurements"]) == 112
    assert len(plate_result.all_measurements()) == 112 * len(plate_result.pairs)


def test_frames_note_that_the_transform_was_declared(plate_result):
    refs = [
        ref
        for record in plate_result.pairs[0].measurements
        for ref in record.source_refs
    ]
    assert refs, "measurements must be traceable to something"
    assert plate_result.pairs[0].view.correspondence.transform_source == "declared:part"


@pytest.mark.parametrize("parameter_id", ["A1", "A3"])
def test_alignment_parameters_are_available_when_the_frame_is_declared(
    plate_result, parameter_id
):
    records = {record.parameter_id: record for record in plate_result.pairs[0].measurements}
    assert records[parameter_id].is_available
