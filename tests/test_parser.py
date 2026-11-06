"""Parser: instruction-for-instruction traceability, and honesty about unknowns.

The grammar is provisional (config/robot_formats.yaml), so the parser's contract is
not "understand everything" — it is "never silently drop a line". Every motion
carries the source line it came from and the instruction number it was written
with, which is what makes a report reviewable against the program.
"""

from __future__ import annotations

import pytest

from mtra.core.errors import InputError, UnsupportedInput
from mtra.robot.registry import parse_program
from tests.fixtures.synthetic import build_project, build_raster_program


@pytest.fixture(scope="module")
def parsed(plate_project):
    manifest, truth = plate_project
    program_path = manifest.parent / "input" / "robot_programs" / "raster.ls"
    return parse_program(str(program_path), "plate", "fanuc"), truth


def test_every_motion_line_is_reproduced(parsed):
    program, truth = parsed
    assert len(program.motions) == truth["motion_count"]

    lines = [motion.source_line for motion in program.motions]
    indexes = [motion.instruction_index for motion in program.motions]

    assert lines == sorted(lines), "source lines must follow program order"
    assert len(set(lines)) == len(lines), "each instruction appears exactly once"
    assert indexes == sorted(indexes), "instruction numbers must be monotonic"
    assert max(indexes) <= truth["instruction_count"]


def test_motion_records_carry_resolvable_poses(parsed):
    program, truth = parsed
    for motion in program.motions:
        assert motion.position_id is not None
        assert len(motion.tcp_position) == 3
        assert motion.source_line > 0
        assert motion.raw.strip()

    painting = [
        motion
        for motion in program.motions
        if motion.tcp_position[2] == pytest.approx(truth["tcp_z"], abs=1e-6)
    ]
    assert len(painting) >= truth["stroke_count"] * 2
    assert {round(motion.tcp_position[1], 6) for motion in painting} == {
        round(y, 6) for y in truth["y_values"]
    }


def test_rotation_is_taken_from_the_program_not_invented(parsed):
    program, _truth = parsed
    # FANUC W = P = R = 0 points the tool axis along -Z (downward, toward the
    # plate below the pass); the axis must come from the pose, not be assumed.
    for motion in program.motions:
        assert motion.tool_axis == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)


def test_unknown_instruction_is_retained_with_its_line_number(tmp_path):
    manifest, _truth = build_project(tmp_path / "unknown", extra_unknown_instruction=True)
    program_path = manifest.parent / "input" / "robot_programs" / "raster.ls"
    program = parse_program(str(program_path), "plate", "fanuc")

    assert program.unknown_lines, "unknown line must be retained, not skipped"
    line, text = program.unknown_lines[0]
    assert "VISION_FEATURE_OFFSET" in text

    warnings = [d for d in program.diagnostics if d.code == "parser.unknown_instruction"]
    assert warnings, "an unknown line must produce a diagnostic"
    assert warnings[0].line == line


def test_missing_io_is_a_warning_not_an_inference(tmp_path):
    manifest, _truth = build_project(tmp_path / "noio", include_io=False)
    program_path = manifest.parent / "input" / "robot_programs" / "raster.ls"
    program = parse_program(str(program_path), "plate", "fanuc")

    codes = {d.code for d in program.diagnostics}
    assert "parser.no_io" in codes
    assert all(state.paint_on is None for state in program.process_states)


def test_unsupported_dialects_fail_loudly(tmp_path):
    program_path = tmp_path / "program.mod"
    program_path.write_text("MODULE main\nENDMODULE\n", encoding="utf-8")

    with pytest.raises(UnsupportedInput):
        parse_program(str(program_path), "p", "abb")

    with pytest.raises(InputError):
        parse_program(str(program_path), "p", "not-a-controller")


def test_program_name_and_frames_are_preserved(parsed):
    program, _truth = parsed
    assert program.program_name == "RASTER25"
    assert program.robot_type == "fanuc"
    assert program.source_path.endswith("raster.ls")


def test_raw_program_is_untouched(tmp_path):
    source = tmp_path / "raw.ls"
    truth = build_raster_program(source)
    before = source.read_bytes()
    parse_program(str(source), "plate", "fanuc")
    assert source.read_bytes() == before
    assert truth["painting_segments"] == truth["stroke_count"]
