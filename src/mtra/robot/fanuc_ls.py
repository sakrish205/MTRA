"""FANUC LS adapter.

Line-oriented, two-pass: positions are parsed first because motion instructions
reference ``P[n]`` blocks that appear later in ``/POS``. That ordering is a
property of the format, not a shortcut — a streaming parser cannot resolve
targets.

What this adapter refuses to do:

* Skip an unparsed line silently. Every line inside ``/MN`` that matches neither
  a motion, an I/O assignment, a frame assignment nor an ignored pattern becomes
  a diagnostic with its line number (PRD §16 robustness, §19 "parser misses
  instructions").
* Invent a pose for a referenced target that has no block.
* Decide that the process is on because the motion looks like a sweep. Process
  state comes only from configured I/O (PRD §5.2).

The syntax it accepts is the configured grammar, not this module. When a real LS
file disagrees, ``config/robot_formats.yaml`` is the file to change.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from pathlib import Path

from mtra.config import Settings, load_settings
from mtra.core.errors import InputError
from mtra.core.geometry import rotation_from_wpr
from mtra.core.types import (
    Diagnostic,
    MotionType,
    ProcessState,
    RobotMotionRecord,
    Severity,
)
from mtra.robot.common_schema import ParsedProgram
from mtra.robot.grammar import CompiledGrammar, compile_grammar, is_ignored


@dataclass
class _PositionTarget:
    position_id: str
    xyz: tuple[float, float, float]
    wpr: tuple[float, float, float]
    user_frame: str | None = None
    tool_frame: str | None = None
    config_string: str | None = None
    line: int = 0
    missing: list[str] = field(default_factory=list)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _parse_position_block(
    grammar: CompiledGrammar, block_text: str, position_id: str, line: int
) -> _PositionTarget:
    numeric: dict[str, float] = {}
    units: dict[str, str] = {}
    for match in grammar.position_numeric.finditer(block_text):
        name = match.group("field").upper()
        if name in {"X", "Y", "Z", "W", "P", "R"}:
            numeric[name] = float(match.group("value"))
            unit_group = match.groupdict().get("unit")
            if unit_group:
                units[name] = unit_group.lower()

    user_frame = tool_frame = None
    for match in grammar.position_frame.finditer(block_text):
        if match.group("field").upper() == "UF":
            user_frame = match.group("value")
        else:
            tool_frame = match.group("value")

    config_match = grammar.position_config.search(block_text)
    config_string = config_match.group("value") if config_match else None

    missing = [axis for axis in ("X", "Y", "Z") if axis not in numeric]
    for axis in ("W", "P", "R"):
        numeric.setdefault(axis, 0.0)

    # A centimetre/inch unit inside a pose block would silently scale one target
    # and not its neighbours; refuse rather than convert here, because the
    # manifest already declares the project unit.
    inconsistent = {axis: unit for axis, unit in units.items() if unit not in {"mm", "deg"}}
    if inconsistent:
        missing.append(f"non-mm units in pose block: {inconsistent}")

    return _PositionTarget(
        position_id=position_id,
        xyz=(numeric.get("X", 0.0), numeric.get("Y", 0.0), numeric.get("Z", 0.0)),
        wpr=(numeric["W"], numeric["P"], numeric["R"]),
        user_frame=user_frame,
        tool_frame=tool_frame,
        config_string=config_string,
        line=line,
        missing=missing,
    )


def _scan(grammar: CompiledGrammar, text: str) -> tuple[
    list[tuple[int, re.Match[str]]],
    dict[str, _PositionTarget],
    list[tuple[int, str, bool]],
    dict[str, str],
    list[tuple[int, str]],
    str,
]:
    motion_lines: list[tuple[int, re.Match[str]]] = []
    positions: dict[str, _PositionTarget] = {}
    io_events: list[tuple[int, str, bool]] = []
    frames: dict[str, str] = {}
    unknown: list[tuple[int, str]] = []
    program_name = ""

    section = "header"
    block_lines: list[str] = []
    block_id: str | None = None
    block_start_line = 0

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()

        program_match = grammar.program.search(line)
        if program_match:
            program_name = program_match.group("name")
            continue

        if grammar.main_start.search(line):
            section = "main"
            continue
        if grammar.positions_start.search(line):
            section = "positions"
            continue
        if grammar.end.search(line):
            section = "end"
            continue

        if section == "positions":
            if block_id is None:
                start = grammar.position_start.search(line)
                if start:
                    block_id = start.group("id")
                    block_start_line = line_no
                    block_lines = [line[start.end():]]
                continue

            if grammar.position_end.search(line):
                block_lines.append(line)
                positions[block_id] = _parse_position_block(
                    grammar, "\n".join(block_lines), block_id, block_start_line
                )
                block_id = None
                block_lines = []
            else:
                block_lines.append(line)
            continue

        if section != "main":
            continue

        if not stripped or is_ignored(grammar, line):
            continue

        motion = grammar.motion.search(line)
        if motion:
            motion_lines.append((line_no, motion))
            continue

        io_match = grammar.io.search(line)
        if io_match:
            label = f"{io_match.group('signal')}[{io_match.group('number')}]"
            value = io_match.group("value").upper()
            numeric_on = value.isdigit() and float(value) != 0.0
            io_events.append(
                (int(io_match.group("index")), label, value == "ON" or numeric_on)
            )
            continue

        frame_match = grammar.frame_assignment.search(line)
        if frame_match:
            kind = frame_match.group("kind").upper()
            frames["UTOOL_NUM" if kind == "UTOOL_NUM" else "UFRAME_NUM"] = frame_match.group(
                "value"
            )
            continue

        unknown.append((line_no, stripped))

    return motion_lines, positions, io_events, frames, unknown, program_name


def _resolve_motion_type(grammar: CompiledGrammar, letter: str) -> MotionType:
    mapping = grammar.format.raw.get("motion", {}).get("type_map", {})
    value = str(mapping.get(letter, "unknown")).lower()
    try:
        return MotionType(value)
    except ValueError:
        return MotionType.UNKNOWN


def _paint_signal(settings: Settings) -> tuple[str, str]:
    return (
        str(settings.get("process.paint_signal", "DO[1]")),
        str(settings.get("process.paint_on_value", "ON")).upper(),
    )


def parse_program(
    path: str | Path,
    pair_id: str,
    *,
    grammar_key: str = "fanuc",
    settings: Settings | None = None,
) -> ParsedProgram:
    settings = settings or load_settings()
    program_path = Path(path)
    if not program_path.is_file():
        raise InputError(f"robot program not found: {program_path}")

    grammar = compile_grammar(grammar_key)
    text = _read_text(program_path)

    motion_lines, positions, io_events, frames, unknown, program_name = _scan(grammar, text)

    diagnostics: list[Diagnostic] = []
    motions: list[RobotMotionRecord] = []

    for (line_no, match) in motion_lines:
        instruction_index = int(match.group("index"))
        letter = match.group("type").upper()
        target_text = match.group("target")
        id_match = grammar.position_id.search(target_text)
        position_id = id_match.group("id").strip() if id_match else None
        rest = match.group("rest") or ""

        speed_value: float | None = None
        speed_unit: str | None = None
        speed_match = grammar.speed.search(rest)
        if speed_match:
            speed_value = float(speed_match.group("value"))
            unit = speed_match.groupdict().get("unit")
            speed_unit = unit.lower() if unit else None

        termination_match = grammar.termination.search(rest)
        termination = termination_match.group("termination") if termination_match else None

        target = positions.get(position_id) if position_id else None
        if target is None:
            diagnostics.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    code="parser.missing_position",
                    message=(
                        f"motion references {target_text.strip()} with no pose block in /POS; "
                        "the instruction cannot be placed in space"
                    ),
                    stage="robot.parse",
                    source_path=str(program_path),
                    line=line_no,
                )
            )
            continue

        if target.missing:
            diagnostics.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    code="parser.incomplete_position",
                    message=(
                        f"pose block P[{target.position_id}] is missing {target.missing}; "
                        "cannot form a pose"
                    ),
                    stage="robot.parse",
                    source_path=str(program_path),
                    line=target.line,
                )
            )
            continue

        rotation = rotation_from_wpr(*target.wpr)
        motions.append(
            RobotMotionRecord(
                pair_id=pair_id,
                source_file=str(program_path),
                instruction_index=instruction_index,
                motion_type=_resolve_motion_type(grammar, letter),
                tcp_position=target.xyz,
                tcp_orientation=target.wpr,
                rotation_matrix=rotation.tolist(),
                position_id=target.position_id,
                speed=speed_value,
                speed_unit=speed_unit,
                termination=termination,
                tool_frame=target.tool_frame or frames.get("UTOOL_NUM"),
                user_frame=target.user_frame or frames.get("UFRAME_NUM"),
                source_line=line_no,
                raw=match.group(0).strip(),
            )
        )

    if not motions:
        diagnostics.append(
            Diagnostic(
                severity=Severity.ERROR,
                code="parser.no_motions",
                message="no motion instructions could be parsed; check the grammar and file content",
                stage="robot.parse",
                source_path=str(program_path),
            )
        )

    paint_signal, paint_on_value = _paint_signal(settings)
    process_states = _build_process_states(motions, io_events, paint_signal, paint_on_value)

    if not io_events:
        diagnostics.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="parser.no_io",
                message=(
                    "program contains no I/O assignments; process state is unknown and "
                    "paint classification will be reported as unresolved rather than "
                    "inferred from motion geometry"
                ),
                stage="robot.parse",
                source_path=str(program_path),
            )
        )
    elif not any(state.paint_on is not None for state in process_states):
        diagnostics.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="parser.paint_signal_absent",
                message=(
                    f"no assignment to the configured paint signal {paint_signal} was found; "
                    "process state is unknown for every motion"
                ),
                stage="robot.parse",
                source_path=str(program_path),
            )
        )

    if unknown:
        preview = "; ".join(f"line {line}: {text[:60]}" for line, text in unknown[:5])
        diagnostics.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="parser.unknown_instruction",
                message=(
                    f"{len(unknown)} instruction(s) inside /MN were not understood and were "
                    f"retained as diagnostics: {preview}"
                ),
                stage="robot.parse",
                source_path=str(program_path),
                line=unknown[0][0],
            )
        )

    if grammar.format.provisional:
        diagnostics.append(
            Diagnostic(
                severity=Severity.INFO,
                code="parser.provisional_grammar",
                message=(
                    f"{grammar.format.display_name} grammar is provisional; parser output is "
                    "recorded as observed but its dialect assumptions are unverified"
                ),
                stage="robot.parse",
                source_path=str(program_path),
            )
        )

    return ParsedProgram(
        pair_id=pair_id,
        source_path=str(program_path),
        robot_type=grammar.key,
        program_name=program_name,
        motions=motions,
        process_states=process_states,
        diagnostics=diagnostics,
        tool_frame=frames.get("UTOOL_NUM"),
        user_frame=frames.get("UFRAME_NUM"),
        unknown_lines=unknown,
    )


def _build_process_states(
    motions: list[RobotMotionRecord],
    io_events: list[tuple[int, str, bool]],
    paint_signal: str,
    paint_on_value: str,
) -> list[ProcessState]:
    """Paint state per motion, resolved by instruction order.

    Events are indexed by source instruction index; motions keep their order of
    appearance. An event that follows the last motion cannot affect any segment,
    so it is correctly ignored.
    """
    events = sorted(io_events, key=lambda item: item[0])
    paint_events = [
        (index, signal, value)
        for index, signal, value in events
        if signal == paint_signal or _normalise_signal(signal) == _normalise_signal(paint_signal)
    ]
    event_indices = [index for index, _, _ in paint_events]

    states: list[ProcessState] = []
    for motion in motions:
        insertion = bisect.bisect_right(event_indices, motion.instruction_index)
        if insertion == 0:
            states.append(
                ProcessState(
                    paint_on=None,
                    signal=paint_signal,
                    evidence=["no paint-signal assignment before this motion"],
                )
            )
            continue
        index, signal, is_on = paint_events[insertion - 1]
        paint_on = bool(is_on) if paint_on_value == "ON" else not is_on
        states.append(
            ProcessState(
                paint_on=paint_on,
                signal=signal,
                evidence=[f"{signal} last set at instruction {index} before instruction "
                          f"{motion.instruction_index}"],
            )
        )
    return states


def _normalise_signal(signal: str) -> str:
    return signal.replace(" ", "").upper()
