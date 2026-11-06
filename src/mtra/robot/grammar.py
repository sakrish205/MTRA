"""Compiled grammars.

``config/robot_formats.yaml`` holds regular expressions as strings; this module
compiles them once and exposes typed accessors. Nothing here knows what a FANUC
program means — it only knows how the configured grammar is structured, so a new
dialect is a YAML entry plus an adapter that consumes this object.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from mtra.config import RobotFormat, load_robot_format
from mtra.core.errors import InputError


@dataclass(frozen=True)
class CompiledGrammar:
    format: RobotFormat
    motion: re.Pattern[str]
    position_id: re.Pattern[str]
    speed: re.Pattern[str]
    termination: re.Pattern[str]
    position_start: re.Pattern[str]
    position_end: re.Pattern[str]
    position_numeric: re.Pattern[str]
    position_frame: re.Pattern[str]
    position_config: re.Pattern[str]
    io: re.Pattern[str]
    frame_assignment: re.Pattern[str]
    program: re.Pattern[str]
    main_start: re.Pattern[str]
    positions_start: re.Pattern[str]
    end: re.Pattern[str]
    ignored: tuple[re.Pattern[str], ...]

    @property
    def key(self) -> str:
        return self.format.key


def _compile(pattern: str, label: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:  # pragma: no cover - configuration error path
        raise InputError(f"invalid regex for {label}: {exc}") from exc


@lru_cache(maxsize=None)
def compile_grammar(key: str) -> CompiledGrammar:
    fmt = load_robot_format(key)
    if not fmt.supported:
        raise InputError(f"{fmt.display_name} adapter is not implemented: {fmt.note}")
    raw = fmt.raw
    sections = raw.get("sections", {})
    position = raw.get("position_block", {})
    io = raw.get("io", {})
    frames = raw.get("frame_assignment", {})
    io_pattern = io.get("pattern")
    frame_pattern = frames.get("both") or frames.get("tool_frame")
    for label, value in {
        "sections.program": sections.get("program"),
        "sections.main": sections.get("main"),
        "sections.positions": sections.get("positions"),
        "sections.end": sections.get("end"),
        "motion.pattern": raw.get("motion", {}).get("pattern"),
        "position_block.start": position.get("start"),
        "position_block.end": position.get("end"),
        "io.pattern": io_pattern,
        "frame_assignment.both": frame_pattern,
    }.items():
        if not value:
            raise InputError(f"grammar {key}: required pattern {label} is missing")

    return CompiledGrammar(
        format=fmt,
        motion=_compile(str(raw["motion"]["pattern"]), f"{key}.motion"),
        position_id=_compile(
            str(raw["motion"].get("position_id_pattern", r"\[\s*(?P<id>[^\]]+?)\s*\]")),
            f"{key}.position_id",
        ),
        speed=_compile(
            str(raw["motion"].get("speed_pattern", r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>\w+)")),
            f"{key}.speed",
        ),
        termination=_compile(
            str(raw["motion"].get("termination_pattern", r"(?P<termination>FINE|CNT\d+)")),
            f"{key}.termination",
        ),
        position_start=_compile(str(position["start"]), f"{key}.position_start"),
        position_end=_compile(str(position["end"]), f"{key}.position_end"),
        position_numeric=_compile(
            str(position.get("numeric_field", r"(?P<field>[XYZ])\s*=\s*(?P<value>-?\d+(?:\.\d+)?)")),
            f"{key}.position_numeric",
        ),
        position_frame=_compile(
            str(position.get("frame_field", r"(?P<field>UF|UT)\s*:\s*(?P<value>-?\d+)")),
            f"{key}.position_frame",
        ),
        position_config=_compile(
            str(position.get("config_field", r"CONFIG\s*:\s*'(?P<value>[^']*)'")),
            f"{key}.position_config",
        ),
        io=_compile(str(io_pattern), f"{key}.io"),
        frame_assignment=_compile(str(frame_pattern), f"{key}.frame_assignment"),
        program=_compile(str(sections["program"]), f"{key}.program"),
        main_start=_compile(str(sections["main"]), f"{key}.main_start"),
        positions_start=_compile(str(sections["positions"]), f"{key}.positions_start"),
        end=_compile(str(sections["end"]), f"{key}.end"),
        ignored=tuple(
            _compile(str(pattern), f"{key}.ignored[{index}]")
            for index, pattern in enumerate(raw.get("ignored", []))
        ),
    )


def is_ignored(grammar: CompiledGrammar, line: str) -> bool:
    return any(pattern.search(line) for pattern in grammar.ignored)
