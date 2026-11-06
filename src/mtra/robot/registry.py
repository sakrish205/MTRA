"""Adapter registry.

One dispatch point, so no stage above it imports a controller module. Adding a
dialect means adding an adapter here and flipping ``supported`` in the grammar
config — and the failure mode for an unimplemented dialect is a loud
``UnsupportedInput`` naming the PRD gate, not a half-parse.
"""

from __future__ import annotations

from mtra.config import Settings, load_settings
from mtra.core.errors import UnsupportedInput
from mtra.project.validation import require_supported_format
from mtra.robot.common_schema import ParsedProgram
from mtra.robot.fanuc_ls import parse_program as parse_fanuc


def parse_program(
    path: str,
    pair_id: str,
    robot_type: str,
    *,
    settings: Settings | None = None,
) -> ParsedProgram:
    settings = settings or load_settings()
    key = robot_type.strip().lower()
    require_supported_format(key)

    if key == "fanuc":
        return parse_fanuc(path, pair_id, grammar_key="fanuc", settings=settings)

    raise UnsupportedInput(f"no adapter registered for robot type {robot_type!r}")
