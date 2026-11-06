"""Parser output contract.

Every controller adapter produces a ``ParsedProgram``. Downstream stages know
nothing about LS or RAPID, which is what PRD §16 (maintainability) asks for.

``ProcessState`` is stored per motion instruction, not per I/O event, because
classification asks "was the process on while moving from pose i to pose i+1?" —
and an I/O event that never precedes a motion has no trajectory meaning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mtra.core.types import Diagnostic, ProcessState, RobotMotionRecord


@dataclass
class ParsedProgram:
    pair_id: str
    source_path: str
    robot_type: str
    program_name: str
    motions: list[RobotMotionRecord] = field(default_factory=list)
    process_states: list[ProcessState] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    tool_frame: str | None = None
    user_frame: str | None = None
    unknown_lines: list[tuple[int, str]] = field(default_factory=list)

    @property
    def motion_count(self) -> int:
        return len(self.motions)

    @property
    def covered_instruction_count(self) -> int:
        return len(self.motions) + len(self.unknown_lines)

    def positions(self) -> np.ndarray:
        if not self.motions:
            return np.zeros((0, 3))
        return np.vstack([m.tcp for m in self.motions])

    def has_process_state(self) -> bool:
        """True when at least one motion has a resolved paint on/off state."""
        return any(state.paint_on is not None for state in self.process_states)
