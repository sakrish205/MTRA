"""Errors that mean "stop", not "try harder".

MTRA's failure mode of concern is not an exception — it is a plausible number
produced from an unjustified assumption. So blocked analysis is a typed error
carrying the reason code that will appear in the report, rather than a log line
and a default.
"""

from __future__ import annotations

from mtra.core.types import ReasonCode


class MTRAError(Exception):
    """Base class for all MTRA errors."""


class InputError(MTRAError):
    """The input is malformed, missing or self-contradictory."""


class UnsupportedInput(MTRAError):
    """The input is well-formed but this build cannot handle it.

    Raised for robot dialects without a validated grammar. PRD Phase 0 gate:
    never guess a controller syntax.
    """


class AnalysisBlocked(MTRAError):
    """A precondition is unmet, so downstream numbers would be meaningless."""

    def __init__(self, reason: ReasonCode, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")
