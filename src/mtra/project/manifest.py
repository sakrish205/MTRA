"""Project manifest: the only place a mesh is bound to a robot program.

PRD §5.3 and §19 make pairing an explicit, reviewed act. MTRA does not guess a
pair from filenames, because a wrong pair silently invalidates every relationship
downstream — and it would look exactly like a real, surprising finding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mtra.core.errors import InputError
from mtra.core.units import LengthUnit

Vector3 = tuple[float, float, float]


class ToolSpec(BaseModel):
    """Tool/TCP data. Absence is meaningful, so every field defaults to None."""

    model_config = ConfigDict(extra="forbid")

    tcp_offset_mm: Vector3 | None = None
    tool_axis: Vector3 | None = None
    known_standoff_mm: float | None = None
    workobject: str | None = None
    notes: str = ""


class FrameSpec(BaseModel):
    """A declared coordinate frame.

    ``transform`` is the 4x4 mapping *from* the mesh/part frame *to* the robot
    world frame; when it is None the frame is declared but uncalibrated, which
    downgrades correspondence to an estimate.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    parent: str | None = None
    transform: list[list[float]] | None = None
    source: str = "manifest"

    @field_validator("transform")
    @classmethod
    def _check_4x4(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return None
        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("frame transform must be a 4x4 matrix")
        return value


class PairSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_id: str
    mesh: str
    robot_program: str
    robot_type: str
    mesh_frame: str = "part"
    robot_frame: str = "world"
    units: str | None = None
    tool: ToolSpec | None = None
    notes: str = ""

    @field_validator("pair_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("pair_id must not be empty")
        return value


class ProjectSpec(BaseModel):
    """The manifest as written, before path resolution."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    units: str = "unknown"
    coordinate_convention: Literal["right_handed", "left_handed", "unknown"] = "unknown"
    pairs: list[PairSpec] = Field(default_factory=list)
    frames: list[FrameSpec] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _check_pairs(self) -> ProjectSpec:
        if not self.pairs:
            raise ValueError("manifest declares no mesh/robot pairs")
        seen: set[str] = set()
        for pair in self.pairs:
            if pair.pair_id in seen:
                raise ValueError(f"duplicate pair_id: {pair.pair_id}")
            seen.add(pair.pair_id)
        return self

    @property
    def declared_unit(self) -> LengthUnit:
        return LengthUnit.parse(self.units)

    def unit_for(self, pair: PairSpec) -> LengthUnit:
        """Pair override, else project declaration, else unknown."""
        if pair.units is not None:
            return LengthUnit.parse(pair.units)
        return self.declared_unit

    def frame(self, name: str) -> FrameSpec | None:
        for frame in self.frames:
            if frame.name == name:
                return frame
        return None


class ProjectPaths(BaseModel):
    """Resolved filesystem locations, so no stage has to do path arithmetic."""

    base_dir: Path
    manifest_path: Path
    output_dir: Path

    def mesh_path(self, pair: PairSpec) -> Path:
        return (self.base_dir / pair.mesh).resolve()

    def program_path(self, pair: PairSpec) -> Path:
        return (self.base_dir / pair.robot_program).resolve()


class Project(BaseModel):
    spec: ProjectSpec
    paths: ProjectPaths

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def project_id(self) -> str:
        return self.spec.project_id


def load_manifest(path: str | Path) -> Project:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise InputError(f"project manifest not found: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"{manifest_path} is not valid JSON: {exc}") from exc

    try:
        spec = ProjectSpec.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError, expressed in MTRA terms
        raise InputError(f"{manifest_path} fails schema validation: {exc}") from exc

    base_dir = manifest_path.parent
    output_dir = base_dir / "output"
    return Project(
        spec=spec,
        paths=ProjectPaths(
            base_dir=base_dir, manifest_path=manifest_path.resolve(), output_dir=output_dir
        ),
    )
