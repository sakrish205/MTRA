"""Canonical data contract.

Every stage of MTRA speaks in these records. Source files are read exactly once,
by an adapter, and never again downstream — which is what makes robot-format
adapters replaceable and the report a pure function of records.

Two type-level rules carry most of the tool's honesty:

* A measurement either has a value *or* an ``Unavailable``. There is no third
  state, so a missing number can never be rendered as zero or silently dropped.
* ``EvidenceStatus`` travels with every record, so "observed", "inferred" and
  "validated" are structural properties rather than adjectives in prose.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class EvidenceStatus(StrEnum):
    """PRD §10. Drives the wording a report is allowed to use."""

    OBSERVED = "observed"
    INFERRED = "inferred"
    VALIDATED = "validated"
    UNRESOLVED = "unresolved"
    REJECTED = "rejected"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class MotionType(StrEnum):
    JOINT = "joint"
    LINEAR = "linear"
    CIRCULAR = "circular"
    UNKNOWN = "unknown"


class SegmentClass(StrEnum):
    PAINTING = "painting"
    APPROACH = "approach"
    DEPARTURE = "departure"
    TRANSFER = "transfer"
    UNKNOWN = "unknown"


class ReasonCode(StrEnum):
    """Why a catalogue parameter has no value.

    These are report-visible vocabulary, not log strings: when an engineer asks
    "why is standoff blank?", the answer is one of these, with a detail line.
    """

    OUT_OF_PHASE_SCOPE = "out_of_phase_scope"
    MISSING_TOOL_DEFINITION = "missing_tool_definition"
    UNKNOWN_TOOL_AXIS = "unknown_tool_axis"
    UNKNOWN_UNITS = "unknown_units"
    UNKNOWN_FRAME = "unknown_frame"
    NO_PROCESS_STATE = "no_process_state"
    NO_PAINTING_SEGMENTS = "no_painting_segments"
    INSUFFICIENT_STROKES = "insufficient_strokes"
    NO_SURFACE_CORRESPONDENCE = "no_surface_correspondence"
    NO_RAY_INTERSECTION = "no_ray_intersection"
    MULTIPLE_RAY_INTERSECTIONS = "multiple_ray_intersections"
    DEGENERATE_GEOMETRY = "degenerate_geometry"
    OPTIONAL_DEPENDENCY_MISSING = "optional_dependency_missing"
    COMPUTATION_FAILED = "computation_failed"
    # A parameter that is a property of the declared calibration (or of a later
    # phase's method) and therefore cannot be measured from the supplied inputs.
    NO_INDEPENDENT_MEASUREMENT = "no_independent_measurement"
    PROCESS_STATE_UNKNOWN = "process_state_unknown"


class Unavailable(BaseModel):
    """A first-class absence. Never rendered as a value."""

    model_config = ConfigDict(frozen=True)

    reason: ReasonCode
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.reason}: {self.detail}" if self.detail else str(self.reason)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


class Diagnostic(BaseModel):
    """A factual note about the input or the analysis.

    Severity ``error`` means the stage could not do its job. ``warning`` means it
    did, but something is being assumed or approximated that the engineer should
    know about.
    """

    severity: Severity
    code: str
    message: str
    stage: str | None = None
    source_path: str | None = None
    line: int | None = None


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


class SourceFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    role: str
    sha256: str
    size_bytes: int
    detected_format: str


class Provenance(BaseModel):
    """Run-level reproducibility record (PRD §11.3)."""

    generated_at_utc: str
    mtra_version: str
    python_version: str
    package_versions: dict[str, str] = Field(default_factory=dict)
    source_files: list[SourceFile] = Field(default_factory=list)
    configuration: dict[str, Any] = Field(default_factory=dict)
    random_seed: int = 0


# --------------------------------------------------------------------------- #
# Canonical records (PRD §11.2)
# --------------------------------------------------------------------------- #


class MeshRecord(BaseModel):
    """A loaded mesh, in canonical millimetres, with its own history.

    ``vertices``/``faces`` are arrays rather than lists because every consumer is
    numeric. They are excluded from serialisation; ``summary()`` is what goes into
    JSON, so artefacts stay diffable and small.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pair_id: str
    source_path: str
    format: str
    declared_units: str
    canonical_units: str = "mm"
    vertices: np.ndarray = Field(exclude=True)
    faces: np.ndarray = Field(exclude=True)
    transform_history: list[str] = Field(default_factory=list)
    component_ids: np.ndarray | None = Field(default=None, exclude=True)
    boundary_loops: list[np.ndarray] = Field(default_factory=list, exclude=True)
    mesh_quality: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "source_path": self.source_path,
            "format": self.format,
            "declared_units": self.declared_units,
            "canonical_units": self.canonical_units,
            "vertex_count": int(self.vertices.shape[0]),
            "face_count": int(self.faces.shape[0]),
            "transform_history": self.transform_history,
            "mesh_quality": self.mesh_quality,
        }


class RobotMotionRecord(BaseModel):
    """One instruction that moves the robot.

    ``source_line`` and ``instruction_index`` are mandatory: PRD FR-04 makes
    traceability a requirement, and an untraceable parse cannot be reviewed.
    """

    pair_id: str
    source_file: str
    instruction_index: int
    motion_type: MotionType
    tcp_position: tuple[float, float, float]
    tcp_orientation: tuple[float, float, float]  # raw W,P,R as written by the controller
    rotation_matrix: list[list[float]]
    position_id: str | None = None
    speed: float | None = None
    speed_unit: str | None = None
    termination: str | None = None
    tool_frame: str | None = None
    user_frame: str | None = None
    source_line: int
    raw: str

    @property
    def tcp(self) -> np.ndarray:
        return np.asarray(self.tcp_position, dtype=float)

    @property
    def rotation(self) -> np.ndarray:
        return np.asarray(self.rotation_matrix, dtype=float)

    @property
    def tool_axis(self) -> np.ndarray:
        """Tool +Z in world coordinates — the only defensible spray direction."""
        return self.rotation[:, 2]


class ProcessState(BaseModel):
    """I/O-derived process state at a point in the program."""

    paint_on: bool | None = None
    signal: str | None = None
    source_line: int | None = None
    evidence: list[str] = Field(default_factory=list)


class TrajectorySegment(BaseModel):
    """A run of motion records with one classification and its justification."""

    segment_id: str
    pair_id: str
    segment_class: SegmentClass
    ordered_motion_ids: list[int]
    start_pose: tuple[float, float, float]
    end_pose: tuple[float, float, float]
    length_mm: float
    confidence: float = Field(ge=0.0, le=1.0)
    classification_evidence: list[str] = Field(default_factory=list)
    status: EvidenceStatus = EvidenceStatus.OBSERVED

    @property
    def start(self) -> np.ndarray:
        return np.asarray(self.start_pose, dtype=float)

    @property
    def end(self) -> np.ndarray:
        return np.asarray(self.end_pose, dtype=float)


class CorrespondenceRecord(BaseModel):
    """Where one trajectory sample lands on the mesh.

    ``tcp_position`` and ``impact_point`` are deliberately separate fields
    (PRD FR-07, §19): standoff lives between them and confusing the two is the
    single most common way to invent a spacing rule that does not exist.
    """

    sample_id: str
    pair_id: str
    segment_id: str | None
    tcp_position: tuple[float, float, float]
    closest_point: tuple[float, float, float]
    mesh_face_id: int
    barycentric: tuple[float, float, float]
    distance_mm: float
    impact_point: tuple[float, float, float] | None = None
    impact_distance_mm: float | None = None
    ray_hit_count: int = 0
    method: str = "closest_point"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class MeasurementRecord(BaseModel):
    """One catalogue entry, resolved or explicitly unresolved.

    Exactly one of ``value`` and ``unavailable`` is set — enforced in
    ``model_post_init`` so a half-filled record cannot be constructed.
    """

    parameter_id: str
    group: str
    name: str
    pair_id: str
    value: Any | None = None
    unit: str = ""
    method: str = ""
    region_id: str | None = None
    stroke_id: str | None = None
    location: tuple[float, float, float] | None = None
    status: EvidenceStatus = EvidenceStatus.OBSERVED
    uncertainty: float | None = None
    source_refs: list[str] = Field(default_factory=list)
    unavailable: Unavailable | None = None

    def model_post_init(self, __context: Any) -> None:
        if (self.value is None) == (self.unavailable is None):
            raise ValueError(
                f"{self.parameter_id}: exactly one of value/unavailable must be set"
            )

    @property
    def is_available(self) -> bool:
        return self.value is not None

    def as_row(self) -> dict[str, Any]:
        return {
            "parameter_id": self.parameter_id,
            "group": self.group,
            "name": self.name,
            "pair_id": self.pair_id,
            "value": self.value if self.is_available else "",
            "unit": self.unit,
            "status": str(self.status),
            "reason": str(self.unavailable.reason) if self.unavailable else "",
            "detail": self.unavailable.detail if self.unavailable else "",
            "method": self.method,
            "uncertainty": self.uncertainty,
            "region_id": self.region_id,
            "stroke_id": self.stroke_id,
            "source_refs": "; ".join(self.source_refs),
        }


class StageContext(BaseModel):
    """Everything one stage may read, and everything the next stage inherits."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    manifest: Any
    meshes: dict[str, MeshRecord] = Field(default_factory=dict)
    motions: dict[str, list[RobotMotionRecord]] = Field(default_factory=dict)
    process_states: dict[str, list[ProcessState]] = Field(default_factory=dict)
    segments: dict[str, list[TrajectorySegment]] = Field(default_factory=dict)
    transforms: dict[str, list[list[float]]] = Field(default_factory=dict)
    correspondences: dict[str, list[CorrespondenceRecord]] = Field(default_factory=dict)
    measurements: list[MeasurementRecord] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    provenance: Provenance | None = None
    stage_log: list[dict[str, Any]] = Field(default_factory=list)
    output_dir: str | None = None
    blocked: bool = False

    def note(self, diagnostic: Diagnostic) -> None:
        self.diagnostics.append(diagnostic)

    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == Severity.ERROR]

    def has_blocking_errors(self) -> bool:
        return self.blocked or bool(self.errors())
