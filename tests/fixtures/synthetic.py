"""Synthetic fixtures with analytically known answers.

The synthetic raster exists because every real reference set is a single
observation: if the tool reports 25.3 mm on a program whose true pitch is 25 mm,
there is nothing to check it against. Here the ground truth is constructed, so a
regression in the measurement layer shows up as a number moving, not as a
plausible-looking report.

The plate is a closed 200 x 300 x 5 mm box with its top face at z = 5 mm. The
program is a serpentine raster in the XY plane at a 15 mm standoff, written in
FANUC LS so the parser is exercised on the same path the measurements are checked
against.

Everything about the geometry is stated once, in ``RasterSpec``; the mesh, the
program and the expected measurements are all derived from it, so they cannot
drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class RasterSpec:
    """Ground truth for the synthetic fixture."""

    plate_x: float = 200.0
    plate_y: float = 300.0
    plate_thickness: float = 5.0
    standoff: float = 15.0
    pitch: float = 25.0
    x_span: tuple[float, float] = (-80.0, 80.0)
    y_span: tuple[float, float] = (-100.0, 100.0)
    lead_in_mm: float = 10.0
    speed: float = 200.0
    paint_signal: int = 1
    # Target triangle edge. The plate is a box subdivided to this edge length
    # before it is written, because a 12-triangle box has averaged vertex normals
    # at every corner and two giant faces: normals, curvature and geodesics would
    # all describe the tessellation rather than the geometry. Real meshes are
    # tessellated, so the fixture is too.
    mesh_edge_mm: float = 12.5

    @property
    def top_z(self) -> float:
        return self.plate_thickness

    @property
    def tcp_z(self) -> float:
        return self.plate_thickness + self.standoff

    @property
    def stroke_count(self) -> int:
        span = self.y_span[1] - self.y_span[0]
        return int(round(span / self.pitch)) + 1

    @property
    def stroke_length(self) -> float:
        return self.x_span[1] - self.x_span[0]

    @property
    def y_values(self) -> np.ndarray:
        return np.linspace(self.y_span[0], self.y_span[1], self.stroke_count)


@dataclass(frozen=True)
class CylinderSpec:
    """Ground truth for the curved-surface fixture.

    Five axial strokes on a cylinder of radius 40 mm, 20° apart. The cylinder is the
    simplest surface where the measurements that are degenerate on a flat plate
    become meaningful: curvature is known exactly (k1 = 1/40, k2 = 0), the surface
    normal is radial, and the geodesic pitch differs from the straight-line pitch
    between the TCP poses.
    """

    radius: float = 40.0
    height: float = 120.0
    ring_step: float = 10.0
    sections: int = 64
    standoff: float = 15.0
    angle_span_deg: tuple[float, float] = (-40.0, 40.0)
    stroke_count: int = 5
    z_span: tuple[float, float] = (-50.0, 50.0)
    lead_in_mm: float = 10.0
    speed: float = 200.0
    paint_signal: int = 1

    @property
    def angles_deg(self) -> np.ndarray:
        return np.linspace(self.angle_span_deg[0], self.angle_span_deg[1], self.stroke_count)

    @property
    def tcp_radius(self) -> float:
        return self.radius + self.standoff

    @property
    def angle_step_deg(self) -> float:
        return (self.angle_span_deg[1] - self.angle_span_deg[0]) / (self.stroke_count - 1)

    @property
    def geodesic_pitch_mm(self) -> float:
        """Arc distance along the surface between neighbouring strokes."""
        return float(np.radians(self.angle_step_deg) * self.radius)

    @property
    def tcp_chord_pitch_mm(self) -> float:
        """Straight-line distance between the neighbouring TCP lines."""
        return float(
            2.0 * self.tcp_radius * np.sin(np.radians(self.angle_step_deg) / 2.0)
        )

    @property
    def curvature_per_mm(self) -> float:
        return 1.0 / self.radius


# --------------------------------------------------------------------------- #
# Meshes
# --------------------------------------------------------------------------- #


def build_plate(path: str | Path, spec: RasterSpec | None = None) -> trimesh.Trimesh:
    """Closed box with the top face at z = ``plate_thickness``, subdivided.

    The top face is flat by construction, so the true surface normal is +Z and the
    true principal curvatures are zero everywhere. Subdividing is what makes those
    truths observable: on the raw 12-triangle box, vertex normals at the corners are
    averaged across the top and both sides, and libigl reports the corner as curved.
    """
    spec = spec or RasterSpec()
    mesh = trimesh.creation.box(
        extents=[spec.plate_x, spec.plate_y, spec.plate_thickness]
    )
    mesh.apply_translation([0.0, 0.0, spec.plate_thickness / 2.0])
    while float(mesh.edges_unique_length.max()) > spec.mesh_edge_mm * 1.01:
        mesh = mesh.subdivide()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type="stl")
    return mesh


def build_cylinder_mesh(
    path: str | Path,
    spec: CylinderSpec | None = None,
) -> trimesh.Trimesh:
    """Closed faceted cylinder about +Z with its mid-height at z = 0.

    Built as an explicit ring grid rather than with ``trimesh.creation.cylinder``
    because that primitive has vertices only at the two rims: every side vertex is
    shared with a cap, so every vertex normal is a blend of radial and axial and the
    "surface normal" would be wrong everywhere. Rings make interior vertices exist,
    which is what makes the analytic truth (radial normal, k1 = 1/radius, k2 = 0)
    actually recoverable from the mesh.
    """
    spec = spec or CylinderSpec()
    sections = spec.sections
    ring_count = int(round(spec.height / spec.ring_step)) + 1
    z_values = np.linspace(-spec.height / 2.0, spec.height / 2.0, ring_count)

    vertices: list[list[float]] = [[0.0, 0.0, float(z_values[0])]]
    for z_value in z_values:
        for section in range(sections):
            angle = 2.0 * np.pi * section / sections
            vertices.append(
                [
                    spec.radius * np.cos(angle),
                    spec.radius * np.sin(angle),
                    float(z_value),
                ]
            )
    top_centre = len(vertices)
    vertices.append([0.0, 0.0, float(z_values[-1])])

    def ring(ring_index: int, section: int) -> int:
        return 1 + ring_index * sections + (section % sections)

    faces: list[list[int]] = []
    for ring_index in range(ring_count - 1):
        for section in range(sections):
            a = ring(ring_index, section)
            b = ring(ring_index, section + 1)
            c = ring(ring_index + 1, section + 1)
            d = ring(ring_index + 1, section)
            faces.append([a, d, c])
            faces.append([a, c, b])
    bottom_centre = 0
    for section in range(sections):
        faces.append([bottom_centre, ring(0, section + 1), ring(0, section)])
        faces.append([top_centre, ring(ring_count - 1, section), ring(ring_count - 1, section + 1)])

    mesh = trimesh.Trimesh(
        vertices=np.array(vertices, dtype=float),
        faces=np.array(faces, dtype=np.int64),
        process=True,
    )
    mesh.fix_normals()
    if mesh.volume < 0:
        mesh.invert()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type="stl")
    return mesh


# --------------------------------------------------------------------------- #
# Robot program
# --------------------------------------------------------------------------- #


@dataclass
class _Instruction:
    index: int
    text: str


def _wpr() -> tuple[float, float, float]:
    """W=0, P=0, R=180: rotation about X by 180°, so tool +Z points down (-Z)."""
    return 0.0, 0.0, 180.0


def build_raster_program(
    path: str | Path,
    spec: RasterSpec | None = None,
    *,
    include_io: bool = True,
    extra_unknown_instruction: bool = False,
    program_name: str = "RASTER25",
) -> dict[str, object]:
    """Write a FANUC LS serpentine raster and return its expected measurements."""
    spec = spec or RasterSpec()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    positions: list[tuple[float, float, float]] = []
    instructions: list[str] = []
    index = 0

    def emit(text: str) -> None:
        nonlocal index
        index += 1
        instructions.append(f"{index}:  {text} ;")

    def add_position(xyz: tuple[float, float, float]) -> int:
        positions.append(xyz)
        return len(positions)

    approach = add_position((0.0, 0.0, spec.tcp_z + 80.0))

    if include_io:
        emit(f"DO[{spec.paint_signal}]=OFF")
    emit(f"J P[{approach}] 100% FINE")

    if extra_unknown_instruction:
        emit("VISION_FEATURE_OFFSET")

    z = spec.tcp_z
    for stroke in range(spec.stroke_count):
        y = float(spec.y_values[stroke])
        direction = 1.0 if stroke % 2 == 0 else -1.0
        start_x = spec.x_span[0] if direction > 0 else spec.x_span[1]
        end_x = spec.x_span[1] if direction > 0 else spec.x_span[0]
        lead_x = start_x - direction * spec.lead_in_mm

        lead_in = add_position((lead_x, y, z))
        start = add_position((start_x, y, z))
        end = add_position((end_x, y, z))

        if include_io:
            emit(f"DO[{spec.paint_signal}]=OFF")
        emit(f"L P[{lead_in}] {spec.speed:.0f}mm/sec CNT100")
        if include_io:
            emit(f"DO[{spec.paint_signal}]=ON")
        emit(f"L P[{start}] {spec.speed:.0f}mm/sec CNT100")
        emit(f"L P[{end}] {spec.speed:.0f}mm/sec CNT100")

    if include_io:
        emit(f"DO[{spec.paint_signal}]=OFF")
    retreat = add_position((0.0, 0.0, spec.tcp_z + 80.0))
    emit(f"J P[{retreat}] 100% FINE")

    lines = [
        f"/PROG {program_name}",
        "/ATTR",
        "OWNER          = MNEDITOR;",
        'COMMENT        = "synthetic raster";',
        "DEFAULT_GROUP  = 1,*,*,*,*;",
        "/MN",
        *instructions,
        "/POS",
    ]

    for position_index, (x, y_value, z_value) in enumerate(positions, start=1):
        w, p, r = _wpr()
        lines.extend(
            [
                f"P[{position_index}]{{",
                "   GP1:",
                f"\tUF : 0, UT : 1,\t\tCONFIG : 'N U T, 0, 0, 0',",
                f"\tX = {x:9.3f}  mm,\tY = {y_value:9.3f}  mm,\tZ = {z_value:9.3f}  mm,",
                f"\tW = {w:9.3f} deg,\tP = {p:9.3f} deg,\tR = {r:9.3f} deg",
                "};",
            ]
        )
    lines.append("/END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "pitch_mm": spec.pitch,
        "stroke_count": spec.stroke_count,
        "stroke_length_mm": spec.stroke_length,
        "standoff_mm": spec.standoff,
        "direction_deg": 0.0,
        "y_values": [float(v) for v in spec.y_values],
        "tcp_z": spec.tcp_z,
        "top_z": spec.top_z,
        "painting_segments": spec.stroke_count,
        "instruction_count": index,
    }


# --------------------------------------------------------------------------- #
# Curved fixture: axial strokes on a cylinder
# --------------------------------------------------------------------------- #


def build_cylinder_program(
    path: str | Path,
    spec: CylinderSpec | None = None,
) -> dict[str, object]:
    """Write a FANUC LS program of axial strokes at fixed angles on the cylinder.

    Each stroke is a straight vertical pass at radius ``radius + standoff``, with the
    tool pointing radially inward. That inward attitude is expressed through W/P/R
    (W = angle + 180°, P = 90°), not through a declared tool axis, so the parser's
    Euler convention is exercised rather than bypassed.
    """
    spec = spec or CylinderSpec()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    positions: list[tuple[float, float, float, float, float, float]] = []
    instructions: list[str] = []
    index = 0

    def emit(text: str) -> None:
        nonlocal index
        index += 1
        instructions.append(f"{index}:  {text} ;")

    def add_position(
        radius: float, angle_deg: float, z_value: float, wpr: tuple[float, float, float]
    ) -> int:
        angle = np.radians(angle_deg)
        positions.append(
            (
                radius * float(np.cos(angle)),
                radius * float(np.sin(angle)),
                float(z_value),
                *wpr,
            )
        )
        return len(positions)

    down_wpr = (0.0, 0.0, 180.0)
    approach = add_position(spec.radius, 0.0, spec.height / 2.0 + 60.0, down_wpr)
    emit(f"DO[{spec.paint_signal}]=OFF")
    emit(f"J P[{approach}] 100% FINE")

    z_start, z_end = spec.z_span
    for stroke, angle_deg in enumerate(spec.angles_deg):
        direction = 1.0 if stroke % 2 == 0 else -1.0
        start_z = z_start if direction > 0 else z_end
        end_z = z_end if direction > 0 else z_start
        lead_z = start_z - direction * spec.lead_in_mm
        # Tool +Z points inward: W = angle + 180°, P = 90°.
        wpr = (float(angle_deg) + 180.0, 90.0, 0.0)

        lead_in = add_position(spec.tcp_radius, angle_deg, lead_z, wpr)
        start = add_position(spec.tcp_radius, angle_deg, start_z, wpr)
        end = add_position(spec.tcp_radius, angle_deg, end_z, wpr)

        emit(f"DO[{spec.paint_signal}]=OFF")
        emit(f"L P[{lead_in}] {spec.speed:.0f}mm/sec CNT100")
        emit(f"DO[{spec.paint_signal}]=ON")
        emit(f"L P[{start}] {spec.speed:.0f}mm/sec CNT100")
        emit(f"L P[{end}] {spec.speed:.0f}mm/sec CNT100")

    emit(f"DO[{spec.paint_signal}]=OFF")
    retreat = add_position(spec.radius, 0.0, spec.height / 2.0 + 60.0, down_wpr)
    emit(f"J P[{retreat}] 100% FINE")

    lines = [
        "/PROG CYLINDER5",
        "/ATTR",
        "OWNER          = MNEDITOR;",
        'COMMENT        = "synthetic cylinder strokes";',
        "DEFAULT_GROUP  = 1,*,*,*,*;",
        "/MN",
        *instructions,
        "/POS",
    ]
    for position_index, (x, y, z, w, p, r) in enumerate(positions, start=1):
        lines.extend(
            [
                f"P[{position_index}]{{",
                "   GP1:",
                "\tUF : 0, UT : 1,\t\tCONFIG : 'N U T, 0, 0, 0',",
                f"\tX = {x:9.3f}  mm,\tY = {y:9.3f}  mm,\tZ = {z:9.3f}  mm,",
                f"\tW = {w:9.3f} deg,\tP = {p:9.3f} deg,\tR = {r:9.3f} deg",
                "};",
            ]
        )
    lines.append("/END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "radius_mm": spec.radius,
        "standoff_mm": spec.standoff,
        "stroke_count": spec.stroke_count,
        "angles_deg": [float(v) for v in spec.angles_deg],
        "geodesic_pitch_mm": spec.geodesic_pitch_mm,
        "tcp_chord_pitch_mm": spec.tcp_chord_pitch_mm,
        "curvature_per_mm": spec.curvature_per_mm,
        "stroke_length_mm": float(spec.z_span[1] - spec.z_span[0]),
    }


def build_cylinder_project(
    directory: str | Path, *, spec: CylinderSpec | None = None
) -> tuple[Path, dict[str, object]]:
    """Create a cylinder project manifest plus its inputs."""
    spec = spec or CylinderSpec()
    root = Path(directory)
    mesh_path = root / "input" / "meshes" / "cylinder.stl"
    program_path = root / "input" / "robot_programs" / "cylinder.ls"

    build_cylinder_mesh(mesh_path, spec)
    truth = build_cylinder_program(program_path, spec)

    manifest: dict[str, object] = {
        "project_id": "synthetic_cylinder",
        "units": "mm",
        "coordinate_convention": "right_handed",
        "frames": [
            {"name": "part", "parent": "world", "transform": _IDENTITY_4X4, "source": "synthetic fixture"},
            {"name": "world", "parent": None, "transform": _IDENTITY_4X4},
        ],
        "pairs": [
            {
                "pair_id": "cylinder",
                "mesh": "input/meshes/cylinder.stl",
                "robot_program": "input/robot_programs/cylinder.ls",
                "robot_type": "fanuc",
                "mesh_frame": "part",
                "robot_frame": "world",
                "notes": "synthetic cylinder: 5 axial strokes 20 deg apart, 15 mm standoff",
            }
        ],
        "notes": "Generated by tests/fixtures/synthetic.py — curvature and pitch are analytic.",
    }
    manifest_path = root / "project.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, truth


# --------------------------------------------------------------------------- #
# Project manifest
# --------------------------------------------------------------------------- #

_IDENTITY_4X4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def build_project(
    directory: str | Path,
    *,
    spec: RasterSpec | None = None,
    declare_units: bool = True,
    declare_frame: bool = True,
    include_io: bool = True,
    robot_type: str = "fanuc",
    extra_unknown_instruction: bool = False,
) -> tuple[Path, dict[str, object]]:
    """Create ``directory/project.json`` plus inputs; return its path and truth."""
    spec = spec or RasterSpec()
    root = Path(directory)
    mesh_path = root / "input" / "meshes" / "plate.stl"
    program_path = root / "input" / "robot_programs" / "raster.ls"

    build_plate(mesh_path, spec)
    truth = build_raster_program(
        program_path,
        spec,
        include_io=include_io,
        extra_unknown_instruction=extra_unknown_instruction,
    )

    manifest: dict[str, object] = {
        "project_id": "synthetic_raster",
        "units": "mm" if declare_units else "unknown",
        "coordinate_convention": "right_handed",
        "frames": [
            {
                "name": "part",
                "parent": "world",
                "transform": _IDENTITY_4X4 if declare_frame else None,
                "source": "synthetic fixture",
            },
            {"name": "world", "parent": None, "transform": _IDENTITY_4X4},
        ],
        "pairs": [
            {
                "pair_id": "plate",
                "mesh": "input/meshes/plate.stl",
                "robot_program": "input/robot_programs/raster.ls",
                "robot_type": robot_type,
                "mesh_frame": "part",
                "robot_frame": "world",
                "notes": "synthetic 25 mm serpentine raster, 15 mm standoff",
            }
        ],
        "notes": "Generated by tests/fixtures/synthetic.py — ground truth is analytical.",
    }
    manifest_path = root / "project.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, truth
