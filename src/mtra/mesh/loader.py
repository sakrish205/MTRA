"""Mesh ingestion.

Three commitments, all traceable to PRD §5.1 and §11.3:

* The source file is never modified (read-only load, raw counts preserved).
* Every modification MTRA makes — vertex merge, degenerate-face removal, unit
  scaling — is appended to ``transform_history`` so the report can state the
  analysis representation differs from the source.
* Units are applied from an explicit declaration. There is no default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from mtra.core.errors import InputError
from mtra.core.types import Diagnostic, MeshRecord, Severity
from mtra.core.units import LengthUnit, factor_to_millimetres
from mtra.mesh import quality as quality_mod
from mtra.mesh import topology as topology_mod
from mtra.project.inventory import detect_mesh_format


def _as_trimesh_object(loaded: object) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise InputError("scene contains no geometry")
        dumped = loaded.dump(concatenate=True)
        if not isinstance(dumped, trimesh.Trimesh):
            raise InputError("could not concatenate scene geometry into a single mesh")
        return dumped
    raise InputError(f"trimesh returned an unsupported object type: {type(loaded)!r}")


def load_mesh(
    path: str | Path,
    pair_id: str,
    unit: LengthUnit,
    *,
    diagnostics: list[Diagnostic] | None = None,
) -> MeshRecord:
    """Load a mesh into canonical millimetres for analysis.

    ``unit`` must be declared; ``LengthUnit.UNKNOWN`` raises ``AnalysisBlocked``
    via ``factor_to_millimetres`` rather than defaulting to millimetres.
    """
    diagnostics = diagnostics if diagnostics is not None else []
    mesh_path = Path(path)
    guess = detect_mesh_format(mesh_path)
    if guess.format not in {"stl-binary", "stl-ascii", "obj"}:
        raise InputError(f"unsupported mesh format for {mesh_path}: {guess.format}")
    if not guess.confident:
        diagnostics.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="mesh.format_uncertain",
                message=f"{mesh_path.name}: {guess.detail}",
                stage="mesh.load",
                source_path=str(mesh_path),
            )
        )

    try:
        loaded = trimesh.load(mesh_path, force="mesh", process=False)
    except Exception as exc:  # trimesh raises a variety of types
        raise InputError(f"failed to read mesh {mesh_path}: {exc}") from exc

    raw = _as_trimesh_object(loaded)
    raw_vertex_count = int(len(raw.vertices))
    raw_face_count = int(len(raw.faces))

    analysis = raw.copy()
    analysis.merge_vertices()
    merged_vertices = int(len(analysis.vertices))
    analysis.update_faces(analysis.nondegenerate_faces())
    analysis.remove_unreferenced_vertices()
    final_vertices = int(len(analysis.vertices))
    final_faces = int(len(analysis.faces))

    scale = factor_to_millimetres(unit)
    history = [
        f"loaded {guess.format} vertex_count={raw_vertex_count} face_count={raw_face_count}",
        f"merge_vertices: {raw_vertex_count} -> {merged_vertices} vertices",
        f"remove_degenerate_faces: {raw_face_count} -> {final_faces} faces",
        f"unit {unit} -> mm: scale={scale}",
    ]

    vertices = np.asarray(analysis.vertices, dtype=float) * scale
    faces = np.asarray(analysis.faces, dtype=np.int64)

    if final_faces == 0 or final_vertices == 0:
        raise InputError(f"{mesh_path.name}: no usable triangles after cleaning")

    scaled = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh_quality = quality_mod.analyse(raw, scaled)
    component_ids, boundary_loops = topology_mod.analyse(scaled, faces)

    if not scaled.is_watertight:
        diagnostics.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="mesh.not_watertight",
                message=(
                    f"{mesh_path.name}: mesh is not watertight "
                    f"({len(boundary_loops)} boundary loop(s)); surface-interior "
                    "assumptions (inside/outside, ray parity) are unreliable"
                ),
                stage="mesh.load",
                source_path=str(mesh_path),
            )
        )
    if mesh_quality["duplicate_vertex_count"] > 0:
        diagnostics.append(
            Diagnostic(
                severity=Severity.INFO,
                code="mesh.duplicate_vertices",
                message=(
                    f"{mesh_path.name}: {mesh_quality['duplicate_vertex_count']} duplicate "
                    "vertex positions in source; merged for analysis"
                ),
                stage="mesh.load",
                source_path=str(mesh_path),
            )
        )
    if mesh_quality["degenerate_face_count"] > 0:
        diagnostics.append(
            Diagnostic(
                severity=Severity.INFO,
                code="mesh.degenerate_faces",
                message=(
                    f"{mesh_path.name}: {mesh_quality['degenerate_face_count']} degenerate "
                    "triangles removed from the analysis representation"
                ),
                stage="mesh.load",
                source_path=str(mesh_path),
            )
        )

    return MeshRecord(
        pair_id=pair_id,
        source_path=str(mesh_path),
        format=guess.format,
        declared_units=str(unit),
        vertices=vertices,
        faces=faces,
        transform_history=history,
        component_ids=component_ids,
        boundary_loops=boundary_loops,
        mesh_quality=mesh_quality,
    )


def as_trimesh(record: MeshRecord) -> trimesh.Trimesh:
    """Rebuild a trimesh object from a record, at analysis resolution.

    Rebuilt rather than cached: ``MeshRecord`` is the contract, and a stale scene
    cache is a subtle way for two stages to disagree about the geometry.
    """
    return trimesh.Trimesh(
        vertices=np.asarray(record.vertices, dtype=float),
        faces=np.asarray(record.faces, dtype=np.int64),
        process=False,
    )
