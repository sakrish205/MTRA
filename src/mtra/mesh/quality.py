"""Mesh quality and resolution statistics.

These numbers exist to stop tessellation artefacts being read as engineer intent.
PRD §19 ("Mesh tessellation artefacts") and parameter D9 depend on being able to
say, in the report, what the surface representation actually looked like.
"""

from __future__ import annotations

import numpy as np
import trimesh

DEGENERATE_AREA_MM2 = 1e-10


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def _edge_counts(mesh: trimesh.Trimesh) -> tuple[int, int]:
    """Boundary edges appear once; non-manifold edges belong to three or more faces."""
    if len(mesh.faces) == 0:
        return 0, 0
    edges = np.sort(mesh.edges_sorted, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = int(np.count_nonzero(counts == 1))
    non_manifold = int(np.count_nonzero(counts > 2))
    return boundary, non_manifold


def _duplicate_vertex_count(raw: trimesh.Trimesh) -> int:
    """Duplicate *positions*, not indices: STL has no vertex identity at all."""
    vertices = np.asarray(raw.vertices, dtype=float)
    if vertices.size == 0:
        return 0
    rounded = np.round(vertices, decimals=9)
    unique = np.unique(rounded, axis=0)
    return int(len(vertices) - len(unique))


def analyse(raw: trimesh.Trimesh, analysis: trimesh.Trimesh) -> dict[str, object]:
    """Quality of the analysis representation, with source counts for comparison."""
    face_areas = np.asarray(analysis.area_faces, dtype=float)
    boundary_edges, non_manifold_edges = _edge_counts(analysis)
    edge_lengths = (
        np.linalg.norm(
            analysis.vertices[analysis.edges_unique[:, 0]]
            - analysis.vertices[analysis.edges_unique[:, 1]],
            axis=1,
        )
        if len(analysis.edges_unique)
        else np.array([])
    )

    bounds = analysis.bounds if len(analysis.vertices) else np.zeros((2, 3))
    quality: dict[str, object] = {
        "source_vertex_count": int(len(raw.vertices)),
        "source_face_count": int(len(raw.faces)),
        "vertex_count": int(len(analysis.vertices)),
        "face_count": int(len(analysis.faces)),
        "duplicate_vertex_count": _duplicate_vertex_count(raw),
        "degenerate_face_count": int(np.count_nonzero(face_areas < DEGENERATE_AREA_MM2)),
        "boundary_edge_count": boundary_edges,
        "non_manifold_edge_count": non_manifold_edges,
        "component_count": int(len(analysis.split(only_watertight=False))),
        "watertight": bool(analysis.is_watertight),
        "winding_consistent": bool(analysis.is_winding_consistent),
        "volume_mm3": float(analysis.volume) if analysis.is_watertight else None,
        "surface_area_mm2": float(analysis.area),
        "bbox_min": [float(v) for v in bounds[0]],
        "bbox_max": [float(v) for v in bounds[1]],
        "extents_mm": [float(v) for v in analysis.extents],
        "edge_length_mm": _percentiles(edge_lengths),
        "triangle_area_mm2": _percentiles(face_areas),
    }
    return quality
