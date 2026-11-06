"""Mesh ingestion: format, canonical units, quality facts, and refusal.

The plate fixture is a subdivided box, so its properties are known: extents
200 x 300 x 5 mm, watertight, and a triangle edge length under the target
tessellation size. A regression in scaling or triangulation moves these numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtra.core.errors import AnalysisBlocked
from mtra.core.types import ReasonCode
from mtra.core.units import LengthUnit
from mtra.mesh.loader import as_trimesh, load_mesh


@pytest.fixture(scope="module")
def plate_mesh(plate_project):
    manifest, _truth = plate_project
    mesh_path = manifest.parent / "input" / "meshes" / "plate.stl"
    return load_mesh(mesh_path, "plate", LengthUnit.MILLIMETRE)


def test_plate_is_loaded_in_millimetres_with_declared_units(plate_mesh):
    assert plate_mesh.declared_units == "mm"
    assert plate_mesh.canonical_units == "mm"
    assert plate_mesh.format in {"stl-binary", "stl-ascii"}


def test_plate_geometry_matches_the_spec(plate_mesh):
    vertices = np.asarray(plate_mesh.vertices)
    faces = np.asarray(plate_mesh.faces)
    assert vertices.shape[0] > 1000, "fixture must be tessellated, not a 12-triangle box"
    assert faces.shape[1] == 3

    extents = vertices.max(axis=0) - vertices.min(axis=0)
    assert extents == pytest.approx([200.0, 300.0, 5.0], abs=1e-6)
    assert vertices[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    assert vertices[:, 2].max() == pytest.approx(5.0, abs=1e-9)


def test_plate_quality_claims(plate_mesh):
    quality = plate_mesh.mesh_quality
    assert quality["watertight"] is True
    assert quality["component_count"] == 1
    assert quality["degenerate_face_count"] == 0
    assert quality["surface_area_mm2"] == pytest.approx(
        2 * (200 * 300 + 200 * 5 + 300 * 5), rel=1e-6
    )
    assert quality["edge_length_mm"]["p95"] <= 12.6


def test_unknown_units_block_rather_than_defaulting(plate_project):
    manifest, _truth = plate_project
    mesh_path = manifest.parent / "input" / "meshes" / "plate.stl"
    with pytest.raises(AnalysisBlocked) as excinfo:
        load_mesh(mesh_path, "plate", LengthUnit.UNKNOWN)
    assert excinfo.value.reason is ReasonCode.UNKNOWN_UNITS


def test_trimesh_view_is_the_same_surface(plate_mesh):
    mesh = as_trimesh(plate_mesh)
    assert mesh.vertices.shape == (len(plate_mesh.vertices), 3)
    assert mesh.faces.shape == (len(plate_mesh.faces), 3)
    assert mesh.is_watertight
