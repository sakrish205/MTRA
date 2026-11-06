"""Frame algebra and rigid fitting.

Two claims are pinned here:

* correspondence is driven by a *declared* transform; an undeclared one blocks
  rather than being estimated (a wrong fit looks exactly like a finding);
* the rigid-fit arithmetic that Phase 2 will use is correct, and refuses when the
  point set cannot constrain a rotation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mtra.core.errors import AnalysisBlocked
from mtra.core.types import ReasonCode
from mtra.correspondence import frames as frame_ops
from mtra.correspondence.registration import conditioning_ratio, rigid_fit
from mtra.project.manifest import load_manifest


def _project(tmp_path, *, transform, mesh_frame="part", robot_frame="world", same=False):
    manifest = {
        "project_id": "frames",
        "units": "mm",
        "coordinate_convention": "right_handed",
        "frames": [
            {"name": "part", "parent": "world", "transform": transform},
            {"name": "world", "parent": None, "transform": np.eye(4).tolist()},
        ],
        "pairs": [
            {
                "pair_id": "p",
                "mesh": "input/meshes/plate.stl",
                "robot_program": "input/robot_programs/raster.ls",
                "robot_type": "fanuc",
                "mesh_frame": "world" if same else mesh_frame,
                "robot_frame": robot_frame,
            }
        ],
    }
    path = tmp_path / "project.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return load_manifest(path)


def _transform(z_offset: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[2, 3] = z_offset
    return matrix


def test_declared_transform_is_used_verbatim(tmp_path):
    declared = _transform(37.5)
    project = _project(tmp_path, transform=declared.tolist())
    pair = project.spec.pairs[0]

    matrix, source, is_declared, notes = frame_ops.resolve_pair_transform(project, pair)

    assert matrix == pytest.approx(declared)
    assert source == "declared:part"
    assert is_declared is True
    assert notes and "manifest" in notes[0]


def test_same_frame_names_use_identity_and_say_so(tmp_path):
    project = _project(tmp_path, transform=None, same=True)
    pair = project.spec.pairs[0]

    matrix, source, _declared, notes = frame_ops.resolve_pair_transform(project, pair)

    assert matrix == pytest.approx(np.eye(4))
    assert "identity" in source
    assert any("identity" in note for note in notes)


def test_undeclared_transform_blocks_instead_of_estimating(tmp_path):
    project = _project(tmp_path, transform=None)
    pair = project.spec.pairs[0]

    with pytest.raises(AnalysisBlocked) as excinfo:
        frame_ops.resolve_pair_transform(project, pair)

    assert excinfo.value.reason is ReasonCode.UNKNOWN_FRAME
    assert "will not estimate" in excinfo.value.detail


def test_invert_and_compose_are_consistent():
    matrix = frame_ops.compose(
        _transform(10.0),
        np.array(
            [
                [0.0, -1.0, 0.0, 5.0],
                [1.0, 0.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert frame_ops.compose(matrix, frame_ops.invert(matrix)) == pytest.approx(np.eye(4))

    points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 2.0]])
    round_trip = frame_ops.apply(
        frame_ops.invert(matrix), frame_ops.apply(matrix, points)
    )
    assert round_trip == pytest.approx(points)


def test_rotation_difference_reports_degrees_not_matrix_deltas():
    quarter = np.eye(4)
    quarter[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert frame_ops.rotation_difference_deg(np.eye(4), quarter) == pytest.approx(90.0)
    assert frame_ops.rotation_difference_deg(quarter, quarter) == pytest.approx(0.0)


def test_rigid_fit_recovers_a_known_transform_exactly():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(40, 3)) * 50.0
    angle = np.radians(30.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([12.0, -7.5, 3.25])
    target = source @ rotation.T + translation

    fit = rigid_fit(source, target)

    assert fit.is_degenerate is False
    assert fit.rotation == pytest.approx(rotation, abs=1e-9)
    assert fit.translation == pytest.approx(translation, abs=1e-9)
    assert fit.matrix[:3, :3] == pytest.approx(rotation, abs=1e-9)
    assert fit.matrix[:3, 3] == pytest.approx(translation, abs=1e-9)
    assert fit.matrix[3] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert fit.rms_mm == pytest.approx(0.0, abs=1e-9)


def test_rigid_fit_flags_collinear_points_as_degenerate():
    line = np.linspace(0.0, 100.0, 20)
    points = np.column_stack([line, line * 0.5, line * 0.25])

    _, ratio = conditioning_ratio(points)
    fit = rigid_fit(points, points + 1.0)

    assert ratio < 1e-9
    assert fit.is_degenerate is True


def test_rigid_fit_flags_planar_points_as_degenerate():
    rng = np.random.default_rng(11)
    planar = rng.normal(size=(30, 2)) * 40.0
    planar = np.column_stack([planar, np.zeros(len(planar))])

    fit = rigid_fit(planar, planar + np.array([0.0, 0.0, 5.0]))

    assert fit.conditioning_ratio < 0.02
    assert fit.is_degenerate is True


def test_rigid_fit_rejects_an_underdetermined_input():
    with pytest.raises(ValueError):
        rigid_fit(np.zeros((2, 3)), np.zeros((2, 3)))
