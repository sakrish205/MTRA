"""Measurements against constructed ground truth.

Every number asserted here is known by construction in ``tests.fixtures.synthetic``
(pitch 25 mm, standoff 15 mm, cylinder radius 40 mm), so a regression in a
measurement module shows up as a changed value rather than as a plausible report.
Tolerances are stated per group and reflect the geometry, not convenience.
"""

from __future__ import annotations

import pytest

from mtra.core.types import ReasonCode


def _by_id(result):
    return {record.parameter_id: record for record in result.pairs[0].measurements}


def _value(records, parameter_id):
    record = records[parameter_id]
    assert record.is_available, f"{parameter_id} unexpectedly unavailable: {record.unavailable}"
    return record.value


def test_every_catalogue_parameter_is_accounted_for_exactly_once(plate_result, cylinder_result):
    for result in (plate_result, cylinder_result):
        records = result.pairs[0].measurements
        ids = [record.parameter_id for record in records]
        assert len(ids) == len(set(ids)) == 112
        assert all(record.is_available or record.unavailable is not None for record in records)


def test_phase_two_parameters_are_named_not_blank(plate_result):
    records = _by_id(plate_result)
    for parameter_id in ("D9", "F1", "G1"):
        assert records[parameter_id].unavailable.reason is ReasonCode.OUT_OF_PHASE_SCOPE


def test_plate_standoff_is_measured_from_tcp_to_surface(plate_result, plate_project):
    _manifest, truth = plate_project
    a3 = _value(_by_id(plate_result), "A3")
    assert a3["count"] > 0
    assert a3["median"] == pytest.approx(truth["standoff_mm"], abs=1e-6)
    assert a3["std"] == pytest.approx(0.0, abs=1e-9)


def test_plate_tool_ray_hits_the_painted_face(plate_result, plate_project):
    _manifest, truth = plate_project
    a4 = _value(_by_id(plate_result), "A4")
    assert a4["fraction"] == pytest.approx(1.0)
    assert a4["bbox_min"][2] == pytest.approx(truth["top_z"], abs=1e-6)
    assert a4["bbox_max"][2] == pytest.approx(truth["top_z"], abs=1e-6)
    # RasterSpec spans x in [-80, 80]: every ray lands inside the painted face.
    assert a4["bbox_min"][0] == pytest.approx(-80.0, abs=1e-6)
    assert a4["bbox_max"][0] == pytest.approx(80.0, abs=1e-6)


def test_plate_nearest_point_and_ray_intersection_agree_on_a_plane(plate_result):
    # On a plane, the nearest surface point and the axis intersection coincide;
    # their disagreement is the diagnostic A5 exists to expose on curved parts.
    a5 = _value(_by_id(plate_result), "A5")
    assert a5["max"] < 1e-6


def test_plate_direction_is_axial_with_zero_bearing(plate_result, plate_project):
    _manifest, truth = plate_project
    b1 = _value(_by_id(plate_result), "B1")
    assert b1["mean_axis"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert b1["bearing_deg"] == pytest.approx(truth["direction_deg"], abs=1e-9)
    assert b1["axial_stability"] == pytest.approx(1.0, abs=1e-9)

    b2 = _value(_by_id(plate_result), "B2")
    assert b2["stroke_count"] == truth["stroke_count"]
    bearings = [stroke["bearing_deg"] for stroke in b2["strokes"]]
    # Reversed passes are antiparallel but axially identical, so bearing is unchanged.
    assert bearings == pytest.approx([0.0] * truth["stroke_count"], abs=1e-9)


def test_plate_normal_is_the_top_face_normal(plate_result):
    b4 = _value(_by_id(plate_result), "B4")
    assert b4["mean_deg"] == pytest.approx(90.0, abs=1e-6)
    assert b4["plane_normal"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    assert b4["plane_tilt_deg"] == pytest.approx(0.0, abs=1e-6)


def test_plate_path_meets_the_border_at_ninety_degrees(plate_result):
    b6 = _value(_by_id(plate_result), "B6")
    assert b6["path_deg"] == pytest.approx(0.0, abs=1e-9)
    assert b6["boundary_deg"] == pytest.approx(90.0, abs=1e-9)
    assert b6["difference_deg"] == pytest.approx(90.0, abs=1e-9)


def test_plate_pitch_is_the_constructed_spacing(plate_result, plate_project):
    _manifest, truth = plate_project
    records = _by_id(plate_result)
    for parameter_id in ("C1", "C2", "C3", "J3"):
        stats = _value(records, parameter_id)
        assert stats["median"] == pytest.approx(truth["pitch_mm"], abs=1e-6), parameter_id
        assert stats["count"] == truth["stroke_count"] - 1, parameter_id


def test_plate_curvature_and_normal_deviation_are_zero(plate_result):
    records = _by_id(plate_result)
    assert _value(records, "E2")["max"] == pytest.approx(0.0, abs=1e-9)
    assert _value(records, "I2")["max"] == pytest.approx(0.0, abs=1e-9)


def test_plate_standoff_parameters_agree(plate_result, plate_project):
    _manifest, truth = plate_project
    assert _value(_by_id(plate_result), "I5")["median"] == pytest.approx(
        truth["standoff_mm"], abs=1e-6
    )


def test_plate_pass_order_alternates(plate_result, plate_project):
    _manifest, truth = plate_project
    records = _by_id(plate_result)
    j1 = _value(records, "J1")
    assert j1["count"] == truth["stroke_count"]
    j2 = _value(records, "J2")
    assert j2["class"] == "alternating"
    assert j2["reversed_pairs"] == truth["stroke_count"] - 1


def test_cylinder_geodesic_pitch_differs_from_tcp_chord(cylinder_result, cylinder_project):
    _manifest, truth = cylinder_project
    records = _by_id(cylinder_result)
    c1 = _value(records, "C1")["median"]
    c3 = _value(records, "C3")["median"]

    assert c1 == pytest.approx(truth["tcp_chord_pitch_mm"], rel=1e-3)
    assert c3 < c1, "surface arc must be shorter than the TCP chord on a convex part"
    # The endpoint correction is first order, so the analytic arc is approximate.
    assert c3 == pytest.approx(truth["geodesic_pitch_mm"], rel=0.03)
    assert _value(records, "C3")["max_snap_offset_mm"] > 0.0


def test_cylinder_curvature_matches_one_over_radius(cylinder_result, cylinder_project):
    _manifest, truth = cylinder_project
    e2 = _value(_by_id(cylinder_result), "E2")
    assert e2["count"] > 0
    assert e2["min"] > 0.0
    # Discrete mean curvature on a coarse mesh overestimates 1/r; the ordering and
    # magnitude are what the report claims, not an exact analytic identity.
    assert e2["median"] == pytest.approx(truth["curvature_per_mm"], rel=0.6)
    assert e2["resolution_ratio"] == pytest.approx(
        e2["mean"] * e2["mesh_median_edge_mm"], abs=1e-6
    )


def test_cylinder_standoff_survives_curvature(cylinder_result, cylinder_project):
    _manifest, truth = cylinder_project
    records = _by_id(cylinder_result)
    assert _value(records, "I5")["mean"] == pytest.approx(
        truth["standoff_mm"], abs=0.1
    )
    assert _value(records, "A3")["median"] == pytest.approx(
        truth["standoff_mm"], abs=0.1
    )


def test_cylinder_direction_is_axial_so_bearing_is_null_not_zero(cylinder_result):
    records = _by_id(cylinder_result)
    b1 = _value(records, "B1")
    assert b1["mean_axis"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    assert b1["bearing_deg"] is None
    assert b1["elevation_deg"] == pytest.approx(90.0, abs=1e-6)

    b2 = _value(records, "B2")
    assert b2["strokes_with_bearing"] == 0
    assert b2["bearing_stats"] is None


def test_cylinder_has_no_planar_bearing_so_b6_is_unavailable(cylinder_result):
    record = _by_id(cylinder_result)["B6"]
    assert not record.is_available
    assert record.unavailable.reason is ReasonCode.DEGENERATE_GEOMETRY
    assert "bearing" in record.unavailable.detail


def test_cylinder_normal_deviation_is_reported_as_a_distribution(cylinder_result):
    i2 = _value(_by_id(cylinder_result), "I2")
    assert i2["count"] > 0
    assert i2["min"] >= 0.0
    assert i2["max"] <= 90.0
    assert i2["max"] > i2["median"], "curved surface must show a spread in tilt"


def test_cylinder_pass_order_alternates(cylinder_result, cylinder_project):
    _manifest, truth = cylinder_project
    j2 = _value(_by_id(cylinder_result), "J2")
    assert j2["class"] == "alternating"
    assert j2["reversed_pairs"] == truth["stroke_count"] - 1


def test_measurement_methods_state_what_they_measured(plate_result):
    for record in plate_result.pairs[0].measurements:
        if record.is_available:
            assert record.method, f"{record.parameter_id} has a value but no method"


def test_no_measurement_is_a_bare_mean_for_a_distribution(plate_result):
    for record in plate_result.pairs[0].measurements:
        if not record.is_available or not isinstance(record.value, dict):
            continue
        if "mean" in record.value and "unit" in record.value:
            assert {"median", "std", "min", "max", "p05", "p95", "count"} <= set(
                record.value
            ), record.parameter_id


def test_distribution_algebra_is_internally_consistent(plate_result):
    a3 = _value(_by_id(plate_result), "A3")
    assert a3["p05"] <= a3["median"] <= a3["p95"]
    assert a3["min"] <= a3["p05"] and a3["p95"] <= a3["max"]
