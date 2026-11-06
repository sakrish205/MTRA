"""Reporting: one model, four artefacts, and an accounting guard.

The report is the product. These tests check that it cannot silently omit a
catalogue row, that HTML and PDF are rendered from the same model, and that a
saved ``report.json`` can be re-rendered without analysing anything again.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mtra.cli import report as report_command
from mtra.reporting import (
    build_report,
    format_value,
    load_report_json,
    write_all,
    write_measurements_csv,
    write_strokes_csv,
)
from mtra.reporting.report_data import write_report_json
from tests.fixtures.synthetic import build_project


@pytest.fixture(scope="module")
def plate_report(plate_result):
    return build_report(plate_result)


# --------------------------------------------------------------------------- #
# Value rendering
# --------------------------------------------------------------------------- #


def test_stats_render_median_with_spread_and_sample_count():
    value = {
        "count": 9,
        "mean": 25.0,
        "median": 25.0,
        "std": 0.0,
        "min": 24.0,
        "max": 26.0,
        "p05": 24.0,
        "p95": 26.0,
        "unit": "mm",
    }
    text = format_value(value, "mm")
    assert "25 mm" in text
    assert "median" in text
    assert "n=9" in text
    assert "p05-p95 24-26" in text


def test_vectors_and_booleans_render_without_python_syntax():
    assert format_value([1.0, 2.5, -3.0]) == "[1, 2.5, -3]"
    assert format_value(True) == "yes"
    assert format_value(False) == "no"
    assert format_value(None) == ""
    assert format_value(np.float64(1.5)) == "1.5"
    assert format_value([[1.0, 2.0], [3.0, 4.0]]) == "2x2 matrix"


def test_small_dicts_render_key_value_pairs():
    text = format_value({"fraction": 1.0, "samples": 729, "note": "ok"})
    assert "fraction=1" in text
    assert "samples=729" in text
    assert "note=ok" in text


# --------------------------------------------------------------------------- #
# Model and accounting
# --------------------------------------------------------------------------- #


def test_report_covers_every_catalogue_row_in_catalogue_order(plate_result, plate_report):
    catalogue = plate_result.catalogue
    pair = plate_report.pairs[0]

    assert pair.coverage == {"parameters": 112, "with_value": 61, "without_value": 51}
    assert len(pair.rows) == 112
    assert [row.parameter_id for row in pair.rows] == [
        spec.id for spec in catalogue.parameters
    ]
    assert len(pair.sections) == len({spec.group for spec in catalogue.parameters})


def test_report_rows_separate_values_from_absences(plate_report):
    pair = plate_report.pairs[0]
    assert len(pair.unavailable) == pair.coverage["without_value"]
    for row in pair.rows:
        if row.available:
            assert row.value_text
            assert row.reason == ""
        else:
            assert row.value_text == ""
            assert row.reason
            assert row.detail


def test_build_report_refuses_to_render_a_missing_catalogue_row(plate_result):
    pair = plate_result.pairs[0]
    original = list(pair.measurements)
    dropped = original.pop()
    try:
        pair.measurements = original
        with pytest.raises(ValueError, match=dropped.parameter_id):
            build_report(plate_result)
    finally:
        pair.measurements = original + [dropped]


def test_blocked_pair_still_renders_the_whole_catalogue(tmp_path):
    manifest, _truth = build_project(tmp_path / "nounits", declare_units=False)
    from mtra.pipeline.run import analyse_project

    report = build_report(analyse_project(manifest))
    pair = report.pairs[0]

    assert pair.blocked_reason == "unknown_units"
    assert len(pair.rows) == 112
    assert len(pair.unavailable) == 112
    assert {row.reason for row in pair.rows} == {"unknown_units"}


# --------------------------------------------------------------------------- #
# Artefacts
# --------------------------------------------------------------------------- #


def test_write_all_produces_four_artefacts_and_html_shows_every_id(
    plate_report, plate_result, tmp_path
):
    paths = write_all(plate_report, tmp_path)
    assert set(paths) == {"json", "html", "pdf"}
    for path in paths.values():
        assert Path(path).is_file()

    html = Path(paths["html"]).read_text(encoding="utf-8")
    for spec in plate_result.catalogue.parameters:
        assert f">{spec.id}<" in html, spec.id
    assert "Analysis blocked" not in html
    assert "out_of_phase_scope" in html

    pdf = Path(paths["pdf"]).read_bytes()
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 10_000


def test_report_json_is_the_rendered_model(plate_report, plate_result, tmp_path):
    target = write_report_json(plate_report, tmp_path / "report.json")
    payload = load_report_json(target)

    assert payload["project_id"] == plate_report.project_id
    assert payload["metrics"]["parameters"] == 112
    assert len(payload["pairs"][0]["sections"]) == len(plate_report.pairs[0].sections)
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert str(plate_result.project.paths.base_dir) not in target.read_text(encoding="utf-8")


def test_saved_report_can_be_rerendered_without_reanalysis(plate_report, tmp_path):
    source = write_report_json(plate_report, tmp_path / "model" / "report.json")
    rendered = tmp_path / "rendered"

    report_command(report_json=source, output=rendered)

    html = (rendered / "report.html").read_text(encoding="utf-8")
    assert plate_report.project_id in html
    assert (rendered / "report.pdf").read_bytes().startswith(b"%PDF-")


def test_measurements_csv_has_one_row_per_catalogue_entry(plate_result, tmp_path):
    target = write_measurements_csv(
        plate_result.all_measurements(), tmp_path / "measurements.csv"
    )
    with target.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 112
    assert rows[0]["parameter_id"] == "A1"
    assert {"parameter_id", "value", "reason", "method", "pair_id"} <= set(rows[0])
    unavailable = [row for row in rows if row["reason"]]
    assert len(unavailable) == 51


def test_strokes_csv_lists_every_painted_pass(plate_result, plate_project, tmp_path):
    _manifest, truth = plate_project
    target = write_strokes_csv(plate_result, tmp_path / "strokes.csv", frame="mesh")
    assert target is not None

    with target.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == truth["stroke_count"]
    assert all(int(row["samples"]) > 0 for row in rows)
    assert len({row["stroke_id"] for row in rows}) == truth["stroke_count"]
