"""Reporting: one model, four artefacts (HTML, PDF, JSON, CSV).

The renderers never read the pipeline directly. They consume ``ReportData``, which
is built once and serialised to ``report.json`` — so the HTML, the PDF and any
later re-render are provably the same numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mtra.reporting.csv_export import write_measurements_csv, write_strokes_csv
from mtra.reporting.figures import render as render_figures
from mtra.reporting.html_report import render_html, write_html
from mtra.reporting.pdf_report import write_pdf
from mtra.reporting.report_data import (
    GroupSection,
    PairReport,
    ReportData,
    ReportRow,
    build_report,
    format_value,
    load_report_json,
    write_report_json,
)

__all__ = [
    "GroupSection",
    "PairReport",
    "ReportData",
    "ReportRow",
    "build_report",
    "format_value",
    "load_report_json",
    "render_figures",
    "render_html",
    "write_all",
    "write_html",
    "write_measurements_csv",
    "write_pdf",
    "write_report_json",
    "write_strokes_csv",
]


def write_all(
    report: ReportData,
    output_dir: str | Path,
    *,
    figure_base: str | Path | None = None,
) -> dict[str, str]:
    """Write every artefact into ``output_dir`` and return their paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    produced: dict[str, str] = {}

    produced["json"] = str(write_report_json(report, directory / "report.json"))
    produced["html"] = str(write_html(report, directory / "report.html"))
    produced["pdf"] = str(
        write_pdf(report, directory / "report.pdf", base_dir=figure_base or directory)
    )
    return produced


def report_payload(report: ReportData) -> dict[str, Any]:
    """The model as a plain dict, for callers that do not want a file."""
    return report.to_dict()
