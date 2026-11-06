"""PDF rendering: the same report model as HTML, in a form that survives printing.

Long tables are the norm here (112 rows per pair), so the document is landscape
A4 with repeating header rows and per-row tinting for absent values. Nothing is
summarised away: a printed report that omits the absences is the failure mode this
module exists to avoid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from mtra.reporting.report_data import PairReport, ReportData, ReportRow

_INK = colors.HexColor("#1b1f24")
_MUTED = colors.HexColor("#5c6773")
_LINE = colors.HexColor("#dfe3e8")
_TINT = colors.HexColor("#f6f8fa")
_WARN_BG = colors.HexColor("#fff8e6")
_WARN = colors.HexColor("#8a5a00")
_BAD_BG = colors.HexColor("#fdeeee")
_BAD = colors.HexColor("#8e1b1b")

_REPLACEMENTS = {
    "\u2013": "-",
    "\u2014": "-",
    "\u00d7": "x",
    "\u2265": ">=",
    "\u2264": "<=",
    "\u00b7": "-",
    "\u2026": "...",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
}


def _clean(text: Any) -> str:
    """Make text safe for the built-in Type-1 fonts (WinAnsi only)."""
    value = "" if text is None else str(text)
    for source, target in _REPLACEMENTS.items():
        value = value.replace(source, target)
    return value.encode("latin-1", "replace").decode("latin-1")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "MtraTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=18, leading=22, textColor=_INK, alignment=TA_LEFT, spaceAfter=2,
        ),
        "eyebrow": ParagraphStyle(
            "MtraEyebrow", parent=base["Normal"], fontName="Helvetica",
            fontSize=7.5, leading=10, textColor=_MUTED, spaceAfter=6,
        ),
        "meta": ParagraphStyle(
            "MtraMeta", parent=base["Normal"], fontName="Helvetica",
            fontSize=8.5, leading=11, textColor=_MUTED, spaceAfter=4,
        ),
        "h2": ParagraphStyle(
            "MtraH2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=12, leading=15, textColor=_INK, spaceBefore=10, spaceAfter=5,
        ),
        "h3": ParagraphStyle(
            "MtraH3", parent=base["Heading3"], fontName="Helvetica-Bold",
            fontSize=9.5, leading=12, textColor=_INK, spaceBefore=8, spaceAfter=3,
        ),
        "cell": ParagraphStyle(
            "MtraCell", parent=base["Normal"], fontName="Helvetica",
            fontSize=7.4, leading=9.4, textColor=_INK, splitLongWords=True,
        ),
        "cell_muted": ParagraphStyle(
            "MtraCellMuted", parent=base["Normal"], fontName="Helvetica",
            fontSize=7.2, leading=9.2, textColor=_MUTED, splitLongWords=True,
        ),
        "reason": ParagraphStyle(
            "MtraReason", parent=base["Normal"], fontName="Courier",
            fontSize=7.0, leading=9.2, textColor=_WARN, splitLongWords=True,
        ),
        "header": ParagraphStyle(
            "MtraHeader", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=7.0, leading=9, textColor=_MUTED,
        ),
        "notice": ParagraphStyle(
            "MtraNotice", parent=base["Normal"], fontName="Helvetica",
            fontSize=8.6, leading=11.5, textColor=_BAD,
        ),
    }


def _value_paragraph(row: ReportRow, style: dict[str, ParagraphStyle]) -> Paragraph:
    if row.available:
        unit = f" {row.unit}" if row.unit and row.unit not in row.value_text else ""
        return Paragraph(escape(_clean(f"{row.value_text}{unit}")), style["cell"])
    text = f'<font name="Courier" color="#8a5a00">{escape(_clean(row.reason))}</font>'
    if row.detail:
        text += f'<br/><font color="#5c6773">{escape(_clean(row.detail))}</font>'
    return Paragraph(text, style["cell"])


def _group_table(section, style) -> Table:
    rows: list[list[Any]] = [
        [
            Paragraph("ID", style["header"]),
            Paragraph("PARAMETER", style["header"]),
            Paragraph("VALUE / ABSENCE", style["header"]),
            Paragraph("STATUS", style["header"]),
        ]
    ]
    shaded: list[int] = []
    for index, row in enumerate(section.rows, start=1):
        name = escape(_clean(row.name))
        if row.method:
            name += f'<br/><font size="6.6" color="#5c6773">{escape(_clean(row.method))}</font>'
        rows.append(
            [
                Paragraph(escape(_clean(row.parameter_id)), style["cell"]),
                Paragraph(name, style["cell"]),
                _value_paragraph(row, style),
                Paragraph(escape(_clean(row.status)), style["cell_muted"]),
            ]
        )
        if not row.available:
            shaded.append(index)

    table = Table(rows, colWidths=[16 * mm, 92 * mm, 132 * mm, 20 * mm], repeatRows=1)
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), _TINT),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, _LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    for row_index in shaded:
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), _WARN_BG))
    table.setStyle(TableStyle(commands))
    return table


def _summary_table(report: ReportData, style) -> Table:
    metrics = report.metrics
    data = [
        ["Catalogue rows", str(metrics["parameters"])],
        ["With a value", str(metrics["with_value"])],
        ["Coverage", f"{metrics['coverage_ratio'] * 100:.0f}%"],
        ["Pairs analysed", f"{metrics['pairs_analysed']} / {metrics['pairs']}"],
    ]
    rows = [
        [Paragraph(escape(_clean(label)), style["header"]), Paragraph(escape(_clean(value)), style["cell"])]
        for label, value in data
    ]
    table = Table(rows, colWidths=[42 * mm, 32 * mm])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, _LINE),
                ("BACKGROUND", (0, 0), (0, -1), _TINT),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _absences_table(report: ReportData, style) -> Table | None:
    histogram = report.metrics.get("unavailable_by_reason") or {}
    if not histogram:
        return None
    rows = [[Paragraph("REASON CODE", style["header"]), Paragraph("COUNT", style["header"])]]
    for reason, count in histogram.items():
        rows.append(
            [
                Paragraph(escape(_clean(reason)), style["reason"]),
                Paragraph(str(count), style["cell"]),
            ]
        )
    table = Table(rows, colWidths=[70 * mm, 20 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _TINT),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, _LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    return table


def _pair_figure(pair: PairReport, base_dir: Path, style) -> list[Any]:
    flowables: list[Any] = []
    for relative in pair.figures:
        candidate = base_dir / relative
        if not candidate.is_file():
            continue
        try:
            flowables.append(Image(str(candidate), width=170 * mm, height=127.5 * mm))
            flowables.append(
                Paragraph(
                    escape(
                        _clean(
                            f"{pair.pair_id}: mesh with painted passes (red), corresponded "
                            "surface points (blue) and normals (green)."
                        )
                    ),
                    style["meta"],
                )
            )
        except Exception:  # pragma: no cover - corrupt image should not kill the report
            continue
    return flowables


def _diagnostics_table(diagnostics: list[dict[str, Any]], style) -> Table | None:
    if not diagnostics:
        return None
    rows = [
        [
            Paragraph("SEVERITY", style["header"]),
            Paragraph("CODE", style["header"]),
            Paragraph("MESSAGE", style["header"]),
            Paragraph("STAGE", style["header"]),
        ]
    ]
    for diagnostic in diagnostics:
        rows.append(
            [
                Paragraph(escape(_clean(diagnostic.get("severity"))), style["cell_muted"]),
                Paragraph(escape(_clean(diagnostic.get("code"))), style["reason"]),
                Paragraph(escape(_clean(diagnostic.get("message"))), style["cell"]),
                Paragraph(escape(_clean(diagnostic.get("stage"))), style["cell_muted"]),
            ]
        )
    table = Table(rows, colWidths=[16 * mm, 46 * mm, 178 * mm, 20 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _TINT),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, _LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    return table


def build_story(report: ReportData, *, base_dir: str | Path | None = None) -> list[Any]:
    style = _styles()
    base = Path(base_dir) if base_dir else Path(".")
    story: list[Any] = []

    story.append(Paragraph(escape("Mesh-Trajectory Relationship Analysis"), style["eyebrow"]))
    story.append(Paragraph(escape(_clean(report.project_id)), style["title"]))
    story.append(
        Paragraph(
            escape(
                _clean(
                    f"Generated {report.generated_at_utc} - MTRA {report.mtra_version}"
                )
            ),
            style["meta"],
        )
    )
    story.append(_summary_table(report, style))
    story.append(Spacer(1, 4 * mm))

    absences = _absences_table(report, style)
    if absences is not None:
        story.append(KeepTogether([Paragraph("Absences across all pairs", style["h3"]), absences]))

    for pair in report.pairs:
        story.append(PageBreak())
        story.append(
            Paragraph(
                escape(_clean(f"{pair.pair_id} - {pair.status}")),
                style["h2"],
            )
        )
        if pair.blocked_reason:
            story.append(
                Paragraph(
                    escape(
                        _clean(
                            f"Analysis blocked: {pair.blocked_reason}. Every parameter below is "
                            "unavailable for one blocking reason; this is an absence of "
                            "measurement, not a measured zero."
                        )
                    ),
                    style["notice"],
                )
            )
            story.append(Spacer(1, 2 * mm))

        coverage = pair.coverage
        story.append(
            Paragraph(
                escape(
                    _clean(
                        f"{coverage['with_value']} of {coverage['parameters']} parameters carry a value."
                    )
                ),
                style["meta"],
            )
        )
        story.extend(_pair_figure(pair, base, style))

        for section in pair.sections:
            heading_text = (
                f"{section.group} - {section.title} ({section.available}/{len(section.rows)})"
            )
            story.append(Paragraph(escape(_clean(heading_text)), style["h3"]))
            story.append(_group_table(section, style))

        diagnostics = _diagnostics_table(pair.diagnostics, style)
        if diagnostics is not None:
            story.append(Paragraph("Diagnostics for this pair", style["h3"]))
            story.append(diagnostics)

    run_diagnostics = _diagnostics_table(report.diagnostics, style)
    if run_diagnostics is not None:
        story.append(PageBreak())
        story.append(Paragraph("Run diagnostics", style["h2"]))
        story.append(run_diagnostics)

    story.append(PageBreak())
    story.append(Paragraph("Provenance", style["h2"]))
    provenance = report.provenance
    story.append(
        Paragraph(
            escape(
                _clean(
                    f"MTRA {provenance.get('mtra_version', '')} - Python "
                    f"{provenance.get('python_version', '')} - seed "
                    f"{provenance.get('random_seed', 0)}"
                )
            ),
            style["meta"],
        )
    )

    version_rows = [[Paragraph("PACKAGE", style["header"]), Paragraph("VERSION", style["header"])]]
    for name, version in (provenance.get("package_versions") or {}).items():
        version_rows.append(
            [
                Paragraph(escape(_clean(name)), style["cell"]),
                Paragraph(escape(_clean(version)), style["cell_muted"]),
            ]
        )
    version_table = Table(version_rows, colWidths=[60 * mm, 60 * mm], repeatRows=1)
    version_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _TINT),
                ("LINEBELOW", (0, 0), (-1, -1), 0.25, _LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    story.append(version_table)

    source_files = provenance.get("source_files") or []
    if source_files:
        story.append(Paragraph("Input files", style["h3"]))
        source_rows = [
            [
                Paragraph("ROLE", style["header"]),
                Paragraph("PATH", style["header"]),
                Paragraph("SHA-256", style["header"]),
            ]
        ]
        for source in source_files:
            source_rows.append(
                [
                    Paragraph(escape(_clean(source.get("role"))), style["cell"]),
                    Paragraph(escape(_clean(source.get("path"))), style["cell_muted"]),
                    Paragraph(escape(_clean(source.get("sha256"))), style["reason"]),
                ]
            )
        source_table = Table(source_rows, colWidths=[26 * mm, 130 * mm, 104 * mm], repeatRows=1)
        source_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _TINT),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.25, _LINE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ]
            )
        )
        story.append(source_table)

    story.append(Spacer(1, 5 * mm))
    story.append(
        Paragraph(
            escape(
                _clean(
                    f"Every one of the {report.metrics['parameters']} catalogue rows in this "
                    "document carries either a value or an explicit reason code. Scope: analysis "
                    "only - no toolpath generation, no robot program output, no input modified."
                )
            ),
            style["meta"],
        )
    )
    return story


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(_MUTED)
    canvas.drawString(15 * mm, 9 * mm, "MTRA - Mesh-Trajectory Relationship Analyser")
    canvas.drawRightString(
        document.pagesize[0] - 15 * mm, 9 * mm, f"page {document.page}"
    )
    canvas.restoreState()


def write_pdf(
    report: ReportData, path: str | Path, *, base_dir: str | Path | None = None
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(target),
        pagesize=landscape(A4),
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=12 * mm,
        bottomMargin=16 * mm,
        title=f"MTRA report - {report.project_id}",
        author="MTRA",
        subject="Mesh-trajectory relationship analysis",
    )
    document.build(build_story(report, base_dir=base_dir), onFirstPage=_footer, onLaterPages=_footer)
    return target
