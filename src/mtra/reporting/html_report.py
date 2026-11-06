"""HTML rendering: one self-contained file, no external assets.

Self-contained matters here for the same reason the JSON artefact does: a report
that breaks when it is emailed, archived, or opened on a machine without network
access is not evidence. The only external references are the figure PNGs written
next to it, and the report says so when they are absent.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from mtra.reporting.report_data import ReportData

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "report.html.j2"


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.policies["json.dumps_kwargs"] = {"sort_keys": False, "ensure_ascii": False}
    return env


def render_html(report: ReportData) -> str:
    template = _environment().get_template(_TEMPLATE_NAME)
    return template.render(report=report)


def write_html(report: ReportData, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(report), encoding="utf-8")
    return target
