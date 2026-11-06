"""Off-screen figures: mesh, painted passes, and the normals they were measured against.

Rendering is optional by design. A headless machine without a usable VTK backend
must still produce a report, so every failure here degrades to a warning and an
empty figure list rather than aborting the run. The report states which figures
were produced, so an absent image is visible rather than merely missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_MESH_COLOUR = "#d7d9dd"
_STROKE_COLOUR = "#c0392b"
_POINT_COLOUR = "#1f4e79"
_NORMAL_COLOUR = "#2e7d32"
_NORMAL_SCALE_MM = 3.0


def render(
    result,
    output_dir: str | Path,
    *,
    dpi: int = 110,
    enabled: bool = True,
    window_size: tuple[int, int] = (1200, 900),
) -> tuple[dict[str, list[str]], list[str]]:
    """Render one figure per analysed pair.

    Returns ``({pair_id: [relative paths]}, warnings)``. Paths are relative to
    ``output_dir`` so the HTML stays portable.
    """
    if not enabled:
        return {}, []

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures: dict[str, list[str]] = {}
    warnings: list[str] = []

    try:
        import pyvista as pv
    except Exception as exc:  # pragma: no cover - optional dependency
        return {}, [f"figures disabled: pyvista unavailable ({exc})"]

    for pair in result.pairs:
        view = getattr(pair, "view", None)
        if view is None:
            continue
        figures_dir.mkdir(parents=True, exist_ok=True)
        target = figures_dir / f"{_safe(pair.pair_id)}.png"
        try:
            _render_pair(pv, view, target, dpi=dpi, window_size=window_size)
        except Exception as exc:  # pragma: no cover - native/backend failure
            warnings.append(f"figure for pair {pair.pair_id} failed: {type(exc).__name__}: {exc}")
            continue
        figures[pair.pair_id] = [f"figures/{target.name}"]

    return figures, warnings


def _safe(name: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in name)


def _polyline(pv, points: np.ndarray):
    """One connected polyline cell.

    ``Plotter.add_lines`` interprets its input as independent segments, which
    splits a pass into disconnected two-point stubs; building the cell explicitly
    keeps a pass one continuous curve in the figure.
    """
    points = np.asarray(points, dtype=float)
    connectivity = np.empty(len(points) + 1, dtype=np.int64)
    connectivity[0] = len(points)
    connectivity[1:] = np.arange(len(points))
    return pv.PolyData(points, lines=connectivity)


def _render_pair(pv, view, target: Path, *, dpi: int, window_size: tuple[int, int]) -> None:
    record = view.surface.record
    vertices = np.asarray(record.vertices, dtype=float)
    faces = np.asarray(record.faces, dtype=np.int64)
    padded = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces])
    mesh = pv.PolyData(vertices, padded)

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    try:
        plotter.add_mesh(
            mesh,
            color=_MESH_COLOUR,
            opacity=0.85,
            smooth_shading=False,
            show_edges=False,
            lighting=True,
        )

        for stroke in view.painting_strokes:
            points = view.stroke_points_mesh(stroke)
            if len(points) >= 2:
                plotter.add_mesh(
                    _polyline(pv, points), color=_STROKE_COLOUR, line_width=4
                )

        surface_points = view.surface_points()
        if len(surface_points):
            plotter.add_points(
                surface_points,
                color=_POINT_COLOUR,
                point_size=4,
                render_points_as_spheres=True,
            )
            normals = view.surface_normals()
            if len(normals) == len(surface_points):
                tips = surface_points + normals * _NORMAL_SCALE_MM
                endpoints = np.empty((len(surface_points) * 2, 3), dtype=float)
                endpoints[0::2] = surface_points
                endpoints[1::2] = tips
                cells = np.empty((len(surface_points), 3), dtype=np.int64)
                cells[:, 0] = 2
                cells[:, 1] = np.arange(0, len(endpoints), 2)
                cells[:, 2] = cells[:, 1] + 1
                plotter.add_mesh(
                    pv.PolyData(endpoints, lines=cells),
                    color=_NORMAL_COLOUR,
                    line_width=1,
                )

        plotter.add_text(f"{view.pair_id}: mesh + painted passes (mm)", font_size=10)
        plotter.view_isometric()
        plotter.camera.zoom(1.15)
        plotter.screenshot(str(target), scale=1)
    finally:
        plotter.close()


def figures_summary(figures: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "count": sum(len(paths) for paths in figures.values()),
        "pairs": sorted(figures),
    }
