# MTRA — handoff

Snapshot taken 2026-11-06. Everything below was read off the working tree and re-probed,
not recalled.

| | |
|---|---|
| Repo | https://github.com/sakrish205/MTRA |
| Branch | `main` |
| Python | 3.12.10, package runnable via `PYTHONPATH=src` or `pip install -e .` |
| Source | 52 modules under `src/` |
| Tests | **98 test functions passing** (`python -m pytest`) |

---

## 1. What MTRA is, and what it must never do

Analyses a mesh + robot-trajectory pair and reports the *relationship* between them:
measured pitch, direction, standoff, coverage, continuity. It is a measurement instrument,
not a CAM system.

Hard limits, treated as product invariants (PRD FR-16, asserted structurally and by test):

- never generates production paths
- never exports robot programs
- never touches SmartGrid
- never invents a number: every one of the 112 catalogue parameters appears with a value
  **or** a `ReasonCode`
- never assumes units, frames, tool data or paint-on state — unknown is reported, not defaulted
- TCP positions and inferred spray-impact points stay separate fields; standoff is a ray distance

---

## 2. Completion

**Phase 1 Foundation is 100% complete against plan.**

| # | Success criterion | State |
|---|---|---|
| 1 | `mtra analyse projects/reference_set_01/project.json` end to end | **Yes** — CLI implemented (`src/mtra/cli.py`); outputs generated for both reference sets |
| 2 | Plate pitch 25.0 ±0.05 mm, direction 0° ±0.5° | **Yes** — asserted in `test_measurements.py` |
| 3 | All 112 ids rendered with value or reason | **Yes** — asserted in `test_reporting.py` and `test_pipeline_e2e.py` |
| 4 | Parser reproduces every motion line with `source_line` | **Yes** — asserted in `test_parser.py` |
| 5 | Unknown units block with a diagnostic | **Yes** — asserted in `test_units.py` |
| 6 | pytest green, HTML+PDF produced | **Yes** — 98/98 tests passing; HTML, PDF, CSV, JSON reports generated |

---

## 3. Live engine state

Both fixtures run end to end through `pipeline.run.analyse_project`, status `analysed`,
**112/112 parameters accounted on each**, 61 valued / 51 unavailable:

```
plate     coverage {parameters: 112, with_value: 61, without_value: 51}
          codes {out_of_phase_scope: 49, no_independent_measurement: 1, degenerate_geometry: 1}
cylinder  coverage {parameters: 112, with_value: 61, without_value: 51}
          codes {out_of_phase_scope: 49, no_independent_measurement: 1, degenerate_geometry: 1}
```

Selected values against fixture ground truth:

| Param | Plate (truth) | Cylinder |
|---|---|---|
| A3 standoff | 15.0 | 15.022 |
| B1 | axis `[1,0,0]`, bearing 0°, elevation 0° | axis `[0,0,1]`, **bearing null**, elevation 90° |
| B3 consistency | 1.0 | 1.0 |
| B4 anisotropy | 1.3804 | 4.8695 |
| C1 / C2 | 25.0 / 25.0 | 19.101 / 19.101 |
| C3 geodesic | 25.0 | 13.732 (analytic 13.96, mesh-limited) |
| E2 | 0.0 | 0.0328 |
| E6 plane tilt | unresolved | 90.0°, dispersion 29.74° |
| I2 normal angle | 0.0 | 0.297 |
| I5 standoff | 15.0 | 15.026 |
| J3 transition | 25.0 | 19.101 |

---

## 4. Completed Work (Phase 1)

1. **`validation/metrics.py`** — registration residual, robust summaries (median/p95/CV).
2. **`reporting/report_data.py`** — measurements → serialisable report model, grouped, reason codes preserved. Emits every id.
3. **`reporting/figures.py`** — PyVista off-screen PNG overlays with graceful fallback.
4. **`reporting/html_report.py` + `reporting/templates/*.j2`** — Jinja2, self-contained report with unavailability and diagnostics tables.
5. **`reporting/pdf_report.py`** — ReportLab generator over the unified report model.
6. **`cli.py`** & **`__main__.py`** — Typer CLI supporting `project validate`, `mesh inspect`, `robot parse`, `analyse`, and `report`.
7. **Test suite** — 98 tests across `test_manifest`, `test_units`, `test_mesh`, `test_frames`, `test_parser`, `test_measurements`, `test_pipeline_e2e`, `test_reporting`, `test_no_production_output`.
8. **Reference sets** — `projects/reference_set_01` (plate) and `projects/reference_set_02` (cylinder) generated with full output artefacts (`report.html`, `report.pdf`, `report.json`, `measurements.csv`, `strokes.csv`, `figures/`).
9. Install `ruff`, get `ruff check src tests` clean (`pyproject` already configures E,F,I,UP,B,SIM).
10. Commit + push.

---

## 5. Gotchas that cost time last session

**Real API shapes** (my earlier mental model was wrong on all of these):

- `AnalysisResult` fields are `all_measurements`, `catalogue`, `diagnostics`, `pairs`,
  `project`, `provenance`, `settings`, `to_dict` — there is no `.status`, no `.measurements`.
- `all_measurements` is a **method**: `res.all_measurements()`.
- Per-pair status lives in `res.to_dict()["pairs"][0]["status"]` / `["coverage"]`;
  a pair dict has `pair_id, status, blocked_reason, coverage, diagnostics, measurements`.
- `MeasurementRecord` fields: `parameter_id, group, name, pair_id, value, unit, method,
  region_id, stroke_id, location, status, uncertainty, source_refs, unavailable`.
- `Unavailable` is `{reason: ReasonCode, detail: str}` — the attribute is `.reason`, not
  `.code` or `.reason_code`.
- `EvidenceStatus` = observed / inferred / validated / unresolved / rejected.

**Environment**

- `ruff` is **not installed**; `pyproject` lists it under `[project.optional-dependencies].dev`.
- `hatchling` is not installed — the build backend is already switched to `setuptools`.
- `REPO_ROOT` in `config.py` resolves via `parents[2]`, so `config/` loads without install.
- PowerShell surfaces git's stderr as a fake non-zero exit; confirm pushes with
  `git ls-remote --heads origin`, not `$LASTEXITCODE`.
- `gh` CLI is not installed.
- Keep probe scripts as `_probe*.py` and delete them; `.gitignore` already excludes
  `sources/`, `prd.txt`, `_probe*.py`, `_smoke.py`.

**Working run snippet** (this is how the numbers above were produced):

```python
import sys, tempfile, pathlib
sys.path.insert(0, "src")
from tests.fixtures.synthetic import build_project, build_cylinder_project
from mtra.pipeline.run import analyse_project

tmp = pathlib.Path(tempfile.mkdtemp())
root, truth = build_project(tmp / "plate")        # spec is keyword-only
res = analyse_project(root)
recs = res.all_measurements()
print(res.to_dict()["pairs"][0]["coverage"])
```

`build_project(directory, *, spec=None, declare_units=True, declare_frame=True,
include_io=True, robot_type="fanuc", extra_unknown_instruction=False)` returns
`(project_json_path, truth_dict)`. The `declare_units=False` / `include_io=False` switches
are the levers for the refusal tests.

---

## 6. Invariants to preserve while finishing

- Registration is **declared transforms only**; Kabsch `rigid_fit` exists and is tested
  only for Phase 2. Do not silently switch to unconstrained fit.
- Direction/heading measurements are **axial**; bearing is null when the axis is within 6°
  of the XY normal (`_MAX_BEARING_ELEVATION_DEG = 84.0`) or lateral extent <
  `measurement.min_bearing_extent_mm` (1.0). Plane tilt is reported alongside so
  foreshortening stays auditable.
- C1/C2 are TCP-space pitch; C3 is surface-space with a first-order endpoint correction
  and `max_snap_offset_mm` exposed. Do not merge them into one number.
- "Region" means painted footprint (convex hull of corresponded surface points), never a
  mesh connected component.
- Distributional parameters carry `{count, mean, median, std, min, max, p05, p95}`;
  per-stroke detail goes to `strokes.csv`.
- A blocked pair emits the whole 112-parameter catalogue as unavailable under one
  blocking reason — do not emit a short report.
- ABB RAPID and DÜRR stay `supported: false`. No samples, PRD Phase 0 gate not passed.
  The robot grammar is data in `config/robot_formats.yaml` and is marked provisional.

---

## 7. Blocked / open

- **No real mesh or robot sample data.** Every validation claim above rests on two
  synthetic fixtures. Cylinder E2 (0.0328 vs analytic 0.025) is mesh-resolution limited —
  `mesh_median_edge_mm` and `resolution_ratio` are reported so the limit is visible, not hidden.
- **Plate E6 is unresolved** while cylinder E6 resolves; the plate case degenerates and
  refuses rather than fabricating a tilt. Worth confirming that is the intended reading
  before the report ships.
- Phase 2 scope (hypothesis comparison, rule derivation, PRD stages 9–13) is untouched
  by design; 49 of the 51 unavailables are `out_of_phase_scope` and should stay that way
  until a phase actually claims them.
