# Implementation Plan: MTRA Phase 1 — trustworthy foundation

Companion to `docs/design/2026-11-06-phase-1-foundation-design.md`.
Delivers PRD stages 1–8 on one matched pair; stages 9–13 are explicitly deferred.

## Overview

Build a modular Python package that takes a project manifest, loads a mesh and a
robot program, proves the parser instruction-for-instruction, aligns frames,
builds trajectory→surface correspondence, and computes the measurable subset of
the 112-parameter catalogue — with every unavailable value carrying a reason
code. Exit gate (PRD §25): *one matched pair verified end-to-end, including
units, frames, parser correctness, painting-state classification and
TCP-to-surface correspondence.*

## Architecture

```
manifest ──▶ mesh loader ──▶ MeshRecord ─┐
   │                                      ├──▶ frames/registration ──▶ Correspondence ──▶ measurements ──▶ report
   └──────▶ robot parser ──▶ MotionRecord ─┘
                    │
                    └──▶ segment classifier ──▶ TrajectorySegment
```

Six canonical records (PRD §11.2) are the only inter-stage contract. Stage
functions are pure `StageContext → StageContext`; the CLI is a thin shell over
`pipeline.run`.

## Requirements traceability

| PRD | Requirement | Where |
|---|---|---|
| FR-01 | STL ASCII/binary + OBJ import | `mesh/loader.py` |
| FR-02 | Units recorded; refuse when unknown | `core/units.py`, `pipeline/run.py` |
| FR-03 | Controller-specific parser adapters | `robot/fanuc_ls.py`, `robot/registry.py` |
| FR-04 | Source instruction/line references preserved | `robot/common_schema.py` |
| FR-05 | Explicit manifest pairing | `project/manifest.py` |
| FR-06 | Painting/transition/approach/departure/unknown with evidence | `traj/segments.py` |
| FR-07 | TCP separate from spray target | `correspondence/spray_axis.py` |
| FR-08 | Declared transforms + registration residuals | `correspondence/registration.py` |
| FR-09 | Catalogue measurements with reason codes | `measurements/catalogue.py`, `measurements/*.py` |
| FR-14 | HTML + PDF + JSON/CSV | `reporting/*` |
| FR-15 | Never modify inputs | `project/inventory.py` (read-only, hash) |
| FR-16 | No path generation, no SmartGrid | structurally absent; asserted in tests |

## Implementation steps

### Phase 1.1 — Repo and contracts (blocking everything)

1. **`pyproject.toml`** — hatchling, `src` layout, pinned deps, `mtra` console
   script, ruff/pytest config. *Risk: low.*
2. **`core/types.py`** — the six canonical records + `EvidenceStatus`,
   `Unavailable`, `Diagnostic`, `Provenance`. *Why first: every module imports
   it; churn here is churn everywhere. Risk: medium — model shape is load-bearing.*
3. **`core/units.py`** — `LengthUnit` enum, explicit conversion to mm, refusal on
   `unknown`. *Risk: low, but the PRD calls this a top risk class.*
4. **`core/errors.py`** — `MTRAError`, `InputError`, `UnsupportedInput`,
   `AnalysisBlocked` (carries the reason code that blocked the run).

### Phase 1.2 — Intake

5. **`project/manifest.py`** — Pydantic `ProjectManifest`/`PairSpec`/`ToolSpec`,
   matching PRD §5.3 JSON exactly, with `unknown` as a first-class value.
6. **`project/inventory.py`** — SHA-256, size, format sniff (binary vs ASCII
   STL), read-only guard. *Invariant: source bytes never touched.*
7. **`project/validation.py`** — pair existence, duplicate pair ids, unit/frame
   declaration status, aggregated `Diagnostic`s.

### Phase 1.3 — Mesh

8. **`mesh/loader.py`** — Trimesh ingestion, raw copy retained, triangulation,
   merge-tolerance reporting (not silent), `MeshRecord` with `transform_history`.
9. **`mesh/quality.py`** — bbox, degenerate faces, duplicate vertices, normal
   consistency, watertightness, component count, area, resolution statistics.
10. **`mesh/topology.py`** — connected components, boundary loops.
11. **`mesh/features.py`** — vertex normals, curvature via `libigl`
    (`principal_curvature`: mean/Gaussian/principal directions), graceful
    `Unavailable` when libigl is absent.

### Phase 1.4 — Robot programs

12. **`config/robot_formats.yaml`** — LS grammar as data: motion regex, position
    block regex, field maps, I/O regex, `PROVISIONAL: true`.
13. **`robot/grammar.py`** — YAML → compiled grammar objects; validation of the
    grammar itself.
14. **`robot/fanuc_ls.py`** — line-oriented parser: `/MN` section, motion
    records, `P[n]` blocks from `/POS`, `DO[n]=ON/OFF`, registry of unknown
    lines. Every record carries `source_line` and `instruction_index`.
15. **`robot/registry.py`** — `--type fanuc` → adapter; clear failure for ABB and
    DÜRR pointing at the Phase 0 gate rather than pretending.

### Phase 1.5 — Segments and sampling

16. **`traj/segments.py`** — classify into painting / approach / departure /
    transfer / unknown using I/O state first, motion type second; emit
    `classification_evidence`. Never infer paint-on from geometry alone (PRD §5.2).
17. **`traj/sampling.py`** — resample segments at a declared arc-length step;
    record method so pitch is reproducible.

### Phase 1.6 — Correspondence

18. **`correspondence/frames.py`** — 4×4 rigid transforms, frame registry,
    composition, `<frame>/<frame>` resolution.
19. **`correspondence/registration.py`** — declared-transform path and
    Kabsch/SVD estimation with residual percentiles + degeneracy test (collinear
    or coplanar point sets must be flagged, not silently accepted).
20. **`correspondence/projection.py`** — closest-point projection via
    `trimesh.proximity`; barycentric coordinates and face id per sample.
21. **`correspondence/spray_axis.py`** — ray/mesh intersection along the tool
    axis; explicit `no_intersection` / `multiple_intersections` outcomes.

### Phase 1.7 — Measurements

22. **`measurements/catalogue.py`** — all 112 parameters as data (group, id,
    unit, requirement inputs). `MeasurementSet.compute()` fills values or
    `Unavailable`.
23. **`measurements/direction.py`** — B1–B6: axial PCA per stroke, circular
    statistics for consistency, tangent-plane and boundary comparisons.
24. **`measurements/spacing.py`** — C1–C6: projected / 3D / geodesic pitch kept
    separate; local pitch, distribution, CV. Geodesic via `pygeodesic`.
25. **`measurements/stroke_shape.py`** — D1–D7: length, straightness RMS,
    discrete curvature and its variation, smoothness, fragmentation.
26. **`measurements/orientation.py`** — I1–I7: tool axis from rotation, normal
    angle, lead/lag under a declared sign convention, standoff, continuity.
27. **`measurements/continuity.py`** — J1–J8: order, alternation, transition
    distance, connection class, region order.
28. **`measurements/robot_motion.py`** — K1–K9: sampling interval/density, curve
    representation class, process state, classification summary.

### Phase 1.8 — Validation, report, CLI

29. **`validation/metrics.py`** — registration residual, reconstruction-style
    metrics used later, robust summaries (median/p95/CV).
30. **`reporting/report_data.py`** — measurements → serialisable report model,
    grouped, with reason codes preserved.
31. **`reporting/figures.py`** — PyVista off-screen overlays (mesh + strokes +
    normals), exported PNG; disabled gracefully when VTK is unavailable.
32. **`reporting/html_report.py` + template** — Jinja2, self-contained, includes
    unavailability table and diagnostics.
33. **`reporting/pdf_report.py`** — ReportLab, same model.
34. **`cli.py`** — `project validate`, `mesh inspect`, `robot parse`,
    `analyse`, `report` (PRD §21 subset that Phase 1 can honour). Non-zero exit
    on blocked analysis.

### Phase 1.9 — Fixtures and tests

35. **`tests/fixtures/synthetic.py`** — plate (exact analytic truth: 25 mm
    pitch, 0°, 15 mm standoff) + cylinder (curved case) + LS generator emitting
    the provisional grammar.
36. **Unit tests** — manifest, units refusal, mesh quality, parser
    (instruction-by-instruction, unknown-line diagnostics), frames, Kabsch
    (including degenerate case), spacing (geodesic ≥ 3D ≥ projected), direction,
    orientation, continuity.
37. **`test_pipeline_e2e.py`** — one pair, end to end, asserting pitch/direction/
    standoff against analytic values within tolerance, and asserting that a
    second pair without tool data yields `Unavailable`, not a number.
38. **`test_no_production_output.py`** — asserts no module writes robot programs
    or references SmartGrid (FR-16 guard).

## Testing strategy

- **Unit** — synthetic geometry with closed-form answers; wrong-answer-proof
  assertions (`pytest.approx`) rather than snapshot files.
- **Integration** — `pipeline.run` on the fixture project; report files exist,
  HTML contains every parameter id, JSON round-trips.
- **Regression** — the same fixtures stay in the suite; any change to pitch or
  direction maths trips them.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Provisional LS grammar is wrong | Grammar in YAML; parser records unknown lines instead of skipping; fixture tests import the same grammar |
| Native wheels break on another machine | Pinned versions + documented wheel availability; libigl/pygeodesic degrade to `Unavailable` rather than crash |
| Degenerate registration accepted | Explicit degeneracy test on the correspondence point set |
| Geodesic pitch expensive | pygeodesic per-stroke source sets; guard on vertex count with a declared limit |
| Report implies more certainty than exists | Every row carries status; unavailability section is mandatory |
| Fixtures encode my assumptions as truth | Fixtures assert *geometry*, not *the engineer's rule*; docs mark parser output provisional |

## Success criteria

- [ ] `mtra analyse projects/reference_set_01/project.json` runs end to end
- [ ] Pitch measured as 25.0 mm ± 0.05 on the plate fixture; direction 0° ± 0.5°
- [ ] Every one of the 112 catalogue ids appears in the report with value or reason
- [ ] Parser reproduces every motion line with its source line number
- [ ] Unknown units block the run with an explicit diagnostic, not a default
- [ ] `pytest` green; HTML and PDF generated; `ruff` clean
