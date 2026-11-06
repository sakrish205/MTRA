# MTRA — Mesh–Trajectory Relationship Analyser

Reference-driven reverse engineering of robotic surface path-generation methods.

MTRA takes one or more matched pairs of (part mesh, robot program) and works out
what relationship exists between the part's geometry and the trajectory an
engineer generated for it. It measures, it compares competing explanations, and
it reports — with evidence grades and explicit unknowns.

**MTRA does not generate production paths, does not export robot programs, and
does not integrate with SmartGrid.** Its responsibility ends at analysis and
reporting.

## Status

Phase 1 — trustworthy foundation. Ingestion, parsing, frame alignment,
trajectory→surface correspondence and the measurable part of the parameter
catalogue. Hypothesis comparison and rule derivation (PRD stages 9–13) are
Phase 2 and later.

## Requirements

Python 3.12, Windows 11 (portable elsewhere). Core: numpy, scipy, trimesh,
pandas, sympy, pydantic, jinja2, reportlab, pyyaml, typer. Optional: libigl,
pygeodesic (curvature and geodesic spacing), pyvista (figures).

```powershell
pip install -e ".[geometry,viz,dev]"
```

## Quick start

```powershell
# 1. validate a project manifest and its source files
mtra project validate projects/reference_set_01/project.json

# 2. inspect a mesh
mtra mesh inspect projects/reference_set_01/input/meshes/plate.stl --units mm

# 3. parse a robot program, proving instruction coverage
mtra robot parse projects/reference_set_01/input/robot_programs/raster.LS --type fanuc

# 4. run the analysis and write artefacts
mtra analyse projects/reference_set_01/project.json

# 5. render reports
mtra report projects/reference_set_01/project.json --format html,pdf
```

## What the report guarantees

- Every catalogue parameter appears, with either a value and its method, or a
  reason code explaining why it could not be measured.
- Every value is traceable to a file hash, an instruction index and a mesh face.
- TCP positions and inferred spray-impact points are separate fields; standoff is
  computed as a ray distance, never as a nearest-point distance.
- Unknown units, frames, tool data and process states block or downgrade the
  analysis instead of being defaulted.
- Paint on/off is never inferred from motion geometry alone.

## Layout

```
config/        parameter catalogue, robot grammars, defaults (grammar is data, not code)
src/mtra/      core types, ingestion, parsers, correspondence, measurements, reporting
projects/      one directory per reference set: inputs versioned, outputs generated
docs/          design record and phase plans
tests/         synthetic fixtures with analytically known answers
```

## Documentation

- `docs/design/2026-11-06-phase-1-foundation-design.md` — why Phase 1 is shaped this way
- `docs/plans/phase-1-foundation.md` — step plan and traceability to PRD requirements
- `sources/MTRA_Product_Requirements_Document.docx` — the PRD this implements

## Licence

Application code is proprietary. Third-party dependencies retain their own
licences; see `LICENSES/THIRD_PARTY.md`. libigl in particular is a mixed
GPL/MPL-2.0 project and is used here only through its Python bindings for
curvature; review before redistribution.
