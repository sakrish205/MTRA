# Design: MTRA Phase 1 — trustworthy foundation

Date: 2026-11-06
Status: approved approach (A), design recorded for review
PRD: `sources/MTRA_Product_Requirements_Document.docx`

## Intent and why

MTRA exists to explain *why* an experienced robotic-process engineer's trajectory
looks the way it does — to reverse-engineer the reusable rules, not to copy XYZ
points. Phase 1 does not attempt that explanation. It makes the *inputs to* that
explanation trustworthy: what the mesh is, what the program says, how the two
frames relate, and where on the surface each motion point lands.

The PRD is explicit about the ordering (§25): *"Do not begin formula discovery
until this foundation is trustworthy."* Every later phase — measurement,
hypothesis comparison, rule derivation — is arithmetic on top of this layer. An
error here is silent and diffuse: a wrong unit assumption looks exactly like a
real spacing rule, and a TCP confused with an impact point looks exactly like a
real standoff rule. Phase 1 therefore spends its budget on provenance,
explicitness and refusals rather than on coverage.

## Goals

1. One matched mesh/program pair can be loaded, parsed, aligned and corresponded
   end to end, with every stage emitting either a value *or* a reason code.
2. Every value is traceable to a source file hash, instruction index and mesh
   face.
3. The measurement layer computes real, analytically checkable numbers for
   direction, pitch, stroke shape, orientation and continuity.
4. The pipeline is a library first and a CLI second, so tests exercise modules
   directly and the CLI is a thin shell.
5. Deterministic: same inputs + same versions reproduce identical outputs.

## Non-goals (Phase 1)

- Hypothesis generation, pattern/model comparison, child rules, parent
  algorithm (PRD stages 9–13). Planned, not built.
- Spiral/contour/flowline models, parameterisation, flattening.
- Any GUI, any network access, any SmartGrid interaction, any path export.
- ABB RAPID and DÜRR adapters — no representative samples exist yet, and the
  PRD forbids coding unsupported parser assumptions (Phase 0 gate).

## Anti-goals

- A report that renders convincingly while containing invented values. If a
  parameter cannot be measured, the report says *why*, in the row itself.
- Silently assuming millimetres, right-handedness, or that TCP lies on the
  surface.
- A parser that skips instructions it does not understand. Unknown lines must
  surface as diagnostics with line numbers.
- Treating the 112 catalogue entries as 112 independent variables.

## Constraints

- Windows 11 primary target, Python 3.12.10, local/offline.
- Dependency set from PRD §12, pinned. Native wheels verified on this machine:
  `libigl 2.6.3` (cp312 abi3), `pygeodesic 0.1.11` (cp312), `pyvista 0.48.4`
  (VTK). libigl is GPL/MPL module-split — notices recorded in `LICENSES/`.
- No sample data supplied. Fixtures are synthetic and analytically known.

## Approaches considered

**A — Foundation on synthetic fixtures, built to the PRD's own milestone.**
Manifest → ingestion → parser → frames → correspondence → measurements, proven
on a generated plate with a 25 mm raster synthesised into FANUC LS syntax, plus
a cylinder for a curved case. Synthetic fixtures become the permanent regression
suite. *Chosen.* Fixtures are the only way to have analytically known answers
before real data exists, and they survive: `assert pitch == 25.0` stays a valid
test forever, whereas a placeholder measurement layer does not survive the first
real `.LS` file.

**B — Full-width skeleton, all 14 stages wired shallowly.** Demos well, teaches
nothing, and every module above the schema layer is rewritten when real data
arrives. Rejected: cost is paid twice, and interim output is misleading.

**C — Data-audit tooling only.** Cheapest, honest, but produces nothing that can
be corresponded. Rejected as a whole milestone; folded in as Phase 1's intake
stage, which is where it belongs.

## Decided design

### Data contract, not data flow

Six canonical records (PRD §11.2) are Pydantic models in one module. Stages
extend them; nothing downstream re-reads source files. This keeps adapters
replaceable and makes the report a pure function of records.

### Evidence is a first-class type

`EvidenceStatus` (observed / inferred / validated / unresolved / rejected) and
`Unavailable(reason_code, detail)` are core types, not report strings. A
measurement either carries a value with method + provenance, or carries an
`Unavailable`. There is no third state, so "missing" cannot be silently rendered
as zero.

### Grammar as data, parser as engine

FANUC LS syntax lives in `config/robot_formats.yaml` (regex + field maps), not
in Python branches. Consequences: the dialect question becomes a config edit
rather than a refactor; every fixture imports its own grammar in tests; and the
`PROVISIONAL` flag is reviewable in one file. Unmatched non-blank lines are
collected as diagnostics rather than dropped.

### Units and frames are explicit gates

The manifest declares units and coordinate convention; `pipeline.run` refuses to
proceed to correspondence when either is `unknown`. Registration uses a declared
transform when present; when absent it estimates one from correspondences and
reports residual distribution *and* a degeneracy check, never a bare matrix.

### TCP and impact point are different objects

`TrajectorySample.tcp_position` and `SampleCorrespondence.impact_point` never
share a field. Standoff is computed *as* the ray distance, so the distinction is
structural rather than a naming convention.

### Pipeline is a list of stages over one context

`StageContext` carries records + diagnostics + provenance. Each stage is a
callable `(ctx) -> ctx`, individually testable, and `run()` records stage
durations and versions for reproducibility.

## Left to the builder

- Exact FANUC dialect: if samples contradict the provisional grammar, edit YAML.
- Whether SVD-based Kabsch is sufficient for registration or ICP refinement is
  needed (Phase 2).
- Region segmentation strategy (connected components vs. user-specified) — v1
  uses components; region *semantics* are Phase 3.
- Numeric tolerances for "continuity" and "smoothness" scores, which are
  engineering judgements to be calibrated on real data.
