# Reference projects

These are **synthetic** reference sets with analytically known answers, not
production cells. MTRA was built without real mesh/robot sample data; each set is
constructed by `tests/fixtures/synthetic.py`, so the same specification produces
the mesh, the program and the expected measurements.

| Set | Surface | What it exercises |
| --- | --- | --- |
| `reference_set_01` | Flat plate, 9 serpentine passes | Pitch, standoff, direction, alternation, zero curvature |
| `reference_set_02` | Cylinder R = 40 mm, 5 axial passes | Surface arc vs TCP chord, curvature, normal deviation, null bearing |

Each directory contains the inputs, a `project.json`, a `TRUTH.md` stating the
constructed ground truth, and the `output/` artefacts produced by:

```powershell
$env:PYTHONPATH = "src"
python -m mtra analyse projects/reference_set_01/project.json
python -m mtra analyse projects/reference_set_02/project.json
```

`output/report.html` and `output/report.pdf` are the readable reports;
`output/report.json` is the exact model they were rendered from; `measurements.csv`
and `strokes.csv` are flat exports. Every report accounts for all 112 catalogue
parameters per pair, with a value or a reason code.

To re-render a report without analysing anything again:

```powershell
python -m mtra report projects/reference_set_01/output/report.json
```

These sets validate the pipeline against constructed geometry. They do **not**
validate a controller dialect, a real tool definition, a paint process, or a
scanned surface.
