# Third-party components

MTRA depends on the following projects at runtime. Each retains its own licence.
This file records what MTRA uses and the licence position, per PRD §13 (repository
reuse policy) and §19 (licensing risk).

| Component | Link | Used for | Licence position |
|---|---|---|---|
| Trimesh | https://github.com/mikedh/trimesh | STL/OBJ ingestion, proximity queries, ray casting, components | MIT |
| NumPy | https://github.com/numpy/numpy | Array and vector maths throughout | BSD-3-Clause |
| SciPy | https://github.com/scipy/scipy | Rotation algebra, spatial indexing, statistics | BSD-3-Clause |
| pandas | https://github.com/pandas-dev/pandas | Measurement tables, CSV export | BSD-3-Clause |
| SymPy | https://github.com/sympy/sympy | Symbolic equation handling in the discovery layer (Phase 2) | BSD-3-Clause |
| Pydantic | https://github.com/pydantic/pydantic | Manifest and record validation | MIT |
| Jinja2 | https://github.com/pallets/jinja | HTML report templating | BSD-3-Clause |
| ReportLab | https://github.com/MrBitBucket/reportlab-mirror | PDF report generation | BSD-3-Clause |
| PyYAML | https://github.com/yaml/pyyaml | Grammar and catalogue configuration | MIT |
| Typer | https://github.com/fastapi/typer | CLI | MIT |
| libigl Python bindings | https://github.com/libigl/libigl-python-bindings | Principal curvature, normals, point-mesh distance | **Mixed GPL / MPL-2.0** — module-specific. MTRA uses `principal_curvature`, `per_vertex_normals`, `point_mesh_squared_distance`, `gaussian_curvature`. Review before redistribution. |
| pygeodesic | https://github.com/mhogg/pygeodesic | Exact geodesic distances on triangle meshes | MIT |
| PyVista / VTK | https://github.com/pyvista/pyvista | Off-screen overlay figures | MIT / BSD-3-Clause |
| pytest | https://github.com/pytest-dev/pytest | Test suite | MIT |
| PyVista Qt / VTK wheels | https://pypi.org/project/vtk/ | Rendering backend | BSD-3-Clause |

## Referenced but not depended upon

| Project | Link | Why it is not a dependency |
|---|---|---|
| ROS-Industrial Noether | https://github.com/ros-industrial/noether | Studied as an architecture reference (separation of mesh preparation, toolpath planning and toolpath modification — PRD stages 7–10 mirror that split). Apache-2.0, but it is a ROS/ROS-Industrial stack; installing it would add a large unrelated runtime for no analytical gain. No Noether code is copied into MTRA. |

## Notice

MTRA is an independent analysis tool. It contains no SmartGrid code, links
against no SmartGrid component, and produces no robot programs or production
paths.
