"""Surface differential geometry at arbitrary trajectory points.

Trajectory samples almost never coincide with mesh vertices, so every per-vertex
field is interpolated through the barycentric coordinates of the closest surface
point. Doing it any other way — nearest-vertex snapping, for instance — quietly
quantises normals and curvature to the mesh resolution and then reports the
quantisation as geometry.

libigl supplies principal curvature; trimesh supplies the closest-point search.
If libigl is unavailable, curvature becomes an explicit ``Unavailable`` and
normal-based measurements still run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from mtra.core.geometry import barycentric_coordinates, unit
from mtra.mesh.loader import as_trimesh


@dataclass
class SurfaceSample:
    """Surface quantities at a set of query points."""

    closest_points: np.ndarray
    face_ids: np.ndarray
    barycentric: np.ndarray
    distances: np.ndarray
    normals: np.ndarray
    k1: np.ndarray | None = None
    k2: np.ndarray | None = None
    principal_dir_1: np.ndarray | None = None
    principal_dir_2: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def mean_curvature(self) -> np.ndarray | None:
        if self.k1 is None or self.k2 is None:
            return None
        return 0.5 * (self.k1 + self.k2)

    @property
    def gaussian_curvature(self) -> np.ndarray | None:
        if self.k1 is None or self.k2 is None:
            return None
        return self.k1 * self.k2


class SurfaceField:
    """Per-mesh cached vertex fields plus a point query."""

    def __init__(self, record) -> None:
        self.record = record
        self._mesh: trimesh.Trimesh = as_trimesh(record)
        self._vertices = np.asarray(self._mesh.vertices, dtype=float)
        self._faces = np.asarray(self._mesh.faces, dtype=np.int64)
        self._vertex_normals = np.asarray(self._mesh.vertex_normals, dtype=float)
        self._k1: np.ndarray | None = None
        self._k2: np.ndarray | None = None
        self._pd1: np.ndarray | None = None
        self._pd2: np.ndarray | None = None
        self._curvature_error: str | None = None
        self._load_curvature()

    # -- curvature ---------------------------------------------------------- #

    def _load_curvature(self) -> None:
        try:
            import igl
        except ImportError as exc:  # optional dependency
            self._curvature_error = f"libigl unavailable: {exc}"
            return

        vertices = self._vertices.astype(np.float64)
        faces = self._faces.astype(np.int32)
        try:
            pd1, pd2, k1, k2 = igl.principal_curvature(vertices, faces)[:4]
        except Exception as exc:  # pragma: no cover - native failure mode
            self._curvature_error = f"libigl principal_curvature failed: {exc}"
            return

        self._pd1 = np.asarray(pd1, dtype=float)
        self._pd2 = np.asarray(pd2, dtype=float)
        self._k1 = np.asarray(k1, dtype=float)
        self._k2 = np.asarray(k2, dtype=float)

    @property
    def curvature_available(self) -> bool:
        return self._k1 is not None

    @property
    def curvature_error(self) -> str | None:
        return self._curvature_error

    @property
    def mesh(self) -> trimesh.Trimesh:
        return self._mesh

    @property
    def mesh_record(self):
        """The ``MeshRecord`` this field was built from (boundary loops live there)."""
        return self.record

    # -- query -------------------------------------------------------------- #

    def sample(self, points: np.ndarray) -> SurfaceSample:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must be an (N, 3) array")
        if len(pts) == 0:
            empty = np.zeros((0, 3))
            return SurfaceSample(
                closest_points=empty,
                face_ids=np.zeros(0, dtype=np.int64),
                barycentric=np.zeros((0, 3)),
                distances=np.zeros(0),
                normals=empty,
            )

        closest, distance, face_ids = self._mesh.nearest.on_surface(pts)
        closest = np.asarray(closest, dtype=float)
        face_ids = np.asarray(face_ids, dtype=np.int64)

        triangles = self._vertices[self._faces[face_ids]]  # (N, 3, 3)
        bary = np.array(
            [
                barycentric_coordinates(closest[i], triangles[i])
                for i in range(len(closest))
            ]
        )

        normals = np.array(
            [unit(self._interpolate(b, self._vertex_normals, face_id))
             for b, face_id in zip(bary, face_ids, strict=True)]
        )

        sample = SurfaceSample(
            closest_points=closest,
            face_ids=face_ids,
            barycentric=bary,
            distances=np.asarray(distance, dtype=float),
            normals=normals,
        )
        if self.curvature_available:
            assert self._k1 is not None and self._k2 is not None
            assert self._pd1 is not None and self._pd2 is not None
            sample.k1 = np.array(
                [self._interpolate(b, self._k1, f) for b, f in zip(bary, face_ids, strict=True)]
            )
            sample.k2 = np.array(
                [self._interpolate(b, self._k2, f) for b, f in zip(bary, face_ids, strict=True)]
            )
            sample.principal_dir_1 = np.array(
                [
                    unit(self._interpolate(b, self._pd1, f))
                    for b, f in zip(bary, face_ids, strict=True)
                ]
            )
            sample.principal_dir_2 = np.array(
                [
                    unit(self._interpolate(b, self._pd2, f))
                    for b, f in zip(bary, face_ids, strict=True)
                ]
            )
        elif self._curvature_error:
            sample.notes.append(self._curvature_error)
        return sample

    def _interpolate(self, bary: np.ndarray, field_values: np.ndarray, face_id: int) -> np.ndarray:
        corner_indices = self._faces[face_id]
        return np.asarray(bary, dtype=float) @ np.asarray(field_values[corner_indices], dtype=float)

    def nearest_vertex(self, points: np.ndarray) -> np.ndarray:
        """Nearest vertex index per point.

        Used only for geodesic queries, where pygeodesic's API is vertex-to-vertex.
        The approximation is recorded in the measurement method string so the
        report can distinguish it from an exact surface distance.
        """
        _, vertex_ids = self._mesh.kdtree.query(np.asarray(points, dtype=float))
        return np.asarray(vertex_ids, dtype=np.int64)
