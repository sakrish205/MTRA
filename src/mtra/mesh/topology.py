"""Topology: connected components and boundary loops.

Both are needed by the region parameters (group H) and by the honesty warnings:
a mesh with several components has several *candidate* painting regions, and a
mesh with boundary loops is not closed, which makes inside/outside tests — and
therefore ray-parity intersection counting — unreliable.
"""

from __future__ import annotations

import numpy as np
import trimesh


def face_components(mesh: trimesh.Trimesh) -> np.ndarray:
    """Component id per face, ordered by descending face count.

    Faces rather than vertices: two shells that touch at a single vertex are
    separate surfaces, and vertex connectivity would merge them into one region.
    """
    n_faces = len(mesh.faces)
    labels = np.full(n_faces, -1, dtype=np.int64)
    if n_faces == 0:
        return labels

    adjacency = mesh.face_adjacency
    components = trimesh.graph.connected_components(
        adjacency, nodes=np.arange(n_faces), min_len=1
    )
    components = sorted(components, key=len, reverse=True)
    for component_id, faces in enumerate(components):
        labels[np.asarray(faces, dtype=np.int64)] = component_id
    return labels


def boundary_loops(mesh: trimesh.Trimesh, max_loops: int = 200) -> list[np.ndarray]:
    """Ordered boundary loops as vertex-index arrays.

    Walks the once-used edges. Loops that cannot be closed (non-manifold
    junctions) are returned as open chains; the walk is bounded so a pathological
    mesh cannot hang a run.
    """
    if len(mesh.faces) == 0:
        return []

    edges = np.sort(mesh.edges_sorted, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique_edges[counts == 1]
    if len(boundary) == 0:
        return []

    neighbours: dict[int, list[int]] = {}
    for a, b in boundary:
        neighbours.setdefault(int(a), []).append(int(b))
        neighbours.setdefault(int(b), []).append(int(a))

    unused = {frozenset((int(a), int(b))) for a, b in boundary}
    loops: list[np.ndarray] = []
    for start in list(neighbours):
        if len(loops) >= max_loops:
            break
        if not any(frozenset((start, n)) in unused for n in neighbours[start]):
            continue
        loop = [start]
        previous = None
        current = start
        while True:
            candidates = [
                n
                for n in neighbours[current]
                if frozenset((current, n)) in unused and n != previous
            ]
            if not candidates:
                break
            nxt = candidates[0]
            unused.discard(frozenset((current, nxt)))
            previous, current = current, nxt
            loop.append(current)
            if current == start:
                break
        if len(loop) >= 3:
            loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def analyse(mesh: trimesh.Trimesh, faces: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return ``(component_id_per_face, boundary_loops)``."""
    del faces  # retained in the signature for callers that pass the record's arrays
    return face_components(mesh), boundary_loops(mesh)
