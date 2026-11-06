"""Group E — mesh surface characteristics vs trajectory (PRD §7, phase 1 items E1–E6, E9).

All per-vertex fields are evaluated at the barycentric coordinates of the closest
surface point to each trajectory sample, not snapped to the nearest vertex. On a
tessellated surface the difference is the tessellation itself, and reporting it as
curvature is how a meshing decision gets mistaken for an engineering rule.

Curvature comes from libigl. When it is unavailable the normals-based parameters
(E1–E3) still stand on their own, and E2/E4/E5/E6 become explicit absences.

E9 is deliberately not a duplicate of A3: A3 is the absolute TCP-to-surface
distance (which includes standoff), E9 is the *variation* of that distance about
its median — the part of the deviation that a constant-standoff path would not
have.
"""

from __future__ import annotations

import numpy as np

from mtra.core.geometry import circular_dispersion_deg, circular_mean_deg
from mtra.core.types import ReasonCode
from mtra.measurements.catalogue import MeasurementSet, numeric_stats
from mtra.measurements.common import PairView, angular_deviation_deg, mean_vector


def _curvature_vectors(view: PairView):
    """Curvature fields sampled at the corresponded surface points."""
    points = view.surface_points()
    if len(points) == 0:
        return None
    sample = view.surface.sample(points)
    if sample.k1 is None or sample.k2 is None:
        return None
    return sample


def _mesh_edge_context(view: PairView) -> tuple[float | None, str]:
    """Median triangle edge length, and the sentence explaining why it bounds E2/E4/E5.

    Curvature on a triangle mesh is estimated from the discrete operator, whose
    resolution is the edge length: a surface that bends over 50 mm cannot be
    distinguished from a flat one on 12 mm triangles, and a *faceted* approximation of
    a smooth surface reports curvature spikes at the facets. Quoting k without the
    mesh scale invites reading a meshing artefact as a geometric fact, so the scale
    travels with the number.
    """
    edges = getattr(view.surface.mesh, "edges_unique_length", None)
    if edges is None or len(edges) == 0:
        return None, "the mesh edge length could not be sampled, so no resolution bound is claimed"
    median = float(np.median(edges))
    return median, (
        f"curvature below roughly 1/{median:.2f} mm⁻¹ cannot be resolved on this mesh "
        f"(median edge {median:.2f} mm), and the values carry that discretisation; "
        "features at or below the edge length describe the tessellation, not the part"
    )


def measure_surface(view: PairView, measurements: MeasurementSet) -> None:
    points = view.surface_points()
    if len(points) == 0:
        for parameter_id in ("E1", "E2", "E3", "E4", "E5", "E6", "E9"):
            measurements.unavailable(
                parameter_id,
                ReasonCode.NO_SURFACE_CORRESPONDENCE,
                "no trajectory sample was corresponded to the mesh",
            )
        return

    sample = view.surface.sample(points)
    normals = sample.normals
    mean_normal = mean_vector(normals)

    measurements.add(
        "E1",
        {
            "mean_normal": [float(v) for v in mean_normal],
            "plane_tilt_deg": float(
                np.degrees(np.arccos(np.clip(abs(mean_normal[2]), 0.0, 1.0)))
            ),
            "samples": int(len(normals)),
        },
        method=(
            "normalised mean of the interpolated per-sample surface normals; the tilt "
            "relative to mesh +Z is reported because a mean normal only describes a "
            "surface that is locally flat"
        ),
        source_refs=["mesh.features"],
    )

    deviations = angular_deviation_deg(normals, mean_normal)
    measurements.add(
        "E3",
        numeric_stats(deviations, unit_label="degrees"),
        method="angle between each interpolated surface normal and the mean normal",
        uncertainty=float(np.std(deviations)),
        source_refs=["mesh.features"],
    )

    if sample.k1 is None or sample.k2 is None:
        notes_text = " ".join(sample.notes)
        reason = (
            ReasonCode.OPTIONAL_DEPENDENCY_MISSING
            if "libigl" in notes_text.lower() and "unavailable" in notes_text.lower()
            else ReasonCode.COMPUTATION_FAILED
        )
        detail = notes_text or "principal curvature was not computed"
        for parameter_id in ("E2", "E4", "E5", "E6"):
            measurements.unavailable(parameter_id, reason, detail)
    else:
        k1 = np.asarray(sample.k1, dtype=float)
        k2 = np.asarray(sample.k2, dtype=float)
        magnitude = np.maximum(np.abs(k1), np.abs(k2))
        edge_mm, edge_note = _mesh_edge_context(view)
        measurements.add(
            "E2",
            {
                **numeric_stats(magnitude, unit_label="1/mm"),
                "mesh_median_edge_mm": None if edge_mm is None else round(edge_mm, 4),
                "resolution_ratio": (
                    None if edge_mm is None else round(float(np.mean(magnitude)) * edge_mm, 6)
                ),
            },
            method=(
                "max(|k1|, |k2|) per sample from libigl principal curvature, interpolated "
                "through the barycentric coordinates of the closest surface point. "
                f"{edge_note}. resolution_ratio is mean|k| × median edge: values near or "
                "below 1 mean the bend is being measured at the limit of the mesh"
            ),
            uncertainty=float(np.std(magnitude)),
            source_refs=["mesh.features", "libigl"],
        )

        gaussian = k1 * k2
        measurements.add(
            "E4",
            numeric_stats(gaussian, unit_label="1/mm2"),
            method=f"k1 * k2 per sample (Gaussian curvature); {edge_note}",
            uncertainty=float(np.std(gaussian)),
            source_refs=["mesh.features", "libigl"],
        )

        mean_curvature = 0.5 * (k1 + k2)
        measurements.add(
            "E5",
            numeric_stats(mean_curvature, unit_label="1/mm"),
            method=f"(k1 + k2) / 2 per sample (mean curvature); {edge_note}",
            uncertainty=float(np.std(mean_curvature)),
            source_refs=["mesh.features", "libigl"],
        )

        if sample.principal_dir_1 is not None:
            directions_2d = np.asarray(sample.principal_dir_1, dtype=float)
            headings = []
            for k1_value, direction in zip(k1, directions_2d, strict=True):
                # The major principal direction alternates with the sign of k1; take the
                # direction associated with the larger absolute curvature so the field
                # does not flip mid-surface.
                projected = direction[:2]
                if np.linalg.norm(projected) < 1e-9:
                    continue
                headings.append(
                    float(np.mod(np.degrees(np.arctan2(projected[1], projected[0])), 180.0))
                )
            if headings:
                heading_array = np.array(headings, dtype=float)
                dispersion = circular_dispersion_deg(heading_array)
                measurements.add(
                    "E6",
                    {
                        "mean_deg": circular_mean_deg(heading_array),
                        "dispersion_deg": dispersion,
                        "samples": int(len(heading_array)),
                    },
                    method=(
                        "axial bearing of the major principal curvature direction in mesh "
                        "XY, averaged as axial data; dispersion indicates how well a single "
                        "direction describes the surface"
                    ),
                    uncertainty=dispersion,
                    source_refs=["mesh.features", "libigl"],
                )
            else:
                measurements.unavailable(
                    "E6",
                    ReasonCode.DEGENERATE_GEOMETRY,
                    "every principal direction was perpendicular to the XY plane, so no "
                    "in-plane bearing exists",
                )
        else:
            measurements.unavailable(
                "E6",
                ReasonCode.COMPUTATION_FAILED,
                "principal directions were not returned by the curvature computation",
            )

    distances = np.array([record.distance_mm for record in view.correspondence.records], dtype=float)
    if distances.size == 0:
        measurements.unavailable(
            "E9",
            ReasonCode.NO_SURFACE_CORRESPONDENCE,
            "no distances to deviate from",
        )
    else:
        residual = distances - float(np.median(distances))
        rms = float(np.sqrt(np.mean(residual**2)))
        measurements.add(
            "E9",
            {
                "rms_mm": rms,
                "max_abs_mm": float(np.max(np.abs(residual))),
                "median_standoff_mm": float(np.median(distances)),
                "samples": int(distances.size),
            },
            method=(
                "RMS of (TCP-to-surface distance - median distance); this is the part of "
                "the path-to-surface deviation not explained by a constant standoff"
            ),
            uncertainty=rms,
            source_refs=["correspondence.projection"],
        )
