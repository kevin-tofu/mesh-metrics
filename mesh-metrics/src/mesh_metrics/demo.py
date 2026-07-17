from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mesh_metrics.geometry import FacetLabels, MeshGeometry
from mesh_metrics.regions import AutoCurveSpec, RegionSpecSet


@dataclass(frozen=True)
class DemoCase:
    mesh: MeshGeometry
    facet_sizes: dict[str, float]

    def write(self, directory: str | Path) -> dict[str, Any]:
        import meshio

        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        mesh_path = output / "notched_channel.mesh"
        vtk_path = output / "notched_channel.vtk"
        labels_path = output / "facet_labels.json"
        sizes_path = output / "facet_sizes.json"
        regions_path = output / "regions.json"

        points = self.mesh.points.T
        cells = [("tetra", self.mesh.elements.T)]
        if self.mesh.facets is not None and self.mesh.facets.shape[0] == 3:
            cells.insert(0, ("triangle", self.mesh.facets.T))
        meshio_mesh = meshio.Mesh(points, cells)
        meshio.write(mesh_path, meshio_mesh)
        meshio.write(vtk_path, meshio_mesh)

        labels_path.write_text(json.dumps(self.mesh.facet_labels.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        sizes_path.write_text(json.dumps(self.facet_sizes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        spec = RegionSpecSet.from_facet_labels(
            self.mesh.facet_labels,
            max_distance=0.08,
            max_angle_deg=35.0,
            auto_curves=AutoCurveSpec(from_surface_boundaries=True, include_exterior=True, max_distance=0.04),
        )
        regions_path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "mesh": str(mesh_path),
            "vtk": str(vtk_path),
            "facet_labels": str(labels_path),
            "facet_sizes": str(sizes_path),
            "regions": str(regions_path),
            "npoints": self.mesh.npoints,
            "nelements": self.mesh.nelements,
            "nfacets": self.mesh.nfacets,
        }


def create_notched_channel(nx: int = 12, ny: int = 8, nz: int = 8) -> DemoCase:
    return _create_channel(
        nx=nx,
        ny=ny,
        nz=nz,
        length=4.0,
        half_width=1.0,
        half_height=1.0,
        predicate=_notched_channel_region,
        labeler=_label_notched_channel_facets,
        facet_sizes={
            "inlet": 0.20,
            "outlet": 0.20,
            "hole": 0.10,
            "notch": 0.12,
            "wall": 0.24,
        },
    )


def create_complex_channel(nx: int = 36, ny: int = 24, nz: int = 24) -> DemoCase:
    return _create_channel(
        nx=nx,
        ny=ny,
        nz=nz,
        length=6.0,
        half_width=1.5,
        half_height=1.5,
        predicate=_complex_channel_region,
        labeler=_label_complex_channel_facets,
        facet_sizes={
            "inlet": 0.22,
            "outlet": 0.22,
            "main_bore": 0.12,
            "offset_bore": 0.11,
            "cross_bore": 0.10,
            "slot": 0.14,
            "pocket": 0.16,
            "notch": 0.14,
            "wall": 0.30,
        },
    )


def _create_channel(
    *,
    nx: int,
    ny: int,
    nz: int,
    length: float,
    half_width: float,
    half_height: float,
    predicate,
    labeler,
    facet_sizes: dict[str, float],
) -> DemoCase:
    x_values = np.linspace(0.0, length, nx + 1)
    y_values = np.linspace(-half_width, half_width, ny + 1)
    z_values = np.linspace(-half_height, half_height, nz + 1)
    points = np.asarray([[x, y, z] for x in x_values for y in y_values for z in z_values], dtype=float)

    def node(i: int, j: int, k: int) -> int:
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    tets: list[list[int]] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                center = np.asarray(
                    [
                        0.5 * (x_values[i] + x_values[i + 1]),
                        0.5 * (y_values[j] + y_values[j + 1]),
                        0.5 * (z_values[k] + z_values[k + 1]),
                    ]
                )
                if predicate(center):
                    continue
                v000 = node(i, j, k)
                v100 = node(i + 1, j, k)
                v010 = node(i, j + 1, k)
                v110 = node(i + 1, j + 1, k)
                v001 = node(i, j, k + 1)
                v101 = node(i + 1, j, k + 1)
                v011 = node(i, j + 1, k + 1)
                v111 = node(i + 1, j + 1, k + 1)
                tets.extend(
                    [
                        [v000, v100, v110, v111],
                        [v000, v110, v010, v111],
                        [v000, v010, v011, v111],
                        [v000, v011, v001, v111],
                        [v000, v001, v101, v111],
                        [v000, v101, v100, v111],
                    ]
                )

    mesh = MeshGeometry(points=points.T, elements=np.asarray(tets, dtype=np.int64).T, backend="demo")
    labels = labeler(mesh)
    mesh = mesh.with_facet_labels(labels)
    return DemoCase(mesh=mesh, facet_sizes=facet_sizes)


def _notched_channel_region(center: np.ndarray) -> bool:
    hole_radius = 0.34
    if center[1] * center[1] + center[2] * center[2] < hole_radius * hole_radius:
        return True
    return bool(1.45 <= center[0] <= 2.55 and center[1] > 0.25 and center[2] > 0.25)


def _complex_channel_region(center: np.ndarray) -> bool:
    x, y, z = center
    main_y = 0.18 * np.sin(2.0 * np.pi * x / 6.0)
    main_z = 0.16 * np.cos(3.0 * np.pi * x / 6.0)
    main_radius = 0.30 + 0.05 * np.sin(4.0 * np.pi * x / 6.0)
    if (y - main_y) ** 2 + (z - main_z) ** 2 < main_radius**2:
        return True
    if 0.8 <= x <= 5.4 and (y + 0.82) ** 2 + (z - 0.55) ** 2 < 0.22**2:
        return True
    if (x - 2.0) ** 2 + (z + 0.28) ** 2 < 0.24**2 and -1.45 <= y <= 1.45:
        return True
    if (x - 4.45) ** 2 + (z - 0.34) ** 2 < 0.20**2 and -1.45 <= y <= 1.45:
        return True
    if 0.9 <= x <= 2.1 and y > 0.78 and -0.42 <= z <= 0.42:
        return True
    if 3.1 <= x <= 5.2 and -0.25 <= y <= 0.72 and z > 0.78:
        return True
    if 2.5 <= x <= 3.45 and y < -0.85 and z < -0.30:
        return True
    return False


def _label_notched_channel_facets(mesh: MeshGeometry) -> FacetLabels:
    assert mesh.facets is not None
    hole_radius = 0.34
    groups: dict[str, list[int]] = {"inlet": [], "outlet": [], "hole": [], "notch": [], "wall": []}
    for index, facet in enumerate(mesh.facets.T):
        centroid = mesh.points[:, facet].mean(axis=1)
        x, y, z = centroid
        radius = float(np.hypot(y, z))
        if x < 1.0e-9:
            groups["inlet"].append(index)
        elif x > 4.0 - 1.0e-9:
            groups["outlet"].append(index)
        elif radius < hole_radius + 0.16:
            groups["hole"].append(index)
        elif 1.35 <= x <= 2.65 and y > 0.15 and z > 0.15:
            groups["notch"].append(index)
        else:
            groups["wall"].append(index)
    return FacetLabels({label: np.asarray(indices, dtype=np.int64) for label, indices in groups.items()})


def _label_complex_channel_facets(mesh: MeshGeometry) -> FacetLabels:
    assert mesh.facets is not None
    groups: dict[str, list[int]] = {
        "inlet": [],
        "outlet": [],
        "main_bore": [],
        "offset_bore": [],
        "cross_bore": [],
        "slot": [],
        "pocket": [],
        "notch": [],
        "wall": [],
    }
    for index, facet in enumerate(mesh.facets.T):
        centroid = mesh.points[:, facet].mean(axis=1)
        x, y, z = centroid
        main_y = 0.18 * np.sin(2.0 * np.pi * x / 6.0)
        main_z = 0.16 * np.cos(3.0 * np.pi * x / 6.0)
        main_radius = 0.30 + 0.05 * np.sin(4.0 * np.pi * x / 6.0)
        if x < 1.0e-9:
            groups["inlet"].append(index)
        elif x > 6.0 - 1.0e-9:
            groups["outlet"].append(index)
        elif (y - main_y) ** 2 + (z - main_z) ** 2 < (main_radius + 0.12) ** 2:
            groups["main_bore"].append(index)
        elif 0.65 <= x <= 5.55 and (y + 0.82) ** 2 + (z - 0.55) ** 2 < (0.22 + 0.12) ** 2:
            groups["offset_bore"].append(index)
        elif (
            ((x - 2.0) ** 2 + (z + 0.28) ** 2 < (0.24 + 0.12) ** 2)
            or ((x - 4.45) ** 2 + (z - 0.34) ** 2 < (0.20 + 0.12) ** 2)
        ):
            groups["cross_bore"].append(index)
        elif 0.8 <= x <= 2.2 and y > 0.68 and -0.55 <= z <= 0.55:
            groups["slot"].append(index)
        elif 3.0 <= x <= 5.3 and -0.35 <= y <= 0.85 and z > 0.68:
            groups["pocket"].append(index)
        elif 2.4 <= x <= 3.55 and y < -0.75 and z < -0.18:
            groups["notch"].append(index)
        else:
            groups["wall"].append(index)
    return FacetLabels({label: np.asarray(indices, dtype=np.int64) for label, indices in groups.items()})
