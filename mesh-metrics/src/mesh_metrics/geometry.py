from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]
IndexArray = NDArray[np.int64]


@dataclass(frozen=True)
class FacetLabels:
    """Named groups of facet indices."""

    groups: dict[str, IndexArray]

    def __post_init__(self) -> None:
        normalized = {str(name): np.asarray(indices, dtype=np.int64) for name, indices in self.groups.items()}
        for name, indices in normalized.items():
            if indices.ndim != 1:
                raise ValueError(f"facet label {name!r} must be a 1D index array")
            if indices.size and indices.min() < 0:
                raise ValueError(f"facet label {name!r} contains negative indices")
        object.__setattr__(self, "groups", normalized)

    @classmethod
    def from_mapping(cls, labels: dict[str, Any] | None) -> FacetLabels | None:
        if labels is None:
            return None
        return cls({name: np.asarray(indices, dtype=np.int64) for name, indices in labels.items()})

    @classmethod
    def from_per_facet(cls, labels: Any, *, prefix: str = "label") -> FacetLabels:
        values = np.asarray(labels)
        if values.ndim != 1:
            raise ValueError("per-facet labels must be a 1D array")
        groups: dict[str, IndexArray] = {}
        for value in np.unique(values):
            groups[f"{prefix}_{value}"] = np.flatnonzero(values == value).astype(np.int64)
        return cls(groups)

    def validate(self, nfacets: int) -> None:
        for name, indices in self.groups.items():
            if indices.size and indices.max() >= nfacets:
                raise ValueError(f"facet label {name!r} contains indices outside facets")

    def to_dict(self) -> dict[str, list[int]]:
        return {name: indices.tolist() for name, indices in self.groups.items()}


@dataclass(frozen=True)
class FacetGeometry:
    """Facet connectivity and optional label groups."""

    connectivity: IndexArray
    labels: FacetLabels | None = None

    def __post_init__(self) -> None:
        connectivity = np.asarray(self.connectivity, dtype=np.int64)
        labels = self.labels if isinstance(self.labels, FacetLabels) else FacetLabels.from_mapping(self.labels)
        if connectivity.ndim != 2:
            raise ValueError("facet connectivity must have shape (nodes_per_facet, nfacets)")
        if labels is not None:
            labels.validate(connectivity.shape[1])
        object.__setattr__(self, "connectivity", connectivity)
        object.__setattr__(self, "labels", labels)

    @property
    def nfacets(self) -> int:
        return int(self.connectivity.shape[1])

    @property
    def nodes_per_facet(self) -> int:
        return int(self.connectivity.shape[0])

    def label_indices(self) -> dict[str, IndexArray]:
        if self.labels is None:
            return {}
        return self.labels.groups

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes_per_facet": self.nodes_per_facet,
            "nfacets": self.nfacets,
            "labels": self.labels.to_dict() if self.labels is not None else {},
        }


@dataclass(frozen=True)
class MeshGeometry:
    """Minimal mesh representation used by the statistics engine."""

    points: Array
    elements: IndexArray
    facets: IndexArray | None = None
    facet_labels: FacetLabels | None = None
    source: str | None = None
    backend: str | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        elements = np.asarray(self.elements, dtype=np.int64)
        facets = _infer_boundary_facets(elements, points.shape[0]) if self.facets is None else np.asarray(self.facets, dtype=np.int64)
        facet_labels = self.facet_labels if isinstance(self.facet_labels, FacetLabels) else FacetLabels.from_mapping(self.facet_labels)

        if points.ndim != 2:
            raise ValueError("points must have shape (dimension, npoints)")
        if elements.ndim != 2:
            raise ValueError("elements must have shape (nodes_per_element, nelements)")
        if facets is not None and facets.ndim != 2:
            raise ValueError("facets must have shape (nodes_per_facet, nfacets)")
        if elements.size and (elements.min() < 0 or elements.max() >= points.shape[1]):
            raise ValueError("elements contain point indices outside points")
        if facets is not None and facets.size and (facets.min() < 0 or facets.max() >= points.shape[1]):
            raise ValueError("facets contain point indices outside points")
        if facet_labels is not None:
            facet_labels.validate(0 if facets is None else facets.shape[1])

        object.__setattr__(self, "points", points)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "facets", facets)
        object.__setattr__(self, "facet_labels", facet_labels)

    @property
    def dimension(self) -> int:
        return int(self.points.shape[0])

    @property
    def npoints(self) -> int:
        return int(self.points.shape[1])

    @property
    def nelements(self) -> int:
        return int(self.elements.shape[1])

    @property
    def nfacets(self) -> int:
        if self.facets is None:
            return 0
        return int(self.facets.shape[1])

    @property
    def facet_geometry(self) -> FacetGeometry | None:
        if self.facets is None:
            return None
        return FacetGeometry(self.facets, self.facet_labels)

    @classmethod
    def from_skfem(cls, mesh: Any, *, source: str | None = None, backend: str | None = "skfem") -> MeshGeometry:
        facets = getattr(mesh, "facets", None)
        facet_labels = _skfem_facet_labels(mesh)
        return cls(points=np.asarray(mesh.p), elements=np.asarray(mesh.t), facets=facets, facet_labels=facet_labels, source=source, backend=backend)

    @classmethod
    def from_object(cls, mesh: Any, *, source: str | None = None, backend: str | None = None) -> MeshGeometry:
        """Create geometry from common skfem/fluxfem-style mesh objects."""

        if hasattr(mesh, "p") and hasattr(mesh, "t"):
            return cls.from_skfem(mesh, source=source, backend=backend)

        point_names = ("points", "vertices", "nodes", "coords", "coordinates")
        element_names = ("elements", "cells", "connectivity", "t")
        facet_names = ("facets", "faces", "edges", "boundary_facets")
        facet_label_names = ("facet_labels", "boundary_labels", "boundaries")

        point_name, points = _first_named_attr(mesh, point_names)
        element_name, elements = _first_named_attr(mesh, element_names)
        facet_name, facets = _first_named_attr(mesh, facet_names, required=False)
        facet_labels = _first_attr(mesh, facet_label_names, required=False)
        if points is None or elements is None:
            raise TypeError("mesh object must expose p/t or points/elements-style arrays")

        points_array = _as_dimension_first(points, prefer_point_rows=point_name != "p")
        elements_array = _as_index_connectivity(elements, points_array.shape[1], prefer_item_rows=element_name != "t")
        facets_array = None if facets is None else _as_index_connectivity(facets, points_array.shape[1], prefer_item_rows=facet_name in {"faces", "edges", "boundary_facets"})
        return cls(points=points_array, elements=elements_array, facets=facets_array, facet_labels=facet_labels, source=source, backend=backend)

    @classmethod
    def load(cls, path: str | Path, *, backend: str = "skfem") -> MeshGeometry:
        """Load a mesh through scikit-fem's meshio-backed loader."""

        from mesh_metrics.backends import get_backend

        return get_backend(backend).load(Path(path))

    def element_measures(self) -> Array:
        return _connectivity_measures(self.points, self.elements, domain_dimension=self.dimension)

    def facet_measures(self) -> Array:
        if self.facets is None:
            return np.asarray([], dtype=float)
        return _connectivity_measures(self.points, self.facets, domain_dimension=max(self.dimension - 1, 0))

    def element_diameters(self) -> Array:
        return _diameters(self.points, self.elements)

    def facet_diameters(self) -> Array:
        if self.facets is None:
            return np.asarray([], dtype=float)
        return _diameters(self.points, self.facets)

    def element_edge_aspect_ratios(self) -> Array:
        return _edge_aspect_ratios(self.points, self.elements)

    def facet_edge_aspect_ratios(self) -> Array:
        if self.facets is None:
            return np.asarray([], dtype=float)
        return _edge_aspect_ratios(self.points, self.facets)

    def facet_label_indices(self) -> dict[str, IndexArray]:
        if self.facet_labels is None:
            return {}
        return self.facet_labels.groups

    def with_facet_labels(self, labels: FacetLabels | dict[str, Any]) -> MeshGeometry:
        return MeshGeometry(
            points=self.points,
            elements=self.elements,
            facets=self.facets,
            facet_labels=labels if isinstance(labels, FacetLabels) else FacetLabels.from_mapping(labels),
            source=self.source,
            backend=self.backend,
        )


def _skfem_facet_labels(mesh: Any) -> FacetLabels | None:
    boundaries = getattr(mesh, "boundaries", None)
    if not boundaries:
        return None
    groups = {name: np.asarray(indices, dtype=np.int64) for name, indices in boundaries.items()}
    return FacetLabels(groups)


def _first_attr(obj: Any, names: tuple[str, ...], *, required: bool = True) -> Any:
    _, value = _first_named_attr(obj, names, required=required)
    return value


def _first_named_attr(obj: Any, names: tuple[str, ...], *, required: bool = True) -> tuple[str | None, Any]:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            return name, value() if callable(value) else value
    if required:
        raise AttributeError(f"none of {names!r} found")
    return None, None


def _as_dimension_first(values: Any, *, prefer_point_rows: bool = False) -> Array:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("point coordinates must be a 2D array")
    if prefer_point_rows and array.shape[1] <= 3 and array.shape[0] > array.shape[1]:
        return array.T
    if array.shape[0] <= 3:
        return array
    if array.shape[1] <= 3:
        return array.T
    raise ValueError("cannot infer coordinate array orientation")


def _as_index_connectivity(values: Any, npoints: int, *, prefer_item_rows: bool = True) -> IndexArray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("connectivity must be a 2D array")
    if array.size == 0:
        return array
    if array.max() < npoints:
        return _orient_connectivity(array, prefer_item_rows=prefer_item_rows)
    if array.T.max() < npoints:
        return _orient_connectivity(array.T, prefer_item_rows=prefer_item_rows)
    raise ValueError("connectivity contains indices outside points")


def _orient_connectivity(array: IndexArray, *, prefer_item_rows: bool) -> IndexArray:
    """Return connectivity as (nodes_per_item, nitems)."""

    common_node_counts = {1, 2, 3, 4, 6, 8, 10, 20, 27}
    rows, cols = array.shape
    rows_common = rows in common_node_counts
    cols_common = cols in common_node_counts

    if rows_common and not cols_common:
        return array
    if cols_common and not rows_common:
        return array.T
    if rows == 1 and cols > 1:
        return array.T
    if cols == 1:
        return array
    if prefer_item_rows and cols_common:
        return array.T
    if not prefer_item_rows and rows_common:
        return array
    if rows > cols and cols_common:
        return array.T
    if rows <= cols:
        return array
    return array.T


def _infer_boundary_facets(elements: IndexArray, dimension: int) -> IndexArray | None:
    if elements.ndim != 2 or elements.shape[1] == 0:
        return None
    nodes_per_element = elements.shape[0]
    if dimension == 2 and nodes_per_element == 3:
        local_facets = ((0, 1), (1, 2), (2, 0))
    elif dimension == 2 and nodes_per_element == 4:
        local_facets = ((0, 1), (1, 2), (2, 3), (3, 0))
    elif dimension == 3 and nodes_per_element == 4:
        local_facets = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    elif dimension == 3 and nodes_per_element == 8:
        local_facets = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    else:
        return None

    counts: dict[tuple[int, ...], int] = {}
    oriented: dict[tuple[int, ...], tuple[int, ...]] = {}
    for element in elements.T:
        for local in local_facets:
            facet = tuple(int(element[i]) for i in local)
            key = tuple(sorted(facet))
            counts[key] = counts.get(key, 0) + 1
            oriented.setdefault(key, facet)

    boundary = [oriented[key] for key, count in counts.items() if count == 1]
    if not boundary:
        return None
    return np.asarray(boundary, dtype=np.int64).T


def _connectivity_measures(points: Array, connectivity: IndexArray, *, domain_dimension: int) -> Array:
    if connectivity.shape[1] == 0:
        return np.asarray([], dtype=float)
    return np.asarray(
        [_cell_measure(points[:, cell], domain_dimension=domain_dimension) for cell in connectivity.T],
        dtype=float,
    )


def _diameters(points: Array, connectivity: IndexArray) -> Array:
    if connectivity.shape[1] == 0:
        return np.asarray([], dtype=float)
    values: list[float] = []
    for cell in connectivity.T:
        vertices = points[:, cell]
        diffs = vertices[:, :, None] - vertices[:, None, :]
        distances = np.linalg.norm(diffs, axis=0)
        values.append(float(distances.max()))
    return np.asarray(values, dtype=float)


def _edge_aspect_ratios(points: Array, connectivity: IndexArray) -> Array:
    if connectivity.shape[1] == 0:
        return np.asarray([], dtype=float)
    values: list[float] = []
    for cell in connectivity.T:
        vertices = _unique_columns(points[:, cell])
        edge_lengths = _pairwise_distances(vertices)
        positive = edge_lengths[edge_lengths > 0.0]
        if positive.size == 0:
            values.append(float("nan"))
        else:
            values.append(float(positive.max() / positive.min()))
    return np.asarray(values, dtype=float)


def _pairwise_distances(vertices: Array) -> Array:
    diffs = vertices[:, :, None] - vertices[:, None, :]
    distances = np.linalg.norm(diffs, axis=0)
    return distances[np.triu_indices(vertices.shape[1], k=1)]


def _cell_measure(vertices: Array, *, domain_dimension: int) -> float:
    vertices = _unique_columns(vertices)
    if domain_dimension <= 0:
        return 0.0
    if domain_dimension == 1:
        return _diameter_of_vertices(vertices)
    if domain_dimension == 2:
        return _polygon_area(vertices)
    if domain_dimension == 3:
        return _polyhedron_volume(vertices)
    raise ValueError(f"unsupported domain dimension: {domain_dimension}")


def _unique_columns(vertices: Array) -> Array:
    _, indices = np.unique(vertices.T, axis=0, return_index=True)
    return vertices[:, np.sort(indices)]


def _diameter_of_vertices(vertices: Array) -> float:
    if vertices.shape[1] <= 1:
        return 0.0
    diffs = vertices[:, :, None] - vertices[:, None, :]
    return float(np.linalg.norm(diffs, axis=0).max())


def _polygon_area(vertices: Array) -> float:
    if vertices.shape[1] < 3:
        return 0.0
    if vertices.shape[0] == 2:
        ordered = _order_planar_vertices(vertices)
        x = ordered[0]
        y = ordered[1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)

    centered = vertices - vertices.mean(axis=1, keepdims=True)
    _, _, vh = np.linalg.svd(centered.T, full_matrices=False)
    basis = vh[:2, :]
    projected = basis @ centered
    return _polygon_area(projected)


def _order_planar_vertices(vertices: Array) -> Array:
    center = vertices.mean(axis=1, keepdims=True)
    angles = np.arctan2(vertices[1] - center[1, 0], vertices[0] - center[0, 0])
    return vertices[:, np.argsort(angles)]


def _polyhedron_volume(vertices: Array) -> float:
    if vertices.shape[1] < 4:
        return 0.0
    if vertices.shape[1] == 4:
        matrix = vertices[:, 1:4] - vertices[:, [0]]
        return float(abs(np.linalg.det(matrix)) / 6.0)

    # Exact for parallelepiped-style cells such as axis-aligned or affine hexes.
    mins = vertices.min(axis=1)
    maxs = vertices.max(axis=1)
    return float(np.prod(maxs - mins))
