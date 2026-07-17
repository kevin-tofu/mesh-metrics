from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mesh_metrics.geometry import MeshGeometry
from mesh_metrics.histogram import Histogram


@dataclass(frozen=True)
class SamplingConfig:
    """Random sampling settings for large mesh statistics.

    Exact statistics remain the default. When this config is passed to
    ``MeshStatistics.from_mesh``, quantity statistics and histograms are
    computed from sampled element/facet subsets.
    """

    max_elements: int | None = None
    max_facets: int | None = None
    max_facets_per_label: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_elements", "max_facets", "max_facets_per_label"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SamplingSummary:
    exact: bool
    seed: int | None
    element_count: int
    sampled_elements: int
    facet_count: int
    sampled_facets: int
    max_elements: int | None
    max_facets: int | None
    max_facets_per_label: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuantityStats:
    name: str
    count: int
    min: float
    max: float
    mean: float
    median: float
    std: float
    p05: float
    p25: float
    p75: float
    p95: float
    total: float

    @classmethod
    def from_values(cls, name: str, values: NDArray[np.float64]) -> QuantityStats:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            nan = float("nan")
            return cls(name=name, count=0, min=nan, max=nan, mean=nan, median=nan, std=nan, p05=nan, p25=nan, p75=nan, p95=nan, total=0.0)

        return cls(
            name=name,
            count=int(finite.size),
            min=float(np.min(finite)),
            max=float(np.max(finite)),
            mean=float(np.mean(finite)),
            median=float(np.median(finite)),
            std=float(np.std(finite)),
            p05=float(np.percentile(finite, 5)),
            p25=float(np.percentile(finite, 25)),
            p75=float(np.percentile(finite, 75)),
            p95=float(np.percentile(finite, 95)),
            total=float(np.sum(finite)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FacetLabelStats:
    label: str
    indices: NDArray[np.int64]
    measure: QuantityStats
    diameter: QuantityStats
    histograms: dict[str, Histogram]

    @classmethod
    def from_values(
        cls,
        label: str,
        indices: NDArray[np.int64],
        facet_measures: NDArray[np.float64],
        facet_diameters: NDArray[np.float64],
        *,
        bins: int = 30,
        include_histograms: bool = True,
    ) -> FacetLabelStats:
        selected = np.asarray(indices, dtype=np.int64)
        measures = facet_measures[selected]
        diameters = facet_diameters[selected]
        measure_name = f"facet_measure:{label}"
        diameter_name = f"facet_diameter:{label}"
        return cls(
            label=label,
            indices=selected,
            measure=QuantityStats.from_values(measure_name, measures),
            diameter=QuantityStats.from_values(diameter_name, diameters),
            histograms={
                "facet_measure": Histogram.from_values(measure_name, measures, bins=bins),
                "facet_diameter": Histogram.from_values(diameter_name, diameters, bins=bins),
            } if include_histograms else {},
        )

    @classmethod
    def from_sampled_values(
        cls,
        label: str,
        indices: NDArray[np.int64],
        measures: NDArray[np.float64],
        diameters: NDArray[np.float64],
        *,
        bins: int = 30,
        include_histograms: bool = True,
    ) -> FacetLabelStats:
        selected = np.asarray(indices, dtype=np.int64)
        measure_name = f"facet_measure:{label}"
        diameter_name = f"facet_diameter:{label}"
        return cls(
            label=label,
            indices=selected,
            measure=QuantityStats.from_values(measure_name, measures),
            diameter=QuantityStats.from_values(diameter_name, diameters),
            histograms={
                "facet_measure": Histogram.from_values(measure_name, measures, bins=bins),
                "facet_diameter": Histogram.from_values(diameter_name, diameters, bins=bins),
            } if include_histograms else {},
        )

    def to_dict(self, *, include_histograms: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "label": self.label,
            "indices": self.indices.tolist(),
            "measure": self.measure.to_dict(),
            "diameter": self.diameter.to_dict(),
        }
        if include_histograms:
            result["histograms"] = {name: histogram.to_dict() for name, histogram in self.histograms.items()}
        return result


@dataclass(frozen=True)
class ElementStatistics:
    measure: QuantityStats
    diameter: QuantityStats
    edge_aspect_ratio: QuantityStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "measure": self.measure.to_dict(),
            "diameter": self.diameter.to_dict(),
            "edge_aspect_ratio": self.edge_aspect_ratio.to_dict(),
        }


@dataclass(frozen=True)
class FacetStatistics:
    measure: QuantityStats
    diameter: QuantityStats
    edge_aspect_ratio: QuantityStats
    labels: dict[str, FacetLabelStats]

    def to_dict(self, *, include_histograms: bool = True) -> dict[str, Any]:
        return {
            "measure": self.measure.to_dict(),
            "diameter": self.diameter.to_dict(),
            "edge_aspect_ratio": self.edge_aspect_ratio.to_dict(),
            "labels": {
                label: stats.to_dict(include_histograms=include_histograms)
                for label, stats in self.labels.items()
            },
        }


@dataclass(frozen=True)
class MeshStatistics:
    mesh: MeshGeometry
    elements: ElementStatistics
    facets: FacetStatistics
    element_measure: QuantityStats
    element_diameter: QuantityStats
    element_edge_aspect_ratio: QuantityStats
    facet_measure: QuantityStats
    facet_diameter: QuantityStats
    facet_edge_aspect_ratio: QuantityStats
    histograms: dict[str, Histogram]
    facet_labels: dict[str, FacetLabelStats]
    sampling: SamplingSummary | None = None

    @classmethod
    def from_mesh(
        cls,
        mesh: MeshGeometry,
        *,
        bins: int = 30,
        include_histograms: bool = True,
        sampling: SamplingConfig | None = None,
    ) -> MeshStatistics:
        if sampling is None:
            element_indices = np.arange(mesh.nelements, dtype=np.int64)
            facet_indices = np.arange(mesh.nfacets, dtype=np.int64)
        else:
            rng = np.random.default_rng(sampling.seed)
            element_indices = _sample_indices(mesh.nelements, sampling.max_elements, rng)
            facet_indices = _sample_indices(mesh.nfacets, sampling.max_facets, rng)

        element_measures = _connectivity_measures_for_indices(mesh.points, mesh.elements, element_indices, domain_dimension=mesh.dimension)
        element_diameters = _diameters_for_indices(mesh.points, mesh.elements, element_indices)
        element_edge_aspect_ratios = _edge_aspect_ratios_for_indices(mesh.points, mesh.elements, element_indices)
        facet_measures = _connectivity_measures_for_indices(mesh.points, mesh.facets, facet_indices, domain_dimension=max(mesh.dimension - 1, 0))
        facet_diameters = _diameters_for_indices(mesh.points, mesh.facets, facet_indices)
        facet_edge_aspect_ratios = _edge_aspect_ratios_for_indices(mesh.points, mesh.facets, facet_indices)
        histogram_values = {
            "element_measure": element_measures,
            "element_diameter": element_diameters,
            "element_edge_aspect_ratio": element_edge_aspect_ratios,
            "facet_measure": facet_measures,
            "facet_diameter": facet_diameters,
            "facet_edge_aspect_ratio": facet_edge_aspect_ratios,
        }

        element_measure_stats = QuantityStats.from_values("element_measure", element_measures)
        element_diameter_stats = QuantityStats.from_values("element_diameter", element_diameters)
        element_aspect_stats = QuantityStats.from_values("element_edge_aspect_ratio", element_edge_aspect_ratios)
        facet_measure_stats = QuantityStats.from_values("facet_measure", facet_measures)
        facet_diameter_stats = QuantityStats.from_values("facet_diameter", facet_diameters)
        facet_aspect_stats = QuantityStats.from_values("facet_edge_aspect_ratio", facet_edge_aspect_ratios)
        label_stats = _facet_label_stats(
            mesh,
            sampling=sampling,
            bins=bins,
            include_histograms=include_histograms,
        )
        elements = ElementStatistics(
            measure=element_measure_stats,
            diameter=element_diameter_stats,
            edge_aspect_ratio=element_aspect_stats,
        )
        facets = FacetStatistics(
            measure=facet_measure_stats,
            diameter=facet_diameter_stats,
            edge_aspect_ratio=facet_aspect_stats,
            labels=label_stats,
        )

        sampling_summary = None if sampling is None else SamplingSummary(
            exact=False,
            seed=sampling.seed,
            element_count=mesh.nelements,
            sampled_elements=int(element_indices.size),
            facet_count=mesh.nfacets,
            sampled_facets=int(facet_indices.size),
            max_elements=sampling.max_elements,
            max_facets=sampling.max_facets,
            max_facets_per_label=sampling.max_facets_per_label,
        )

        return cls(
            mesh=mesh,
            elements=elements,
            facets=facets,
            element_measure=element_measure_stats,
            element_diameter=element_diameter_stats,
            element_edge_aspect_ratio=element_aspect_stats,
            facet_measure=facet_measure_stats,
            facet_diameter=facet_diameter_stats,
            facet_edge_aspect_ratio=facet_aspect_stats,
            histograms={name: Histogram.from_values(name, values, bins=bins) for name, values in histogram_values.items()} if include_histograms else {},
            facet_labels=label_stats,
            sampling=sampling_summary,
        )

    def to_dict(self, *, include_histograms: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mesh": {
                "source": self.mesh.source,
                "dimension": self.mesh.dimension,
                "npoints": self.mesh.npoints,
                "nelements": self.mesh.nelements,
                "nfacets": self.mesh.nfacets,
                "backend": self.mesh.backend,
                "facet_labels": self.mesh.facet_labels.to_dict() if self.mesh.facet_labels is not None else {},
            },
            "mesh_quality": {
                "element_measure": self.element_measure.to_dict(),
                "element_diameter": self.element_diameter.to_dict(),
                "element_edge_aspect_ratio": self.element_edge_aspect_ratio.to_dict(),
            },
            "elements": self.elements.to_dict(),
            "facets": self.facets.to_dict(include_histograms=include_histograms),
            "facet_size": {
                "facet_measure": self.facet_measure.to_dict(),
                "facet_diameter": self.facet_diameter.to_dict(),
                "facet_edge_aspect_ratio": self.facet_edge_aspect_ratio.to_dict(),
            },
            "element_measure": self.element_measure.to_dict(),
            "element_diameter": self.element_diameter.to_dict(),
            "element_edge_aspect_ratio": self.element_edge_aspect_ratio.to_dict(),
            "facet_measure": self.facet_measure.to_dict(),
            "facet_diameter": self.facet_diameter.to_dict(),
            "facet_edge_aspect_ratio": self.facet_edge_aspect_ratio.to_dict(),
            "facet_label_stats": {
                label: stats.to_dict(include_histograms=include_histograms)
                for label, stats in self.facet_labels.items()
            },
        }
        if include_histograms:
            result["histograms"] = {name: histogram.to_dict() for name, histogram in self.histograms.items()}
        if self.sampling is not None:
            result["sampling"] = self.sampling.to_dict()
        return result

    def save_histograms(self, directory: str, *, dpi: int = 150) -> list[str]:
        paths: list[str] = []
        for name, histogram in self.histograms.items():
            paths.append(str(histogram.savefig(f"{directory}/{name}.png", dpi=dpi)))
        for label, stats in self.facet_labels.items():
            safe_label = _safe_filename(label)
            for name, histogram in stats.histograms.items():
                paths.append(str(histogram.savefig(f"{directory}/facet_labels/{safe_label}_{name}.png", dpi=dpi)))
        return paths


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def _facet_label_stats(
    mesh: MeshGeometry,
    *,
    sampling: SamplingConfig | None,
    bins: int,
    include_histograms: bool,
) -> dict[str, FacetLabelStats]:
    rng = np.random.default_rng(None if sampling is None else sampling.seed)
    label_stats = {}
    for label, indices in mesh.facet_label_indices().items():
        selected = np.asarray(indices, dtype=np.int64)
        if sampling is not None:
            selected = _sample_from_array(selected, sampling.max_facets_per_label, rng)
        measures = _connectivity_measures_for_indices(mesh.points, mesh.facets, selected, domain_dimension=max(mesh.dimension - 1, 0))
        diameters = _diameters_for_indices(mesh.points, mesh.facets, selected)
        label_stats[label] = FacetLabelStats.from_sampled_values(
            label,
            selected,
            measures,
            diameters,
            bins=bins,
            include_histograms=include_histograms,
        )
    return label_stats


def _sample_indices(count: int, limit: int | None, rng: np.random.Generator) -> NDArray[np.int64]:
    if count <= 0:
        return np.asarray([], dtype=np.int64)
    if limit is None or count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.sort(rng.choice(count, size=limit, replace=False).astype(np.int64))


def _sample_from_array(values: NDArray[np.int64], limit: int | None, rng: np.random.Generator) -> NDArray[np.int64]:
    array = np.asarray(values, dtype=np.int64)
    if limit is None or array.size <= limit:
        return array
    selected = rng.choice(array.size, size=limit, replace=False)
    return np.sort(array[selected].astype(np.int64))


def _connectivity_measures_for_indices(
    points: NDArray[np.float64],
    connectivity: NDArray[np.int64] | None,
    indices: NDArray[np.int64],
    *,
    domain_dimension: int,
) -> NDArray[np.float64]:
    if connectivity is None or indices.size == 0:
        return np.asarray([], dtype=float)
    return np.asarray(
        [_cell_measure(points[:, connectivity[:, index]], domain_dimension=domain_dimension) for index in indices],
        dtype=float,
    )


def _diameters_for_indices(
    points: NDArray[np.float64],
    connectivity: NDArray[np.int64] | None,
    indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    if connectivity is None or indices.size == 0:
        return np.asarray([], dtype=float)
    values = np.empty(indices.size, dtype=float)
    for output_index, item_index in enumerate(indices):
        vertices = points[:, connectivity[:, item_index]]
        diffs = vertices[:, :, None] - vertices[:, None, :]
        values[output_index] = float(np.linalg.norm(diffs, axis=0).max())
    return values


def _edge_aspect_ratios_for_indices(
    points: NDArray[np.float64],
    connectivity: NDArray[np.int64] | None,
    indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    if connectivity is None or indices.size == 0:
        return np.asarray([], dtype=float)
    values = np.empty(indices.size, dtype=float)
    for output_index, item_index in enumerate(indices):
        vertices = _unique_columns(points[:, connectivity[:, item_index]])
        edge_lengths = _pairwise_distances(vertices)
        positive = edge_lengths[edge_lengths > 0.0]
        values[output_index] = float("nan") if positive.size == 0 else float(positive.max() / positive.min())
    return values


def _pairwise_distances(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    diffs = vertices[:, :, None] - vertices[:, None, :]
    distances = np.linalg.norm(diffs, axis=0)
    return distances[np.triu_indices(vertices.shape[1], k=1)]


def _cell_measure(vertices: NDArray[np.float64], *, domain_dimension: int) -> float:
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


def _unique_columns(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    _, indices = np.unique(vertices.T, axis=0, return_index=True)
    return vertices[:, np.sort(indices)]


def _diameter_of_vertices(vertices: NDArray[np.float64]) -> float:
    if vertices.shape[1] <= 1:
        return 0.0
    diffs = vertices[:, :, None] - vertices[:, None, :]
    return float(np.linalg.norm(diffs, axis=0).max())


def _polygon_area(vertices: NDArray[np.float64]) -> float:
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


def _order_planar_vertices(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    center = vertices.mean(axis=1, keepdims=True)
    angles = np.arctan2(vertices[1] - center[1, 0], vertices[0] - center[0, 0])
    return vertices[:, np.argsort(angles)]


def _polyhedron_volume(vertices: NDArray[np.float64]) -> float:
    if vertices.shape[1] < 4:
        return 0.0
    if vertices.shape[1] == 4:
        matrix = vertices[:, 1:4] - vertices[:, [0]]
        return float(abs(np.linalg.det(matrix)) / 6.0)
    center = vertices.mean(axis=1)
    volume = 0.0
    for i in range(1, vertices.shape[1] - 1):
        matrix = np.column_stack((vertices[:, 0] - center, vertices[:, i] - center, vertices[:, i + 1] - center))
        volume += abs(np.linalg.det(matrix)) / 6.0
    return float(volume)
