from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mesh_metrics.geometry import MeshGeometry
from mesh_metrics.histogram import Histogram


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

    @classmethod
    def from_mesh(cls, mesh: MeshGeometry, *, bins: int = 30, include_histograms: bool = True) -> MeshStatistics:
        element_measures = mesh.element_measures()
        element_diameters = mesh.element_diameters()
        element_edge_aspect_ratios = mesh.element_edge_aspect_ratios()
        facet_measures = mesh.facet_measures()
        facet_diameters = mesh.facet_diameters()
        facet_edge_aspect_ratios = mesh.facet_edge_aspect_ratios()
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
        label_stats = {
            label: FacetLabelStats.from_values(label, indices, facet_measures, facet_diameters, bins=bins, include_histograms=include_histograms)
            for label, indices in mesh.facet_label_indices().items()
        }
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
