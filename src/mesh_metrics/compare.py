from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mesh_metrics.stats import MeshStatistics, QuantityStats


@dataclass(frozen=True)
class QuantityDelta:
    before: QuantityStats
    after: QuantityStats
    count_delta: int
    min_delta: float
    max_delta: float
    mean_delta: float
    median_delta: float
    std_delta: float
    total_delta: float
    count_ratio: float
    mean_ratio: float

    @classmethod
    def from_stats(cls, before: QuantityStats, after: QuantityStats) -> QuantityDelta:
        return cls(
            before=before,
            after=after,
            count_delta=after.count - before.count,
            min_delta=after.min - before.min,
            max_delta=after.max - before.max,
            mean_delta=after.mean - before.mean,
            median_delta=after.median - before.median,
            std_delta=after.std - before.std,
            total_delta=after.total - before.total,
            count_ratio=_ratio(after.count, before.count),
            mean_ratio=_ratio(after.mean, before.mean),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["before"] = self.before.to_dict()
        result["after"] = self.after.to_dict()
        return result


@dataclass(frozen=True)
class MeshComparison:
    before: MeshStatistics
    after: MeshStatistics
    element_measure: QuantityDelta
    element_diameter: QuantityDelta
    element_edge_aspect_ratio: QuantityDelta
    facet_measure: QuantityDelta
    facet_diameter: QuantityDelta
    facet_edge_aspect_ratio: QuantityDelta
    facet_labels: dict[str, dict[str, QuantityDelta]]

    @classmethod
    def from_stats(cls, before: MeshStatistics, after: MeshStatistics) -> MeshComparison:
        label_names = sorted(set(before.facet_labels) | set(after.facet_labels))
        label_deltas: dict[str, dict[str, QuantityDelta]] = {}
        for label in label_names:
            if label not in before.facet_labels or label not in after.facet_labels:
                continue
            before_label = before.facet_labels[label]
            after_label = after.facet_labels[label]
            label_deltas[label] = {
                "measure": QuantityDelta.from_stats(before_label.measure, after_label.measure),
                "diameter": QuantityDelta.from_stats(before_label.diameter, after_label.diameter),
            }

        return cls(
            before=before,
            after=after,
            element_measure=QuantityDelta.from_stats(before.element_measure, after.element_measure),
            element_diameter=QuantityDelta.from_stats(before.element_diameter, after.element_diameter),
            element_edge_aspect_ratio=QuantityDelta.from_stats(before.element_edge_aspect_ratio, after.element_edge_aspect_ratio),
            facet_measure=QuantityDelta.from_stats(before.facet_measure, after.facet_measure),
            facet_diameter=QuantityDelta.from_stats(before.facet_diameter, after.facet_diameter),
            facet_edge_aspect_ratio=QuantityDelta.from_stats(before.facet_edge_aspect_ratio, after.facet_edge_aspect_ratio),
            facet_labels=label_deltas,
        )

    def to_dict(self, *, include_mesh_stats: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mesh": {
                "before": self.before.to_dict(include_histograms=False)["mesh"],
                "after": self.after.to_dict(include_histograms=False)["mesh"],
                "npoints_delta": self.after.mesh.npoints - self.before.mesh.npoints,
                "nelements_delta": self.after.mesh.nelements - self.before.mesh.nelements,
                "nfacets_delta": self.after.mesh.nfacets - self.before.mesh.nfacets,
                "nelements_ratio": _ratio(self.after.mesh.nelements, self.before.mesh.nelements),
                "nfacets_ratio": _ratio(self.after.mesh.nfacets, self.before.mesh.nfacets),
            },
            "mesh_quality": {
                "element_measure": self.element_measure.to_dict(),
                "element_diameter": self.element_diameter.to_dict(),
                "element_edge_aspect_ratio": self.element_edge_aspect_ratio.to_dict(),
            },
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
            "facet_labels": {
                label: {
                    name: delta.to_dict()
                    for name, delta in deltas.items()
                }
                for label, deltas in self.facet_labels.items()
            },
        }
        if include_mesh_stats:
            result["before_stats"] = self.before.to_dict(include_histograms=False)
            result["after_stats"] = self.after.to_dict(include_histograms=False)
        return result


def _ratio(after: float, before: float) -> float:
    if before == 0:
        return float("nan")
    return float(after / before)
