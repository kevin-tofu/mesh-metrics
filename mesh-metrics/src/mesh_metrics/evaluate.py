from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mesh_metrics.compare import MeshComparison
from mesh_metrics.stats import MeshStatistics, QuantityStats


@dataclass(frozen=True)
class ElementCountEvaluation:
    target: int
    actual: int
    error: int
    ratio: float
    error_percent: float

    @classmethod
    def from_stats(cls, after: MeshStatistics, target: int) -> ElementCountEvaluation:
        if target <= 0:
            raise ValueError("target element count must be positive")
        actual = after.mesh.nelements
        error = actual - target
        return cls(
            target=int(target),
            actual=int(actual),
            error=int(error),
            ratio=float(actual / target),
            error_percent=float(100.0 * error / target),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FacetSizeEvaluation:
    label: str
    target_size: float
    count: int
    mean_ratio: float
    median_ratio: float
    p95_ratio: float
    too_small_count: int
    too_large_count: int
    too_small_fraction: float
    too_large_fraction: float
    diameter: QuantityStats

    @classmethod
    def from_diameters(
        cls,
        label: str,
        target_size: float,
        diameters: NDArray[np.float64],
        *,
        tolerance: float,
    ) -> FacetSizeEvaluation:
        if target_size <= 0.0 or not np.isfinite(target_size):
            raise ValueError(f"target size for {label!r} must be positive and finite")
        finite = np.asarray(diameters, dtype=float)
        finite = finite[np.isfinite(finite)]
        lower = target_size * (1.0 - tolerance)
        upper = target_size * (1.0 + tolerance)
        count = int(finite.size)
        too_small = int(np.count_nonzero(finite < lower))
        too_large = int(np.count_nonzero(finite > upper))
        stats = QuantityStats.from_values(f"facet_diameter:{label}", finite)
        return cls(
            label=label,
            target_size=float(target_size),
            count=count,
            mean_ratio=_ratio(stats.mean, target_size),
            median_ratio=_ratio(stats.median, target_size),
            p95_ratio=_ratio(stats.p95, target_size),
            too_small_count=too_small,
            too_large_count=too_large,
            too_small_fraction=_ratio(too_small, count),
            too_large_fraction=_ratio(too_large, count),
            diameter=stats,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["diameter"] = self.diameter.to_dict()
        return result


@dataclass(frozen=True)
class QualityEvaluation:
    before_p95: float
    after_p95: float
    before_max: float
    after_max: float
    p95_ratio: float
    max_ratio: float

    @classmethod
    def from_stats(cls, before: MeshStatistics, after: MeshStatistics) -> QualityEvaluation:
        before_ar = before.element_edge_aspect_ratio
        after_ar = after.element_edge_aspect_ratio
        return cls(
            before_p95=before_ar.p95,
            after_p95=after_ar.p95,
            before_max=before_ar.max,
            after_max=after_ar.max,
            p95_ratio=_ratio(after_ar.p95, before_ar.p95),
            max_ratio=_ratio(after_ar.max, before_ar.max),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RemeshEvaluation:
    before: MeshStatistics
    after: MeshStatistics
    comparison: MeshComparison
    target_elements: ElementCountEvaluation | None
    facet_sizes: dict[str, FacetSizeEvaluation]
    quality: QualityEvaluation
    tolerance: float
    unmatched_facet_size_labels: list[str]

    @classmethod
    def from_stats(
        cls,
        before: MeshStatistics,
        after: MeshStatistics,
        *,
        target_elements: int | None = None,
        facet_size_map: dict[str, float] | None = None,
        tolerance: float = 0.2,
    ) -> RemeshEvaluation:
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        size_map = {} if facet_size_map is None else {str(label): float(size) for label, size in facet_size_map.items()}
        facet_evaluations: dict[str, FacetSizeEvaluation] = {}
        after_facet_diameters = after.mesh.facet_diameters()
        for label, target_size in size_map.items():
            indices = after.mesh.facet_label_indices().get(label)
            if indices is None:
                continue
            facet_evaluations[label] = FacetSizeEvaluation.from_diameters(
                label,
                target_size,
                after_facet_diameters[indices],
                tolerance=tolerance,
            )

        return cls(
            before=before,
            after=after,
            comparison=MeshComparison.from_stats(before, after),
            target_elements=None if target_elements is None else ElementCountEvaluation.from_stats(after, target_elements),
            facet_sizes=facet_evaluations,
            quality=QualityEvaluation.from_stats(before, after),
            tolerance=float(tolerance),
            unmatched_facet_size_labels=sorted(set(size_map) - set(after.mesh.facet_label_indices())),
        )

    def to_dict(self, *, include_mesh_stats: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "target_elements": None if self.target_elements is None else self.target_elements.to_dict(),
            "facet_size_targets": {
                label: evaluation.to_dict()
                for label, evaluation in self.facet_sizes.items()
            },
            "mesh_quality": self.quality.to_dict(),
            "comparison": self.comparison.to_dict(include_mesh_stats=False),
            "tolerance": self.tolerance,
            "unmatched_facet_size_labels": self.unmatched_facet_size_labels,
        }
        if include_mesh_stats:
            result["before_stats"] = self.before.to_dict(include_histograms=False)
            result["after_stats"] = self.after.to_dict(include_histograms=False)
        return result

    def diagnose_constraints(
        self,
        *,
        max_relaxation: float = 3.0,
        min_relaxation: float = 0.25,
    ) -> ConstraintDiagnosis:
        return ConstraintDiagnosis.from_evaluation(
            self,
            max_relaxation=max_relaxation,
            min_relaxation=min_relaxation,
        )


@dataclass(frozen=True)
class FacetRelaxationSuggestion:
    label: str
    current_size: float
    suggested_size: float
    relaxation_factor: float
    mean_ratio: float
    p95_ratio: float
    too_large_fraction: float
    priority: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConstraintDiagnosis:
    element_ratio: float | None
    global_size_scale: float | None
    suggested_facet_sizes: dict[str, float]
    facet_suggestions: dict[str, FacetRelaxationSuggestion]
    notes: list[str]

    @classmethod
    def from_evaluation(
        cls,
        evaluation: RemeshEvaluation,
        *,
        max_relaxation: float = 3.0,
        min_relaxation: float = 0.25,
    ) -> ConstraintDiagnosis:
        element_ratio = None if evaluation.target_elements is None else evaluation.target_elements.ratio
        global_scale = None
        notes: list[str] = []
        if element_ratio is not None and np.isfinite(element_ratio) and element_ratio > 0.0:
            global_scale = float(element_ratio ** (1.0 / max(evaluation.after.mesh.dimension, 1)))
            if element_ratio > 1.2:
                notes.append("actual element count is above target; relax small facet sizes or raise target-elements")
            elif element_ratio < 0.8:
                notes.append("actual element count is below target; tighten sizes or lower target-elements")

        suggestions: dict[str, FacetRelaxationSuggestion] = {}
        suggested_sizes: dict[str, float] = {}
        for label, facet_eval in evaluation.facet_sizes.items():
            factor = _relaxation_factor(
                facet_eval,
                global_scale=global_scale,
                min_relaxation=min_relaxation,
                max_relaxation=max_relaxation,
            )
            suggested = float(facet_eval.target_size * factor)
            priority = _relaxation_priority(facet_eval, factor)
            suggestions[label] = FacetRelaxationSuggestion(
                label=label,
                current_size=facet_eval.target_size,
                suggested_size=suggested,
                relaxation_factor=factor,
                mean_ratio=facet_eval.mean_ratio,
                p95_ratio=facet_eval.p95_ratio,
                too_large_fraction=facet_eval.too_large_fraction,
                priority=priority,
            )
            suggested_sizes[label] = suggested

        if evaluation.unmatched_facet_size_labels:
            notes.append("some facet size labels were not found after transfer")
        return cls(
            element_ratio=element_ratio,
            global_size_scale=global_scale,
            suggested_facet_sizes=suggested_sizes,
            facet_suggestions=suggestions,
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_ratio": self.element_ratio,
            "global_size_scale": self.global_size_scale,
            "suggested_facet_sizes": self.suggested_facet_sizes,
            "facet_suggestions": {
                label: suggestion.to_dict()
                for label, suggestion in self.facet_suggestions.items()
            },
            "notes": self.notes,
        }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _relaxation_factor(
    facet_eval: FacetSizeEvaluation,
    *,
    global_scale: float | None,
    min_relaxation: float,
    max_relaxation: float,
) -> float:
    if global_scale is None or not np.isfinite(global_scale):
        return 1.0
    factor = global_scale
    if facet_eval.mean_ratio > 1.2:
        factor *= 0.85
    elif facet_eval.mean_ratio < 0.8:
        factor *= 1.10
    if facet_eval.too_large_fraction > 0.5:
        factor *= 0.9
    return float(np.clip(factor, min_relaxation, max_relaxation))


def _relaxation_priority(facet_eval: FacetSizeEvaluation, factor: float) -> str:
    if 0.95 <= factor <= 1.05:
        return "keep"
    if factor < 0.95:
        return "tighten"
    if facet_eval.too_large_fraction > 0.5 or facet_eval.p95_ratio > 1.5:
        return "relax-carefully"
    return "relax"
