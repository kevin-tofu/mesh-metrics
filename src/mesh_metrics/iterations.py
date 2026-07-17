from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class IterationMetrics:
    iteration: int
    source: str
    target_elements: int | None
    actual_elements: int | None
    element_ratio: float | None
    element_error_percent: float | None
    aspect_p95: float | None
    aspect_max: float | None
    aspect_p05: float | None
    aspect_p25: float | None
    aspect_median: float | None
    aspect_p75: float | None
    aspect_p95_ratio: float | None
    aspect_max_ratio: float | None
    facet_mean_ratio: float | None
    facet_p95_ratio: float | None
    facet_mean_ratio_min: float | None
    facet_mean_ratio_p25: float | None
    facet_mean_ratio_median: float | None
    facet_mean_ratio_p75: float | None
    facet_mean_ratio_max: float | None
    facet_too_small_fraction: float | None
    facet_too_large_fraction: float | None

    @classmethod
    def from_evaluation_payload(cls, payload: dict[str, Any], *, iteration: int, source: str) -> IterationMetrics:
        target = payload.get("target_elements") or {}
        quality = payload.get("mesh_quality") or {}
        aspect_stats = _comparison_after_stats(payload, "mesh_quality", "element_edge_aspect_ratio")
        facet_targets = payload.get("facet_size_targets") or {}
        facets = list(facet_targets.values()) if isinstance(facet_targets, dict) else []
        facet_mean_ratios = _metric_values(facets, "mean_ratio")
        return cls(
            iteration=int(iteration),
            source=source,
            target_elements=_optional_int(target.get("target")),
            actual_elements=_optional_int(target.get("actual")),
            element_ratio=_optional_float(target.get("ratio")),
            element_error_percent=_optional_float(target.get("error_percent")),
            aspect_p95=_optional_float(quality.get("after_p95")),
            aspect_max=_optional_float(quality.get("after_max")),
            aspect_p05=_optional_float(aspect_stats.get("p05")),
            aspect_p25=_optional_float(aspect_stats.get("p25")),
            aspect_median=_optional_float(aspect_stats.get("median")),
            aspect_p75=_optional_float(aspect_stats.get("p75")),
            aspect_p95_ratio=_optional_float(quality.get("p95_ratio")),
            aspect_max_ratio=_optional_float(quality.get("max_ratio")),
            facet_mean_ratio=_mean_metric(facets, "mean_ratio"),
            facet_p95_ratio=_mean_metric(facets, "p95_ratio"),
            facet_mean_ratio_min=_percentile_or_none(facet_mean_ratios, 0),
            facet_mean_ratio_p25=_percentile_or_none(facet_mean_ratios, 25),
            facet_mean_ratio_median=_percentile_or_none(facet_mean_ratios, 50),
            facet_mean_ratio_p75=_percentile_or_none(facet_mean_ratios, 75),
            facet_mean_ratio_max=_percentile_or_none(facet_mean_ratios, 100),
            facet_too_small_fraction=_mean_metric(facets, "too_small_fraction"),
            facet_too_large_fraction=_mean_metric(facets, "too_large_fraction"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IterationSeries:
    metrics: list[IterationMetrics]

    @classmethod
    def load(cls, paths: Iterable[str | Path]) -> IterationSeries:
        metrics: list[IterationMetrics] = []
        for iteration, raw_path in enumerate(paths):
            path = Path(raw_path)
            evaluation_path = path / "evaluation.json" if path.is_dir() else path
            payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
            metrics.append(IterationMetrics.from_evaluation_payload(payload, iteration=iteration, source=str(evaluation_path)))
        return cls(metrics=metrics)

    def to_dict(self) -> dict[str, Any]:
        return {"iterations": [metric.to_dict() for metric in self.metrics]}

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
        return output

    def write_csv(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(IterationMetrics.__dataclass_fields__)
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for metric in self.metrics:
                writer.writerow(metric.to_dict())
        return output

    def save_plot(self, path: str | Path, *, dpi: int = 150) -> Path:
        from mesh_metrics._plotting import configure_matplotlib_cache

        configure_matplotlib_cache()
        import matplotlib.pyplot as plt

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        iterations = np.asarray([metric.iteration for metric in self.metrics], dtype=float)

        fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=True, constrained_layout=True)
        _plot_series(axes[0], iterations, self.metrics, "actual_elements", label="actual elements", marker="o")
        _plot_series(axes[0], iterations, self.metrics, "target_elements", label="target elements", marker="x", linestyle="--")
        axes[0].set_ylabel("elements")
        axes[0].set_title("Element Count")
        axes[0].legend()

        _plot_series(axes[1], iterations, self.metrics, "aspect_p95", label="aspect p95", marker="o")
        _plot_series(axes[1], iterations, self.metrics, "aspect_max", label="aspect max", marker="x")
        _plot_band(axes[1], iterations, self.metrics, "aspect_p05", "aspect_p95", label="aspect p05-p95", alpha=0.12)
        _plot_band(axes[1], iterations, self.metrics, "aspect_p25", "aspect_p75", label="aspect p25-p75", alpha=0.20)
        _plot_series(axes[1], iterations, self.metrics, "aspect_median", label="aspect median", marker=".", linestyle=":")
        axes[1].set_ylabel("edge aspect ratio")
        axes[1].set_title("Mesh Quality")
        axes[1].legend()

        _plot_series(axes[2], iterations, self.metrics, "facet_mean_ratio", label="facet mean/target", marker="o")
        _plot_series(axes[2], iterations, self.metrics, "facet_p95_ratio", label="facet p95/target", marker="x")
        _plot_band(axes[2], iterations, self.metrics, "facet_mean_ratio_min", "facet_mean_ratio_max", label="label mean min-max", alpha=0.12)
        _plot_band(axes[2], iterations, self.metrics, "facet_mean_ratio_p25", "facet_mean_ratio_p75", label="label mean p25-p75", alpha=0.20)
        _plot_series(axes[2], iterations, self.metrics, "facet_mean_ratio_median", label="label mean median", marker=".", linestyle=":")
        axes[2].axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
        axes[2].set_ylabel("ratio")
        axes[2].set_xlabel("iteration")
        axes[2].set_title("Facet Size Target Fit")
        axes[2].legend()

        for axis in axes:
            axis.grid(alpha=0.25)
        fig.savefig(output, dpi=dpi)
        plt.close(fig)
        return output


def _plot_series(axis, iterations: np.ndarray, metrics: list[IterationMetrics], field: str, **kwargs: Any) -> None:
    values = np.asarray([_nan_if_none(getattr(metric, field)) for metric in metrics], dtype=float)
    if np.all(np.isnan(values)):
        return
    axis.plot(iterations, values, **kwargs)


def _plot_band(axis, iterations: np.ndarray, metrics: list[IterationMetrics], lower_field: str, upper_field: str, **kwargs: Any) -> None:
    lower = np.asarray([_nan_if_none(getattr(metric, lower_field)) for metric in metrics], dtype=float)
    upper = np.asarray([_nan_if_none(getattr(metric, upper_field)) for metric in metrics], dtype=float)
    if np.all(np.isnan(lower)) or np.all(np.isnan(upper)):
        return
    axis.fill_between(iterations, lower, upper, **kwargs)


def _mean_metric(items: list[Any], key: str) -> float | None:
    values = _metric_values(items, key)
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _metric_values(items: list[Any], key: str) -> list[float | None]:
    return [_optional_float(item.get(key)) for item in items if isinstance(item, dict)]


def _percentile_or_none(values: list[float | None], percentile: float) -> float | None:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return None
    return float(np.percentile(finite, percentile))


def _comparison_after_stats(payload: dict[str, Any], group: str, quantity: str) -> dict[str, Any]:
    comparison = payload.get("comparison") or {}
    grouped = comparison.get(group) or {}
    quantity_payload = grouped.get(quantity) or comparison.get(quantity) or {}
    after = quantity_payload.get("after") if isinstance(quantity_payload, dict) else None
    return after if isinstance(after, dict) else {}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _nan_if_none(value: float | int | None) -> float:
    return float("nan") if value is None else float(value)
