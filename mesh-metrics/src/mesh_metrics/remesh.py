from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mesh_metrics.geometry import MeshGeometry


@dataclass(frozen=True)
class SizeWindow:
    hmin: float
    htarget: float
    hmax: float
    lower_percentile: float
    target_percentile: float
    upper_percentile: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutlierIndices:
    too_small: NDArray[np.int64]
    too_large: NDArray[np.int64]

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "too_small": self.too_small.tolist(),
            "too_large": self.too_large.tolist(),
        }


@dataclass(frozen=True)
class LabelSizeTarget:
    label: str
    count: int
    htarget: float
    hmin: float
    hmax: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ElementCountTarget:
    current_elements: int
    target_elements: int
    dimension: int
    base_hsiz: float
    recommended_hsiz: float
    size_scale: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Mmg3dRecommendation:
    """Quantile-based sizing hints for MMG3D-style remeshing."""

    element_size: SizeWindow
    facet_size: SizeWindow
    element_outliers: OutlierIndices
    facet_outliers: OutlierIndices
    facet_label_targets: dict[str, LabelSizeTarget]
    suggested_args: dict[str, float]
    element_count_target: ElementCountTarget | None = None

    @classmethod
    def from_mesh(
        cls,
        mesh: MeshGeometry,
        *,
        lower_percentile: float = 5.0,
        target_percentile: float = 50.0,
        upper_percentile: float = 95.0,
        target_elements: int | None = None,
    ) -> Mmg3dRecommendation:
        element_sizes = mesh.element_diameters()
        facet_sizes = mesh.facet_diameters()
        element_window = _size_window(
            element_sizes,
            lower_percentile=lower_percentile,
            target_percentile=target_percentile,
            upper_percentile=upper_percentile,
        )
        facet_window = _size_window(
            facet_sizes,
            lower_percentile=lower_percentile,
            target_percentile=target_percentile,
            upper_percentile=upper_percentile,
        )
        hmin = _finite_min(element_window.hmin, facet_window.hmin)
        htarget = _finite_min(element_window.htarget, facet_window.htarget)
        hmax = _finite_max(element_window.hmax, facet_window.hmax)
        count_target = _element_count_target(mesh, htarget, target_elements)
        if count_target is not None:
            htarget = count_target.recommended_hsiz

        return cls(
            element_size=element_window,
            facet_size=facet_window,
            element_outliers=_outlier_indices(element_sizes, element_window),
            facet_outliers=_outlier_indices(facet_sizes, facet_window),
            facet_label_targets=_facet_label_targets(mesh, facet_sizes, lower_percentile, target_percentile, upper_percentile),
            suggested_args={
                "hmin": hmin,
                "hsiz": htarget,
                "hmax": hmax,
            },
            element_count_target=count_target,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_size": self.element_size.to_dict(),
            "facet_size": self.facet_size.to_dict(),
            "element_outliers": self.element_outliers.to_dict(),
            "facet_outliers": self.facet_outliers.to_dict(),
            "facet_label_targets": {
                label: target.to_dict()
                for label, target in self.facet_label_targets.items()
            },
            "suggested_args": self.suggested_args,
            "element_count_target": None if self.element_count_target is None else self.element_count_target.to_dict(),
            "mmg3d_hint": _format_mmg3d_hint(self.suggested_args),
        }


@dataclass(frozen=True)
class Mmg3dRun:
    command: list[str]
    input_path: str
    output_path: str
    dry_run: bool
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""

    @property
    def succeeded(self) -> bool:
        return self.dry_run or self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "dry_run": self.dry_run,
            "returncode": self.returncode,
            "succeeded": self.succeeded,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def run_mmg3d(
    input_path: str | Path,
    output_path: str | Path,
    *,
    mmg3d_bin: str = "mmg3d",
    hmin: float | None = None,
    hsiz: float | None = None,
    hmax: float | None = None,
    sol_path: str | Path | None = None,
    extra_args: list[str] | None = None,
    dry_run: bool = False,
) -> Mmg3dRun:
    input_mesh = Path(input_path)
    output_mesh = Path(output_path)
    command = [mmg3d_bin, str(input_mesh), str(output_mesh)]
    size_args = (("hmin", hmin), ("hmax", hmax)) if sol_path is not None else (("hmin", hmin), ("hsiz", hsiz), ("hmax", hmax))
    for key, value in size_args:
        if value is not None and np.isfinite(value):
            command.extend([f"-{key}", f"{value:.17g}"])
    if sol_path is not None:
        command.extend(["-sol", str(sol_path)])
    if extra_args:
        command.extend(extra_args)

    if dry_run:
        return Mmg3dRun(command=command, input_path=str(input_mesh), output_path=str(output_mesh), dry_run=True)

    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return Mmg3dRun(
        command=command,
        input_path=str(input_mesh),
        output_path=str(output_mesh),
        dry_run=False,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _size_window(
    values: NDArray[np.float64],
    *,
    lower_percentile: float,
    target_percentile: float,
    upper_percentile: float,
) -> SizeWindow:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        nan = float("nan")
        return SizeWindow(nan, nan, nan, lower_percentile, target_percentile, upper_percentile)
    return SizeWindow(
        hmin=float(np.percentile(finite, lower_percentile)),
        htarget=float(np.percentile(finite, target_percentile)),
        hmax=float(np.percentile(finite, upper_percentile)),
        lower_percentile=lower_percentile,
        target_percentile=target_percentile,
        upper_percentile=upper_percentile,
    )


def _outlier_indices(values: NDArray[np.float64], window: SizeWindow) -> OutlierIndices:
    finite = np.asarray(values, dtype=float)
    too_small = np.flatnonzero(finite < window.hmin).astype(np.int64)
    too_large = np.flatnonzero(finite > window.hmax).astype(np.int64)
    return OutlierIndices(too_small=too_small, too_large=too_large)


def _facet_label_targets(
    mesh: MeshGeometry,
    facet_sizes: NDArray[np.float64],
    lower_percentile: float,
    target_percentile: float,
    upper_percentile: float,
) -> dict[str, LabelSizeTarget]:
    targets: dict[str, LabelSizeTarget] = {}
    for label, indices in mesh.facet_label_indices().items():
        selected = facet_sizes[indices]
        window = _size_window(
            selected,
            lower_percentile=lower_percentile,
            target_percentile=target_percentile,
            upper_percentile=upper_percentile,
        )
        targets[label] = LabelSizeTarget(
            label=label,
            count=int(indices.size),
            htarget=window.htarget,
            hmin=window.hmin,
            hmax=window.hmax,
        )
    return targets


def _element_count_target(mesh: MeshGeometry, base_hsiz: float, target_elements: int | None) -> ElementCountTarget | None:
    if target_elements is None:
        return None
    if target_elements <= 0:
        raise ValueError("target_elements must be positive")
    if mesh.nelements <= 0 or not np.isfinite(base_hsiz):
        return None

    dimension = max(mesh.dimension, 1)
    size_scale = float((mesh.nelements / target_elements) ** (1.0 / dimension))
    return ElementCountTarget(
        current_elements=mesh.nelements,
        target_elements=int(target_elements),
        dimension=dimension,
        base_hsiz=float(base_hsiz),
        recommended_hsiz=float(base_hsiz * size_scale),
        size_scale=size_scale,
    )


def _finite_min(*values: float) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(min(finite)) if finite else float("nan")


def _finite_max(*values: float) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(max(finite)) if finite else float("nan")


def _format_mmg3d_hint(args: dict[str, float]) -> str:
    parts = []
    for key in ("hmin", "hsiz", "hmax"):
        value = args.get(key, float("nan"))
        if np.isfinite(value):
            parts.append(f"-{key} {value:.6g}")
    return "mmg3d input.mesh output.mesh " + " ".join(parts)
