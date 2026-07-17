from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mesh_metrics.geometry import MeshGeometry


@dataclass(frozen=True)
class SizeField:
    """Isotropic scalar size field stored at mesh vertices."""

    values: NDArray[np.float64]
    default_size: float
    facet_size_map: dict[str, float]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1:
            raise ValueError("size field values must be a 1D array")
        if not np.all(np.isfinite(values)):
            raise ValueError("size field values must be finite")
        if np.any(values <= 0.0):
            raise ValueError("size field values must be positive")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "facet_size_map", {str(label): float(size) for label, size in self.facet_size_map.items()})

    @classmethod
    def from_mesh(
        cls,
        mesh: MeshGeometry,
        *,
        default_size: float,
        facet_size_map: dict[str, float] | None = None,
    ) -> SizeField:
        if default_size <= 0.0 or not np.isfinite(default_size):
            raise ValueError("default_size must be a positive finite value")

        values = np.full(mesh.npoints, float(default_size), dtype=float)
        sizes = {} if facet_size_map is None else {str(label): float(size) for label, size in facet_size_map.items()}
        if mesh.facets is not None:
            for label, size in sizes.items():
                if size <= 0.0 or not np.isfinite(size):
                    raise ValueError(f"facet size for {label!r} must be a positive finite value")
                indices = mesh.facet_label_indices().get(label)
                if indices is None:
                    continue
                node_indices = np.unique(mesh.facets[:, indices])
                values[node_indices] = np.minimum(values[node_indices], size)

        return cls(values=values, default_size=float(default_size), facet_size_map=sizes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nvertices": int(self.values.size),
            "default_size": self.default_size,
            "facet_size_map": self.facet_size_map,
            "min": float(np.min(self.values)) if self.values.size else float("nan"),
            "max": float(np.max(self.values)) if self.values.size else float("nan"),
            "mean": float(np.mean(self.values)) if self.values.size else float("nan"),
        }

    def write_mmg_sol(self, path: str | Path, *, dimension: int) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "MeshVersionFormatted 2",
            f"Dimension {dimension}",
            "",
            "SolAtVertices",
            str(self.values.size),
            "1 1",
        ]
        lines.extend(f"{value:.17g}" for value in self.values)
        lines.extend(["", "End", ""])
        output.write_text("\n".join(lines), encoding="ascii")
        return output


def load_facet_size_map(path: str | Path) -> dict[str, float]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("facet size map JSON must be an object mapping labels to sizes")
    return {str(label): float(size) for label, size in payload.items()}
