from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Histogram:
    quantity: str
    counts: NDArray[np.int64]
    bin_edges: NDArray[np.float64]
    bins: int

    @classmethod
    def from_values(cls, quantity: str, values: NDArray[np.float64], *, bins: int = 30) -> Histogram:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            counts, bin_edges = np.histogram(finite, bins=bins)
        elif float(np.min(finite)) == float(np.max(finite)):
            center = float(finite[0])
            width = max(abs(center) * 1.0e-6, 1.0e-12)
            counts, bin_edges = np.histogram(finite, bins=bins, range=(center - width, center + width))
        else:
            try:
                counts, bin_edges = np.histogram(finite, bins=bins)
            except ValueError:
                lower = float(np.min(finite))
                upper = float(np.max(finite))
                width = max(abs(upper - lower) * 1.0e-6, max(abs(lower), abs(upper)) * 1.0e-12, 1.0e-12)
                counts, bin_edges = np.histogram(finite, bins=bins, range=(lower - width, upper + width))
        return cls(quantity=quantity, counts=counts.astype(np.int64), bin_edges=bin_edges.astype(float), bins=bins)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["counts"] = self.counts.tolist()
        result["bin_edges"] = self.bin_edges.tolist()
        return result

    def savefig(self, path: str | Path, *, title: str | None = None, dpi: int = 150) -> Path:
        from mesh_metrics._plotting import configure_matplotlib_cache

        configure_matplotlib_cache()
        import matplotlib.pyplot as plt

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        width = np.diff(self.bin_edges)
        ax.bar(self.bin_edges[:-1], self.counts, width=width, align="edge", edgecolor="#222222", linewidth=0.6)
        ax.set_title(title or self.quantity.replace("_", " ").title())
        ax.set_xlabel(self.quantity)
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(output, dpi=dpi)
        plt.close(fig)
        return output
