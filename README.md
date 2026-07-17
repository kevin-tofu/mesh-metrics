# mesh-metrics

Mesh quality metrics, facet statistics, visualization, and MMG3D remeshing workflow helpers for Python.

`mesh-metrics` is built for workflows where you inspect a mesh, decide how to control facet sizes and element counts, run MMG3D, and evaluate whether the remeshed result improved. It separates element-level mesh quality metrics from facet-level size metrics, and it keeps facet labels and geometric regions available across remeshing iterations.

## Features

- Element statistics: measure, diameter, edge aspect ratio, histograms.
- Facet statistics: facet measure, diameter, edge aspect ratio, label-wise stats, histograms.
- Dataclass API for meshes, facets, labels, histograms, regions, iteration metrics, and remeshing evaluation.
- Backends for `skfem`, `fluxfem`, and `meshio`.
- Matplotlib histogram and metrics-vs-iteration plots.
- PyVista VTU export and PNG rendering for mesh quality and labeled surface regions.
- MMG3D helpers for sizing recommendations, `.sol` size fields, remeshing, comparison, evaluation, and iterative workflows.

## Installation

Install the base package:

```bash
pip install mesh-metrics
```

Install optional mesh backends as needed:

```bash
pip install "mesh-metrics[skfem]"
pip install "mesh-metrics[fluxfem]"
pip install "mesh-metrics[all]"
```

For local development:

```bash
git clone git@github.com:kevin-tofu/mesh-metrics.git
cd mesh-metrics
poetry install --with dev
poetry run pytest -q
```

MMG3D is not bundled. Install `mmg3d` separately and make sure it is available on `PATH`, or pass `--mmg3d-bin`.

## Command Line

Compute statistics and write JSON plus histogram PNGs:

```bash
mesh-metrics stats mesh.mesh \
  --backend meshio \
  --json result/stats.json \
  --hist-dir result/histograms
```

Use facet labels:

```json
{
  "wall": [0, 1, 2, 3],
  "inlet": [4, 5],
  "outlet": [6, 7]
}
```

```bash
mesh-metrics stats mesh.mesh \
  --backend meshio \
  --facet-labels facet_labels.json \
  --json result/stats.json \
  --hist-dir result/histograms
```

The JSON separates element quality and facet size data:

- `elements`: element measure, diameter, and edge aspect ratio.
- `facets`: facet measure, diameter, edge aspect ratio, and label-wise statistics.
- `mesh_quality`: compatibility view focused on element quality.
- `facet_size`: compatibility view focused on facet sizing.

## Python API

```python
import numpy as np

from mesh_metrics import FacetLabels, MeshGeometry, MeshStatistics

mesh = MeshGeometry(
    points=np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    elements=np.asarray([[0], [1], [2], [3]]),
    facets=np.asarray([[0, 0, 0, 1], [1, 1, 2, 2], [2, 3, 3, 3]]),
    facet_labels=FacetLabels({"wall": np.asarray([0, 1, 2]), "outlet": np.asarray([3])}),
)

stats = MeshStatistics.from_mesh(mesh, bins=30)

print(stats.elements.edge_aspect_ratio.p95)
print(stats.facets.measure.mean)
print(stats.facets.labels["wall"].diameter.median)

stats.save_histograms("result/histograms")
```

Load from a file through a backend:

```python
from mesh_metrics import MeshGeometry, MeshStatistics

mesh = MeshGeometry.load("mesh.mesh", backend="meshio")
stats = MeshStatistics.from_mesh(mesh)
payload = stats.to_dict()
```

## Visualization

Export mesh and facet VTU files:

```bash
mesh-metrics export-vtu mesh.mesh \
  --backend meshio \
  --facet-labels facet_labels.json \
  --mesh-vtu result/mesh.vtu \
  --facets-vtu result/facets.vtu
```

Render element quality or facet size:

```bash
mesh-metrics render-vtu result/mesh.vtu result/mesh_aspect.png \
  --scalar element_edge_aspect_ratio

mesh-metrics render-vtu result/facets.vtu result/facet_size.png \
  --scalar facet_diameter
```

Render selected labeled regions:

```bash
mesh-metrics render-regions result/facets.vtu result/wall_region.png \
  --legend result/facets_vtu.json \
  --label wall
```

## MMG3D Workflow

Generate a size field from facet labels:

```bash
mesh-metrics size-field input.mesh metric.sol \
  --backend meshio \
  --facet-labels facet_labels.json \
  --facet-size-map facet_sizes.json \
  --target-elements 50000 \
  --json result/size_field.json
```

Run MMG3D with automatic sizing recommendations:

```bash
mesh-metrics remesh input.mesh remeshed.mesh \
  --backend meshio \
  --auto \
  --target-elements 50000 \
  --json result/remesh.json
```

Evaluate a remeshed result:

```bash
mesh-metrics evaluate input.mesh remeshed.mesh \
  --backend meshio \
  --facet-labels original_labels.json \
  --after-facet-labels remeshed_labels.json \
  --facet-size-map facet_sizes.json \
  --target-elements 50000 \
  --json result/evaluation.json \
  --suggest-relaxation \
  --suggested-facet-size-map result/next_facet_sizes.json
```

Plot iteration metrics with ranges:

```bash
mesh-metrics plot-iterations result/iter_000 result/iter_001 result/iter_002 \
  result/metrics_vs_iteration_with_ranges.png \
  --csv result/metrics_with_ranges.csv \
  --json result/metrics_with_ranges.json
```

## Region Transfer

After remeshing, facet IDs usually change. `mesh-metrics` provides region metadata and transfer tools so that original surface labels can be mapped onto the remeshed facets.

Create region metadata from labeled facets:

```bash
mesh-metrics init-regions input.mesh regions.json \
  --backend meshio \
  --facet-labels facet_labels.json \
  --include-exterior-curves
```

Transfer labels to a remeshed mesh:

```bash
mesh-metrics transfer-regions input.mesh remeshed.mesh remeshed_labels.json \
  --backend meshio \
  --regions regions.json \
  --source-facet-labels facet_labels.json \
  --target-facets-vtu result/remeshed_facets.vtu \
  --json result/transfer.json
```

This is useful when one selected surface should remain fine while the rest of the mesh is allowed to coarsen toward a target element count.

## Development

```bash
poetry install --with dev
poetry run pytest -q
poetry check
poetry build
```

The package is licensed under the Apache License 2.0.
