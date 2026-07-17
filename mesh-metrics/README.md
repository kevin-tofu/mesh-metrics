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

For local development with Poetry:

```bash
git clone git@github.com:kevin-tofu/mesh-metrics.git
cd mesh-metrics
python -m pip install --user pipx
pipx install poetry
poetry install --with dev --extras all
poetry run python -c "import mesh_metrics; print(mesh_metrics.__name__)"
poetry run pytest -q
```

`poetry install --with dev --extras all` installs the package in editable mode together with the development dependency group and optional mesh backends used by the backend tests.

MMG3D is not bundled. Install `mmg3d` separately and make sure it is available on `PATH`, or pass `--mmg3d-bin`.

## Python API

The core API has two steps:

1. Convert a mesh object or file into `MeshGeometry`.
2. Build `MeshStatistics` from that geometry.

`MeshStatistics` intentionally separates element quality from facet sizing:

- `stats.elements`: element measure, diameter, and edge aspect ratio.
- `stats.facets`: facet measure, diameter, edge aspect ratio, and label-wise facet statistics.
- `stats.histograms`: histogram data for element and facet quantities.
- `stats.to_dict()`: JSON-serializable output for reports, optimization loops, or remeshing drivers.

### From arrays

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

`MeshGeometry` expects dimension-first arrays:

- `points`: shape `(dimension, npoints)`
- `elements`: shape `(nodes_per_element, nelements)`
- `facets`: shape `(nodes_per_facet, nfacets)`

If `facets` is omitted, boundary facets are inferred for common triangle, quad, tetrahedron, and hexahedron meshes.

### Sampled Statistics For Large Meshes

For large meshes, compute representative statistics from random samples instead of traversing every element and facet. This is useful inside optimization loops, where mean, p95, max, and histograms are needed quickly and exact totals are less important than iteration speed.

```python
from mesh_metrics import MeshGeometry, MeshStatistics, SamplingConfig

mesh = MeshGeometry.load("large.mesh", backend="meshio")

stats = MeshStatistics.from_mesh(
    mesh,
    bins=40,
    sampling=SamplingConfig(
        max_elements=50_000,
        max_facets=30_000,
        max_facets_per_label=5_000,
        seed=7,
    ),
)

print(stats.elements.edge_aspect_ratio.p95)
print(stats.facets.diameter.p95)
print(stats.facets.labels["contact"].diameter.mean)
print(stats.to_dict()["sampling"])
```

Without `sampling=...`, `MeshStatistics.from_mesh` remains exact. With sampling enabled, quantity `count`, histograms, and label-wise statistics describe the sampled subset; `stats.to_dict()["sampling"]` records the original mesh counts and sampled counts so downstream optimization tools can distinguish sampled diagnostics from exact reports.

### From a scikit-fem mesh

Install the optional backend first:

```bash
pip install "mesh-metrics[skfem]"
```

Then pass a `skfem` mesh directly:

```python
from skfem import MeshTet

from mesh_metrics import MeshGeometry, MeshStatistics

sk_mesh = MeshTet().refined(2)

mesh = MeshGeometry.from_skfem(sk_mesh)
stats = MeshStatistics.from_mesh(mesh, bins=40)

print("elements:", mesh.nelements)
print("facets:", mesh.nfacets)
print("element volume mean:", stats.elements.measure.mean)
print("element aspect p95:", stats.elements.edge_aspect_ratio.p95)
print("facet area median:", stats.facets.measure.median)
```

`MeshGeometry.from_skfem` reads `mesh.p`, `mesh.t`, `mesh.facets`, and `mesh.boundaries` when present. `mesh.boundaries` becomes `FacetLabels`, so named boundaries are available in `stats.facets.labels`.

For labeled `skfem` boundaries:

```python
from skfem import MeshTri

from mesh_metrics import MeshGeometry, MeshStatistics

sk_mesh = MeshTri().refined(3).with_boundaries(
    {
        "left": lambda x: x[0] == 0.0,
        "right": lambda x: x[0] == 1.0,
    }
)

mesh = MeshGeometry.from_skfem(sk_mesh)
stats = MeshStatistics.from_mesh(mesh)

left = stats.facets.labels["left"]
print(left.measure.count)
print(left.diameter.mean)
```

### From a fluxfem-style mesh object

Install the optional backend first:

```bash
pip install "mesh-metrics[fluxfem]"
```

For in-memory mesh objects, use `MeshGeometry.from_object`. In the examples below, `flux_mesh` is the mesh object you already have in a `fluxfem` application, for example one returned by your mesh generator, file-loading layer, preprocessing step, or solver setup. `mesh-metrics` does not create a `fluxfem` mesh; it adapts the mesh-like object you pass in.

`MeshGeometry.from_object` accepts common `fluxfem`-style attribute names:

- coordinates: `points`, `vertices`, `nodes`, `coords`, or `coordinates`
- element connectivity: `elements`, `cells`, `connectivity`, or `t`
- facet connectivity: `facets`, `faces`, `edges`, or `boundary_facets`
- labels: `facet_labels`, `boundary_labels`, or `boundaries`

```python
from mesh_metrics import MeshGeometry, MeshStatistics

# flux_mesh comes from your fluxfem-side setup code.
mesh = MeshGeometry.from_object(flux_mesh, backend="fluxfem")
stats = MeshStatistics.from_mesh(mesh)

print(stats.elements.measure.total)
print(stats.elements.edge_aspect_ratio.max)
print(stats.facets.diameter.p95)
```

If your fluxfem workflow keeps mesh arrays separately, wrap them in a small adapter object:

```python
from dataclasses import dataclass

import numpy as np

from mesh_metrics import MeshGeometry, MeshStatistics


@dataclass
class FluxMeshAdapter:
    points: np.ndarray
    elements: np.ndarray
    faces: np.ndarray | None = None
    facet_labels: dict[str, list[int]] | None = None


flux_mesh = FluxMeshAdapter(
    points=np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    elements=np.asarray([[0, 1, 2, 3]]),
    faces=np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
    facet_labels={"wall": [0, 1, 2], "outlet": [3]},
)

mesh = MeshGeometry.from_object(flux_mesh, backend="fluxfem")
stats = MeshStatistics.from_mesh(mesh)
print(stats.facets.labels["wall"].measure.mean)
```

If your object does not expose labels in a supported attribute, attach them explicitly:

```python
from mesh_metrics import FacetLabels, MeshGeometry, MeshStatistics

mesh = MeshGeometry.from_object(flux_mesh, backend="fluxfem")
mesh = mesh.with_facet_labels(
    FacetLabels(
        {
            "wall": [0, 1, 2, 3],
            "inlet": [4, 5],
            "outlet": [6, 7],
        }
    )
)

stats = MeshStatistics.from_mesh(mesh)
print(stats.facets.labels["wall"].measure.mean)
```

### From mesh files

Use `MeshGeometry.load` when you want the backend to read a file:

```python
from mesh_metrics import MeshGeometry, MeshStatistics

mesh = MeshGeometry.load("mesh.mesh", backend="meshio")
stats = MeshStatistics.from_mesh(mesh)
payload = stats.to_dict()
```

Available file backends are `meshio`, `skfem`, and `fluxfem`.

```python
mesh = MeshGeometry.load("mesh.vtu", backend="meshio")
mesh = MeshGeometry.load("mesh.msh", backend="skfem")
mesh = MeshGeometry.load("mesh.mesh", backend="fluxfem")
```

The `fluxfem` file backend currently uses `meshio` as the file adapter while keeping `backend="fluxfem"` in the resulting `MeshGeometry`.

### Reading statistics

```python
stats = MeshStatistics.from_mesh(mesh, bins=50)

element_quality = stats.elements
facet_size = stats.facets

print(element_quality.measure.min)
print(element_quality.measure.max)
print(element_quality.edge_aspect_ratio.mean)
print(element_quality.edge_aspect_ratio.p95)

print(facet_size.measure.mean)
print(facet_size.diameter.p05)
print(facet_size.diameter.p95)
```

Each quantity is a `QuantityStats` dataclass:

```python
q = stats.elements.edge_aspect_ratio

print(q.count)
print(q.min, q.max)
print(q.mean, q.median, q.std)
print(q.p05, q.p25, q.p75, q.p95)
```

Label-wise facet statistics are stored under `stats.facets.labels`:

```python
for label, label_stats in stats.facets.labels.items():
    print(label)
    print("facet count:", label_stats.measure.count)
    print("area mean:", label_stats.measure.mean)
    print("diameter p95:", label_stats.diameter.p95)
```

### Histograms and report data

Save all histograms as PNG files:

```python
stats = MeshStatistics.from_mesh(mesh, bins=40)
paths = stats.save_histograms("result/histograms")
```

Use `to_dict` for optimization tools, reports, or JSON output:

```python
import json
from pathlib import Path

payload = stats.to_dict()
Path("result/stats.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
```

Omit histogram arrays when the mesh is large:

```python
payload = stats.to_dict(include_histograms=False)
```

### Visualization and remeshing helpers

The Python API also exposes VTU export, PNG rendering, MMG3D execution metadata, size fields, region transfer, and iteration plots:

```python
from mesh_metrics import SizeField, export_vtu

field = SizeField.from_mesh(
    mesh,
    default_size=0.5,
    facet_size_map={"wall": 0.1, "inlet": 0.2},
)
field.write_mmg_sol("metric.sol", dimension=mesh.dimension)

export_vtu(mesh, mesh_path="result/mesh.vtu", facets_path="result/facets.vtu")
```

The command line interface is available as `mesh-metrics`, but the library is designed so optimization and remeshing loops can use these dataclasses directly.

## Development

Set up the repository:

```bash
git clone git@github.com:kevin-tofu/mesh-metrics.git
cd mesh-metrics
poetry install --with dev --extras all
```

Run the local checks:

```bash
poetry run python -c "from mesh_metrics import MeshGeometry, MeshStatistics; print(MeshGeometry, MeshStatistics)"
poetry run pytest -q
poetry check
poetry build
```

When dependencies change, update the lock file and verify the package again:

```bash
poetry lock
poetry install --with dev --extras all
poetry run pytest -q
poetry build
```

The GitHub Actions workflow runs `poetry install --with dev --extras all`, `poetry check`, and `pytest` on Python 3.12.

The package is licensed under the Apache License 2.0.
