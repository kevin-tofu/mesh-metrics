from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mesh_metrics.geometry import MeshGeometry
from mesh_metrics.geometry import FacetLabels


@dataclass(frozen=True)
class VtkExportResult:
    mesh_path: str | None
    facets_path: str | None
    label_legend: dict[str, int]
    curve_label_legend: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mesh_path": self.mesh_path,
            "facets_path": self.facets_path,
            "label_legend": self.label_legend,
            "curve_label_legend": self.curve_label_legend,
        }


@dataclass(frozen=True)
class VtuRenderResult:
    input_path: str
    output_path: str
    scalar: str | None
    view: str
    window_size: tuple[int, int]
    show_edges: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "scalar": self.scalar,
            "view": self.view,
            "window_size": list(self.window_size),
            "show_edges": self.show_edges,
        }


@dataclass(frozen=True)
class RegionRenderResult:
    input_path: str
    output_path: str
    selected_labels: list[str]
    selected_region_ids: list[int]
    label_legend: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "selected_labels": self.selected_labels,
            "selected_region_ids": self.selected_region_ids,
            "label_legend": self.label_legend,
        }


def export_vtu(
    mesh: MeshGeometry,
    *,
    mesh_path: str | Path | None = None,
    facets_path: str | Path | None = None,
    curve_labels: FacetLabels | None = None,
) -> VtkExportResult:
    written_mesh = None
    written_facets = None
    legend = _label_legend(mesh)
    curve_legend = _labels_legend(curve_labels)
    if mesh_path is not None:
        grid = _mesh_grid(mesh)
        grid.save(str(mesh_path))
        written_mesh = str(mesh_path)
    if facets_path is not None:
        grid = _facet_grid(mesh, legend, curve_labels, curve_legend)
        grid.save(str(facets_path))
        written_facets = str(facets_path)
    return VtkExportResult(mesh_path=written_mesh, facets_path=written_facets, label_legend=legend, curve_label_legend=curve_legend)


def render_vtu_png(
    vtu_path: str | Path,
    output_path: str | Path,
    *,
    scalar: str | None = None,
    view: str = "isometric",
    show_edges: bool = True,
    cmap: str = "viridis",
    window_size: tuple[int, int] = (1200, 900),
    background: str = "white",
) -> VtuRenderResult:
    from mesh_metrics._plotting import configure_matplotlib_cache

    configure_matplotlib_cache()
    import pyvista as pv

    input_path = Path(vtu_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    pv.OFF_SCREEN = True
    dataset = pv.read(str(input_path))
    selected_scalar = scalar or _default_scalar(dataset)
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background)
    plotter.add_mesh(
        dataset,
        scalars=selected_scalar,
        show_edges=show_edges,
        cmap=cmap,
        edge_color="#333333",
        line_width=0.6,
        scalar_bar_args={"title": selected_scalar or ""},
    )
    _set_camera(plotter, view)
    plotter.show(screenshot=str(output), auto_close=True)
    return VtuRenderResult(
        input_path=str(input_path),
        output_path=str(output),
        scalar=selected_scalar,
        view=view,
        window_size=window_size,
        show_edges=show_edges,
    )


def render_regions_png(
    vtu_path: str | Path,
    output_path: str | Path,
    *,
    label_legend: dict[str, int],
    labels: list[str] | None = None,
    view: str = "isometric",
    show_edges: bool = True,
    window_size: tuple[int, int] = (1200, 900),
    background: str = "white",
) -> RegionRenderResult:
    from mesh_metrics._plotting import configure_matplotlib_cache

    configure_matplotlib_cache()
    import pyvista as pv

    input_path = Path(vtu_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = pv.read(str(input_path))
    if "region_id" not in dataset.cell_data:
        raise ValueError("VTU has no cell_data['region_id']")

    selected_labels = sorted(label_legend) if labels is None else labels
    selected_region_ids = [label_legend[label] for label in selected_labels if label in label_legend]
    region_ids = np.asarray(dataset.cell_data["region_id"])
    selected_mask = np.isin(region_ids, selected_region_ids)

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background)
    plotter.add_mesh(dataset, color="#d8d8d8", opacity=0.20, show_edges=False)
    if np.any(selected_mask):
        selected = dataset.extract_cells(np.flatnonzero(selected_mask))
        plotter.add_mesh(
            selected,
            scalars="region_id",
            cmap="tab20",
            categories=True,
            show_edges=show_edges,
            edge_color="#262626",
            line_width=0.5,
            scalar_bar_args={"title": "region_id"},
        )
    _set_camera(plotter, view)
    plotter.show(screenshot=str(output), auto_close=True)
    return RegionRenderResult(
        input_path=str(input_path),
        output_path=str(output),
        selected_labels=selected_labels,
        selected_region_ids=[int(value) for value in selected_region_ids],
        label_legend=label_legend,
    )


def _mesh_grid(mesh: MeshGeometry):
    import pyvista as pv

    cells = _vtk_cells(mesh.elements)
    cell_type = _element_cell_type(mesh.dimension, mesh.elements.shape[0])
    grid = pv.UnstructuredGrid(cells, np.full(mesh.nelements, cell_type, dtype=np.uint8), _points_xyz(mesh))
    grid.cell_data["element_measure"] = mesh.element_measures()
    grid.cell_data["element_diameter"] = mesh.element_diameters()
    grid.cell_data["element_edge_aspect_ratio"] = mesh.element_edge_aspect_ratios()
    return grid


def _facet_grid(mesh: MeshGeometry, legend: dict[str, int], curve_labels: FacetLabels | None, curve_legend: dict[str, int]):
    import pyvista as pv

    if mesh.facets is None:
        raise ValueError("mesh has no facets")
    cells = _vtk_cells(mesh.facets)
    cell_type = _facet_cell_type(mesh.dimension, mesh.facets.shape[0])
    grid = pv.UnstructuredGrid(cells, np.full(mesh.nfacets, cell_type, dtype=np.uint8), _points_xyz(mesh))
    grid.cell_data["facet_measure"] = mesh.facet_measures()
    grid.cell_data["facet_diameter"] = mesh.facet_diameters()
    grid.cell_data["facet_edge_aspect_ratio"] = mesh.facet_edge_aspect_ratios()
    region_id = np.full(mesh.nfacets, -1, dtype=np.int64)
    for label, value in legend.items():
        indices = mesh.facet_label_indices().get(label)
        if indices is not None:
            region_id[indices] = value
    grid.cell_data["region_id"] = region_id
    curve_region_id = np.full(mesh.nfacets, -1, dtype=np.int64)
    if curve_labels is not None:
        for label, value in curve_legend.items():
            indices = curve_labels.groups.get(label)
            if indices is not None:
                curve_region_id[indices] = value
    grid.cell_data["curve_region_id"] = curve_region_id
    return grid


def _vtk_cells(connectivity: np.ndarray) -> np.ndarray:
    nitems, ncells = connectivity.shape
    cells = np.empty((ncells, nitems + 1), dtype=np.int64)
    cells[:, 0] = nitems
    cells[:, 1:] = connectivity.T
    return cells.ravel()


def _points_xyz(mesh: MeshGeometry) -> np.ndarray:
    points = mesh.points.T
    if points.shape[1] == 3:
        return points
    if points.shape[1] == 2:
        return np.column_stack([points, np.zeros(points.shape[0])])
    if points.shape[1] == 1:
        return np.column_stack([points, np.zeros((points.shape[0], 2))])
    raise ValueError("only dimensions 1, 2, and 3 can be exported to VTK")


def _label_legend(mesh: MeshGeometry) -> dict[str, int]:
    return {label: index for index, label in enumerate(sorted(mesh.facet_label_indices()))}


def _labels_legend(labels: FacetLabels | None) -> dict[str, int]:
    if labels is None:
        return {}
    return {label: index for index, label in enumerate(sorted(labels.groups))}


def _default_scalar(dataset) -> str | None:
    preferred = [
        "element_edge_aspect_ratio",
        "facet_diameter",
        "region_id",
        "element_diameter",
        "facet_edge_aspect_ratio",
    ]
    for name in preferred:
        if name in dataset.cell_data:
            return name
    if dataset.cell_data:
        return next(iter(dataset.cell_data.keys()))
    if dataset.point_data:
        return next(iter(dataset.point_data.keys()))
    return None


def _set_camera(plotter, view: str) -> None:
    normalized = view.lower().replace("-", "_")
    if normalized in {"xy", "top"}:
        plotter.view_xy()
    elif normalized in {"xz", "front"}:
        plotter.view_xz()
    elif normalized in {"yz", "side"}:
        plotter.view_yz()
    elif normalized in {"isometric", "iso", "3d"}:
        plotter.view_isometric()
    else:
        raise ValueError("view must be one of: isometric, xy, xz, yz")
    plotter.reset_camera()


def _element_cell_type(dimension: int, nodes_per_element: int) -> int:
    if dimension == 1 and nodes_per_element == 2:
        return 3
    if dimension == 2 and nodes_per_element == 3:
        return 5
    if dimension == 2 and nodes_per_element == 4:
        return 9
    if dimension == 3 and nodes_per_element == 4:
        return 10
    if dimension == 3 and nodes_per_element == 8:
        return 12
    raise ValueError(f"unsupported element type: dimension={dimension}, nodes={nodes_per_element}")


def _facet_cell_type(dimension: int, nodes_per_facet: int) -> int:
    if dimension == 2 and nodes_per_facet == 2:
        return 3
    if dimension == 3 and nodes_per_facet == 3:
        return 5
    if dimension == 3 and nodes_per_facet == 4:
        return 9
    raise ValueError(f"unsupported facet type: dimension={dimension}, nodes={nodes_per_facet}")
