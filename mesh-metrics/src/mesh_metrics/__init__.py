"""Mesh statistics tools for skfem/fluxfem-style meshes."""

from mesh_metrics.geometry import FacetGeometry, FacetLabels, MeshGeometry
from mesh_metrics.histogram import Histogram
from mesh_metrics.iterations import IterationMetrics, IterationSeries
from mesh_metrics.remesh import Mmg3dRecommendation, Mmg3dRun, run_mmg3d
from mesh_metrics.regions import CurveRegion, Region, RegionSet, RegionSpec, RegionSpecSet, SurfaceRegion
from mesh_metrics.size_field import SizeField
from mesh_metrics.stats import ElementStatistics, FacetLabelStats, FacetStatistics, MeshStatistics, QuantityStats, SamplingConfig, SamplingSummary
from mesh_metrics.evaluate import RemeshEvaluation
from mesh_metrics.visualize import RegionRenderResult, VtkExportResult, VtuRenderResult, export_vtu, render_regions_png, render_vtu_png

__all__ = [
    "FacetLabels",
    "FacetGeometry",
    "FacetLabelStats",
    "FacetStatistics",
    "ElementStatistics",
    "CurveRegion",
    "Histogram",
    "IterationMetrics",
    "IterationSeries",
    "MeshGeometry",
    "MeshStatistics",
    "Mmg3dRecommendation",
    "Mmg3dRun",
    "QuantityStats",
    "SamplingConfig",
    "SamplingSummary",
    "Region",
    "RegionRenderResult",
    "RegionSet",
    "RegionSpec",
    "RegionSpecSet",
    "RemeshEvaluation",
    "SizeField",
    "SurfaceRegion",
    "VtkExportResult",
    "VtuRenderResult",
    "export_vtu",
    "render_regions_png",
    "render_vtu_png",
    "run_mmg3d",
]
