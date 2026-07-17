from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mesh_metrics.compare import MeshComparison
from mesh_metrics.demo import create_complex_channel, create_notched_channel
from mesh_metrics.evaluate import ConstraintDiagnosis, RemeshEvaluation
from mesh_metrics.geometry import FacetLabels, MeshGeometry
from mesh_metrics.iterations import IterationSeries
from mesh_metrics.remesh import Mmg3dRecommendation, run_mmg3d
from mesh_metrics.regions import RegionSet, RegionSpecSet
from mesh_metrics.size_field import SizeField, load_facet_size_map
from mesh_metrics.stats import MeshStatistics
from mesh_metrics.visualize import export_vtu, render_regions_png, render_vtu_png


COMMANDS = {
    "stats",
    "remesh",
    "compare",
    "evaluate",
    "size-field",
    "export-vtu",
    "render-vtu",
    "render-regions",
    "plot-iterations",
    "transfer-regions",
    "init-regions",
    "demo-case",
    "mmg3d-workflow",
    "iterate",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mesh statistics and MMG3D remeshing helpers.")
    subparsers = parser.add_subparsers(dest="command")
    _add_stats_arguments(subparsers.add_parser("stats", help="Compute mesh statistics."))
    _add_remesh_arguments(subparsers.add_parser("remesh", help="Run MMG3D with explicit or recommended size options."))
    _add_compare_arguments(subparsers.add_parser("compare", help="Compare mesh statistics before and after remeshing."))
    _add_evaluate_arguments(subparsers.add_parser("evaluate", help="Evaluate remeshing targets after MMG3D."))
    _add_size_field_arguments(subparsers.add_parser("size-field", help="Write an MMG .sol scalar size field."))
    _add_export_vtu_arguments(subparsers.add_parser("export-vtu", help="Export mesh and facets to VTU for PyVista/ParaView."))
    _add_render_vtu_arguments(subparsers.add_parser("render-vtu", help="Render a VTU file to a PNG image."))
    _add_render_regions_arguments(subparsers.add_parser("render-regions", help="Render labeled facet regions from a VTU file."))
    _add_plot_iterations_arguments(subparsers.add_parser("plot-iterations", help="Plot remeshing metrics across iteration outputs."))
    _add_transfer_regions_arguments(subparsers.add_parser("transfer-regions", help="Transfer original facet labels to remeshed facets."))
    _add_init_regions_arguments(subparsers.add_parser("init-regions", help="Create a region metadata JSON from facet labels."))
    _add_demo_case_arguments(subparsers.add_parser("demo-case", help="Generate a notched-channel demo mesh and metadata."))
    _add_mmg3d_workflow_arguments(subparsers.add_parser("mmg3d-workflow", help="Run stats, metric, MMG3D, transfer, evaluate, and VTU export."))
    _add_iterate_arguments(subparsers.add_parser("iterate", help="Run repeated MMG3D workflows with updated facet size maps."))
    return parser


def build_stats_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute mesh element/facet statistics and histograms.")
    _add_stats_arguments(parser)
    return parser


def _add_common_mesh_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("skfem", "fluxfem", "meshio"), default="skfem", help="Mesh loading backend.")
    parser.add_argument("--facet-labels", help="JSON file mapping facet label names to facet index arrays.")


def _add_stats_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("mesh", help="Mesh file readable by the selected backend.")
    _add_common_mesh_arguments(parser)
    parser.add_argument("--bins", type=int, default=30, help="Number of histogram bins.")
    parser.add_argument("--json", dest="json_path", help="Write statistics JSON to this path.")
    parser.add_argument("--hist-dir", help="Write histogram PNG files to this directory.")
    parser.add_argument("--recommend-mmg3d", action="store_true", help="Add quantile-based MMG3D sizing recommendations.")
    parser.add_argument("--target-elements", type=int, help="Adjust recommended hsiz toward a target element count.")
    parser.add_argument("--lower-percentile", type=float, default=5.0, help="Lower size percentile for recommendations.")
    parser.add_argument("--target-percentile", type=float, default=50.0, help="Target size percentile for recommendations.")
    parser.add_argument("--upper-percentile", type=float, default=95.0, help="Upper size percentile for recommendations.")
    parser.add_argument("--no-histograms", action="store_true", help="Omit histogram arrays from JSON/stdout.")
    parser.add_argument("--dpi", type=int, default=150, help="Histogram image DPI.")


def _add_remesh_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_mesh", help="Input mesh passed to MMG3D.")
    parser.add_argument("output_mesh", help="Output mesh produced by MMG3D.")
    _add_common_mesh_arguments(parser)
    parser.add_argument("--mmg3d-bin", default="mmg3d", help="MMG3D executable name or path.")
    parser.add_argument("--hmin", type=float, help="Minimum target edge size.")
    parser.add_argument("--hsiz", type=float, help="Uniform target edge size.")
    parser.add_argument("--hmax", type=float, help="Maximum target edge size.")
    parser.add_argument("--auto", action="store_true", help="Fill missing hmin/hsiz/hmax from mesh statistics.")
    parser.add_argument("--target-elements", type=int, help="When auto-sizing, adjust hsiz toward a target element count.")
    parser.add_argument("--lower-percentile", type=float, default=5.0, help="Auto hmin percentile.")
    parser.add_argument("--target-percentile", type=float, default=50.0, help="Auto hsiz percentile.")
    parser.add_argument("--upper-percentile", type=float, default=95.0, help="Auto hmax percentile.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned command without running MMG3D.")
    parser.add_argument("--json", dest="json_path", help="Write run metadata JSON to this path.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional MMG3D argument, repeatable; use --extra-arg=-flag for flag-like values.")
    parser.add_argument("--facet-size-map", help="JSON mapping facet label names to target sizes.")
    parser.add_argument("--write-sol", help="Write an MMG .sol scalar size field and pass it to MMG3D with -sol.")


def _add_compare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("before_mesh", help="Mesh before remeshing.")
    parser.add_argument("after_mesh", help="Mesh after remeshing.")
    _add_common_mesh_arguments(parser)
    parser.add_argument("--after-facet-labels", help="Facet labels JSON for the after mesh; defaults to --facet-labels.")
    parser.add_argument("--bins", type=int, default=30, help="Number of histogram bins.")
    parser.add_argument("--json", dest="json_path", help="Write comparison JSON to this path.")
    parser.add_argument("--include-mesh-stats", action="store_true", help="Embed before/after stats in comparison JSON.")


def _add_evaluate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("before_mesh", help="Mesh before remeshing.")
    parser.add_argument("after_mesh", help="Mesh after remeshing.")
    _add_common_mesh_arguments(parser)
    parser.add_argument("--after-facet-labels", help="Facet labels JSON for the after mesh; defaults to --facet-labels.")
    parser.add_argument("--facet-size-map", help="JSON mapping facet label names to target sizes.")
    parser.add_argument("--target-elements", type=int, help="Target element count for the after mesh.")
    parser.add_argument("--tolerance", type=float, default=0.2, help="Relative tolerance for target facet sizes.")
    parser.add_argument("--bins", type=int, default=30, help="Number of histogram bins for internal statistics.")
    parser.add_argument("--json", dest="json_path", help="Write evaluation JSON to this path.")
    parser.add_argument("--include-mesh-stats", action="store_true", help="Embed before/after stats in evaluation JSON.")
    parser.add_argument("--suggest-relaxation", action="store_true", help="Add constraint diagnosis and relaxed facet size suggestions.")
    parser.add_argument("--suggested-facet-size-map", help="Write suggested facet size JSON to this path.")
    parser.add_argument("--max-relaxation", type=float, default=3.0, help="Maximum per-label size relaxation factor.")


def _add_size_field_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("mesh", help="Mesh whose vertices receive the size field.")
    parser.add_argument("sol", help="Output .sol file.")
    _add_common_mesh_arguments(parser)
    parser.add_argument("--default-size", type=float, help="Size for vertices not constrained by facet labels.")
    parser.add_argument("--target-elements", type=int, help="Estimate default size from a target element count.")
    parser.add_argument("--facet-size-map", help="JSON mapping facet label names to target sizes.")
    parser.add_argument("--lower-percentile", type=float, default=5.0, help="Auto hmin percentile.")
    parser.add_argument("--target-percentile", type=float, default=50.0, help="Auto default-size percentile.")
    parser.add_argument("--upper-percentile", type=float, default=95.0, help="Auto hmax percentile.")
    parser.add_argument("--json", dest="json_path", help="Write size-field metadata JSON to this path.")


def _add_export_vtu_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("mesh", help="Mesh to export.")
    _add_common_mesh_arguments(parser)
    parser.add_argument("--mesh-vtu", help="Output VTU with element quality data.")
    parser.add_argument("--facets-vtu", help="Output VTU with facet size and region_id data.")
    parser.add_argument("--json", dest="json_path", help="Write export metadata JSON to this path.")


def _add_render_vtu_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("vtu", help="Input VTU file.")
    parser.add_argument("png", help="Output PNG screenshot.")
    parser.add_argument("--scalar", help="Cell/point scalar to color by; defaults to a useful quality/region scalar.")
    parser.add_argument("--view", default="isometric", choices=("isometric", "xy", "xz", "yz"), help="Camera view.")
    parser.add_argument("--no-edges", action="store_true", help="Hide mesh edges in the rendered image.")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib/PyVista colormap name.")
    parser.add_argument("--width", type=int, default=1200, help="Image width in pixels.")
    parser.add_argument("--height", type=int, default=900, help="Image height in pixels.")
    parser.add_argument("--json", dest="json_path", help="Write render metadata JSON to this path.")


def _add_render_regions_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("vtu", help="Input facet VTU with region_id cell data.")
    parser.add_argument("png", help="Output PNG screenshot.")
    parser.add_argument("--legend", required=True, help="JSON mapping region labels to region_id integers.")
    parser.add_argument("--label", action="append", default=[], help="Region label to paint; repeatable. Defaults to all labels.")
    parser.add_argument("--view", default="isometric", choices=("isometric", "xy", "xz", "yz"), help="Camera view.")
    parser.add_argument("--no-edges", action="store_true", help="Hide mesh edges on painted regions.")
    parser.add_argument("--width", type=int, default=1200, help="Image width in pixels.")
    parser.add_argument("--height", type=int, default=900, help="Image height in pixels.")
    parser.add_argument("--json", dest="json_path", help="Write render metadata JSON to this path.")


def _add_plot_iterations_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("iterations", nargs="+", help="Iteration directories or evaluation.json files, in order.")
    parser.add_argument("png", help="Output metrics-vs-iteration PNG.")
    parser.add_argument("--csv", dest="csv_path", help="Write flattened metrics CSV.")
    parser.add_argument("--json", dest="json_path", help="Write flattened metrics JSON.")
    parser.add_argument("--dpi", type=int, default=150, help="Output image DPI.")


def _add_transfer_regions_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source_mesh", help="Original mesh with facet labels.")
    parser.add_argument("target_mesh", help="Remeshed mesh whose facets receive transferred labels.")
    parser.add_argument("output_labels", help="Output facet-label JSON for target mesh.")
    parser.add_argument("--backend", choices=("skfem", "fluxfem", "meshio"), default="skfem", help="Mesh loading backend.")
    parser.add_argument("--regions", help="Region metadata JSON defining surface and curve regions.")
    parser.add_argument("--source-facet-labels", help="Facet-label JSON for source mesh, used when --regions is omitted.")
    parser.add_argument("--tolerance", type=float, help="Geometric transfer tolerance; defaults to a small target facet scale.")
    parser.add_argument("--region-max-distance", type=float, help="Per-region source surface search distance.")
    parser.add_argument("--max-angle-deg", type=float, default=45.0, help="Maximum normal angle difference for surface region transfer.")
    parser.add_argument("--region-priorities", help="JSON mapping region labels to integer priorities.")
    parser.add_argument("--include-exterior-curves", action="store_true", help="Also create curve regions for exterior boundaries of each surface label.")
    parser.add_argument("--curve-max-distance", type=float, help="Per-curve source search distance.")
    parser.add_argument("--output-curve-labels", help="Optional output JSON for target facets near transferred curve regions.")
    parser.add_argument("--target-facets-vtu", help="Optional VTU export of target facets with transferred region_id.")
    parser.add_argument("--json", dest="json_path", help="Write transfer metadata JSON to this path.")


def _add_init_regions_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("mesh", help="Mesh used to estimate default distances.")
    parser.add_argument("regions", help="Output region metadata JSON.")
    _add_common_mesh_arguments(parser)
    parser.add_argument("--default-max-distance", type=float, help="Default max_distance for surface regions; inferred from facet median diameter when omitted.")
    parser.add_argument("--distance-scale", type=float, default=0.02, help="Scale times median facet diameter when --default-max-distance is omitted.")
    parser.add_argument("--default-max-angle-deg", type=float, default=45.0, help="Default surface normal angle tolerance.")
    parser.add_argument("--default-priority", type=int, default=0, help="Default region priority.")
    parser.add_argument("--auto-curves", action="store_true", default=True, help="Generate auto_curves metadata from surface boundaries.")
    parser.add_argument("--no-auto-curves", dest="auto_curves", action="store_false", help="Disable auto curve metadata.")
    parser.add_argument("--include-exterior-curves", action="store_true", help="Auto-generate exterior boundary curve regions.")
    parser.add_argument("--curve-max-distance", type=float, help="Default max_distance for auto curve regions; inferred when omitted.")
    parser.add_argument("--curve-distance-scale", type=float, default=0.01, help="Scale times median facet diameter when --curve-max-distance is omitted.")
    parser.add_argument("--json", dest="json_path", help="Write command metadata JSON to this path.")


def _add_demo_case_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("output_dir", help="Directory for demo mesh, labels, sizes, and region spec.")
    parser.add_argument("--kind", choices=("notched", "complex"), default="notched", help="Demo geometry kind.")
    parser.add_argument("--nx", type=int, default=12, help="Number of cells along x.")
    parser.add_argument("--ny", type=int, default=8, help="Number of cells along y.")
    parser.add_argument("--nz", type=int, default=8, help="Number of cells along z.")
    parser.add_argument("--json", dest="json_path", help="Write command metadata JSON to this path.")


def _add_mmg3d_workflow_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_mesh", help="Input mesh.")
    parser.add_argument("output_dir", help="Workflow output directory.")
    parser.add_argument("--backend", choices=("skfem", "fluxfem", "meshio"), default="meshio", help="Mesh loading backend.")
    parser.add_argument("--facet-labels", required=True, help="Original facet-label JSON.")
    parser.add_argument("--regions", required=True, help="Region metadata JSON.")
    parser.add_argument("--facet-size-map", required=True, help="Facet target size JSON.")
    parser.add_argument("--target-elements", type=int, default=1500, help="Target element count after remeshing.")
    parser.add_argument("--mmg3d-bin", default="mmg3d", help="MMG3D executable name or path.")
    parser.add_argument("--dry-run", action="store_true", help="Build inputs and planned command without running MMG3D.")
    parser.add_argument("--json", dest="json_path", help="Write workflow metadata JSON to this path.")


def _add_iterate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_mesh", help="Original input mesh.")
    parser.add_argument("output_dir", help="Directory containing iter_000, iter_001, ... outputs.")
    parser.add_argument("--backend", choices=("skfem", "fluxfem", "meshio"), default="meshio", help="Mesh loading backend.")
    parser.add_argument("--facet-labels", required=True, help="Original facet-label JSON.")
    parser.add_argument("--regions", required=True, help="Region metadata JSON.")
    parser.add_argument("--facet-size-map", required=True, help="Initial facet target size JSON.")
    parser.add_argument("--target-elements", type=int, default=1500, help="Target element count after remeshing.")
    parser.add_argument("--mmg3d-bin", default="mmg3d", help="MMG3D executable name or path.")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum number of workflow evaluations.")
    parser.add_argument("--min-iterations", type=int, default=1, help="Run at least this many successful evaluations before accepting convergence.")
    parser.add_argument("--element-tolerance", type=float, default=0.2, help="Acceptable relative error around target element count.")
    parser.add_argument("--facet-ratio-tolerance", type=float, default=0.25, help="Acceptable average facet mean/target ratio deviation.")
    parser.add_argument("--max-aspect-p95", type=float, help="Optional upper bound for p95 element edge aspect ratio.")
    parser.add_argument("--relaxation-damping", type=float, default=0.7, help="Blend from current sizes to suggested sizes; 1.0 applies suggestions directly.")
    parser.add_argument("--min-facet-size", type=float, help="Lower clamp for generated facet sizes.")
    parser.add_argument("--max-facet-size", type=float, help="Upper clamp for generated facet sizes.")
    parser.add_argument("--paint-region-label", action="append", default=[], help="Facet region label to paint in aggregate region images; repeatable. Defaults to non-wall/inlet/outlet labels.")
    parser.add_argument("--no-region-paint", action="store_true", help="Skip aggregate region paint images.")
    parser.add_argument("--dry-run", action="store_true", help="Build the first iteration inputs and planned MMG3D command without running MMG3D.")
    parser.add_argument("--json", dest="json_path", help="Write iteration summary JSON to this path.")


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and (args_list[0] in COMMANDS or args_list[0] in {"-h", "--help"}):
        args = build_parser().parse_args(args_list)
    else:
        args = build_stats_parser().parse_args(args_list)
        args.command = "stats"

    if args.command == "stats":
        return _run_stats(args)
    if args.command == "remesh":
        return _run_remesh(args)
    if args.command == "compare":
        return _run_compare(args)
    if args.command == "evaluate":
        return _run_evaluate(args)
    if args.command == "size-field":
        return _run_size_field(args)
    if args.command == "export-vtu":
        return _run_export_vtu(args)
    if args.command == "render-vtu":
        return _run_render_vtu(args)
    if args.command == "render-regions":
        return _run_render_regions(args)
    if args.command == "plot-iterations":
        return _run_plot_iterations(args)
    if args.command == "transfer-regions":
        return _run_transfer_regions(args)
    if args.command == "init-regions":
        return _run_init_regions(args)
    if args.command == "demo-case":
        return _run_demo_case(args)
    if args.command == "mmg3d-workflow":
        return _run_mmg3d_workflow(args)
    if args.command == "iterate":
        return _run_iterate(args)
    raise ValueError(f"unknown command: {args.command}")


def _run_stats(args: argparse.Namespace) -> int:
    mesh = _load_mesh(args.mesh, backend=args.backend, facet_labels=args.facet_labels)
    build_histograms = args.hist_dir is not None or not args.no_histograms
    stats = MeshStatistics.from_mesh(mesh, bins=args.bins, include_histograms=build_histograms)
    payload: dict[str, Any] = stats.to_dict(include_histograms=not args.no_histograms)

    if args.hist_dir:
        payload["histogram_files"] = stats.save_histograms(args.hist_dir, dpi=args.dpi)
    if args.recommend_mmg3d:
        payload["mmg3d_recommendation"] = _recommend(mesh, args).to_dict()

    _write_or_print(payload, args.json_path)
    return 0


def _run_remesh(args: argparse.Namespace) -> int:
    hmin = args.hmin
    hsiz = args.hsiz
    hmax = args.hmax
    recommendation = None
    if args.auto or hmin is None or hsiz is None or hmax is None:
        mesh = _load_mesh(args.input_mesh, backend=args.backend, facet_labels=args.facet_labels)
        recommendation = _recommend(mesh, args)
        suggested = recommendation.suggested_args
        hmin = suggested["hmin"] if hmin is None else hmin
        hsiz = suggested["hsiz"] if hsiz is None else hsiz
        hmax = suggested["hmax"] if hmax is None else hmax
    elif args.write_sol:
        mesh = _load_mesh(args.input_mesh, backend=args.backend, facet_labels=args.facet_labels)

    sol_metadata = None
    if args.write_sol:
        if "mesh" not in locals():
            mesh = _load_mesh(args.input_mesh, backend=args.backend, facet_labels=args.facet_labels)
        default_size = hsiz
        if default_size is None:
            recommendation = recommendation or _recommend(mesh, args)
            default_size = recommendation.suggested_args["hsiz"]
        sol_metadata = _write_size_field(
            mesh,
            args.write_sol,
            default_size=default_size,
            facet_size_map_path=args.facet_size_map,
        )

    result = run_mmg3d(
        args.input_mesh,
        args.output_mesh,
        mmg3d_bin=args.mmg3d_bin,
        hmin=_adjust_hmin_for_facet_sizes(hmin, args.facet_size_map) if args.write_sol else hmin,
        hsiz=hsiz,
        hmax=hmax,
        sol_path=args.write_sol,
        extra_args=args.extra_arg,
        dry_run=args.dry_run,
    )
    payload: dict[str, Any] = {"mmg3d_run": result.to_dict()}
    if recommendation is not None:
        payload["mmg3d_recommendation"] = recommendation.to_dict()
    if sol_metadata is not None:
        payload["size_field"] = sol_metadata
    _write_or_print(payload, args.json_path)
    return 0 if result.succeeded else int(result.returncode or 1)


def _run_compare(args: argparse.Namespace) -> int:
    before = _load_mesh(args.before_mesh, backend=args.backend, facet_labels=args.facet_labels)
    after = _load_mesh(args.after_mesh, backend=args.backend, facet_labels=args.after_facet_labels or args.facet_labels)
    comparison = MeshComparison.from_stats(
        MeshStatistics.from_mesh(before, bins=args.bins, include_histograms=False),
        MeshStatistics.from_mesh(after, bins=args.bins, include_histograms=False),
    )
    _write_or_print(comparison.to_dict(include_mesh_stats=args.include_mesh_stats), args.json_path)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    before = _load_mesh(args.before_mesh, backend=args.backend, facet_labels=args.facet_labels)
    after = _load_mesh(args.after_mesh, backend=args.backend, facet_labels=args.after_facet_labels or args.facet_labels)
    facet_size_map = None if args.facet_size_map is None else load_facet_size_map(args.facet_size_map)
    evaluation = RemeshEvaluation.from_stats(
        MeshStatistics.from_mesh(before, bins=args.bins, include_histograms=False),
        MeshStatistics.from_mesh(after, bins=args.bins, include_histograms=False),
        target_elements=args.target_elements,
        facet_size_map=facet_size_map,
        tolerance=args.tolerance,
    )
    payload = evaluation.to_dict(include_mesh_stats=args.include_mesh_stats)
    if args.suggest_relaxation:
        diagnosis = evaluation.diagnose_constraints(max_relaxation=args.max_relaxation)
        payload["constraint_diagnosis"] = diagnosis.to_dict()
        if args.suggested_facet_size_map:
            _write_json_file(args.suggested_facet_size_map, diagnosis.suggested_facet_sizes)
    _write_or_print(payload, args.json_path)
    return 0


def _run_size_field(args: argparse.Namespace) -> int:
    mesh = _load_mesh(args.mesh, backend=args.backend, facet_labels=args.facet_labels)
    default_size = args.default_size
    recommendation = None
    if default_size is None:
        recommendation = _recommend(mesh, args)
        default_size = recommendation.suggested_args["hsiz"]
    metadata = _write_size_field(
        mesh,
        args.sol,
        default_size=default_size,
        facet_size_map_path=args.facet_size_map,
    )
    if recommendation is not None:
        metadata["mmg3d_recommendation"] = recommendation.to_dict()
    _write_or_print(metadata, args.json_path)
    return 0


def _run_export_vtu(args: argparse.Namespace) -> int:
    if args.mesh_vtu is None and args.facets_vtu is None:
        raise ValueError("provide at least one of --mesh-vtu or --facets-vtu")
    mesh = _load_mesh(args.mesh, backend=args.backend, facet_labels=args.facet_labels)
    result = export_vtu(mesh, mesh_path=args.mesh_vtu, facets_path=args.facets_vtu)
    _write_or_print(result.to_dict(), args.json_path)
    return 0


def _run_render_vtu(args: argparse.Namespace) -> int:
    result = render_vtu_png(
        args.vtu,
        args.png,
        scalar=args.scalar,
        view=args.view,
        show_edges=not args.no_edges,
        cmap=args.cmap,
        window_size=(args.width, args.height),
    )
    _write_or_print(result.to_dict(), args.json_path)
    return 0


def _run_render_regions(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.legend).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--legend must be a JSON object mapping labels to ids")
    result = render_regions_png(
        args.vtu,
        args.png,
        label_legend={str(label): int(value) for label, value in payload.items()},
        labels=args.label or None,
        view=args.view,
        show_edges=not args.no_edges,
        window_size=(args.width, args.height),
    )
    _write_or_print(result.to_dict(), args.json_path)
    return 0


def _run_plot_iterations(args: argparse.Namespace) -> int:
    series = IterationSeries.load(args.iterations)
    plot_path = series.save_plot(args.png, dpi=args.dpi)
    payload: dict[str, Any] = {
        "plot": str(plot_path),
        "iterations": [metric.to_dict() for metric in series.metrics],
    }
    if args.csv_path:
        payload["csv"] = str(series.write_csv(args.csv_path))
    _write_or_print(payload, args.json_path)
    return 0


def _run_transfer_regions(args: argparse.Namespace) -> int:
    if args.regions is None and args.source_facet_labels is None:
        raise ValueError("provide --regions or --source-facet-labels")
    source = _load_mesh(args.source_mesh, backend=args.backend, facet_labels=None if args.regions else args.source_facet_labels)
    target = MeshGeometry.load(args.target_mesh, backend=args.backend)
    region_spec = None
    if args.regions:
        region_spec = RegionSpecSet.load(args.regions)
        source = source.with_facet_labels(region_spec.facet_labels())
        region_set = RegionSet.from_spec(source, region_spec)
    else:
        priorities = None if args.region_priorities is None else _load_region_priorities(args.region_priorities)
        region_set = RegionSet.from_mesh(
            source,
            max_distance=args.region_max_distance,
            max_angle_deg=args.max_angle_deg,
            curve_max_distance=args.curve_max_distance,
            include_exterior_curves=args.include_exterior_curves,
            priorities=priorities,
        )
    labels = region_set.transfer_to(target, tolerance=args.tolerance, max_angle_deg=args.max_angle_deg)
    curve_labels = region_set.transfer_curves_to_facets(target, tolerance=args.tolerance)
    output = Path(args.output_labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(labels.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload: dict[str, Any] = {
        "output_labels": str(output),
        "labels": labels.to_dict(),
        "counts": {label: len(indices) for label, indices in labels.to_dict().items()},
        "curve_labels": curve_labels.to_dict(),
        "curve_counts": {label: len(indices) for label, indices in curve_labels.to_dict().items()},
        "regions": region_set.to_dict(),
    }
    if region_spec is not None:
        payload["region_spec"] = region_spec.to_dict()
    if args.output_curve_labels:
        curve_output = Path(args.output_curve_labels)
        curve_output.parent.mkdir(parents=True, exist_ok=True)
        curve_output.write_text(json.dumps(curve_labels.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["output_curve_labels"] = str(curve_output)
    if args.target_facets_vtu:
        target = target.with_facet_labels(labels)
        payload["vtk"] = export_vtu(target, facets_path=args.target_facets_vtu, curve_labels=curve_labels).to_dict()
    _write_or_print(payload, args.json_path)
    return 0


def _run_init_regions(args: argparse.Namespace) -> int:
    mesh = _load_mesh(args.mesh, backend=args.backend, facet_labels=args.facet_labels)
    if mesh.facet_labels is None:
        raise ValueError("init-regions requires --facet-labels or labels embedded in the mesh backend")
    max_distance = args.default_max_distance
    curve_max_distance = args.curve_max_distance
    median_facet_size = _median_facet_size(mesh)
    if max_distance is None:
        max_distance = median_facet_size * args.distance_scale
    if curve_max_distance is None:
        curve_max_distance = median_facet_size * args.curve_distance_scale
    spec = RegionSpecSet.from_facet_labels(
        mesh.facet_labels,
        max_distance=max_distance,
        max_angle_deg=args.default_max_angle_deg,
        priority=args.default_priority,
        auto_curves=RegionSpecSet.from_mapping(
            {
                "regions": {},
                "auto_curves": {
                    "from_surface_boundaries": args.auto_curves,
                    "include_exterior": args.include_exterior_curves,
                    "max_distance": curve_max_distance,
                },
            }
        ).auto_curves,
    )
    output = Path(args.regions)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = {
        "regions": str(output),
        "nregions": len(spec.regions),
        "median_facet_size": median_facet_size,
        "default_max_distance": max_distance,
        "curve_max_distance": curve_max_distance,
    }
    _write_or_print(payload, args.json_path)
    return 0


def _run_demo_case(args: argparse.Namespace) -> int:
    if args.kind == "complex":
        case = create_complex_channel(nx=args.nx, ny=args.ny, nz=args.nz)
    else:
        case = create_notched_channel(nx=args.nx, ny=args.ny, nz=args.nz)
    payload = case.write(args.output_dir)
    payload["kind"] = args.kind
    _write_or_print(payload, args.json_path)
    return 0


def _run_mmg3d_workflow(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    before_stats_path = output / "before_stats.json"
    metric_path = output / "metric.sol"
    remeshed_mesh = output / "remeshed.mesh"
    run_path = output / "mmg3d_run.json"
    transferred_labels = output / "remeshed_labels.json"
    transferred_curves = output / "remeshed_curve_labels.json"
    evaluation_path = output / "evaluation.json"
    diagnosis_path = output / "constraint_diagnosis.json"
    suggested_sizes_path = output / "suggested_facet_sizes.json"
    iteration_metrics_path = output / "iteration_metrics.json"
    before_mesh_vtu = output / "before_mesh.vtu"
    before_facets_vtu = output / "before_facets.vtu"
    after_mesh_vtu = output / "after_mesh.vtu"
    after_facets_vtu = output / "after_facets.vtu"
    before_mesh_png = output / "before_mesh_quality.png"
    before_facets_png = output / "before_facet_size.png"
    before_regions_png = output / "before_regions.png"
    after_mesh_png = output / "after_mesh_quality.png"
    after_facets_png = output / "after_facet_size.png"
    after_regions_png = output / "after_regions.png"
    region_legend_path = output / "region_legend.json"
    before_hist_dir = output / "before_histograms"
    after_hist_dir = output / "after_histograms"

    mesh = _load_mesh(args.input_mesh, backend=args.backend, facet_labels=args.facet_labels)
    before_stats = MeshStatistics.from_mesh(mesh, include_histograms=True)
    before_stats_path.write_text(json.dumps(before_stats.to_dict(include_histograms=False), indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    before_histograms = before_stats.save_histograms(str(before_hist_dir))
    before_vtk = export_vtu(mesh, mesh_path=before_mesh_vtu, facets_path=before_facets_vtu)
    _write_json_file(region_legend_path, before_vtk.label_legend)
    render_vtu_png(before_mesh_vtu, before_mesh_png, scalar="element_edge_aspect_ratio")
    render_vtu_png(before_facets_vtu, before_facets_png, scalar="facet_diameter")
    render_regions_png(before_facets_vtu, before_regions_png, label_legend=before_vtk.label_legend)

    recommendation = Mmg3dRecommendation.from_mesh(mesh, target_elements=args.target_elements)
    hsiz = recommendation.suggested_args["hsiz"]
    sol_metadata = _write_size_field(
        mesh,
        str(metric_path),
        default_size=hsiz,
        facet_size_map_path=args.facet_size_map,
    )
    mmg_run = run_mmg3d(
        args.input_mesh,
        remeshed_mesh,
        mmg3d_bin=args.mmg3d_bin,
        hmin=_adjust_hmin_for_facet_sizes(recommendation.suggested_args["hmin"], args.facet_size_map),
        hmax=recommendation.suggested_args["hmax"],
        sol_path=metric_path,
        dry_run=args.dry_run,
    )
    run_payload = {
        "mmg3d_run": mmg_run.to_dict(),
        "mmg3d_recommendation": recommendation.to_dict(),
        "size_field": sol_metadata,
    }
    run_path.write_text(json.dumps(run_payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")

    payload: dict[str, Any] = {
        "before_stats": str(before_stats_path),
        "metric": str(metric_path),
        "mmg3d_run": str(run_path),
        "before_mesh_vtu": str(before_mesh_vtu),
        "before_facets_vtu": str(before_facets_vtu),
        "before_mesh_png": str(before_mesh_png),
        "before_facets_png": str(before_facets_png),
        "before_regions_png": str(before_regions_png),
        "region_legend": str(region_legend_path),
        "before_histograms": before_histograms,
        "remeshed_mesh": str(remeshed_mesh),
    }
    if mmg_run.succeeded and not args.dry_run and remeshed_mesh.exists():
        source = _load_mesh(args.input_mesh, backend=args.backend, facet_labels=None)
        region_spec = RegionSpecSet.load(args.regions)
        source = source.with_facet_labels(region_spec.facet_labels())
        target = MeshGeometry.load(remeshed_mesh, backend=args.backend)
        region_set = RegionSet.from_spec(source, region_spec)
        labels = region_set.transfer_to(target)
        curve_labels = region_set.transfer_curves_to_facets(target)
        transferred_labels.write_text(json.dumps(labels.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        transferred_curves.write_text(json.dumps(curve_labels.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        target = target.with_facet_labels(labels)
        after_vtk = export_vtu(target, mesh_path=after_mesh_vtu, facets_path=after_facets_vtu, curve_labels=curve_labels)
        render_vtu_png(after_mesh_vtu, after_mesh_png, scalar="element_edge_aspect_ratio")
        render_vtu_png(after_facets_vtu, after_facets_png, scalar="facet_diameter")
        render_regions_png(after_facets_vtu, after_regions_png, label_legend=after_vtk.label_legend)
        after_stats = MeshStatistics.from_mesh(target, include_histograms=True)
        after_histograms = after_stats.save_histograms(str(after_hist_dir))
        evaluation = RemeshEvaluation.from_stats(
            before_stats,
            after_stats,
            target_elements=args.target_elements,
            facet_size_map=load_facet_size_map(args.facet_size_map),
        )
        diagnosis = evaluation.diagnose_constraints()
        evaluation_payload = evaluation.to_dict(include_mesh_stats=False)
        evaluation_payload["constraint_diagnosis"] = diagnosis.to_dict()
        evaluation_path.write_text(json.dumps(evaluation_payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
        diagnosis_path.write_text(json.dumps(diagnosis.to_dict(), indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
        suggested_sizes_path.write_text(json.dumps(diagnosis.suggested_facet_sizes, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
        iteration_metrics = IterationSeries.load([evaluation_path])
        iteration_metrics.write_json(iteration_metrics_path)
        payload.update(
            {
                "remeshed_labels": str(transferred_labels),
                "remeshed_curve_labels": str(transferred_curves),
                "after_mesh_vtu": str(after_mesh_vtu),
                "after_facets_vtu": str(after_facets_vtu),
                "after_mesh_png": str(after_mesh_png),
                "after_facets_png": str(after_facets_png),
                "after_regions_png": str(after_regions_png),
                "after_histograms": after_histograms,
                "evaluation": str(evaluation_path),
                "constraint_diagnosis": str(diagnosis_path),
                "suggested_facet_sizes": str(suggested_sizes_path),
                "iteration_metrics": str(iteration_metrics_path),
            }
        )
    _write_or_print(payload, args.json_path)
    return 0 if mmg_run.succeeded else int(mmg_run.returncode or 1)


def _run_iterate(args: argparse.Namespace) -> int:
    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive")
    if args.min_iterations <= 0 or args.min_iterations > args.max_iterations:
        raise ValueError("--min-iterations must be in [1, max-iterations]")
    if not (0.0 < args.relaxation_damping <= 1.0):
        raise ValueError("--relaxation-damping must be in (0, 1]")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    current_sizes = load_facet_size_map(args.facet_size_map)
    iteration_dirs: list[Path] = []
    iteration_records: list[dict[str, Any]] = []
    converged = False
    stop_reason = "max-iterations"
    return_code = 0

    for iteration in range(args.max_iterations):
        iteration_dir = output / f"iter_{iteration:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        input_sizes_path = iteration_dir / "input_facet_sizes.json"
        _write_json_file(input_sizes_path, current_sizes)
        iteration_input = {
            "iteration": iteration,
            "input_mesh": args.input_mesh,
            "target_elements": args.target_elements,
            "facet_size_map": str(input_sizes_path),
            "facet_sizes": current_sizes,
            "backend": args.backend,
            "mmg3d_bin": args.mmg3d_bin,
        }
        _write_json_file(iteration_dir / "iteration_input.json", iteration_input)

        workflow_args = argparse.Namespace(
            input_mesh=args.input_mesh,
            output_dir=str(iteration_dir),
            backend=args.backend,
            facet_labels=args.facet_labels,
            regions=args.regions,
            facet_size_map=str(input_sizes_path),
            target_elements=args.target_elements,
            mmg3d_bin=args.mmg3d_bin,
            dry_run=args.dry_run,
            json_path=str(iteration_dir / "workflow.json"),
        )
        code = _run_mmg3d_workflow(workflow_args)
        iteration_dirs.append(iteration_dir)
        record: dict[str, Any] = {
            "iteration": iteration,
            "directory": str(iteration_dir),
            "input_facet_sizes": str(input_sizes_path),
            "workflow": str(iteration_dir / "workflow.json"),
            "returncode": code,
        }
        if code != 0:
            stop_reason = "workflow-failed"
            return_code = code
            iteration_records.append(record)
            break
        if args.dry_run:
            stop_reason = "dry-run"
            iteration_records.append(record)
            break

        series = IterationSeries.load([iteration_dir])
        metric = series.metrics[0]
        converged, checks = _iteration_convergence_checks(metric, args)
        metric_payload = metric.to_dict()
        metric_payload["iteration"] = iteration
        record["metrics"] = metric_payload
        record["convergence"] = checks
        iteration_records.append(record)
        if converged and iteration + 1 >= args.min_iterations:
            stop_reason = "converged"
            break

        suggested_path = iteration_dir / "suggested_facet_sizes.json"
        if not suggested_path.exists():
            stop_reason = "missing-suggested-facet-sizes"
            return_code = 1
            break
        suggested_sizes = load_facet_size_map(suggested_path)
        next_sizes = _damped_facet_sizes(
            current_sizes,
            suggested_sizes,
            damping=args.relaxation_damping,
            min_size=args.min_facet_size,
            max_size=args.max_facet_size,
        )
        _write_json_file(iteration_dir / "next_facet_sizes.json", next_sizes)
        if next_sizes == current_sizes and iteration + 1 >= args.min_iterations:
            stop_reason = "facet-sizes-unchanged"
            break
        current_sizes = next_sizes

    successful_dirs = [path for path in iteration_dirs if (path / "evaluation.json").exists()]
    metrics_plot = None
    metrics_csv = None
    metrics_json = None
    aggregate_artifacts: dict[str, Any] = {}
    if successful_dirs:
        series = IterationSeries.load(successful_dirs)
        metrics_plot = output / "metrics_vs_iteration.png"
        metrics_csv = output / "metrics.csv"
        metrics_json = output / "metrics.json"
        series.save_plot(metrics_plot)
        series.write_csv(metrics_csv)
        series.write_json(metrics_json)
        aggregate_artifacts = _collect_iteration_artifacts(output, successful_dirs, paint_labels=args.paint_region_label, paint_regions=not args.no_region_paint)

    final_sizes_path = output / "final_facet_sizes.json"
    _write_json_file(final_sizes_path, current_sizes)
    summary = {
        "converged": converged,
        "stop_reason": stop_reason,
        "returncode": return_code,
        "iterations": iteration_records,
        "iteration_dirs": [str(path) for path in iteration_dirs],
        "successful_iteration_dirs": [str(path) for path in successful_dirs],
        "final_facet_sizes": str(final_sizes_path),
        "metrics_plot": None if metrics_plot is None else str(metrics_plot),
        "metrics_csv": None if metrics_csv is None else str(metrics_csv),
        "metrics_json": None if metrics_json is None else str(metrics_json),
        "artifacts": aggregate_artifacts,
        "optimizer_interface": {
            "inputs": ["input_facet_sizes.json", "iteration_input.json"],
            "outputs": ["evaluation.json", "constraint_diagnosis.json", "suggested_facet_sizes.json", "iteration_metrics.json"],
            "objective_hint": "minimize abs(element_ratio - 1), keep aspect_p95 bounded, keep facet_mean_ratio near 1",
        },
    }
    summary_path = output / "summary.json"
    summary["summary"] = str(summary_path)
    _write_json_file(summary_path, summary)
    _write_or_print(summary, args.json_path)
    return return_code


def _collect_iteration_artifacts(
    output: Path,
    iteration_dirs: list[Path],
    *,
    paint_labels: list[str],
    paint_regions: bool,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    if paint_regions:
        artifacts["region_paint"] = _write_region_paint_series(output, iteration_dirs, paint_labels=paint_labels)
    artifacts["selected_histograms"] = _collect_selected_histograms(output, iteration_dirs[-1])
    return artifacts


def _write_region_paint_series(output: Path, iteration_dirs: list[Path], *, paint_labels: list[str]) -> dict[str, Any]:
    region_dir = output / "region_paint"
    region_dir.mkdir(parents=True, exist_ok=True)
    first_legend = iteration_dirs[0] / "region_legend.json"
    if not first_legend.exists():
        return {"directory": str(region_dir), "files": [], "labels": []}
    legend_payload = json.loads(first_legend.read_text(encoding="utf-8"))
    if not isinstance(legend_payload, dict):
        return {"directory": str(region_dir), "files": [], "labels": []}
    labels = paint_labels or _default_paint_labels(legend_payload)
    legend_path = region_dir / "region_legend.json"
    _write_json_file(legend_path, {str(label): int(value) for label, value in legend_payload.items()})
    files: list[str] = []
    for index, iteration_dir in enumerate(iteration_dirs):
        facets_vtu = iteration_dir / "after_facets.vtu"
        if not facets_vtu.exists():
            continue
        output_png = region_dir / f"regions_iter{index:03d}.png"
        render_regions_png(facets_vtu, output_png, label_legend={str(label): int(value) for label, value in legend_payload.items()}, labels=labels)
        files.append(str(output_png))
    return {"directory": str(region_dir), "legend": str(legend_path), "labels": labels, "files": files}


def _collect_selected_histograms(output: Path, iteration_dir: Path) -> dict[str, Any]:
    selected_dir = output / "selected_histograms"
    selected_dir.mkdir(parents=True, exist_ok=True)
    after_hist = iteration_dir / "after_histograms"
    selected = {
        "after_facet_area": after_hist / "facet_measure.png",
        "after_mesh_aspect_ratio": after_hist / "element_edge_aspect_ratio.png",
        "after_element_volume": after_hist / "element_measure.png",
    }
    label_hist = after_hist / "facet_labels"
    for label in ("main_bore", "cross_bore", "offset_bore", "slot", "pocket", "notch"):
        path = label_hist / f"{label}_facet_measure.png"
        if path.exists():
            selected[f"after_{label}_facet_area"] = path

    copied: dict[str, str] = {}
    for name, source in selected.items():
        if not source.exists():
            continue
        destination = selected_dir / f"{name}.png"
        shutil.copyfile(source, destination)
        copied[name] = str(destination)
    return {"directory": str(selected_dir), "files": copied}


def _default_paint_labels(legend: dict[str, Any]) -> list[str]:
    background = {"wall", "inlet", "outlet"}
    return sorted(label for label in legend if label not in background)


def _iteration_convergence_checks(metric, args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    element_ok = metric.element_ratio is not None and abs(metric.element_ratio - 1.0) <= args.element_tolerance
    facet_mean_ok = metric.facet_mean_ratio is None or abs(metric.facet_mean_ratio - 1.0) <= args.facet_ratio_tolerance
    facet_p95_ok = metric.facet_p95_ratio is None or metric.facet_p95_ratio <= 1.0 + 2.0 * args.facet_ratio_tolerance
    quality_ok = args.max_aspect_p95 is None or (metric.aspect_p95 is not None and metric.aspect_p95 <= args.max_aspect_p95)
    checks = {
        "element_ok": element_ok,
        "facet_mean_ok": facet_mean_ok,
        "facet_p95_ok": facet_p95_ok,
        "quality_ok": quality_ok,
        "element_ratio": metric.element_ratio,
        "facet_mean_ratio": metric.facet_mean_ratio,
        "facet_p95_ratio": metric.facet_p95_ratio,
        "aspect_p95": metric.aspect_p95,
        "element_tolerance": args.element_tolerance,
        "facet_ratio_tolerance": args.facet_ratio_tolerance,
        "max_aspect_p95": args.max_aspect_p95,
    }
    return bool(element_ok and facet_mean_ok and facet_p95_ok and quality_ok), checks


def _damped_facet_sizes(
    current: dict[str, float],
    suggested: dict[str, float],
    *,
    damping: float,
    min_size: float | None,
    max_size: float | None,
) -> dict[str, float]:
    labels = sorted(set(current) | set(suggested))
    result: dict[str, float] = {}
    for label in labels:
        current_size = float(current.get(label, suggested[label]))
        suggested_size = float(suggested.get(label, current_size))
        value = current_size + damping * (suggested_size - current_size)
        if min_size is not None:
            value = max(value, min_size)
        if max_size is not None:
            value = min(value, max_size)
        result[label] = float(value)
    return result


def _recommend(mesh: MeshGeometry, args: argparse.Namespace) -> Mmg3dRecommendation:
    return Mmg3dRecommendation.from_mesh(
        mesh,
        lower_percentile=args.lower_percentile,
        target_percentile=args.target_percentile,
        upper_percentile=args.upper_percentile,
        target_elements=args.target_elements,
    )


def _load_mesh(mesh_path: str, *, backend: str, facet_labels: str | None) -> MeshGeometry:
    mesh = MeshGeometry.load(mesh_path, backend=backend)
    if facet_labels:
        mesh = mesh.with_facet_labels(_load_facet_labels(facet_labels))
    return mesh


def _median_facet_size(mesh: MeshGeometry) -> float:
    diameters = mesh.facet_diameters()
    finite = diameters[np.isfinite(diameters)]
    return float(np.median(finite)) if finite.size else 1.0


def _adjust_hmin_for_facet_sizes(hmin: float | None, facet_size_map_path: str | None) -> float | None:
    if facet_size_map_path is None:
        return hmin
    sizes = load_facet_size_map(facet_size_map_path)
    finite_sizes = [size for size in sizes.values() if np.isfinite(size) and size > 0.0]
    if not finite_sizes:
        return hmin
    facet_hmin = min(finite_sizes)
    if hmin is None:
        return facet_hmin
    return min(hmin, facet_hmin)


def _write_size_field(
    mesh: MeshGeometry,
    sol_path: str,
    *,
    default_size: float,
    facet_size_map_path: str | None,
) -> dict[str, Any]:
    facet_size_map = {} if facet_size_map_path is None else load_facet_size_map(facet_size_map_path)
    field = SizeField.from_mesh(mesh, default_size=default_size, facet_size_map=facet_size_map)
    written = field.write_mmg_sol(sol_path, dimension=mesh.dimension)
    return {
        "path": str(written),
        **field.to_dict(),
        "unmatched_facet_labels": sorted(set(facet_size_map) - set(mesh.facet_label_indices())),
    }


def _write_or_print(payload: dict[str, Any], json_path: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=True)
    if json_path:
        output = Path(json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _write_json_file(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def _load_facet_labels(path: str) -> FacetLabels:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("facet labels JSON must be an object mapping names to index arrays")
    return FacetLabels.from_mapping(payload) or FacetLabels({})


def _load_region_priorities(path: str) -> dict[str, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("region priorities JSON must be an object mapping labels to priorities")
    return {str(label): int(priority) for label, priority in payload.items()}


if __name__ == "__main__":
    raise SystemExit(main())
