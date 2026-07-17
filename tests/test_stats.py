import json

import numpy as np
import pytest

from mesh_metrics.cli import _damped_facet_sizes, _iteration_convergence_checks, _load_facet_labels
from mesh_metrics.compare import MeshComparison
from mesh_metrics.evaluate import RemeshEvaluation
from mesh_metrics.geometry import FacetLabels, MeshGeometry
from mesh_metrics.iterations import IterationSeries
from mesh_metrics.regions import CurveRegion, RegionSet, RegionSpecSet, SurfaceRegion
from mesh_metrics.remesh import Mmg3dRecommendation, run_mmg3d
from mesh_metrics.size_field import SizeField, load_facet_size_map
from mesh_metrics.stats import MeshStatistics
from mesh_metrics.visualize import export_vtu


def test_triangle_mesh_statistics():
    mesh = MeshGeometry(
        points=np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        elements=np.asarray([[0], [1], [2]]),
        facets=np.asarray([[0, 1, 2], [1, 2, 0]]),
    )

    stats = MeshStatistics.from_mesh(mesh, bins=4)

    assert stats.element_measure.count == 1
    assert stats.element_measure.total == 0.5
    assert stats.elements.measure.count == 1
    assert stats.elements.measure.total == 0.5
    assert stats.facet_measure.count == 3
    assert stats.facets.measure.count == 3
    assert np.isclose(stats.facet_measure.max, np.sqrt(2.0))
    assert stats.histograms["element_measure"].bins == 4


def test_tetrahedron_volume_and_facet_area():
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
    )

    stats = MeshStatistics.from_mesh(mesh)

    assert np.isclose(stats.element_measure.total, 1.0 / 6.0)
    assert stats.facet_measure.count == 4
    assert np.isclose(stats.facet_measure.min, 0.5)


def test_boundary_facets_are_inferred_for_tetrahedra():
    mesh = MeshGeometry(
        points=np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        elements=np.asarray([[0], [1], [2], [3]]),
    )

    assert mesh.nfacets == 4
    assert MeshStatistics.from_mesh(mesh).facet_measure.count == 4


def test_to_dict_contains_mesh_metadata():
    mesh = MeshGeometry(
        points=np.asarray([[0.0, 1.0], [0.0, 0.0]]),
        elements=np.asarray([[0], [1]]),
    )

    payload = MeshStatistics.from_mesh(mesh).to_dict()

    assert payload["mesh"]["dimension"] == 2
    assert payload["mesh"]["nelements"] == 1
    assert "element_diameter" in payload
    assert "mesh_quality" in payload
    assert "facet_size" in payload
    assert "element_edge_aspect_ratio" in payload["mesh_quality"]


def test_facet_label_statistics_are_reported():
    mesh = MeshGeometry(
        points=np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        elements=np.asarray([[0], [1], [2]]),
        facets=np.asarray([[0, 1, 2], [1, 2, 0]]),
        facet_labels=FacetLabels({"axis": np.asarray([0, 2]), "diagonal": np.asarray([1])}),
    )

    payload = MeshStatistics.from_mesh(mesh, bins=3).to_dict()

    assert payload["mesh"]["facet_labels"]["axis"] == [0, 2]
    assert payload["facets"]["labels"]["axis"]["measure"]["count"] == 2
    assert payload["facet_label_stats"]["axis"]["measure"]["count"] == 2
    assert np.isclose(payload["facet_label_stats"]["diagonal"]["measure"]["total"], np.sqrt(2.0))


def test_mesh_geometry_exposes_facet_geometry():
    mesh = MeshGeometry(
        points=np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        elements=np.asarray([[0], [1], [2]]),
        facets=np.asarray([[0, 1, 2], [1, 2, 0]]),
        facet_labels=FacetLabels({"edge": np.asarray([1])}),
    )

    facets = mesh.facet_geometry

    assert facets is not None
    assert facets.nfacets == 3
    assert facets.nodes_per_facet == 2
    assert facets.label_indices()["edge"].tolist() == [1]


def test_from_skfem_accepts_common_mesh_types():
    skfem = pytest.importorskip("skfem")
    mesh_factories = [
        skfem.MeshLine,
        skfem.MeshTri,
        skfem.MeshQuad,
        skfem.MeshTet,
        skfem.MeshHex,
    ]

    for factory in mesh_factories:
        sk_mesh = factory().refined(1)
        mesh = MeshGeometry.from_skfem(sk_mesh)
        stats = MeshStatistics.from_mesh(mesh)

        assert mesh.backend == "skfem"
        assert mesh.dimension == sk_mesh.p.shape[0]
        assert mesh.npoints == sk_mesh.p.shape[1]
        assert mesh.nelements == sk_mesh.t.shape[1]
        assert mesh.nfacets == sk_mesh.facets.shape[1]
        assert stats.elements.measure.count == mesh.nelements
        assert stats.facets.measure.count == mesh.nfacets
        assert np.isfinite(stats.elements.edge_aspect_ratio.mean)


def test_from_skfem_preserves_boundary_labels():
    skfem = pytest.importorskip("skfem")
    sk_mesh = skfem.MeshTri().refined(2).with_boundaries(
        {
            "left": lambda x: np.isclose(x[0], 0.0),
            "right": lambda x: np.isclose(x[0], 1.0),
        }
    )

    mesh = MeshGeometry.from_skfem(sk_mesh)
    stats = MeshStatistics.from_mesh(mesh)

    assert mesh.facet_labels is not None
    assert set(mesh.facet_labels.groups) == {"left", "right"}
    assert stats.facets.labels["left"].measure.count > 0
    assert stats.facets.labels["right"].diameter.count > 0


def test_from_object_accepts_fluxfem_style_row_major_arrays():
    class FluxStyleMesh:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        elements = np.asarray([[0, 1, 2, 3]])
        faces = np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        facet_labels = {"wall": [0, 1, 2], "outlet": [3]}

    mesh = MeshGeometry.from_object(FluxStyleMesh(), backend="fluxfem")
    stats = MeshStatistics.from_mesh(mesh)

    assert mesh.backend == "fluxfem"
    assert mesh.points.shape == (3, 4)
    assert mesh.elements.shape == (4, 1)
    assert mesh.facets is not None
    assert mesh.facets.shape == (3, 4)
    assert np.isclose(stats.elements.measure.total, 1.0 / 6.0)
    assert stats.facets.labels["wall"].measure.count == 3


def test_from_object_accepts_callable_fluxfem_style_accessors():
    class CallableFluxStyleMesh:
        boundary_labels = {"edge": [0, 2]}

        def coordinates(self):
            return np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        def connectivity(self):
            return np.asarray([[0, 1, 2]])

        def edges(self):
            return np.asarray([[0, 1], [1, 2], [2, 0]])

    mesh = MeshGeometry.from_object(CallableFluxStyleMesh(), backend="fluxfem")
    stats = MeshStatistics.from_mesh(mesh)

    assert mesh.points.shape == (2, 3)
    assert mesh.elements.shape == (3, 1)
    assert mesh.facets is not None
    assert mesh.facets.shape == (2, 3)
    assert stats.elements.measure.total == 0.5
    assert stats.facets.labels["edge"].diameter.count == 2


def test_from_object_preserves_t_connectivity_as_nodes_first():
    class TStyleMesh:
        points = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        t = np.asarray([[0], [1], [2]])
        facets = np.asarray([[0, 1, 2], [1, 2, 0]])

    mesh = MeshGeometry.from_object(TStyleMesh(), backend="fluxfem")

    assert mesh.elements.shape == (3, 1)
    assert mesh.facets is not None
    assert mesh.facets.shape == (2, 3)


def test_load_facet_labels_json(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"wall": [0, 2], "inlet": [1]}), encoding="utf-8")

    labels = _load_facet_labels(str(path))

    assert labels.to_dict() == {"wall": [0, 2], "inlet": [1]}


def test_mmg3d_recommendation_contains_size_window_and_label_targets():
    mesh = MeshGeometry(
        points=np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]),
        elements=np.asarray([[0], [1], [2]]),
        facets=np.asarray([[0, 1, 2], [1, 2, 0]]),
        facet_labels=FacetLabels({"base": np.asarray([0]), "long": np.asarray([1])}),
    )

    payload = Mmg3dRecommendation.from_mesh(mesh).to_dict()

    assert set(payload["suggested_args"]) == {"hmin", "hsiz", "hmax"}
    assert payload["facet_label_targets"]["base"]["count"] == 1
    assert payload["mmg3d_hint"].startswith("mmg3d input.mesh output.mesh")


def test_mmg3d_recommendation_can_target_element_count():
    mesh = MeshGeometry(
        points=np.asarray([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]),
        elements=np.asarray([[0, 1], [1, 2]]),
    )

    payload = Mmg3dRecommendation.from_mesh(mesh, target_elements=8).to_dict()

    assert payload["element_count_target"]["current_elements"] == 2
    assert payload["element_count_target"]["target_elements"] == 8
    assert payload["element_count_target"]["size_scale"] == 0.5
    assert payload["suggested_args"]["hsiz"] == payload["element_count_target"]["recommended_hsiz"]


def test_mmg3d_dry_run_builds_command():
    result = run_mmg3d(
        "in.mesh",
        "out.mesh",
        hmin=0.1,
        hsiz=0.2,
        hmax=0.4,
        extra_args=["-hausd", "0.01"],
        dry_run=True,
    )

    assert result.succeeded
    assert result.command == ["mmg3d", "in.mesh", "out.mesh", "-hmin", "0.10000000000000001", "-hsiz", "0.20000000000000001", "-hmax", "0.40000000000000002", "-hausd", "0.01"]


def test_mmg3d_dry_run_can_use_sol_file():
    result = run_mmg3d("in.mesh", "out.mesh", hsiz=0.2, sol_path="metric.sol", dry_run=True)

    assert result.command == ["mmg3d", "in.mesh", "out.mesh", "-sol", "metric.sol"]


def test_size_field_uses_smallest_facet_label_size(tmp_path):
    mesh = MeshGeometry(
        points=np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        elements=np.asarray([[0], [1], [2]]),
        facets=np.asarray([[0, 1, 2], [1, 2, 0]]),
        facet_labels=FacetLabels({"wall": np.asarray([0]), "inlet": np.asarray([0, 1])}),
    )

    field = SizeField.from_mesh(mesh, default_size=0.5, facet_size_map={"wall": 0.1, "inlet": 0.2})
    sol_path = field.write_mmg_sol(tmp_path / "metric.sol", dimension=mesh.dimension)

    assert field.values.tolist() == [0.1, 0.1, 0.2]
    text = sol_path.read_text(encoding="ascii")
    assert "SolAtVertices" in text
    assert "1 1" in text


def test_load_facet_size_map(tmp_path):
    path = tmp_path / "sizes.json"
    path.write_text(json.dumps({"wall": 0.1, "farfield": 0.5}), encoding="utf-8")

    assert load_facet_size_map(path) == {"wall": 0.1, "farfield": 0.5}


def test_mesh_comparison_reports_deltas():
    before_mesh = MeshGeometry(
        points=np.asarray([[0.0, 1.0], [0.0, 0.0]]),
        elements=np.asarray([[0], [1]]),
    )
    after_mesh = MeshGeometry(
        points=np.asarray([[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]]),
        elements=np.asarray([[0, 1], [1, 2]]),
    )

    comparison = MeshComparison.from_stats(
        MeshStatistics.from_mesh(before_mesh),
        MeshStatistics.from_mesh(after_mesh),
    ).to_dict()

    assert comparison["mesh"]["nelements_delta"] == 1
    assert comparison["mesh"]["nelements_ratio"] == 2.0
    assert comparison["element_diameter"]["mean_delta"] == -0.5
    assert "element_edge_aspect_ratio" in comparison["mesh_quality"]


def test_remesh_evaluation_reports_targets():
    before_mesh = MeshGeometry(
        points=np.asarray([[0.0, 1.0], [0.0, 0.0]]),
        elements=np.asarray([[0], [1]]),
        facets=np.asarray([[0], [1]]),
        facet_labels=FacetLabels({"wall": np.asarray([0])}),
    )
    after_mesh = MeshGeometry(
        points=np.asarray([[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]]),
        elements=np.asarray([[0, 1], [1, 2]]),
        facets=np.asarray([[0, 1], [1, 2]]),
        facet_labels=FacetLabels({"wall": np.asarray([0, 1])}),
    )

    payload = RemeshEvaluation.from_stats(
        MeshStatistics.from_mesh(before_mesh),
        MeshStatistics.from_mesh(after_mesh),
        target_elements=4,
        facet_size_map={"wall": 0.5, "missing": 1.0},
        tolerance=0.1,
    ).to_dict()

    assert payload["target_elements"]["actual"] == 2
    assert payload["target_elements"]["ratio"] == 0.5
    assert payload["facet_size_targets"]["wall"]["mean_ratio"] == 1.0
    assert payload["facet_size_targets"]["wall"]["too_large_count"] == 0
    assert payload["unmatched_facet_size_labels"] == ["missing"]
    assert "max_ratio" in payload["mesh_quality"]


def test_constraint_diagnosis_suggests_relaxed_facet_sizes():
    before_mesh = MeshGeometry(
        points=np.asarray([[0.0, 1.0], [0.0, 0.0]]),
        elements=np.asarray([[0], [1]]),
        facets=np.asarray([[0], [1]]),
        facet_labels=FacetLabels({"wall": np.asarray([0])}),
    )
    after_mesh = MeshGeometry(
        points=np.asarray([[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]]),
        elements=np.asarray([[0, 1], [1, 2]]),
        facets=np.asarray([[0, 1], [1, 2]]),
        facet_labels=FacetLabels({"wall": np.asarray([0, 1])}),
    )
    evaluation = RemeshEvaluation.from_stats(
        MeshStatistics.from_mesh(before_mesh),
        MeshStatistics.from_mesh(after_mesh),
        target_elements=1,
        facet_size_map={"wall": 0.25},
    )

    diagnosis = evaluation.diagnose_constraints().to_dict()

    assert diagnosis["element_ratio"] == 2.0
    assert diagnosis["suggested_facet_sizes"]["wall"] > 0.25
    assert diagnosis["facet_suggestions"]["wall"]["priority"] in {"relax", "relax-carefully"}


def test_iteration_series_extracts_and_plots_metrics(tmp_path):
    first = tmp_path / "iter_000"
    second = tmp_path / "iter_001"
    first.mkdir()
    second.mkdir()
    payloads = [
        {
            "target_elements": {"target": 10, "actual": 20, "ratio": 2.0, "error_percent": 100.0},
            "mesh_quality": {"after_p95": 1.5, "after_max": 2.0, "p95_ratio": 0.9, "max_ratio": 1.1},
            "facet_size_targets": {
                "wall": {"mean_ratio": 1.2, "p95_ratio": 1.5, "too_small_fraction": 0.1, "too_large_fraction": 0.2}
            },
        },
        {
            "target_elements": {"target": 10, "actual": 12, "ratio": 1.2, "error_percent": 20.0},
            "mesh_quality": {"after_p95": 1.3, "after_max": 1.8, "p95_ratio": 0.8, "max_ratio": 0.95},
            "facet_size_targets": {
                "wall": {"mean_ratio": 1.05, "p95_ratio": 1.2, "too_small_fraction": 0.05, "too_large_fraction": 0.1}
            },
        },
    ]
    (first / "evaluation.json").write_text(json.dumps(payloads[0]), encoding="utf-8")
    (second / "evaluation.json").write_text(json.dumps(payloads[1]), encoding="utf-8")

    series = IterationSeries.load([first, second])
    plot = series.save_plot(tmp_path / "metrics.png")
    csv_path = series.write_csv(tmp_path / "metrics.csv")

    assert [metric.actual_elements for metric in series.metrics] == [20, 12]
    assert series.metrics[1].facet_mean_ratio == 1.05
    assert plot.exists()
    assert csv_path.read_text(encoding="utf-8").startswith("iteration,source,target_elements")


def test_iterate_helpers_damp_sizes_and_check_convergence():
    current = {"wall": 0.2, "hole": 0.1}
    suggested = {"wall": 0.4, "hole": 0.2}

    damped = _damped_facet_sizes(current, suggested, damping=0.5, min_size=0.12, max_size=0.35)

    assert damped == {"hole": 0.15000000000000002, "wall": 0.30000000000000004}

    class FakeMetric:
        element_ratio = 1.05
        facet_mean_ratio = 1.02
        facet_p95_ratio = 1.2
        aspect_p95 = 2.0

    class FakeArgs:
        element_tolerance = 0.1
        facet_ratio_tolerance = 0.15
        max_aspect_p95 = 2.5

    converged, checks = _iteration_convergence_checks(FakeMetric(), FakeArgs())

    assert converged
    assert checks["element_ok"]


def test_region_set_transfers_source_facet_labels_to_remeshed_facets():
    source = MeshGeometry(
        points=np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        elements=np.asarray([[0], [1], [2]]),
        facets=np.asarray([[0], [1]]),
        facet_labels=FacetLabels({"bottom": np.asarray([0])}),
    )
    target = MeshGeometry(
        points=np.asarray([[0.0, 0.5, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        elements=np.asarray([[0, 1], [1, 2], [3, 3]]),
        facets=np.asarray([[0, 1], [1, 2]]),
    )

    labels = RegionSet.from_mesh(source).transfer_to(target, tolerance=1.0e-12)

    assert labels.to_dict() == {"bottom": [0, 1]}


def test_surface_region_uses_normal_angle_to_avoid_nearby_wrong_surface():
    source = MeshGeometry(
        points=np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        elements=np.asarray([[0], [1], [2], [3]]),
        facets=np.asarray([[0, 0], [1, 1], [2, 3]]),
        facet_labels=FacetLabels({"xy": np.asarray([0]), "xz": np.asarray([1])}),
    )
    target = MeshGeometry(
        points=np.asarray(
            [
                [0.1, 0.8, 0.1],
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.8],
            ]
        ),
        elements=np.asarray([[0], [1], [2]]),
        facets=np.asarray([[0], [1], [2]]),
    )

    labels = RegionSet.from_mesh(source, max_angle_deg=10.0).transfer_to(target, tolerance=0.5, max_angle_deg=10.0)

    assert labels.to_dict() == {"xy": [], "xz": [0]}


def test_surface_region_priority_breaks_ties():
    points = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    facets = np.asarray([[0], [1], [2]])
    low = SurfaceRegion("low", np.asarray([0]), points, facets, max_distance=1.0, priority=0)
    high = SurfaceRegion("high", np.asarray([0]), points, facets, max_distance=1.0, priority=10)
    target = MeshGeometry(points=points, elements=np.asarray([[0], [1], [2]]), facets=facets)

    labels = RegionSet({"low": low, "high": high}).transfer_to(target, tolerance=1.0)

    assert labels.to_dict() == {"low": [], "high": [0]}


def test_curve_regions_are_created_from_facet_label_boundaries():
    mesh = MeshGeometry(
        points=np.asarray(
            [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        elements=np.asarray([[0], [1], [2]]),
        facets=np.asarray([[0, 1], [1, 3], [2, 2]]),
        facet_labels=FacetLabels({"left": np.asarray([0]), "right": np.asarray([1])}),
    )

    region_set = RegionSet.from_mesh(mesh)

    assert "left__right" in region_set.curves
    assert region_set.curves["left__right"].edges.T.tolist() == [[1, 2]]


def test_curve_region_transfers_to_target_facets_near_feature_edge():
    source = MeshGeometry(
        points=np.asarray(
            [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        elements=np.asarray([[0], [1], [2]]),
        facets=np.asarray([[0, 1], [1, 3], [2, 2]]),
        facet_labels=FacetLabels({"left": np.asarray([0]), "right": np.asarray([1])}),
    )
    target = MeshGeometry(
        points=np.asarray(
            [
                [0.0, 1.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 1.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        elements=np.asarray([[0], [1], [2]]),
        facets=np.asarray([[1, 4], [4, 3], [2, 2]]),
    )

    labels = RegionSet.from_mesh(source, curve_max_distance=0.1).transfer_curves_to_facets(target, tolerance=0.1)

    assert labels.to_dict() == {"left__right": [0, 1]}


def test_curve_region_priority_breaks_ties():
    points = np.asarray(
        [
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    edges = np.asarray([[0], [1]])
    low = CurveRegion("low", edges, points, max_distance=1.0, priority=0)
    high = CurveRegion("high", edges, points, max_distance=1.0, priority=10)
    target = MeshGeometry(points=points, elements=np.asarray([[0], [1]]), facets=edges)

    labels = RegionSet({}, {"low": low, "high": high}).transfer_curves_to_facets(target, tolerance=1.0)

    assert labels.to_dict() == {"low": [], "high": [0]}


def test_region_spec_set_builds_surface_and_curve_regions(tmp_path):
    path = tmp_path / "regions.json"
    path.write_text(
        json.dumps(
            {
                "regions": {
                    "left": {
                        "type": "surface",
                        "facet_indices": [0],
                        "max_distance": 0.1,
                        "max_angle_deg": 20,
                        "priority": 5,
                    },
                    "explicit_edge": {
                        "type": "curve",
                        "edge_nodes": [[1, 2]],
                        "max_distance": 0.05,
                        "priority": 10,
                    },
                },
                "auto_curves": {
                    "from_surface_boundaries": False,
                    "include_exterior": False,
                },
            }
        ),
        encoding="utf-8",
    )
    mesh = MeshGeometry(
        points=np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        elements=np.asarray([[0], [1], [2]]),
        facets=np.asarray([[0], [1], [2]]),
    )

    spec = RegionSpecSet.load(path)
    region_set = RegionSet.from_spec(mesh, spec)

    assert spec.facet_labels().to_dict() == {"left": [0]}
    assert region_set.regions["left"].max_angle_deg == 20
    assert region_set.curves["explicit_edge"].priority == 10


def test_region_spec_auto_curves_from_surface_boundaries():
    mesh = MeshGeometry(
        points=np.asarray(
            [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        elements=np.asarray([[0], [1], [2]]),
        facets=np.asarray([[0, 1], [1, 3], [2, 2]]),
    )
    spec = RegionSpecSet.from_mapping(
        {
            "regions": {
                "left": {"type": "surface", "facet_indices": [0]},
                "right": {"type": "surface", "facet_indices": [1]},
            },
            "auto_curves": {"from_surface_boundaries": True},
        }
    )

    region_set = RegionSet.from_spec(mesh, spec)

    assert "left__right" in region_set.curves


def test_region_spec_can_be_initialized_from_facet_labels():
    labels = FacetLabels({"wall": np.asarray([0, 2]), "inlet": np.asarray([1])})

    spec = RegionSpecSet.from_facet_labels(labels, max_distance=0.02, max_angle_deg=30.0, priority=3)
    payload = spec.to_dict()

    assert payload["regions"]["wall"]["facet_indices"] == [0, 2]
    assert payload["regions"]["wall"]["max_distance"] == 0.02
    assert payload["regions"]["wall"]["max_angle_deg"] == 30.0
    assert payload["regions"]["wall"]["priority"] == 3
    assert payload["auto_curves"]["from_surface_boundaries"]


def test_export_vtu_writes_mesh_and_facets(tmp_path):
    mesh = MeshGeometry(
        points=np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        elements=np.asarray([[0], [1], [2], [3]]),
        facet_labels=FacetLabels({"surface": np.asarray([0, 1, 2, 3])}),
    )

    curve_labels = FacetLabels({"edge": np.asarray([0])})
    result = export_vtu(mesh, mesh_path=tmp_path / "mesh.vtu", facets_path=tmp_path / "facets.vtu", curve_labels=curve_labels)

    assert (tmp_path / "mesh.vtu").exists()
    assert (tmp_path / "facets.vtu").exists()
    assert result.label_legend == {"surface": 0}
    assert result.curve_label_legend == {"edge": 0}
