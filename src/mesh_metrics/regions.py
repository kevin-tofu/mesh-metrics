from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from mesh_metrics.geometry import FacetLabels, MeshGeometry


@dataclass(frozen=True)
class RegionSpec:
    label: str
    type: str
    facet_indices: NDArray[np.int64] | None = None
    edge_nodes: NDArray[np.int64] | None = None
    max_distance: float | None = None
    max_angle_deg: float | None = None
    priority: int = 0

    @classmethod
    def from_mapping(cls, label: str, payload: dict[str, Any]) -> RegionSpec:
        region_type = str(payload.get("type", "surface"))
        facet_indices = payload.get("facet_indices")
        edge_nodes = payload.get("edge_nodes")
        return cls(
            label=label,
            type=region_type,
            facet_indices=None if facet_indices is None else np.asarray(facet_indices, dtype=np.int64),
            edge_nodes=None if edge_nodes is None else np.asarray(edge_nodes, dtype=np.int64),
            max_distance=None if payload.get("max_distance") is None else float(payload["max_distance"]),
            max_angle_deg=None if payload.get("max_angle_deg") is None else float(payload["max_angle_deg"]),
            priority=int(payload.get("priority", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "max_distance": self.max_distance,
            "priority": self.priority,
        }
        if self.facet_indices is not None:
            result["facet_indices"] = self.facet_indices.tolist()
        if self.edge_nodes is not None:
            result["edge_nodes"] = self.edge_nodes.tolist()
        if self.max_angle_deg is not None:
            result["max_angle_deg"] = self.max_angle_deg
        return result


@dataclass(frozen=True)
class AutoCurveSpec:
    from_surface_boundaries: bool = True
    include_exterior: bool = False
    max_distance: float | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> AutoCurveSpec:
        if payload is None:
            return cls(from_surface_boundaries=True)
        return cls(
            from_surface_boundaries=bool(payload.get("from_surface_boundaries", True)),
            include_exterior=bool(payload.get("include_exterior", False)),
            max_distance=None if payload.get("max_distance") is None else float(payload["max_distance"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_surface_boundaries": self.from_surface_boundaries,
            "include_exterior": self.include_exterior,
            "max_distance": self.max_distance,
        }


@dataclass(frozen=True)
class RegionSpecSet:
    regions: dict[str, RegionSpec]
    auto_curves: AutoCurveSpec = field(default_factory=AutoCurveSpec)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> RegionSpecSet:
        raw_regions = payload.get("regions", payload)
        if not isinstance(raw_regions, dict):
            raise ValueError("region spec must contain a regions object")
        return cls(
            regions={
                str(label): RegionSpec.from_mapping(str(label), spec)
                for label, spec in raw_regions.items()
                if isinstance(spec, dict)
            },
            auto_curves=AutoCurveSpec.from_mapping(payload.get("auto_curves") if "regions" in payload else None),
        )

    @classmethod
    def load(cls, path: str | Path) -> RegionSpecSet:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("region spec JSON must be an object")
        return cls.from_mapping(payload)

    @classmethod
    def from_facet_labels(
        cls,
        labels: FacetLabels,
        *,
        max_distance: float | None = None,
        max_angle_deg: float = 45.0,
        priority: int = 0,
        auto_curves: AutoCurveSpec | None = None,
    ) -> RegionSpecSet:
        return cls(
            regions={
                label: RegionSpec(
                    label=label,
                    type="surface",
                    facet_indices=indices,
                    max_distance=max_distance,
                    max_angle_deg=max_angle_deg,
                    priority=priority,
                )
                for label, indices in labels.groups.items()
            },
            auto_curves=auto_curves or AutoCurveSpec(from_surface_boundaries=True),
        )

    def facet_labels(self) -> FacetLabels:
        groups = {
            label: spec.facet_indices
            for label, spec in self.regions.items()
            if spec.type == "surface" and spec.facet_indices is not None
        }
        return FacetLabels({label: indices for label, indices in groups.items()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "regions": {
                label: spec.to_dict()
                for label, spec in self.regions.items()
            },
            "auto_curves": self.auto_curves.to_dict(),
        }


@dataclass(frozen=True)
class SurfaceRegion:
    label: str
    facet_indices: NDArray[np.int64]
    points: NDArray[np.float64]
    facets: NDArray[np.int64]
    max_distance: float | None = None
    max_angle_deg: float = 45.0
    priority: int = 0

    def __post_init__(self) -> None:
        indices = np.asarray(self.facet_indices, dtype=np.int64)
        points = np.asarray(self.points, dtype=float)
        facets = np.asarray(self.facets, dtype=np.int64)
        centroids = np.asarray([points[:, facets[:, index]].mean(axis=1) for index in indices], dtype=float)
        normals = np.asarray([_facet_normal(points[:, facets[:, index]]) for index in indices], dtype=float)
        tree = cKDTree(centroids) if centroids.size else None
        object.__setattr__(self, "facet_indices", indices)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "facets", facets)
        object.__setattr__(self, "centroids", centroids)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "_tree", tree)

    def match_score(
        self,
        point: NDArray[np.float64],
        normal: NDArray[np.float64] | None,
        *,
        fallback_distance: float,
    ) -> tuple[float, float] | None:
        if self.facet_indices.size == 0 or self._tree is None:
            return None
        max_distance = self.max_distance if self.max_distance is not None else fallback_distance
        candidate_positions = self._candidate_positions(point, max_distance)
        best: tuple[float, float] | None = None
        for position in candidate_positions:
            source_index = int(self.facet_indices[position])
            distance = _point_to_facet_distance(point, self.points[:, self.facets[:, source_index]])
            if distance > max_distance:
                continue
            angle = _normal_angle_deg(normal, self.normals[position])
            if angle > self.max_angle_deg:
                continue
            score = (distance, angle)
            if best is None or score < best:
                best = score
        return best

    def _candidate_positions(self, point: NDArray[np.float64], max_distance: float) -> list[int]:
        assert self._tree is not None
        if max_distance > 0.0 and np.isfinite(max_distance):
            positions = self._tree.query_ball_point(point, r=max_distance)
            if positions:
                return [int(position) for position in positions]
        _, position = self._tree.query(point, k=1)
        return [int(position)]

    def distance_to_point(self, point: NDArray[np.float64]) -> float:
        score = self.match_score(point, None, fallback_distance=float("inf"))
        return float("inf") if score is None else score[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "type": "surface",
            "facet_indices": self.facet_indices.tolist(),
            "nfacets": int(self.facet_indices.size),
            "max_distance": self.max_distance,
            "max_angle_deg": self.max_angle_deg,
            "priority": self.priority,
        }


Region = SurfaceRegion


@dataclass(frozen=True)
class CurveRegion:
    label: str
    edges: NDArray[np.int64]
    points: NDArray[np.float64]
    max_distance: float | None = None
    priority: int = 0

    def __post_init__(self) -> None:
        edges = np.asarray(self.edges, dtype=np.int64)
        points = np.asarray(self.points, dtype=float)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("curve edges must have shape (2, nedges)")
        midpoints = np.asarray([(points[:, edge[0]] + points[:, edge[1]]) * 0.5 for edge in edges.T], dtype=float)
        tree = cKDTree(midpoints) if midpoints.size else None
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "midpoints", midpoints)
        object.__setattr__(self, "_tree", tree)

    def match_score(self, point: NDArray[np.float64], *, fallback_distance: float) -> float | None:
        if self.edges.shape[1] == 0 or self._tree is None:
            return None
        max_distance = self.max_distance if self.max_distance is not None else fallback_distance
        candidate_positions = self._candidate_positions(point, max_distance)
        best: float | None = None
        for position in candidate_positions:
            edge = self.edges[:, position]
            distance = _point_to_segment_distance(point, self.points[:, edge[0]], self.points[:, edge[1]])
            if distance > max_distance:
                continue
            if best is None or distance < best:
                best = distance
        return best

    def _candidate_positions(self, point: NDArray[np.float64], max_distance: float) -> list[int]:
        assert self._tree is not None
        if max_distance > 0.0 and np.isfinite(max_distance):
            positions = self._tree.query_ball_point(point, r=max_distance)
            if positions:
                return [int(position) for position in positions]
        _, position = self._tree.query(point, k=1)
        return [int(position)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "type": "curve",
            "edge_nodes": self.edges.T.tolist(),
            "nedges": int(self.edges.shape[1]),
            "max_distance": self.max_distance,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class RegionSet:
    regions: dict[str, SurfaceRegion]
    curves: dict[str, CurveRegion] = field(default_factory=dict)

    @classmethod
    def from_mesh(
        cls,
        mesh: MeshGeometry,
        *,
        labels: FacetLabels | None = None,
        max_distance: float | None = None,
        max_angle_deg: float = 45.0,
        curve_max_distance: float | None = None,
        include_exterior_curves: bool = False,
        priorities: dict[str, int] | None = None,
    ) -> RegionSet:
        if mesh.facets is None:
            raise ValueError("mesh has no facets")
        facet_labels = labels or mesh.facet_labels
        if facet_labels is None:
            raise ValueError("mesh has no facet labels")
        priority_map = {} if priorities is None else priorities
        surface_regions = {
                label: SurfaceRegion(
                    label=label,
                    facet_indices=indices,
                    points=mesh.points,
                    facets=mesh.facets,
                    max_distance=max_distance,
                    max_angle_deg=max_angle_deg,
                    priority=int(priority_map.get(label, 0)),
                )
                for label, indices in facet_labels.groups.items()
        }
        curve_regions = _curve_regions_from_facet_labels(
            mesh,
            facet_labels,
            max_distance=curve_max_distance,
            include_exterior=include_exterior_curves,
            priorities=priority_map,
        )
        return cls(surface_regions, curve_regions)

    @classmethod
    def from_spec(cls, mesh: MeshGeometry, spec_set: RegionSpecSet) -> RegionSet:
        if mesh.facets is None:
            raise ValueError("mesh has no facets")
        surfaces: dict[str, SurfaceRegion] = {}
        curves: dict[str, CurveRegion] = {}
        surface_label_groups: dict[str, NDArray[np.int64]] = {}
        for label, spec in spec_set.regions.items():
            if spec.type == "surface":
                if spec.facet_indices is None:
                    raise ValueError(f"surface region {label!r} requires facet_indices")
                surfaces[label] = SurfaceRegion(
                    label=label,
                    facet_indices=spec.facet_indices,
                    points=mesh.points,
                    facets=mesh.facets,
                    max_distance=spec.max_distance,
                    max_angle_deg=45.0 if spec.max_angle_deg is None else spec.max_angle_deg,
                    priority=spec.priority,
                )
                surface_label_groups[label] = spec.facet_indices
            elif spec.type == "curve":
                if spec.edge_nodes is None:
                    raise ValueError(f"curve region {label!r} requires edge_nodes")
                edge_nodes = _edge_nodes_to_connectivity(spec.edge_nodes)
                curves[label] = CurveRegion(
                    label=label,
                    edges=edge_nodes,
                    points=mesh.points,
                    max_distance=spec.max_distance,
                    priority=spec.priority,
                )
            else:
                raise ValueError(f"unsupported region type {spec.type!r} for {label!r}")

        if spec_set.auto_curves.from_surface_boundaries and surface_label_groups:
            auto_curves = _curve_regions_from_facet_labels(
                mesh,
                FacetLabels(surface_label_groups),
                max_distance=spec_set.auto_curves.max_distance,
                include_exterior=spec_set.auto_curves.include_exterior,
                priorities={label: spec.priority for label, spec in spec_set.regions.items()},
            )
            curves = {**auto_curves, **curves}
        return cls(surfaces, curves)

    def transfer_to(
        self,
        mesh: MeshGeometry,
        *,
        tolerance: float | None = None,
        max_angle_deg: float | None = None,
    ) -> FacetLabels:
        if mesh.facets is None:
            raise ValueError("target mesh has no facets")
        fallback_distance = _default_tolerance(mesh) if tolerance is None else tolerance

        groups: dict[str, list[int]] = {label: [] for label in self.regions}
        for facet_index, facet in enumerate(mesh.facets.T):
            vertices = mesh.points[:, facet]
            centroid = vertices.mean(axis=1)
            normal = _facet_normal(vertices)
            match = self._best_region(centroid, normal, fallback_distance=fallback_distance, max_angle_deg=max_angle_deg)
            if match is not None:
                groups[match].append(facet_index)

        return FacetLabels({label: np.asarray(indices, dtype=np.int64) for label, indices in groups.items()})

    def transfer_curves_to_facets(self, mesh: MeshGeometry, *, tolerance: float | None = None) -> FacetLabels:
        if mesh.facets is None:
            raise ValueError("target mesh has no facets")
        fallback_distance = _default_tolerance(mesh) if tolerance is None else tolerance
        groups: dict[str, list[int]] = {label: [] for label in self.curves}
        for facet_index, facet in enumerate(mesh.facets.T):
            best_label = None
            best_key: tuple[int, float] | None = None
            for point in _facet_edge_midpoints(mesh.points[:, facet]).T:
                for label, curve in self.curves.items():
                    distance = curve.match_score(point, fallback_distance=fallback_distance)
                    if distance is None:
                        continue
                    key = (-curve.priority, distance)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_label = label
            if best_label is not None:
                groups[best_label].append(facet_index)
        return FacetLabels({label: np.asarray(indices, dtype=np.int64) for label, indices in groups.items()})

    def _best_region(
        self,
        point: NDArray[np.float64],
        normal: NDArray[np.float64],
        *,
        fallback_distance: float,
        max_angle_deg: float | None,
    ) -> str | None:
        best_label = None
        best_key: tuple[int, float, float] | None = None
        for label, region in self.regions.items():
            effective_region = region
            if max_angle_deg is not None and max_angle_deg != region.max_angle_deg:
                effective_region = SurfaceRegion(
                    label=region.label,
                    facet_indices=region.facet_indices,
                    points=region.points,
                    facets=region.facets,
                    max_distance=region.max_distance,
                    max_angle_deg=max_angle_deg,
                    priority=region.priority,
                )
            score = effective_region.match_score(point, normal, fallback_distance=fallback_distance)
            if score is None:
                continue
            distance, angle = score
            key = (-effective_region.priority, distance, angle)
            if best_key is None or key < best_key:
                best_key = key
                best_label = label
        return best_label

    def to_dict(self) -> dict[str, Any]:
        return {
            "regions": {
                label: region.to_dict()
                for label, region in self.regions.items()
            },
            "curves": {
                label: curve.to_dict()
                for label, curve in self.curves.items()
            },
        }


def _default_tolerance(mesh: MeshGeometry) -> float:
    diameters = mesh.facet_diameters()
    finite = diameters[np.isfinite(diameters)]
    scale = float(np.median(finite)) if finite.size else 1.0
    return max(scale * 1.0e-6, 1.0e-12)


def _curve_regions_from_facet_labels(
    mesh: MeshGeometry,
    facet_labels: FacetLabels,
    *,
    max_distance: float | None,
    include_exterior: bool,
    priorities: dict[str, int],
) -> dict[str, CurveRegion]:
    if mesh.facets is None:
        return {}
    label_by_facet: dict[int, set[str]] = {}
    for label, indices in facet_labels.groups.items():
        for index in indices:
            label_by_facet.setdefault(int(index), set()).add(label)

    edge_labels: dict[tuple[int, int], set[str]] = {}
    edge_counts: dict[tuple[int, int], int] = {}
    for facet_index, facet in enumerate(mesh.facets.T):
        labels = label_by_facet.get(facet_index, set())
        for edge in _facet_edges(facet):
            edge_labels.setdefault(edge, set()).update(labels)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    edges_by_curve: dict[str, list[tuple[int, int]]] = {}
    for edge, labels in edge_labels.items():
        if len(labels) >= 2:
            curve_label = "__".join(sorted(labels))
        elif include_exterior and len(labels) == 1 and edge_counts.get(edge, 0) == 1:
            curve_label = f"{next(iter(labels))}__boundary"
        else:
            continue
        edges_by_curve.setdefault(curve_label, []).append(edge)

    return {
        label: CurveRegion(
            label=label,
            edges=np.asarray(edges, dtype=np.int64).T,
            points=mesh.points,
            max_distance=max_distance,
            priority=int(priorities.get(label, 0)),
        )
        for label, edges in edges_by_curve.items()
    }


def _facet_edges(facet: NDArray[np.int64]) -> list[tuple[int, int]]:
    nodes = [int(node) for node in facet]
    if len(nodes) == 2:
        pairs = [(nodes[0], nodes[1])]
    elif len(nodes) in (3, 4):
        pairs = [(nodes[index], nodes[(index + 1) % len(nodes)]) for index in range(len(nodes))]
    else:
        pairs = [(nodes[index], nodes[next_index]) for index in range(len(nodes)) for next_index in range(index + 1, len(nodes))]
    return [tuple(sorted(pair)) for pair in pairs if pair[0] != pair[1]]


def _edge_nodes_to_connectivity(edge_nodes: NDArray[np.int64]) -> NDArray[np.int64]:
    if edge_nodes.ndim != 2:
        raise ValueError("edge_nodes must be a 2D array")
    if edge_nodes.shape[1] == 2:
        return edge_nodes.T
    if edge_nodes.shape[0] == 2:
        return edge_nodes
    raise ValueError("edge_nodes must have shape (nedges, 2) or (2, nedges)")


def _facet_edge_midpoints(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    edges = _facet_edges(np.arange(vertices.shape[1], dtype=np.int64))
    if not edges:
        return np.empty((vertices.shape[0], 0), dtype=float)
    return np.asarray([(vertices[:, a] + vertices[:, b]) * 0.5 for a, b in edges], dtype=float).T


def _facet_normal(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    unique = _unique_columns(vertices)
    if unique.shape[0] != 3 or unique.shape[1] < 3:
        return np.zeros(unique.shape[0], dtype=float)
    normal = np.cross(unique[:, 1] - unique[:, 0], unique[:, 2] - unique[:, 0])
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        return np.zeros(3, dtype=float)
    return normal / norm


def _normal_angle_deg(a: NDArray[np.float64] | None, b: NDArray[np.float64]) -> float:
    if a is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    anorm = float(np.linalg.norm(a))
    bnorm = float(np.linalg.norm(b))
    if anorm == 0.0 or bnorm == 0.0:
        return 0.0
    cosine = float(np.clip(abs(np.dot(a, b) / (anorm * bnorm)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _point_to_facet_distance(point: NDArray[np.float64], vertices: NDArray[np.float64]) -> float:
    unique = _unique_columns(vertices)
    if unique.shape[1] == 0:
        return float("inf")
    if unique.shape[1] == 1:
        return float(np.linalg.norm(point - unique[:, 0]))
    if unique.shape[1] == 2:
        return _point_to_segment_distance(point, unique[:, 0], unique[:, 1])
    if unique.shape[0] == 3 and unique.shape[1] == 3:
        return _point_to_triangle_distance(point, unique[:, 0], unique[:, 1], unique[:, 2])
    centroid = unique.mean(axis=1)
    return float(min(np.linalg.norm(point - centroid), np.min(np.linalg.norm(unique - point[:, None], axis=0))))


def _point_to_segment_distance(point: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(point - closest))


def _point_to_triangle_distance(point: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64]) -> float:
    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm == 0.0:
        return min(
            _point_to_segment_distance(point, a, b),
            _point_to_segment_distance(point, b, c),
            _point_to_segment_distance(point, c, a),
        )

    projected = point - np.dot(point - a, normal) / np.dot(normal, normal) * normal
    if _point_in_triangle(projected, a, b, c):
        return float(abs(np.dot(point - a, normal)) / normal_norm)
    return min(
        _point_to_segment_distance(point, a, b),
        _point_to_segment_distance(point, b, c),
        _point_to_segment_distance(point, c, a),
    )


def _point_in_triangle(point: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64], c: NDArray[np.float64]) -> bool:
    v0 = c - a
    v1 = b - a
    v2 = point - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = dot00 * dot11 - dot01 * dot01
    if denom == 0.0:
        return False
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    eps = 1.0e-10
    return u >= -eps and v >= -eps and (u + v) <= 1.0 + eps


def _unique_columns(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    _, indices = np.unique(vertices.T, axis=0, return_index=True)
    return vertices[:, np.sort(indices)]
