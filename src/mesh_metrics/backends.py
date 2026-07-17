from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Protocol

from mesh_metrics.geometry import MeshGeometry


class MeshBackend(Protocol):
    name: str

    def load(self, path: str | Path) -> MeshGeometry:
        ...


@dataclass(frozen=True)
class SkfemBackend:
    name: str = "skfem"

    def load(self, path: str | Path) -> MeshGeometry:
        try:
            from skfem import Mesh
        except ImportError as exc:
            raise RuntimeError("skfem backend requires: pip install 'mesh-metrics[skfem]'") from exc

        mesh_path = Path(path)
        return MeshGeometry.from_skfem(Mesh.load(mesh_path), source=str(mesh_path), backend=self.name)


@dataclass(frozen=True)
class FluxfemBackend:
    name: str = "fluxfem"

    def load(self, path: str | Path) -> MeshGeometry:
        try:
            import fluxfem  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("fluxfem backend requires: pip install 'mesh-metrics[fluxfem]'") from exc

        # fluxfem does not expose a stable public file loader here, so use meshio
        # as the file adapter and keep fluxfem as an optional runtime backend.
        return _load_with_meshio(path, backend=self.name)


@dataclass(frozen=True)
class MeshioBackend:
    name: str = "meshio"

    def load(self, path: str | Path) -> MeshGeometry:
        return _load_with_meshio(path, backend=self.name)


def get_backend(name: str) -> MeshBackend:
    normalized = name.lower()
    if normalized == "skfem":
        return SkfemBackend()
    if normalized == "fluxfem":
        return FluxfemBackend()
    if normalized == "meshio":
        return MeshioBackend()
    raise ValueError(f"unknown backend: {name}")


def _load_with_meshio(path: str | Path, *, backend: str) -> MeshGeometry:
    try:
        import meshio
    except ImportError as exc:
        raise RuntimeError("meshio is required to load mesh files") from exc

    mesh_path = Path(path)
    sanitized: Path | None = None
    if mesh_path.suffix.lower() == ".mesh" and _medit_mesh_needs_sanitize(mesh_path):
        sanitized = _sanitize_medit_mesh(mesh_path)
        try:
            mesh = meshio.read(sanitized)
        finally:
            sanitized.unlink(missing_ok=True)
    else:
        try:
            mesh = meshio.read(mesh_path)
        except BaseException:
            if mesh_path.suffix.lower() != ".mesh":
                raise
            sanitized = _sanitize_medit_mesh(mesh_path)
            try:
                mesh = meshio.read(sanitized)
            finally:
                sanitized.unlink(missing_ok=True)
    if not mesh.cells:
        raise ValueError(f"no cells found in {mesh_path}")

    cell_block = max(mesh.cells, key=lambda block: block.data.shape[1])
    points = mesh.points.T
    elements = cell_block.data.T
    return MeshGeometry(points=points, elements=elements, source=str(mesh_path), backend=backend)


def _sanitize_medit_mesh(path: Path) -> Path:
    unsupported_count_sections = {"RequiredEdges"}
    lines = path.read_text(encoding="utf-8").splitlines()
    cleaned: list[str] = []
    index = 0
    while index < len(lines):
        keyword = lines[index].strip()
        if keyword in unsupported_count_sections:
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            if index >= len(lines):
                break
            count = int(lines[index].strip())
            index += count + 1
            continue
        cleaned.append(lines[index])
        index += 1

    with tempfile.NamedTemporaryFile("w", suffix=".mesh", delete=False, encoding="utf-8") as stream:
        stream.write("\n".join(cleaned) + "\n")
        return Path(stream.name)


def _medit_mesh_needs_sanitize(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip() == "RequiredEdges":
                return True
    return False
