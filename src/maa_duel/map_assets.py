from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from maa_duel.map_assembly import StageGrid


@dataclass(frozen=True, slots=True)
class BundleFile:
    path: Path
    relative_path: str
    sha256: str
    source: str
    manifest_index: int

    def as_json(self) -> dict[str, str | int]:
        return {
            "path": self.path.as_posix(),
            "relativePath": self.relative_path,
            "sha256": self.sha256,
            "source": self.source,
            "manifestIndex": self.manifest_index,
        }


@dataclass(frozen=True, slots=True)
class ResourceBundleRecord:
    name: str
    scc_index: int
    dependencies: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ResourceManifest:
    path: Path
    root_offset: int
    version: str | None
    bundle_records: tuple[ResourceBundleRecord, ...]
    bundle_index_by_name: dict[str, int]
    asset_paths_by_bundle: dict[int, frozenset[str]]
    catalog_names: frozenset[str]


@dataclass(frozen=True, slots=True)
class MapAssetPlan:
    bundle_files: tuple[BundleFile, ...]
    tile_bundle_path: str
    tile_prefab_by_key: dict[str, str]
    route_effects: tuple[dict[str, str], ...]
    environment_roots: tuple[dict[str, str], ...]
    manifest_inputs: tuple[dict[str, str], ...]
    manifest_version: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_asset_config() -> dict[str, Any]:
    config_path = _repo_root() / "config" / "map-assembly-assets.json"
    payload = _read_json(config_path, "map assembly asset configuration")
    if payload.get("schemaVersion") != 1:
        raise ValueError(f"unsupported map assembly asset config schemaVersion in {config_path}")
    return payload


def _load_schema_module() -> Any:
    schema_path = _repo_root() / "vendor" / "Ark-Unpacker" / "src" / "fbs" / "CN" / "resource_manifest.py"
    if not schema_path.is_file():
        raise FileNotFoundError(
            f"Ark-Unpacker resource manifest schema is missing at {schema_path}; "
            "initialize vendor submodules with `git submodule update --init --recursive`"
        )
    spec = importlib.util.spec_from_file_location("maa_duel._ark_resource_manifest", schema_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the Ark-Unpacker resource manifest schema: {schema_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as error:
        raise RuntimeError(f"resource manifest schema needs the flatbuffers Python dependency: {error}") from error
    return module


def _safe_bundle_name(name: str) -> str:
    relative = PurePosixPath(name.replace("\\", "/"))
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe or invalid bundle path in resource manifest: {name!r}")
    return relative.as_posix()


def _manifest_index_path(root: Path, label: str) -> tuple[Path, dict[str, Any] | None]:
    catalog_path = root / "hot_update_list.json"
    catalog: dict[str, Any] | None = None
    if catalog_path.is_file():
        catalog = _read_json(catalog_path, f"{label} hot update catalog")
    index_name = catalog.get("manifestName") if catalog else None
    if index_name:
        index_path = root / str(index_name)
        if not index_path.is_file():
            raise FileNotFoundError(f"{label} hot update catalog names a missing index file: {index_path}")
        return index_path, catalog

    index_files = sorted(root.glob("*.idx"))
    if not index_files:
        raise FileNotFoundError(f"no .idx resource manifest found under {root}")
    if len(index_files) != 1:
        raise ValueError(f"expected one resource manifest under {root}, found {len(index_files)} .idx files")
    return index_files[0], catalog


def _load_resource_manifest(root: Path, *, label: str, root_offset: int) -> ResourceManifest:
    index_path, catalog = _manifest_index_path(root, label)
    try:
        schema = _load_schema_module()
        manifest_type = schema.clz_Torappu_Resource_ResourceManifest
        parsed = manifest_type.GetRootAs(index_path.read_bytes(), root_offset)
        bundle_count = parsed.BundlesLength()
        asset_count = parsed.AssetToBundleListLength()
        if bundle_count <= 0 or asset_count <= 0:
            raise ValueError(f"manifest has empty bundle or asset tables ({bundle_count}, {asset_count})")

        bundles: list[ResourceBundleRecord] = []
        bundle_index_by_name: dict[str, int] = {}
        for index in range(bundle_count):
            item = parsed.Bundles(index)
            raw_name = item.Name() if item is not None else None
            if not raw_name:
                raise ValueError(f"bundle table item {index} has no name")
            name = _safe_bundle_name(raw_name.decode("utf-8"))
            normalized_name = name.casefold()
            if normalized_name in bundle_index_by_name:
                raise ValueError(f"resource manifest contains duplicate bundle name {name!r}")
            dependency_count = item.AllDependenciesLength()
            dependencies = tuple(item.AllDependencies(dep_index) for dep_index in range(dependency_count))
            for dependency in dependencies:
                if dependency < 0 or dependency >= bundle_count:
                    raise ValueError(
                        f"bundle {name!r} references dependency index {dependency} outside the bundle table"
                    )
            bundle_index_by_name[normalized_name] = index
            bundles.append(ResourceBundleRecord(name=name, scc_index=item.SccIndex(), dependencies=dependencies))

        asset_paths_by_bundle: dict[int, set[str]] = {}
        for asset_index in range(asset_count):
            item = parsed.AssetToBundleList(asset_index)
            if item is None:
                raise ValueError(f"asset-to-bundle table item {asset_index} is missing")
            bundle_index = item.BundleIndex()
            if bundle_index < 0 or bundle_index >= bundle_count:
                raise ValueError(f"asset-to-bundle item {asset_index} references invalid bundle index {bundle_index}")
            path = item.Path()
            if path:
                asset_path = path.decode("utf-8").replace("\\", "/").casefold()
                asset_paths_by_bundle.setdefault(bundle_index, set()).add(asset_path)

        catalog_names: frozenset[str] = frozenset()
        version: str | None = None
        if catalog is not None:
            entries = catalog.get("abInfos")
            if not isinstance(entries, list):
                raise ValueError(
                    f"{label} hot update catalog has no abInfos array: {index_path.parent / 'hot_update_list.json'}"
                )
            names: set[str] = set()
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                    raise ValueError(f"{label} hot update catalog contains an invalid abInfos entry")
                names.add(_safe_bundle_name(entry["name"]).casefold())
            catalog_names = frozenset(names)
            version_value = catalog.get("versionId")
            version = str(version_value) if version_value else None

        return ResourceManifest(
            path=index_path,
            root_offset=root_offset,
            version=version,
            bundle_records=tuple(bundles),
            bundle_index_by_name=bundle_index_by_name,
            asset_paths_by_bundle={index: frozenset(paths) for index, paths in asset_paths_by_bundle.items()},
            catalog_names=catalog_names,
        )
    except (IndexError, TypeError, ValueError, UnicodeDecodeError, OverflowError) as error:
        raise ValueError(f"cannot parse {label} resource manifest {index_path}: {error}") from error


def _asset_config_records(
    config: dict[str, Any], theme: str
) -> tuple[str, dict[str, str], tuple[dict[str, str], ...], tuple[dict[str, str], ...], tuple[str, ...]]:
    tile = config.get("tilePrefabs")
    if not isinstance(tile, dict) or not isinstance(tile.get("bundle"), str) or not isinstance(tile.get("byKey"), dict):
        raise ValueError("map asset config tilePrefabs must specify bundle and byKey")
    tile_bundle = _safe_bundle_name(tile["bundle"])
    tile_prefabs = {str(key): str(value) for key, value in tile["byKey"].items()}
    if not tile_prefabs or any(not asset_path for asset_path in tile_prefabs.values()):
        raise ValueError("map asset config tilePrefabs.byKey must map tile keys to non-empty asset paths")

    raw_route_effects = config.get("routeEffects")
    if not isinstance(raw_route_effects, list) or not raw_route_effects:
        raise ValueError("map asset config routeEffects must be a non-empty array")
    route_effects = tuple(
        {
            "name": str(record["name"]),
            "bundlePath": _safe_bundle_name(str(record["bundle"])),
            "assetPath": str(record["assetPath"]),
        }
        for record in raw_route_effects
        if isinstance(record, dict) and record.get("name") and record.get("bundle") and record.get("assetPath")
    )
    if len(route_effects) != len(raw_route_effects):
        raise ValueError("map asset config contains an invalid routeEffects record")

    themes = config.get("themes")
    theme_config = themes.get(theme) if isinstance(themes, dict) else None
    if not isinstance(theme_config, dict):
        raise ValueError(f"no map assembly asset roots are configured for theme {theme!r}")
    resource_bundles = theme_config.get("resourceBundles")
    roots = theme_config.get("environmentRoots")
    if not isinstance(resource_bundles, list) or not isinstance(roots, list):
        raise ValueError(f"theme {theme!r} must configure resourceBundles and environmentRoots")
    resource_paths = tuple(_safe_bundle_name(str(path)) for path in resource_bundles)
    environment_roots = tuple(
        {
            "name": str(record["name"]),
            "bundlePath": _safe_bundle_name(str(record["bundle"])),
            "assetPath": str(record["assetPath"]),
            "layer": str(record.get("layer", "environment")),
        }
        for record in roots
        if isinstance(record, dict) and record.get("name") and record.get("bundle") and record.get("assetPath")
    )
    if len(environment_roots) != len(roots):
        raise ValueError(f"theme {theme!r} contains an invalid environmentRoots record")
    return tile_bundle, tile_prefabs, route_effects, environment_roots, resource_paths


def _validate_asset_path(manifest: ResourceManifest, bundle_path: str, asset_path: str) -> int:
    normalized_bundle = bundle_path.casefold()
    if normalized_bundle not in manifest.bundle_index_by_name:
        raise ValueError(f"bundle {bundle_path!r} is absent from resource manifest {manifest.path}")
    bundle_index = manifest.bundle_index_by_name[normalized_bundle]
    normalized_asset = asset_path.replace("\\", "/").casefold()
    if normalized_asset not in manifest.asset_paths_by_bundle.get(bundle_index, frozenset()):
        raise ValueError(f"asset path {asset_path!r} is not indexed in bundle {bundle_path!r} by {manifest.path}")
    return bundle_index


def _resolve_bundle_file(
    bundle_name: str,
    manifest_index: int,
    streaming_root: Path,
    hotfix_root: Path,
    hotfix_names: frozenset[str],
) -> BundleFile:
    relative_path = _safe_bundle_name(bundle_name)
    path_parts = Path(*PurePosixPath(relative_path).parts)
    hotfix_path = hotfix_root / path_parts
    streaming_path = streaming_root / path_parts
    normalized = relative_path.casefold()
    if normalized in hotfix_names and hotfix_path.is_file():
        selected = hotfix_path
        source = "hotfix"
    elif streaming_path.is_file():
        selected = streaming_path
        source = "streaming"
    else:
        raise FileNotFoundError(
            f"dependency bundle {relative_path!r} (manifest index {manifest_index}) is absent from both "
            f"{streaming_root} and {hotfix_root}"
        )
    return BundleFile(
        path=selected.resolve(strict=True),
        relative_path=relative_path,
        sha256=_sha256(selected),
        source=source,
        manifest_index=manifest_index,
    )


def resolve_map_assets(game_assets: Path, theme: str, stage: StageGrid) -> MapAssetPlan:
    game_assets = game_assets.resolve(strict=True)
    if not game_assets.is_dir():
        raise ValueError(f"game-assets must be an Arknights_Data directory: {game_assets}")
    streaming_root = game_assets / "StreamingAssets" / "AB" / "Windows"
    hotfix_root = game_assets / "PersistentData" / "Bundles"
    if not streaming_root.is_dir() or not hotfix_root.is_dir():
        raise FileNotFoundError(f"expected StreamingAssets/AB/Windows and PersistentData/Bundles under {game_assets}")

    config = _load_asset_config()
    root_offset = config.get("flatbufferRootOffset")
    if isinstance(root_offset, bool) or not isinstance(root_offset, int) or root_offset < 0:
        raise ValueError("flatbufferRootOffset must be a non-negative integer")
    tile_bundle, configured_tiles, route_effects, environment_roots, resource_bundles = _asset_config_records(
        config, theme
    )
    required_tile_keys = {cell.tile_key for cell in stage.cells}
    missing_tile_keys = sorted(required_tile_keys - set(configured_tiles))
    if missing_tile_keys:
        raise ValueError(f"no tile prefab mapping configured for stage tile keys: {', '.join(missing_tile_keys)}")

    # Read both installed resource indexes: StreamingAssets can be an older
    # base version while PersistentData contains the currently patched graph.
    streaming_manifest = _load_resource_manifest(streaming_root, label="StreamingAssets", root_offset=root_offset)
    hotfix_manifest: ResourceManifest | None = None
    hotfix_catalog = hotfix_root / "hot_update_list.json"
    if list(hotfix_root.glob("*.idx")):
        hotfix_manifest = _load_resource_manifest(hotfix_root, label="PersistentData", root_offset=root_offset)
    manifest = hotfix_manifest or streaming_manifest
    if manifest.catalog_names and not manifest.version:
        raise ValueError(f"active resource catalog has no versionId: {manifest.path.parent / 'hot_update_list.json'}")

    tile_prefab_by_key: dict[str, str] = {}
    for tile_key in sorted(required_tile_keys):
        asset_path = configured_tiles[tile_key]
        _validate_asset_path(manifest, tile_bundle, asset_path)
        tile_prefab_by_key[tile_key] = asset_path
    for effect in route_effects:
        _validate_asset_path(manifest, effect["bundlePath"], effect["assetPath"])
    for root in environment_roots:
        _validate_asset_path(manifest, root["bundlePath"], root["assetPath"])

    root_names: list[str] = []
    for name in (tile_bundle, *(effect["bundlePath"] for effect in route_effects), *resource_bundles):
        if name.casefold() not in {existing.casefold() for existing in root_names}:
            root_names.append(name)
    for root in environment_roots:
        name = root["bundlePath"]
        if name.casefold() not in {existing.casefold() for existing in root_names}:
            root_names.append(name)

    state: dict[int, int] = {}
    ordered_indices: list[int] = []
    visit_stack: list[int] = []
    stack_position: dict[int, int] = {}

    def visit(index: int) -> None:
        current_state = state.get(index, 0)
        if current_state == 1:
            cycle = visit_stack[stack_position[index] :]
            component_ids = {manifest.bundle_records[cycle_index].scc_index for cycle_index in cycle}
            if len(component_ids) == 1 and next(iter(component_ids)) > 0:
                # ResourceManifest uses SccIndex to mark legitimate cyclic references.
                return
            cycle_names = " -> ".join(manifest.bundle_records[cycle_index].name for cycle_index in (*cycle, index))
            raise ValueError(f"resource manifest has a dependency cycle across SCCs: {cycle_names}")
        if current_state == 2:
            return
        if index < 0 or index >= len(manifest.bundle_records):
            raise ValueError(f"dependency index {index} is outside active resource manifest {manifest.path}")
        state[index] = 1
        stack_position[index] = len(visit_stack)
        visit_stack.append(index)
        for dependency in manifest.bundle_records[index].dependencies:
            visit(dependency)
        visit_stack.pop()
        stack_position.pop(index)
        state[index] = 2
        ordered_indices.append(index)

    for name in root_names:
        bundle_index = manifest.bundle_index_by_name.get(name.casefold())
        if bundle_index is None:
            raise ValueError(f"required bundle {name!r} is absent from active resource manifest {manifest.path}")
        if manifest.catalog_names and name.casefold() not in manifest.catalog_names:
            raise ValueError(f"required bundle {name!r} is not listed in the active hot update catalog")
        visit(bundle_index)

    bundle_files = tuple(
        _resolve_bundle_file(
            manifest.bundle_records[index].name,
            index,
            streaming_root,
            hotfix_root,
            manifest.catalog_names,
        )
        for index in ordered_indices
    )
    manifest_inputs: list[dict[str, str]] = []
    for role, source_path in (
        ("streaming_resource_index", streaming_manifest.path),
        ("streaming_hot_update_catalog", streaming_root / "hot_update_list.json"),
        ("hotfix_resource_index", hotfix_manifest.path if hotfix_manifest else None),
        ("hotfix_update_catalog", hotfix_catalog if hotfix_manifest else None),
    ):
        if source_path is not None and source_path.is_file():
            manifest_inputs.append(
                {"role": role, "path": source_path.resolve().as_posix(), "sha256": _sha256(source_path)}
            )

    return MapAssetPlan(
        bundle_files=bundle_files,
        tile_bundle_path=tile_bundle,
        tile_prefab_by_key=tile_prefab_by_key,
        route_effects=route_effects,
        environment_roots=environment_roots,
        manifest_inputs=tuple(manifest_inputs),
        manifest_version=manifest.version,
    )
