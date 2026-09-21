from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_STAGE_KEY = "activities/act1enemyduel/level_act1enemyduel_02a"
DEFAULT_UNITY_EDITOR = Path(r"D:\Unity\Hub\Editor\2021.3.39f1\Editor\Unity.exe")


@dataclass(frozen=True, slots=True)
class MapAssemblyRequest:
    stage_json: Path
    camera_json: Path
    game_assets: Path
    output_dir: Path
    stage_key: str = DEFAULT_STAGE_KEY
    width: int = 1920
    height: int = 864
    frame_time: float = 0.0


@dataclass(frozen=True, slots=True)
class StageCell:
    row: int
    column: int
    tile_index: int
    tile_key: str
    height_type: int
    buildable_type: int
    passable_mask: int
    player_side_mask: int


@dataclass(frozen=True, slots=True)
class StageGrid:
    rows: int
    columns: int
    cells: tuple[StageCell, ...]

    @property
    def tile_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cell in self.cells:
            counts[cell.tile_key] = counts.get(cell.tile_key, 0) + 1
        return dict(sorted(counts.items()))


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} JSON at {path}: {error}") from error


def _required_int(record: dict[str, Any], key: str, *, row: int, column: int) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"tile at row {row}, column {column} has invalid {key}: {value!r}")
    return value


def parse_stage_grid(stage_json: Path) -> StageGrid:
    payload = _read_json(stage_json, "stage")
    if not isinstance(payload, dict) or not isinstance(payload.get("mapData"), dict):
        raise ValueError("stage JSON must contain an object named mapData")

    map_data = payload["mapData"]
    rows = map_data.get("map")
    tiles = map_data.get("tiles")
    if not isinstance(rows, list) or not rows or not isinstance(tiles, list) or not tiles:
        raise ValueError("mapData.map and mapData.tiles must be non-empty arrays")
    if any(not isinstance(row, list) for row in rows):
        raise ValueError("mapData.map must contain only row arrays")

    column_count = len(rows[0])
    if column_count == 0 or any(len(row) != column_count for row in rows):
        raise ValueError("mapData.map must be a non-empty rectangular grid")

    cells: list[StageCell] = []
    for row_index, row in enumerate(rows):
        for column_index, raw_index in enumerate(row):
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise ValueError(f"map index at row {row_index}, column {column_index} is not an integer")
            if raw_index < 0 or raw_index >= len(tiles):
                raise ValueError(
                    f"map index {raw_index} at row {row_index}, column {column_index} "
                    f"is outside mapData.tiles (length {len(tiles)})"
                )
            tile = tiles[raw_index]
            if not isinstance(tile, dict):
                raise ValueError(f"tile record {raw_index} is not an object")
            tile_key = tile.get("tileKey")
            if not isinstance(tile_key, str) or not tile_key:
                raise ValueError(f"tile record {raw_index} at row {row_index}, column {column_index} has no tileKey")
            cells.append(
                StageCell(
                    row=row_index,
                    column=column_index,
                    tile_index=raw_index,
                    tile_key=tile_key,
                    height_type=_required_int(tile, "heightType", row=row_index, column=column_index),
                    buildable_type=_required_int(tile, "buildableType", row=row_index, column=column_index),
                    passable_mask=_required_int(tile, "passableMask", row=row_index, column=column_index),
                    player_side_mask=_required_int(tile, "playerSideMask", row=row_index, column=column_index),
                )
            )

    return StageGrid(rows=len(rows), columns=column_count, cells=tuple(cells))


def _vector3(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must be an array of three numbers")
    vector: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise ValueError(f"{label} must contain only finite numbers")
        coordinate = float(coordinate)
        if not math.isfinite(coordinate):
            raise ValueError(f"{label} must contain only finite numbers")
        vector.append(coordinate)
    return vector


def parse_camera_profile(camera_json: Path, stage_key: str) -> tuple[str, dict[str, Any]]:
    profiles = _read_json(camera_json, "camera profile")
    summary_path = camera_json.with_name("summary.json")
    summaries = _read_json(summary_path, "camera summary")
    if not isinstance(profiles, dict) or stage_key not in profiles:
        raise ValueError(f"camera profile does not contain exact stage key {stage_key!r}: {camera_json}")
    if not isinstance(summaries, dict) or stage_key not in summaries:
        raise ValueError(f"camera summary does not contain exact stage key {stage_key!r}: {summary_path}")

    raw_profile = profiles[stage_key]
    if not isinstance(raw_profile, list) or len(raw_profile) != 2:
        raise ValueError(f"camera profile for {stage_key!r} must contain default and side view vectors")
    profile = {
        "defaultView": _vector3(raw_profile[0], f"{stage_key}.defaultView"),
        "sideView": _vector3(raw_profile[1], f"{stage_key}.sideView"),
    }
    summary = summaries[stage_key]
    if not isinstance(summary, dict):
        raise ValueError(f"camera summary for {stage_key!r} must be an object")
    debug = summary.get("_debug", {})
    if not isinstance(debug, dict):
        raise ValueError(f"camera summary _debug for {stage_key!r} must be an object")
    theme = summary.get("theme", debug.get("theme"))
    if not isinstance(theme, str) or not theme:
        raise ValueError(f"camera summary for {stage_key!r} has no theme")

    camera_fields = {
        "cameraView": debug.get("cameraView"),
        "layerHeight": debug.get("layerHeight"),
        "highlandHeight": debug.get("highlandHeight"),
        "cameraOffset": debug.get("camera_offset"),
        "cameraFocus": debug.get("camera_focus"),
        "viewDefault": summary.get("view_default"),
        "viewBySide": summary.get("view_by_side"),
    }
    for field in ("cameraOffset", "cameraFocus", "viewDefault", "viewBySide"):
        camera_fields[field] = _vector3(camera_fields[field], f"{stage_key}.{field}")
    for field in ("cameraView", "layerHeight", "highlandHeight"):
        value = camera_fields[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"camera summary field {field} for {stage_key!r} is missing or invalid")
        camera_fields[field] = float(value)
    return theme, {"profile": profile, **camera_fields}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_request(request: MapAssemblyRequest) -> MapAssemblyRequest:
    stage_json = request.stage_json.resolve(strict=True)
    camera_json = request.camera_json.resolve(strict=True)
    game_assets = request.game_assets.resolve(strict=True)
    if not stage_json.is_file() or not camera_json.is_file() or not game_assets.is_dir():
        raise ValueError("stage-json and camera-json must be files; game-assets must be a directory")
    if request.width <= 0 or request.height <= 0:
        raise ValueError("render width and height must be positive integers")
    if not math.isfinite(request.frame_time) or request.frame_time < 0:
        raise ValueError("frame-time must be a finite non-negative number")
    if not request.stage_key.strip():
        raise ValueError("stage-key must not be empty")
    return MapAssemblyRequest(
        stage_json=stage_json,
        camera_json=camera_json,
        game_assets=game_assets,
        output_dir=request.output_dir.resolve(),
        stage_key=request.stage_key,
        width=request.width,
        height=request.height,
        frame_time=request.frame_time,
    )


def prepare_assembly_input(request: MapAssemblyRequest) -> Path:
    request = _resolve_request(request)
    grid = parse_stage_grid(request.stage_json)
    theme, camera = parse_camera_profile(request.camera_json, request.stage_key)
    from maa_duel.map_assets import resolve_map_assets

    asset_plan = resolve_map_assets(request.game_assets, theme, grid)
    summary_path = request.camera_json.with_name("summary.json").resolve(strict=True)
    request.output_dir.mkdir(parents=True, exist_ok=True)

    input_hashes = [
        {"role": role, "path": path.as_posix(), "sha256": _sha256(path)}
        for role, path in (
            ("stage", request.stage_json),
            ("camera_profiles", request.camera_json),
            ("camera_summary", summary_path),
        )
    ]
    input_hashes.extend(asset_plan.manifest_inputs)
    config = {
        "stageKey": request.stage_key,
        "theme": theme,
        "width": request.width,
        "height": request.height,
        "frameTime": request.frame_time,
        "grid": [
            {
                "row": cell.row,
                "column": cell.column,
                "tileIndex": cell.tile_index,
                "tileKey": cell.tile_key,
                "heightType": cell.height_type,
                "buildableType": cell.buildable_type,
                "passableMask": cell.passable_mask,
                "playerSideMask": cell.player_side_mask,
            }
            for cell in grid.cells
        ],
        "camera": camera,
        "worldGrid": {
            "origin": [0.0, 0.0, 0.0],
            "columnDirection": -1,
            "rowDirection": -1,
            "columnPitch": 1.0,
            "rowPitch": 1.0,
            "evidence": "VS-2 cameraFocus (-7,-5) is the center of the 15x11 stage grid.",
        },
        "bundleRoot": request.game_assets.as_posix(),
        "bundleFiles": [bundle.as_json() for bundle in asset_plan.bundle_files],
        "outputDirectory": request.output_dir.as_posix(),
        "inputHashes": input_hashes,
        "tilePrefabByKey": [
            {
                "tileKey": tile_key,
                "bundlePath": asset_plan.tile_bundle_path,
                "assetPath": asset_path,
            }
            for tile_key, asset_path in sorted(asset_plan.tile_prefab_by_key.items())
        ],
        "routeEffects": list(asset_plan.route_effects),
        "environmentRoots": list(asset_plan.environment_roots),
        "resourceManifestVersion": asset_plan.manifest_version,
    }

    run_root = request.output_dir / ".map-assembly"
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=run_root))
    config_path = run_dir / "assembly-input.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config_path


def run_unity_assembler(config_path: Path, output_dir: Path, unity_editor: Path | None = None) -> None:
    config_path = config_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    from .bundle_compat import prepare_unity_compatible_input

    unity_config_path = prepare_unity_compatible_input(config_path, output_dir)
    editor = unity_editor or (Path(os.environ["UNITY_EDITOR"]) if os.environ.get("UNITY_EDITOR") else None)
    if editor is None:
        editor = DEFAULT_UNITY_EDITOR
    editor = editor.resolve(strict=True)
    if not editor.is_file():
        raise ValueError(f"Unity Editor executable is not a file: {editor}")

    project_dir = Path(__file__).resolve().parents[2] / "unity" / "map-assembler"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "map-assembly.log"
    command = [
        str(editor),
        "-batchmode",
        "-quit",
        "-noUpm",
        "-projectPath",
        str(project_dir),
        "-executeMethod",
        "MaaDuelChannel.MapAssembler.DuelMapAssembler.Run",
        "-assemblyConfig",
        str(unity_config_path),
        "-logFile",
        str(log_path),
    ]
    completed = subprocess.run(command, check=False, cwd=project_dir)
    if completed.returncode != 0:
        raise RuntimeError(f"Unity map assembly failed with exit code {completed.returncode}; see {log_path}")
