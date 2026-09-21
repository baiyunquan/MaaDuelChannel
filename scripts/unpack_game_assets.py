from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARK_UNPACKER_ROOT = REPOSITORY_ROOT / "vendor" / "Ark-Unpacker"
REQUIREMENTS_LOCK = REPOSITORY_ROOT / "tools" / "ark-unpacker-requirements.lock.txt"
DEFAULT_GAME_ROOT = Path(r"E:\Program Files\Arknights")
DEFAULT_WORKSPACE = Path(r"D:\MAA-DuelChannel\workspace")

SELECTIONS = {
    "battlefield_spine": "battle",
    "roster_icons": "spritepack",
    "duel_ui": "ui/enemyduel",
    "duel_stage_previews": "arts/ui",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_game_path(path: Path, game_root: Path) -> str:
    return path.relative_to(game_root).as_posix()


def selected_bundles(root: Path, category: str) -> list[Path]:
    source_dir = root / SELECTIONS[category]
    if not source_dir.is_dir():
        return []
    files = sorted(path for path in source_dir.rglob("*.ab") if path.is_file())
    if category == "roster_icons":
        return [path for path in files if re.fullmatch(r"icon_enemies(?:_\d+)?\.ab", path.name, re.IGNORECASE)]
    if category == "duel_stage_previews":
        return [path for path in files if path.name.startswith("stage_mappreview_") and "_duel_" in path.name.lower()]
    return files


def collect_bundle_inventory(
    game_root: Path, streaming_root: Path, hotfix_root: Path
) -> tuple[list[dict], dict[str, Path]]:
    inventory: list[dict] = []
    merged: dict[str, Path] = {}
    for category in SELECTIONS:
        for layer, root in (("streaming", streaming_root), ("hotfix", hotfix_root)):
            for source in selected_bundles(root, category):
                relative = source.relative_to(root).as_posix()
                inventory.append(
                    {
                        "category": category,
                        "layer": layer,
                        "path": relative_game_path(source, game_root),
                        "relative_bundle_path": relative,
                        "size_bytes": source.stat().st_size,
                        "sha256": sha256_file(source),
                        "selected_after_overlay": False,
                    }
                )
                merged[f"{category}/{relative}"] = source
    for item in inventory:
        item["selected_after_overlay"] = merged[f"{item['category']}/{item['relative_bundle_path']}"] == (
            game_root / Path(item["path"])
        )
    return inventory, merged


def ark_unpacker_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(ARK_UNPACKER_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def prepare_runtime(workspace: Path, tool_commit: str) -> Path:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required to prepare Ark-Unpacker's isolated Python 3.12 environment")
    runtime_root = workspace / "tools" / "ark-unpacker"
    environment = runtime_root / ".venv"
    python = environment / "Scripts" / "python.exe"
    runtime_root.mkdir(parents=True, exist_ok=True)
    if not python.exists():
        subprocess.run([uv, "venv", "--python", "3.12", str(environment)], check=True)
    version = subprocess.run(
        [str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version != "3.12":
        raise RuntimeError(f"Ark-Unpacker requires Python 3.12; isolated environment has Python {version}")
    contract = hashlib.sha256((tool_commit + REQUIREMENTS_LOCK.read_text(encoding="utf-8")).encode()).hexdigest()
    marker = runtime_root / "environment.sha256"
    if not marker.is_file() or marker.read_text(encoding="ascii").strip() != contract:
        subprocess.run([uv, "pip", "sync", "--python", str(python), str(REQUIREMENTS_LOCK)], check=True)
        marker.write_text(contract + "\n", encoding="ascii")
    return python


def find_game_spine_matches(catalog_path: Path, spine_root: Path) -> dict:
    if not catalog_path.is_file():
        return {"catalog": str(catalog_path), "matched_enemies": [], "unmatched_enemy_ids": []}
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    skeleton_files = list(spine_root.rglob("*.skel")) if spine_root.is_dir() else []
    exports_by_stem: dict[str, list[str]] = {}
    for path in skeleton_files:
        exports_by_stem.setdefault(path.stem.casefold(), []).append(path.relative_to(spine_root.parent).as_posix())

    matched = []
    unmatched = []
    for enemy in catalog.get("enemies", []):
        variants = enemy.get("battlefield_spine", {}).get("variants", [])
        stems = {
            Path(unquote(urlparse(variant.get("skeleton", {}).get("source", "")).path)).stem.casefold()
            for variant in variants
        }
        stems.discard("")
        paths = sorted({path for stem in stems for path in exports_by_stem.get(stem, [])})
        enemy_id = enemy.get("enemy_id")
        if paths:
            matched.append({"enemy_id": enemy_id, "prts_skeleton_stems": sorted(stems), "game_exports": paths})
        elif enemy_id is not None:
            unmatched.append(enemy_id)
    return {
        "catalog": str(catalog_path),
        "catalog_enemy_count": len(catalog.get("enemies", [])),
        "matched_enemy_count": len(matched),
        "matched_enemies": matched,
        "unmatched_enemy_ids": unmatched,
    }


def find_game_roster_icon_matches(catalog_path: Path, icon_root: Path, combat_path: Path) -> dict:
    if not catalog_path.is_file():
        return {"catalog": str(catalog_path), "matched_enemies": [], "unmatched_enemy_ids": []}
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    exports_by_stem: dict[str, list[str]] = {}
    icon_files = list(icon_root.rglob("*.png")) if icon_root.is_dir() else []
    for path in icon_files:
        exports_by_stem.setdefault(path.stem.casefold(), []).append(path.relative_to(icon_root.parent).as_posix())

    matched = []
    unmatched = []
    for enemy in catalog.get("enemies", []):
        variants = enemy.get("battlefield_spine", {}).get("variants", [])
        model_stems = {
            Path(unquote(urlparse(variant.get("skeleton", {}).get("source", "")).path)).stem.casefold()
            for variant in variants
        }
        model_stems.discard("")
        candidate_stems = set(model_stems)
        candidate_stems.update(stem[:-2] for stem in model_stems if stem.endswith("_2"))
        icon_matches = [
            {"model_stem": stem, "icon_files": sorted(exports_by_stem[stem])}
            for stem in sorted(candidate_stems)
            if exports_by_stem.get(stem)
        ]
        enemy_id = enemy.get("enemy_id")
        if icon_matches:
            matched.append({"enemy_id": enemy_id, "model_stems": sorted(model_stems), "matches": icon_matches})
        elif enemy_id is not None:
            unmatched.append(enemy_id)

    vs2_ids = []
    if combat_path.is_file():
        combat = json.loads(combat_path.read_text(encoding="utf-8"))
        vs2_ids = sorted(
            {enemy["enemy_id"] for enemy in combat.get("enemies", []) if enemy.get("enemy_id") is not None}
        )
    matched_ids = {enemy["enemy_id"] for enemy in matched}
    return {
        "catalog": str(catalog_path),
        "catalog_enemy_count": len(catalog.get("enemies", [])),
        "matched_enemy_count": len(matched),
        "matched_enemies": matched,
        "unmatched_enemy_ids": unmatched,
        "vs2_enemy_count": len(vs2_ids),
        "vs2_matched_enemy_count": len(matched_ids.intersection(vs2_ids)),
        "vs2_unmatched_enemy_ids": sorted(set(vs2_ids).difference(matched_ids)),
    }


def export_category(python: Path, mode: str, input_dir: Path, output_dir: Path, log_dir: Path) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"staged Ark-Unpacker input is missing: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "log_file": str(log_dir / "ArkUnpackerLogs.log"),
        "log_level": 3,
        "min_spare_memory_mb": 1024,
        "performance_level": 1,
        "export_encoding": "utf-8",
        "export_json_indent": 2,
    }
    (log_dir / "ArkUnpackerConfig.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    command = [
        str(python),
        str(ARK_UNPACKER_ROOT / "Main.py"),
        "-m",
        mode,
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-g",
        "-l",
        "3",
    ]
    if mode == "ab":
        command.append("--image")
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=log_dir, env=environment, check=True)


def output_inventory(run_root: Path) -> dict[str, list[dict]]:
    result = {}
    for category in SELECTIONS:
        category_root = run_root / category
        files = sorted(path for path in category_root.rglob("*") if path.is_file()) if category_root.is_dir() else []
        result[category] = [
            {
                "path": path.relative_to(run_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Unpack selected local Arknights PC assets with Ark-Unpacker.")
    parser.add_argument("--game-root", type=Path, default=DEFAULT_GAME_ROOT)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--dry-run", action="store_true", help="list selected bundles and sizes without unpacking")
    args = parser.parse_args()

    if os.name != "nt":
        raise RuntimeError("Ark-Unpacker currently supports Windows only")
    game_root = args.game_root.resolve()
    workspace = args.workspace.resolve()
    data_root = game_root / "Arknights_Data"
    streaming_root = data_root / "StreamingAssets" / "AB" / "Windows"
    hotfix_root = data_root / "PersistentData" / "Bundles"
    if not streaming_root.is_dir() or not hotfix_root.is_dir():
        raise FileNotFoundError(f"expected PC resource roots under {data_root}")
    if not (ARK_UNPACKER_ROOT / "Main.py").is_file() or not (ARK_UNPACKER_ROOT / "src" / "fbs").is_dir():
        raise FileNotFoundError("Ark-Unpacker and its nested Ark-FBS-Py submodule are not initialized")
    tool_commit = ark_unpacker_commit()
    inventory, merged = collect_bundle_inventory(game_root, streaming_root, hotfix_root)
    if not merged:
        raise FileNotFoundError("no selected AssetBundles were found in the game installation")

    input_bytes = sum(item["size_bytes"] for item in inventory if item["selected_after_overlay"])
    if args.dry_run:
        print(f"Ark-Unpacker commit: {tool_commit}")
        print(f"Selected bundles after initial/hotfix overlay: {len(merged)} ({input_bytes / (1024**3):.2f} GiB)")
        for category in SELECTIONS:
            records = [item for item in inventory if item["category"] == category and item["selected_after_overlay"]]
            size = sum(item["size_bytes"] for item in records)
            print(f"{category}: {len(records)} bundles, {size / (1024**2):.1f} MiB")
        return 0

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    assets_root = workspace / "assets" / "game_assets" / "runs"
    run_root = assets_root / run_id
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing extraction output: {run_root}")
    staging_root = workspace / "tools" / "ark-unpacker" / "staging"
    logs_root = workspace / "tools" / "ark-unpacker" / "runs" / run_id
    run_root.mkdir(parents=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    python = prepare_runtime(workspace, tool_commit)

    manifest = {
        "schema_version": 1,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "game_root": str(game_root),
        "streaming_assets": str(streaming_root),
        "hotfix_assets": str(hotfix_root),
        "ark_unpacker_commit": tool_commit,
        "ark_unpacker_version": "5.1",
        "python_version": "3.12",
        "overlay_precedence": ["StreamingAssets/AB/Windows", "PersistentData/Bundles (wins on same bundle path)"],
        "input_bundles": inventory,
        "outputs": {},
    }
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        with tempfile.TemporaryDirectory(prefix=f"{run_id}-", dir=staging_root) as temporary:
            staged_root = Path(temporary)
            overlay = {}
            for category in SELECTIONS:
                for item in inventory:
                    if item["category"] != category:
                        continue
                    if item["selected_after_overlay"]:
                        overlay[f"{category}/{item['relative_bundle_path']}"] = game_root / Path(item["path"])
            for key, source in overlay.items():
                category, relative = key.split("/", 1)
                target = staged_root / SELECTIONS[category] / Path(relative).relative_to(SELECTIONS[category])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

            input_map = {
                "battlefield_spine": ("sp", staged_root / "battle", run_root / "battlefield_spine"),
                "roster_icons": ("ab", staged_root / "spritepack", run_root / "roster_icons"),
                "duel_ui": ("ab", staged_root / "ui" / "enemyduel", run_root / "duel_ui"),
                "duel_stage_previews": ("ab", staged_root / "arts" / "ui", run_root / "duel_stage_previews"),
            }
            for category, (mode, source, destination) in input_map.items():
                print(f"\n[{category}] Ark-Unpacker -m {mode}", flush=True)
                export_category(python, mode, source, destination, logs_root)

        manifest["outputs"] = output_inventory(run_root)
        catalog_path = workspace / "assets" / "catalog.json"
        manifest["prts_enemy_model_matches"] = find_game_spine_matches(catalog_path, run_root / "battlefield_spine")
        manifest["prts_roster_icon_matches"] = find_game_roster_icon_matches(
            catalog_path,
            run_root / "roster_icons",
            workspace / "assets" / "combat" / "vs2_enemy_combat.json",
        )
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Unpacked resources and manifest: {run_root}")
    print(
        "Enemy Spine matches against the existing PRTS catalog: "
        f"{manifest['prts_enemy_model_matches']['matched_enemy_count']}/"
        f"{manifest['prts_enemy_model_matches']['catalog_enemy_count']}"
    )
    print(
        "VS-2 roster icons matched by internal enemy name: "
        f"{manifest['prts_roster_icon_matches']['vs2_matched_enemy_count']}/"
        f"{manifest['prts_roster_icon_matches']['vs2_enemy_count']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
