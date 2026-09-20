from __future__ import annotations

import csv
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AssetFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EnemyAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enemy_id: int = Field(ge=1)
    name: str
    original_name: str
    portrait: AssetFile | None = None
    animation: AssetFile | None = None
    missing: list[Literal["portrait", "animation"]] = Field(default_factory=list)


class AssetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    source_catalog: str
    source_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    enemies: list[EnemyAsset]
    backgrounds: list[AssetFile] = Field(default_factory=list)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_source(value: str, catalog_dir: Path) -> tuple[bytes, str, str]:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme in {"http", "https"}:
        with urllib.request.urlopen(value, timeout=60) as response:
            data = response.read()
        suffix = Path(parsed.path).suffix or ".bin"
        return data, value, suffix
    source_path = (catalog_dir / value).resolve(strict=True)
    return source_path.read_bytes(), source_path.as_uri(), source_path.suffix or ".bin"


def _first(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and value.strip():
            return value.strip().strip("“”")
    return ""


def _read_catalog(path: Path) -> list[dict[str, str]]:
    if path.suffix.casefold() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            payload = payload.get("enemies", [])
        return [{str(key): str(value) if value is not None else "" for key, value in row.items()} for row in payload]
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sync_assets(catalog: Path, workspace: Path, *, background_dir: Path | None = None) -> AssetManifest:
    catalog = catalog.resolve(strict=True)
    asset_root = workspace / "assets"
    raw_root = asset_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    enemies: list[EnemyAsset] = []
    seen_ids: set[int] = set()

    for row in _read_catalog(catalog):
        raw_id = _first(row, "id", "enemy_id", "ID")
        if not raw_id:
            raise ValueError("catalog row is missing id")
        enemy_id = int(raw_id)
        if enemy_id in seen_ids:
            raise ValueError(f"duplicate enemy id in catalog: {enemy_id}")
        seen_ids.add(enemy_id)
        name = _first(row, "name", "名称") or str(enemy_id)
        original_name = _first(row, "original_name", "原始名称") or name
        files: dict[str, AssetFile | None] = {"portrait": None, "animation": None}
        missing: list[Literal["portrait", "animation"]] = []

        for kind, keys in (
            ("portrait", ("portrait", "portrait_url", "头像")),
            ("animation", ("animation", "animation_url", "动画")),
        ):
            source = _first(row, *keys)
            if not source:
                missing.append(kind)
                continue
            data, provenance, suffix = _read_source(source, catalog.parent)
            relative = Path("assets") / "raw" / f"{enemy_id:04d}" / f"{kind}{suffix.casefold()}"
            destination = workspace / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = _sha256_bytes(data)
            if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                destination.write_bytes(data)
            files[kind] = AssetFile(source=provenance, relative_path=relative.as_posix(), sha256=digest)

        enemies.append(
            EnemyAsset(
                enemy_id=enemy_id,
                name=name,
                original_name=original_name,
                portrait=files["portrait"],
                animation=files["animation"],
                missing=missing,
            )
        )

    backgrounds: list[AssetFile] = []
    if background_dir is not None:
        for source_path in sorted(background_dir.resolve(strict=True).iterdir()):
            if not source_path.is_file() or source_path.suffix.casefold() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            data = source_path.read_bytes()
            relative = Path("assets") / "backgrounds" / source_path.name
            destination = workspace / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            backgrounds.append(
                AssetFile(
                    source=source_path.as_uri(),
                    relative_path=relative.as_posix(),
                    sha256=_sha256_bytes(data),
                )
            )

    manifest = AssetManifest(
        source_catalog=catalog.as_uri(),
        source_catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        enemies=sorted(enemies, key=lambda item: item.enemy_id),
        backgrounds=backgrounds,
    )
    manifest_path = asset_root / "catalog.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    return manifest


def load_asset_manifest(path: Path) -> AssetManifest:
    return AssetManifest.model_validate_json(path.read_text(encoding="utf-8"))

