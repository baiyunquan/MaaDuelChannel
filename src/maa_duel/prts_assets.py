from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from maa_duel.assets import (
    AssetFile,
    AssetManifest,
    EnemyAsset,
    SpinePackage,
    SpineVariant,
    _read_catalog,
    load_asset_manifest,
)

PRTS_API = "https://prts.wiki/api.php"
ALLOWED_SPINE_HOSTS = {"static.prts.wiki", "torappu.prts.wiki"}
SPINE_VARIANT_NAMES = {"战斗", "战斗正面", "战斗背面"}
ATLAS_PAGE_RE = re.compile(r"^([^\r\n]+)\r?\nsize:\s*\d+\s*,\s*\d+\s*$", re.MULTILINE)


@dataclass(frozen=True)
class PrtsAssetSyncResult:
    manifest: AssetManifest
    portrait_count: int
    spine_package_count: int
    spine_variant_count: int
    missing_portrait_ids: tuple[int, ...]
    missing_spine_ids: tuple[int, ...]
    errors: tuple[str, ...]


class PrtsClient:
    def __init__(self, request_interval: float = 0.25) -> None:
        self.request_interval = max(0.0, request_interval)
        self.last_request_at = 0.0

    def _wait_for_slot(self) -> None:
        remaining = self.request_interval - (time.monotonic() - self.last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def _read_url(self, url: str, *, accept: str) -> bytes:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": accept,
                "User-Agent": "MaaDuelChannel/0.1 (PRTS Wiki asset downloader)",
            },
        )
        for attempt in range(4):
            self._wait_for_slot()
            self.last_request_at = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                    raise
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                time.sleep(delay)
            except urllib.error.URLError:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError(f"request failed after retries: {url}")

    def query_pages(self, titles: list[str]) -> dict[str, dict[str, Any]]:
        pages: dict[str, dict[str, Any]] = {}
        for offset in range(0, len(titles), 50):
            params = urllib.parse.urlencode(
                {
                    "action": "query",
                    "titles": "|".join(titles[offset : offset + 50]),
                    "prop": "imageinfo|revisions",
                    "iiprop": "url",
                    "rvprop": "ids|timestamp|content",
                    "rvslots": "main",
                    "redirects": "1",
                    "format": "json",
                    "formatversion": "2",
                }
            )
            payload = json.loads(self._read_url(f"{PRTS_API}?{params}", accept="application/json"))
            for page in payload["query"]["pages"]:
                pages[_normalize_title(page["title"])] = page
        return pages

    def download(self, url: str) -> bytes:
        return self._read_url(url, accept="*/*")


def _normalize_title(title: str) -> str:
    return " ".join(title.replace("_", " ").split()).casefold()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _catalog_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value and value.strip():
            return value.strip()
    return ""


def _revision_text(page: dict[str, Any]) -> str | None:
    revisions = page.get("revisions")
    if not revisions:
        return None
    revision = revisions[0]
    slots = revision.get("slots")
    if slots:
        return slots.get("main", {}).get("content")
    return revision.get("*")


def _source_uri_for_page(title: str) -> str:
    path = urllib.parse.quote(title.replace(" ", "_"), safe="/:()")
    return f"https://prts.wiki/w/{path}"


def _spine_variants(page: dict[str, Any]) -> tuple[str, list[tuple[str, str]]]:
    raw = _revision_text(page)
    if not raw:
        raise ValueError("Spine page has no revision")
    model = json.loads(raw)
    prefix = model.get("prefix")
    parsed_prefix = urllib.parse.urlsplit(prefix or "")
    if parsed_prefix.scheme != "https" or parsed_prefix.hostname not in ALLOWED_SPINE_HOSTS:
        raise ValueError(f"unrecognized Spine asset host: {prefix!r}")

    skins = model.get("skin", {})
    default_skin = skins.get("默认") or next(iter(skins.values()), {})
    variants = [
        (name, value["file"])
        for name, value in default_skin.items()
        if name in SPINE_VARIANT_NAMES and isinstance(value, dict) and value.get("file")
    ]
    if not variants:
        raise ValueError("Spine page has no default battle skeleton")
    return prefix if prefix.endswith("/") else f"{prefix}/", variants


def _atlas_texture_names(atlas: bytes) -> list[str]:
    text = atlas.decode("utf-8-sig")
    names = [match.group(1).strip() for match in ATLAS_PAGE_RE.finditer(text)]
    if not names:
        raise ValueError("Spine atlas does not declare any texture pages")
    for name in names:
        suffix = PurePosixPath(name.replace("\\", "/")).suffix.casefold()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError(f"unsupported Spine atlas texture: {name}")
    return list(dict.fromkeys(names))


def _save_asset(
    client: PrtsClient,
    workspace: Path,
    relative_path: Path,
    source: str,
    *,
    previous: AssetFile | None = None,
    force: bool = False,
) -> AssetFile:
    destination = workspace / relative_path
    if not force and previous and previous.source == source and destination.is_file():
        data = destination.read_bytes()
    else:
        data = client.download(source)
    if not data:
        raise ValueError(f"downloaded empty asset: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)
    return AssetFile(source=source, relative_path=relative_path.as_posix(), sha256=_sha256(data))


def _previous_spine_files(package: SpinePackage | None) -> dict[str, AssetFile]:
    files: dict[str, AssetFile] = {}
    if package is None:
        return files
    for variant in package.variants:
        for asset in [variant.skeleton, variant.atlas, *variant.textures]:
            files[asset.source] = asset
    return files


def _download_spine_package(
    client: PrtsClient,
    workspace: Path,
    enemy_id: int,
    page_title: str,
    page: dict[str, Any],
    *,
    previous: SpinePackage | None,
    force: bool,
) -> SpinePackage:
    prefix, variant_specs = _spine_variants(page)
    previous_files = _previous_spine_files(previous)
    variants: list[SpineVariant] = []
    for index, (variant_name, file_stem) in enumerate(variant_specs):
        stem = file_stem if PurePosixPath(file_stem).suffix == "" else file_stem.rsplit(".", 1)[0]
        quoted_stem = urllib.parse.quote(stem, safe="/")
        skeleton_url = urllib.parse.urljoin(prefix, f"{quoted_stem}.skel")
        atlas_url = urllib.parse.urljoin(prefix, f"{quoted_stem}.atlas")
        variant_root = Path("assets") / "battlefield_spine" / f"{enemy_id:04d}" / f"variant-{index:02d}"
        skeleton = _save_asset(
            client,
            workspace,
            variant_root / "skeleton.skel",
            skeleton_url,
            previous=previous_files.get(skeleton_url),
            force=force,
        )
        atlas = _save_asset(
            client,
            workspace,
            variant_root / "model.atlas",
            atlas_url,
            previous=previous_files.get(atlas_url),
            force=force,
        )
        atlas_bytes = (workspace / atlas.relative_path).read_bytes()
        textures: list[AssetFile] = []
        for texture_index, texture_name in enumerate(_atlas_texture_names(atlas_bytes)):
            encoded_name = urllib.parse.quote(texture_name.replace("\\", "/"), safe="/")
            texture_url = urllib.parse.urljoin(atlas_url, encoded_name)
            texture_path = PurePosixPath(texture_name.replace("\\", "/"))
            texture_relative = variant_root / "textures" / f"{texture_index:02d}-{texture_path.name}"
            textures.append(
                _save_asset(
                    client,
                    workspace,
                    texture_relative,
                    texture_url,
                    previous=previous_files.get(texture_url),
                    force=force,
                )
            )
        variants.append(SpineVariant(name=variant_name, skeleton=skeleton, atlas=atlas, textures=textures))

    return SpinePackage(source_page=_source_uri_for_page(page_title), variants=variants)


def sync_prts_assets(
    catalog: Path,
    workspace: Path,
    *,
    force: bool = False,
    request_interval: float = 0.25,
) -> PrtsAssetSyncResult:
    catalog = catalog.resolve(strict=True)
    asset_root = workspace / "assets"
    manifest_path = asset_root / "catalog.json"
    previous_manifest = load_asset_manifest(manifest_path) if manifest_path.is_file() else None
    previous_enemies = {enemy.enemy_id: enemy for enemy in previous_manifest.enemies} if previous_manifest else {}

    rows = _read_catalog(catalog)
    records: list[tuple[int, str, str, str]] = []
    seen_ids: set[int] = set()
    for row in rows:
        raw_id = _catalog_value(row, "id", "enemy_id", "ID")
        if not raw_id:
            raise ValueError("catalog row is missing id")
        enemy_id = int(raw_id)
        if enemy_id in seen_ids:
            raise ValueError(f"duplicate enemy id in catalog: {enemy_id}")
        seen_ids.add(enemy_id)
        name = _catalog_value(row, "name", "名称") or str(enemy_id)
        original_name = _catalog_value(row, "original_name", "原始名称") or name
        records.append((enemy_id, name, original_name, f"文件:头像 敌人 {original_name}.png"))

    requested_titles = [
        title for _, _, original_name, portrait_title in records for title in (portrait_title, f"{original_name}/spine")
    ]
    client = PrtsClient(request_interval=request_interval)
    pages = client.query_pages(requested_titles)

    enemies: list[EnemyAsset] = []
    errors: list[str] = []
    for enemy_id, name, original_name, portrait_title in records:
        previous_enemy = previous_enemies.get(enemy_id)
        row_errors: list[str] = []
        portrait_page = pages.get(_normalize_title(portrait_title), {})
        portrait_info = (portrait_page.get("imageinfo") or [None])[0]
        portrait = previous_enemy.portrait if previous_enemy else None
        if portrait_info and portrait_info.get("url"):
            url = portrait_info["url"]
            suffix = Path(urllib.parse.urlsplit(url).path).suffix.casefold() or ".png"
            relative = Path("assets") / "portraits" / f"{enemy_id:04d}" / f"thumbnail{suffix}"
            try:
                portrait = _save_asset(
                    client,
                    workspace,
                    relative,
                    url,
                    previous=previous_enemy.portrait if previous_enemy else None,
                    force=force,
                )
            except Exception as error:
                message = f"enemy {enemy_id} portrait: {error}"
                row_errors.append(message)
                errors.append(message)

        spine_title = f"{original_name}/spine"
        spine_page = pages.get(_normalize_title(spine_title), {})
        spine = previous_enemy.battlefield_spine if previous_enemy else None
        if _revision_text(spine_page):
            try:
                spine = _download_spine_package(
                    client,
                    workspace,
                    enemy_id,
                    spine_title,
                    spine_page,
                    previous=previous_enemy.battlefield_spine if previous_enemy else None,
                    force=force,
                )
            except Exception as error:
                message = f"enemy {enemy_id} battlefield Spine: {error}"
                row_errors.append(message)
                errors.append(message)

        missing = []
        if portrait is None:
            missing.append("portrait")
        if spine is None:
            missing.append("battlefield_spine")
        if previous_enemy and previous_enemy.animation is not None:
            animation = previous_enemy.animation
        else:
            animation = None
            missing.append("animation")
        enemies.append(
            EnemyAsset(
                enemy_id=enemy_id,
                name=name,
                original_name=original_name,
                portrait=portrait,
                animation=animation,
                battlefield_spine=spine,
                missing=missing,
                errors=row_errors,
            )
        )

    backgrounds = previous_manifest.backgrounds.copy() if previous_manifest else []
    manifest = AssetManifest(
        source_catalog=catalog.as_uri(),
        source_catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        enemies=sorted(enemies, key=lambda enemy: enemy.enemy_id),
        backgrounds=backgrounds,
    )
    asset_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")

    return PrtsAssetSyncResult(
        manifest=manifest,
        portrait_count=sum(enemy.portrait is not None for enemy in manifest.enemies),
        spine_package_count=sum(enemy.battlefield_spine is not None for enemy in manifest.enemies),
        spine_variant_count=sum(
            len(enemy.battlefield_spine.variants) for enemy in manifest.enemies if enemy.battlefield_spine is not None
        ),
        missing_portrait_ids=tuple(enemy.enemy_id for enemy in manifest.enemies if enemy.portrait is None),
        missing_spine_ids=tuple(enemy.enemy_id for enemy in manifest.enemies if enemy.battlefield_spine is None),
        errors=tuple(errors),
    )
