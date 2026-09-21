from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_ark_lz4ak_decompressor() -> None:
    from UnityPy.enums.BundleFile import CompressionFlags
    from UnityPy.helpers import CompressionHelper

    vendor_root = Path(__file__).resolve().parents[2] / "vendor" / "Ark-Unpacker"
    if not vendor_root.is_dir():
        raise FileNotFoundError(f"Ark-Unpacker submodule is required to decode Arknights AssetBundles: {vendor_root}")
    vendor_root_text = str(vendor_root)
    if vendor_root_text not in sys.path:
        sys.path.insert(0, vendor_root_text)
    try:
        from src.lz4ak.Block import decompress_lz4ak
    except ImportError as error:
        raise RuntimeError(f"cannot import Ark-Unpacker's LZ4AK decoder: {error}") from error
    CompressionHelper.DECOMPRESSION_MAP[CompressionFlags.LZHAM] = decompress_lz4ak


def _normalization_metadata_path(bundle_path: Path) -> Path:
    return bundle_path.with_suffix(bundle_path.suffix + ".json")


def _normalize_bundle(source: Path, source_sha256: str, destination: Path) -> tuple[str, str]:
    from UnityPy import load as load_unity_bundle

    _install_ark_lz4ak_decompressor()
    try:
        environment = load_unity_bundle(str(source))
        bundle = environment.file
        if bundle.signature != "UnityFS":
            raise ValueError(f"expected UnityFS bundle, got {bundle.signature!r}")
        normalized_bytes = bundle.save(packer="none")
    except Exception as error:
        raise RuntimeError(f"could not normalize Arknights bundle {source}: {error}") from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(normalized_bytes)
    os.replace(temporary, destination)
    normalized_sha256 = hashlib.sha256(normalized_bytes).hexdigest()
    metadata = {
        "sourcePath": source.as_posix(),
        "sourceSha256": source_sha256,
        "normalizedPath": destination.as_posix(),
        "normalizedSha256": normalized_sha256,
        "format": "UnityFS-uncompressed",
        "unityPyVersion": importlib.metadata.version("UnityPy"),
        "arkUnpackerCommit": _ark_unpacker_commit(),
    }
    _normalization_metadata_path(destination).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    del environment
    return normalized_sha256, metadata["format"]


def _ark_unpacker_commit() -> str:
    head = Path(__file__).resolve().parents[2] / "vendor" / "Ark-Unpacker" / ".git"
    if not head.exists():
        return "unknown"
    if head.is_file():
        git_dir_line = head.read_text(encoding="utf-8").strip()
        if git_dir_line.startswith("gitdir:"):
            git_dir = Path(git_dir_line.partition(":")[2].strip())
            if not git_dir.is_absolute():
                git_dir = (head.parent / git_dir).resolve()
            head_path = git_dir / "HEAD"
        else:
            head_path = head / "HEAD"
    else:
        head_path = head / "HEAD"
    if not head_path.is_file():
        return "unknown"
    value = head_path.read_text(encoding="utf-8").strip()
    if value.startswith("ref:"):
        ref_path = head_path.parent / value.partition(" ")[2]
        return ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else "unknown"
    return value


def _cached_bundle(
    source: Path,
    source_sha256: str,
    cache_directory: Path,
) -> tuple[Path, str, str]:
    destination = cache_directory / f"{source_sha256}.unityfs"
    metadata_path = _normalization_metadata_path(destination)
    if destination.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        normalized_sha256 = metadata.get("normalizedSha256")
        if (
            metadata.get("sourceSha256") == source_sha256
            and metadata.get("format") == "UnityFS-uncompressed"
            and isinstance(normalized_sha256, str)
            and _sha256(destination) == normalized_sha256
        ):
            return destination, normalized_sha256, "UnityFS-uncompressed"

    if _sha256(source) != source_sha256:
        raise ValueError(f"source AssetBundle hash changed after resource resolution: {source}")
    normalized_sha256, normalization = _normalize_bundle(source, source_sha256, destination)
    return destination, normalized_sha256, normalization


def prepare_unity_compatible_input(config_path: Path, output_directory: Path) -> Path:
    if importlib.util.find_spec("UnityPy") is None:
        raise RuntimeError(
            "Unity bundle compatibility needs UnityPy; install the `asset-bundles` extra with "
            "`uv sync --extra asset-bundles`."
        )

    config_path = config_path.resolve(strict=True)
    output_directory = output_directory.resolve()
    try:
        config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read canonical Unity assembly input {config_path}: {error}") from error
    bundle_files = config.get("bundleFiles")
    if not isinstance(bundle_files, list) or not bundle_files:
        raise ValueError("canonical Unity assembly input has no resolved bundle files")

    cache_directory = output_directory / ".map-assembly" / "bundle-cache"
    normalized_bundle_files: list[dict[str, Any]] = []
    total = len(bundle_files)
    for index, item in enumerate(bundle_files, start=1):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
        ):
            raise ValueError(f"canonical assembly bundle file {index} is invalid")
        source = Path(item["path"]).resolve(strict=True)
        normalized_path, normalized_sha256, normalization = _cached_bundle(source, item["sha256"], cache_directory)
        prepared_item = dict(item)
        prepared_item["originalPath"] = source.as_posix()
        prepared_item["path"] = normalized_path.as_posix()
        prepared_item["normalizedSha256"] = normalized_sha256
        prepared_item["normalization"] = normalization
        normalized_bundle_files.append(prepared_item)
        print(f"Normalized {index}/{total}: {item.get('relativePath', source.name)}", flush=True)

    unity_config = dict(config)
    unity_config["bundleFiles"] = normalized_bundle_files
    unity_config["bundleCompatibility"] = {
        "normalization": "UnityFS-uncompressed",
        "unityPyVersion": importlib.metadata.version("UnityPy"),
        "arkUnpackerCommit": _ark_unpacker_commit(),
    }
    unity_config_path = config_path.with_name("assembly-input-unity.json")
    unity_config_path.write_text(json.dumps(unity_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return unity_config_path
