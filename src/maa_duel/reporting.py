from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from maa_duel.assets import load_asset_manifest
from maa_duel.review import ReviewStore
from maa_duel.schema import ReviewStatus, RoundSample, Winner
from maa_duel.store import read_jsonl
from maa_duel.video.inventory import Arena, VideoRecord


def _load_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_report(workspace: Path) -> Path:
    manifest_dir = workspace / "manifests"
    report_dir = workspace / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    videos = read_jsonl(manifest_dir / "videos.jsonl", VideoRecord)
    automatic = read_jsonl(manifest_dir / "rounds.auto.jsonl", RoundSample)
    effective = ReviewStore(workspace / "review" / "corrections.jsonl").overlay(automatic)

    video_summary: dict[str, dict[str, float | int]] = defaultdict(lambda: {"count": 0, "duration_seconds": 0.0})
    for video in videos:
        values = video_summary[video.arena.value]
        values["count"] = int(values["count"]) + 1
        values["duration_seconds"] = round(float(values["duration_seconds"]) + video.duration, 3)
    for arena in Arena:
        video_summary[arena.value]

    statuses = Counter(sample.review_status.value for sample in effective)
    winners = Counter(sample.winner.value for sample in effective if sample.winner is not None)
    predictor = _load_json(manifest_dir / "predictor.meta.json", {})
    extraction_errors = _load_json(report_dir / "extraction-errors.json", [])
    asset_path = workspace / "assets" / "catalog.json"
    asset_summary: dict[str, int] = {}
    if asset_path.exists():
        assets = load_asset_manifest(asset_path)
        asset_summary = {
            "enemies": len(assets.enemies),
            "portraits": sum(enemy.portrait is not None for enemy in assets.enemies),
            "animations": sum(enemy.animation is not None for enemy in assets.enemies),
            "battlefield_spine_packages": sum(enemy.battlefield_spine is not None for enemy in assets.enemies),
            "battlefield_spine_variants": sum(
                len(enemy.battlefield_spine.variants) for enemy in assets.enemies if enemy.battlefield_spine is not None
            ),
            "backgrounds": len(assets.backgrounds),
        }

    payload = {
        "schema_version": 1,
        "videos": dict(sorted(video_summary.items())),
        "rounds": {
            ReviewStatus.ACCEPTED.value: statuses[ReviewStatus.ACCEPTED.value],
            ReviewStatus.PENDING.value: statuses[ReviewStatus.PENDING.value],
            ReviewStatus.REJECTED.value: statuses[ReviewStatus.REJECTED.value],
            "winner_left": winners[Winner.LEFT.value],
            "winner_right": winners[Winner.RIGHT.value],
        },
        "predictor": predictor,
        "assets": asset_summary,
        "extraction_error_count": len(extraction_errors),
    }
    json_path = report_dir / "summary.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Duel Channel pipeline report",
        "",
        "## Videos",
        "",
        "| Arena | Count | Duration (s) |",
        "|---|---:|---:|",
    ]
    for arena, values in payload["videos"].items():
        lines.append(f"| {arena} | {values['count']} | {values['duration_seconds']} |")
    lines.extend(
        [
            "",
            "## Rounds",
            "",
            f"- Accepted: {payload['rounds']['accepted']}",
            f"- Pending: {payload['rounds']['pending']}",
            f"- Rejected: {payload['rounds']['rejected']}",
            f"- Left wins: {payload['rounds']['winner_left']}",
            f"- Right wins: {payload['rounds']['winner_right']}",
            "",
            "## Predictor",
            "",
            f"- Written samples: {predictor.get('written_samples', 0)}",
            f"- Dataset SHA-256: {predictor.get('dataset_sha256', 'not built')}",
            f"- Extraction errors: {len(extraction_errors)}",
            "",
        ]
    )
    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return json_path
