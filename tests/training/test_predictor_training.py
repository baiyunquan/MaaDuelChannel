import json
from datetime import UTC, datetime

import torch

from maa_duel.combat import CombatKnowledge, CombatStats, EnemyCombatProfile, StageRules, save_combat_knowledge
from maa_duel.contracts import ModelVersion, sha256_file
from maa_duel.dataset import PredictorSample, PredictorUnit
from maa_duel.schema import Winner
from maa_duel.store import write_jsonl
from maa_duel.training.predictor import train_predictor_model


def make_sample(index, winner):
    return PredictorSample(
        sample_id=f"{index:032x}",
        source_video_sha256=f"{index + 100:064x}",
        left_units=[PredictorUnit(enemy_id=1 + index % 2, x=0.2, y=0.3)],
        right_units=[PredictorUnit(enemy_id=3 + index % 2, x=0.8, y=0.6)],
        winner=winner,
    )


def write_combat_knowledge(workspace):
    knowledge = CombatKnowledge(
        stage_id="VS-2",
        stage_title="争锋对决！",
        source_page="https://prts.wiki/w/VS-2",
        source_revision=1,
        fetched_at=datetime(2026, 9, 21, tzinfo=UTC),
        rules=StageRules(raw_text=""),
        enemies=[
            EnemyCombatProfile(
                enemy_id=enemy_id,
                display_name=f"enemy-{enemy_id}",
                portrait_name=f"enemy-{enemy_id}",
                page_name=f"enemy-{enemy_id}",
                count_raw="1~99",
                stats=CombatStats(
                    hp=1000 * enemy_id,
                    attack=100 * enemy_id,
                    defense=50 * enemy_id,
                    resistance=10,
                    attack_interval=1.5,
                    weight=1,
                    move_speed=1,
                    attack_radius=1,
                ),
            )
            for enemy_id in range(1, 5)
        ],
    )
    path = workspace / "assets" / "combat" / "vs2_enemy_combat.json"
    save_combat_knowledge(path, knowledge)
    return path


def test_train_predictor_uses_every_row_and_writes_reproducible_metadata(tmp_path):
    workspace = tmp_path / "workspace"
    rows = [
        make_sample(1, Winner.LEFT),
        make_sample(2, Winner.RIGHT),
        make_sample(3, Winner.LEFT),
        make_sample(4, Winner.RIGHT),
    ]
    manifest = workspace / "manifests" / "predictor.jsonl"
    write_jsonl(manifest, rows)
    knowledge_path = write_combat_knowledge(workspace)
    (workspace / "manifests" / "predictor.meta.json").write_text(
        json.dumps(
            {
                "accepted_samples": 4,
                "written_samples": 4,
                "dataset_sha256": "a" * 64,
                "feature_version": "combat-v1",
                "knowledge_sha256": sha256_file(knowledge_path),
            }
        ),
        encoding="utf-8",
    )

    report = train_predictor_model(
        workspace,
        all_samples=True,
        epochs=1,
        batch_size=2,
        embedding_dim=16,
        heads=4,
        layers=1,
        device="cpu",
    )

    assert report["training_samples"] == 4
    assert report["validation_samples"] == 0
    assert report["dataset_sha256"] == "a" * 64
    assert report["feature_version"] == "combat-v1"
    assert report["knowledge_sha256"] == sha256_file(knowledge_path)
    assert report["knowledge_coverage"] == {"known_units": 8, "total_units": 8}
    assert (workspace / "models" / "predictor" / "last.pt").is_file()
    assert (workspace / "models" / "predictor" / "best-train-loss.pt").is_file()
    checkpoint = torch.load(workspace / "models" / "predictor" / "last.pt", weights_only=False)
    assert checkpoint["feature_version"] == "combat-v1"
    assert checkpoint["knowledge_sha256"] == sha256_file(knowledge_path)
    contract = ModelVersion.model_validate_json((workspace / "models" / "predictor" / "model.json").read_text())
    assert contract.base_model == "CombatAwareDuelTransformer"
    assert contract.feature_version == "combat-v1"
    assert contract.knowledge_sha256 == sha256_file(knowledge_path)


def test_train_predictor_rejects_dataset_built_with_different_knowledge(tmp_path):
    workspace = tmp_path / "workspace"
    rows = [make_sample(1, Winner.LEFT), make_sample(2, Winner.RIGHT)]
    write_jsonl(workspace / "manifests" / "predictor.jsonl", rows)
    write_combat_knowledge(workspace)
    (workspace / "manifests" / "predictor.meta.json").write_text(
        json.dumps(
            {
                "accepted_samples": 2,
                "written_samples": 2,
                "dataset_sha256": "a" * 64,
                "feature_version": "combat-v1",
                "knowledge_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )

    try:
        train_predictor_model(workspace, all_samples=True, epochs=1, device="cpu", workers=0)
    except ValueError as exc:
        assert "knowledge hash" in str(exc)
    else:
        raise AssertionError("training must reject a stale combat knowledge contract")


def test_train_predictor_requires_all_flag(tmp_path):
    try:
        train_predictor_model(tmp_path, all_samples=False)
    except ValueError as exc:
        assert "--all" in str(exc)
    else:
        raise AssertionError("training without --all must be rejected")
