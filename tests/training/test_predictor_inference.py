import json
from datetime import UTC, datetime

import pytest

from maa_duel.calibration import BattlefieldCalibration, CalibrationPoint, save_calibration
from maa_duel.combat import (
    DERIVED_FORMULA_VERSION,
    CombatKnowledge,
    CombatStats,
    EnemyCombatProfile,
    StageRules,
    save_combat_knowledge,
)
from maa_duel.contracts import sha256_file
from maa_duel.dataset import BattleState, PredictorUnit
from maa_duel.schema import Winner
from maa_duel.store import write_jsonl
from maa_duel.training.derived import formula_sha256
from maa_duel.training.features import feature_schema_sha256, vocabulary_sha256
from maa_duel.training.inference import predict_duel
from maa_duel.training.predictor import train_predictor_model


def write_calibration(workspace):
    calibration = BattlefieldCalibration(
        calibration_id="green-vine-video-v1",
        image_width=1500,
        image_height=1100,
        map_width=15,
        map_height=11,
        fit_points=[
            CalibrationPoint(image_x=x * 100, image_y=y * 100, ground_x=x, ground_y=y)
            for x, y in ((0, 0), (15, 0), (0, 11), (15, 11), (7.5, 0), (7.5, 11), (0, 5.5), (15, 5.5))
        ],
        check_points=[
            CalibrationPoint(image_x=300, image_y=400, ground_x=3, ground_y=4),
            CalibrationPoint(image_x=1200, image_y=900, ground_x=12, ground_y=9),
        ],
    )
    path = workspace / "assets" / "calibration" / "green-vine-video-v1.json"
    save_calibration(path, calibration)
    return path, sha256_file(path)


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
                count_raw="1",
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
    return path, knowledge


def make_sample(index, winner, calibration_digest):
    from maa_duel.dataset import PredictorSample

    return PredictorSample(
        sample_id=f"{index:032x}",
        source_video_sha256=f"{index + 100:064x}",
        calibration_id="green-vine-video-v1",
        calibration_sha256=calibration_digest,
        left_units=[PredictorUnit(enemy_id=1 + index % 2, x=2, y=3)],
        right_units=[PredictorUnit(enemy_id=3 + index % 2, x=8, y=6)],
        winner=winner,
    )


def metadata_for(knowledge, knowledge_path, calibration_path, calibration_digest, *, count):
    return {
        "accepted_samples": count,
        "written_samples": count,
        "dataset_sha256": "a" * 64,
        "dataset_version": "predictor-test",
        "feature_version": "combat-v2",
        "feature_schema_sha256": feature_schema_sha256(),
        "formula_version": DERIVED_FORMULA_VERSION,
        "formula_sha256": formula_sha256(),
        "knowledge_sha256": sha256_file(knowledge_path),
        "calibration_id": "green-vine-video-v1",
        "calibration_manifest": calibration_path.relative_to(calibration_path.parents[2]).as_posix(),
        "calibration_sha256": calibration_digest,
        "vocabulary_sha256": vocabulary_sha256(knowledge),
    }


def trained_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    calibration_path, calibration_digest = write_calibration(workspace)
    knowledge_path, knowledge = write_combat_knowledge(workspace)
    rows = [
        make_sample(1, Winner.LEFT, calibration_digest),
        make_sample(2, Winner.RIGHT, calibration_digest),
    ]
    write_jsonl(workspace / "manifests" / "predictor.jsonl", rows)
    (workspace / "manifests" / "predictor.meta.json").write_text(
        json.dumps(metadata_for(knowledge, knowledge_path, calibration_path, calibration_digest, count=2)),
        encoding="utf-8",
    )
    train_predictor_model(
        workspace,
        all_samples=True,
        epochs=1,
        batch_size=2,
        embedding_dim=16,
        heads=4,
        layers=1,
        device="cpu",
        workers=0,
    )
    return workspace, calibration_digest


def test_checkpoint_load_prediction_uses_label_free_state_and_exports_relations(tmp_path):
    workspace, calibration_digest = trained_workspace(tmp_path)
    state = BattleState(
        sample_id="f" * 32,
        calibration_id="green-vine-video-v1",
        calibration_sha256=calibration_digest,
        left_units=[PredictorUnit(enemy_id=1, x=2, y=4, instance_id="left-tank")],
        right_units=[PredictorUnit(enemy_id=3, x=8, y=4, instance_id="right-ranged")],
    )

    result = predict_duel(workspace, state, device="cpu", explain=True)

    assert result.left_win_probability + result.right_win_probability == pytest.approx(1.0)
    assert result.feature_version == "combat-v2"
    assert result.formula_version == "derived-v2"
    assert result.explain is not None
    assert result.explain[0]["source_instance_id"] == "left-tank"
    assert {"physical_dps", "arts_dps", "true_dps", "sustained_ttk", "screening_score"} <= set(result.explain[0])


def test_inference_rejects_calibration_mismatch(tmp_path):
    workspace, _ = trained_workspace(tmp_path)
    state = BattleState(
        sample_id="e" * 32,
        calibration_id="wrong-calibration",
        calibration_sha256="0" * 64,
        left_units=[PredictorUnit(enemy_id=1, x=2, y=4)],
        right_units=[PredictorUnit(enemy_id=3, x=8, y=4)],
    )

    with pytest.raises(ValueError, match="calibration"):
        predict_duel(workspace, state, device="cpu")
