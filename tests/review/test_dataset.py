import json
from datetime import UTC, datetime

import pytest

from maa_duel.calibration import BattlefieldCalibration, CalibrationPoint, save_calibration
from maa_duel.combat import CombatKnowledge, StageRules, save_combat_knowledge
from maa_duel.contracts import DatasetVersion, sha256_file
from maa_duel.dataset import BattleState, PredictorSample, build_predictor_dataset
from maa_duel.extraction_state import ActiveExtraction
from maa_duel.review import ReviewCorrection, ReviewStore
from maa_duel.schema import ReviewStatus, Winner
from maa_duel.store import read_jsonl, write_jsonl
from maa_duel.training.derived import formula_sha256
from tests.review.test_review import make_sample


def write_calibration(workspace):
    points = [
        CalibrationPoint(image_x=x * 100, image_y=y * 100, ground_x=x, ground_y=y)
        for x, y in ((0, 0), (15, 0), (0, 11), (15, 11), (7.5, 0), (7.5, 11), (0, 5.5), (15, 5.5))
    ]
    calibration = BattlefieldCalibration(
        calibration_id="green-vine-video-v1",
        image_width=1500,
        image_height=1100,
        map_width=15,
        map_height=11,
        fit_points=points,
        check_points=[
            CalibrationPoint(image_x=300, image_y=400, ground_x=3, ground_y=4),
            CalibrationPoint(image_x=1200, image_y=900, ground_x=12, ground_y=9),
        ],
    )
    path = workspace / "assets" / "calibration" / "green-vine-video-v1.json"
    save_calibration(path, calibration)
    return path, calibration


def test_dataset_builder_uses_manual_overlay_and_only_accepted_samples(tmp_path):
    workspace = tmp_path / "workspace"
    _, calibration = write_calibration(workspace)
    manifest_dir = workspace / "manifests"
    auto_left = make_sample("a" * 32, ReviewStatus.ACCEPTED)
    auto_pending = make_sample("c" * 32, ReviewStatus.PENDING)
    write_jsonl(manifest_dir / "rounds.auto.jsonl", [auto_left, auto_pending])

    corrected = make_sample("c" * 32, ReviewStatus.ACCEPTED)
    corrected.winner = Winner.RIGHT
    ReviewStore(workspace / "review" / "corrections.jsonl").save(
        ReviewCorrection(sample=corrected, note="accepted manually")
    )

    result = build_predictor_dataset(workspace)
    rows = read_jsonl(manifest_dir / "predictor.jsonl", PredictorSample)
    metadata = json.loads((manifest_dir / "predictor.meta.json").read_text(encoding="utf-8"))

    assert len(result) == 2
    assert len(rows) == 2
    assert rows[1].winner is Winner.RIGHT
    assert rows[0].calibration_id == calibration.calibration_id
    assert rows[0].left_units[0].raw_x == 0.2
    assert rows[0].left_units[0].x == pytest.approx(3.0)
    assert metadata["accepted_samples"] == 2
    assert metadata["written_samples"] == 2
    assert len(metadata["dataset_sha256"]) == 64


def test_dataset_builder_uses_active_run_and_ignores_legacy_correction(tmp_path):
    workspace = tmp_path / "workspace"
    write_calibration(workspace)
    sample_id = "f" * 32

    legacy_auto = make_sample(sample_id, ReviewStatus.ACCEPTED)
    legacy_auto.winner = Winner.LEFT
    write_jsonl(workspace / "manifests" / "rounds.auto.jsonl", [legacy_auto])
    legacy_corrected = make_sample(sample_id, ReviewStatus.ACCEPTED)
    legacy_corrected.winner = Winner.LEFT
    ReviewStore(workspace / "review" / "corrections.jsonl").save(
        ReviewCorrection(sample=legacy_corrected, note="legacy correction must be ignored")
    )

    run_id = "20260923T120000000000Z-active02"
    run_root = workspace / "extraction-runs" / run_id
    active_auto = make_sample(sample_id, ReviewStatus.ACCEPTED)
    active_auto.winner = Winner.RIGHT
    write_jsonl(run_root / "manifests" / "rounds.auto.jsonl", [active_auto])
    write_jsonl(run_root / "review" / "corrections.jsonl", [])
    (workspace / "manifests" / "extraction.active.json").write_text(
        ActiveExtraction(run_id=run_id).model_dump_json(),
        encoding="utf-8",
    )

    result = build_predictor_dataset(workspace)
    rows = read_jsonl(workspace / "manifests" / "predictor.jsonl", PredictorSample)

    assert len(result) == 1
    assert len(rows) == 1
    assert result[0].sample_id == sample_id
    assert rows[0].winner is Winner.RIGHT


def test_dataset_contract_binds_combat_knowledge_version_and_hash(tmp_path):
    workspace = tmp_path / "workspace"
    calibration_path, calibration = write_calibration(workspace)
    write_jsonl(workspace / "manifests" / "rounds.auto.jsonl", [make_sample("a" * 32, ReviewStatus.ACCEPTED)])
    knowledge_path = workspace / "assets" / "combat" / "vs2_enemy_combat.json"
    save_combat_knowledge(
        knowledge_path,
        CombatKnowledge(
            stage_id="VS-2",
            stage_title="争锋对决！",
            source_page="https://prts.wiki/w/VS-2",
            source_revision=1,
            fetched_at=datetime(2026, 9, 21, tzinfo=UTC),
            rules=StageRules(raw_text=""),
            enemies=[],
        ),
    )

    build_predictor_dataset(workspace)

    metadata = json.loads((workspace / "manifests" / "predictor.meta.json").read_text(encoding="utf-8"))
    contract_path = next((workspace / "datasets").glob("predictor-*/dataset.json"))
    contract = DatasetVersion.model_validate_json(contract_path.read_text(encoding="utf-8"))
    assert metadata["feature_version"] == "combat-v2"
    assert metadata["knowledge_manifest"] == "assets/combat/vs2_enemy_combat.json"
    assert metadata["knowledge_sha256"] == sha256_file(knowledge_path)
    assert contract.feature_version == "combat-v2"
    assert contract.knowledge_sha256 == sha256_file(knowledge_path)
    assert metadata["calibration_id"] == calibration.calibration_id
    assert metadata["calibration_sha256"] == sha256_file(calibration_path)
    assert contract.calibration_sha256 == sha256_file(calibration_path)
    assert metadata["formula_sha256"] == formula_sha256()
    assert contract.formula_sha256 == formula_sha256()


def test_battle_state_has_no_winner_and_training_sample_wraps_the_label():
    assert "winner" not in BattleState.model_fields
    assert "winner" in PredictorSample.model_fields
    with pytest.raises(ValueError, match="both sides"):
        BattleState(sample_id="e" * 32, left_units=[], right_units=[])


def test_dataset_builder_lists_samples_missing_required_calibration(tmp_path):
    workspace = tmp_path / "workspace"
    sample = make_sample("d" * 32, ReviewStatus.ACCEPTED)
    write_jsonl(workspace / "manifests" / "rounds.auto.jsonl", [sample])

    with pytest.raises(ValueError, match="d{32}.*calibration"):
        build_predictor_dataset(workspace)


def test_dataset_builder_rejects_duplicate_sample_ids(tmp_path):
    workspace = tmp_path / "workspace"
    sample = make_sample()
    write_jsonl(workspace / "manifests" / "rounds.auto.jsonl", [sample, sample])

    try:
        build_predictor_dataset(workspace)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate sample ids must be rejected")
