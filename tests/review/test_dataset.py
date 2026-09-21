import json
from datetime import UTC, datetime

from maa_duel.combat import CombatKnowledge, StageRules, save_combat_knowledge
from maa_duel.contracts import DatasetVersion, sha256_file
from maa_duel.dataset import PredictorSample, build_predictor_dataset
from maa_duel.review import ReviewCorrection, ReviewStore
from maa_duel.schema import ReviewStatus, Winner
from maa_duel.store import read_jsonl, write_jsonl
from tests.review.test_review import make_sample


def test_dataset_builder_uses_manual_overlay_and_only_accepted_samples(tmp_path):
    workspace = tmp_path / "workspace"
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
    assert metadata["accepted_samples"] == 2
    assert metadata["written_samples"] == 2
    assert len(metadata["dataset_sha256"]) == 64


def test_dataset_contract_binds_combat_knowledge_version_and_hash(tmp_path):
    workspace = tmp_path / "workspace"
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
    assert metadata["feature_version"] == "combat-v1"
    assert metadata["knowledge_manifest"] == "assets/combat/vs2_enemy_combat.json"
    assert metadata["knowledge_sha256"] == sha256_file(knowledge_path)
    assert contract.feature_version == "combat-v1"
    assert contract.knowledge_sha256 == sha256_file(knowledge_path)


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
