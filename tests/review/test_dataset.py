import json

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
