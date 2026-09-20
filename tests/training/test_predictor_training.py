import json

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
    (workspace / "manifests" / "predictor.meta.json").write_text(
        json.dumps({"accepted_samples": 4, "written_samples": 4, "dataset_sha256": "a" * 64}),
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
    assert (workspace / "models" / "predictor" / "last.pt").is_file()
    assert (workspace / "models" / "predictor" / "best-train-loss.pt").is_file()


def test_train_predictor_requires_all_flag(tmp_path):
    try:
        train_predictor_model(tmp_path, all_samples=False)
    except ValueError as exc:
        assert "--all" in str(exc)
    else:
        raise AssertionError("training without --all must be rejected")
