import pytest
import torch

from maa_duel.dataset import PredictorSample, PredictorUnit
from maa_duel.schema import Winner
from maa_duel.training.model import DuelTransformer, ModelConfig
from maa_duel.training.predictor import collate_samples


def sample(sample_id, winner, left_x=0.2, right_x=0.8):
    return PredictorSample(
        sample_id=sample_id,
        source_video_sha256="f" * 64,
        left_units=[PredictorUnit(enemy_id=1, x=left_x, y=0.4)],
        right_units=[
            PredictorUnit(enemy_id=2, x=right_x, y=0.4),
            PredictorUnit(enemy_id=3, x=0.7, y=0.5),
        ],
        winner=winner,
    )


def test_collate_dynamically_pads_and_mirrors_right_coordinates():
    batch = collate_samples(
        [
            sample("a" * 32, Winner.LEFT),
            PredictorSample(
                sample_id="b" * 32,
                source_video_sha256="e" * 64,
                left_units=[PredictorUnit(enemy_id=4, x=0.1, y=0.2), PredictorUnit(enemy_id=5, x=0.3, y=0.4)],
                right_units=[PredictorUnit(enemy_id=6, x=0.9, y=0.6)],
                winner=Winner.RIGHT,
            ),
        ]
    )

    assert batch.left_ids.shape == (2, 2)
    assert batch.right_ids.shape == (2, 2)
    assert batch.left_mask.tolist() == [[True, False], [True, True]]
    assert batch.right_mask.tolist() == [[True, True], [True, False]]
    assert batch.right_positions[0, 0, 0].item() == pytest.approx(0.2)
    assert batch.labels.tolist() == [1.0, 0.0]


def test_model_is_antisymmetric_when_sides_are_swapped():
    torch.manual_seed(1)
    model = DuelTransformer(ModelConfig(num_enemy_ids=10, embedding_dim=32, heads=4, layers=1, dropout=0.0))
    model.eval()
    batch = collate_samples([sample("a" * 32, Winner.LEFT)])

    with torch.no_grad():
        forward = model(
            batch.left_ids,
            batch.left_positions,
            batch.left_mask,
            batch.right_ids,
            batch.right_positions,
            batch.right_mask,
        )
        swapped = model(
            batch.right_ids,
            batch.right_positions,
            batch.right_mask,
            batch.left_ids,
            batch.left_positions,
            batch.left_mask,
        )

    assert torch.allclose(forward, -swapped, atol=1e-6)
    assert torch.allclose(torch.sigmoid(forward), 1 - torch.sigmoid(swapped), atol=1e-6)
