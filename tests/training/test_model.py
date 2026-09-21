from datetime import UTC, datetime

import pytest
import torch

from maa_duel.combat import (
    AttackMode,
    CombatKnowledge,
    CombatStats,
    DamageType,
    EnemyCombatProfile,
    MechanicKind,
    StageRules,
)
from maa_duel.dataset import PredictorSample, PredictorUnit
from maa_duel.schema import Winner
from maa_duel.training.features import CombatFeatureIndex, build_combat_feature_table
from maa_duel.training.model import (
    DuelTransformer,
    ModelConfig,
    RelationFeatureIndex,
    pairwise_relation_features,
)
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


def combat_knowledge():
    return CombatKnowledge(
        stage_id="VS-2",
        stage_title="争锋对决！",
        source_page="https://prts.wiki/w/VS-2",
        source_revision=1,
        fetched_at=datetime(2026, 9, 21, tzinfo=UTC),
        rules=StageRules(raw_text=""),
        enemies=[
            EnemyCombatProfile(
                enemy_id=1,
                display_name="测试单位",
                portrait_name="测试单位原型",
                page_name="测试单位原型",
                count_raw="1~99",
                stats=CombatStats(
                    hp=10000,
                    attack=1000,
                    defense=500,
                    resistance=30,
                    attack_interval=2,
                    weight=3,
                    move_speed=1,
                    attack_radius=2.5,
                ),
                attack_modes=[AttackMode.RANGED],
                damage_types=[DamageType.PHYSICAL],
                mechanics=[MechanicKind.DEFENSE_SHRED],
            )
        ],
    )


def test_feature_table_encodes_stats_attributes_and_explicit_unknowns():
    table = build_combat_feature_table(combat_knowledge(), num_enemy_ids=10, knowledge_sha256="a" * 64)

    assert table.features.shape == (10, len(CombatFeatureIndex))
    assert table.known.tolist() == [False, True, False, False, False, False, False, False, False, False]
    assert table.features[1, CombatFeatureIndex.HP].item() > 0
    assert table.features[1, CombatFeatureIndex.PHYSICAL].item() == 1
    assert table.features[1, CombatFeatureIndex.DEFENSE_SHRED].item() == 1
    assert torch.count_nonzero(table.features[2]).item() == 0
    assert table.feature_version == "combat-v1"
    assert table.knowledge_sha256 == "a" * 64


def test_collate_dynamically_pads_and_keeps_shared_battlefield_coordinates():
    combat_table = build_combat_feature_table(combat_knowledge(), num_enemy_ids=10, knowledge_sha256="a" * 64)
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
        ],
        combat_table,
    )

    assert batch.left_ids.shape == (2, 2)
    assert batch.right_ids.shape == (2, 2)
    assert batch.left_mask.tolist() == [[True, False], [True, True]]
    assert batch.right_mask.tolist() == [[True, True], [True, False]]
    assert batch.right_positions[0, 0, 0].item() == pytest.approx(0.8)
    assert batch.left_combat_known.tolist() == [[True, False], [False, False]]
    assert batch.right_combat_known.tolist() == [[False, False], [False, False]]
    assert batch.left_combat[0, 0, CombatFeatureIndex.PHYSICAL].item() == 1
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
            batch.left_combat,
            batch.left_combat_known,
            batch.left_mask,
            batch.right_ids,
            batch.right_positions,
            batch.right_combat,
            batch.right_combat_known,
            batch.right_mask,
        )
        swapped = model(
            batch.right_ids,
            batch.right_positions,
            batch.right_combat,
            batch.right_combat_known,
            batch.right_mask,
            batch.left_ids,
            batch.left_positions,
            batch.left_combat,
            batch.left_combat_known,
            batch.left_mask,
        )

    assert torch.allclose(forward, -swapped, atol=1e-6)
    assert torch.allclose(torch.sigmoid(forward), 1 - torch.sigmoid(swapped), atol=1e-6)


def test_pairwise_relations_express_enemy_matchups_and_friendly_synergy():
    positions = torch.tensor([[[0.2, 0.5], [0.25, 0.5], [0.3, 0.5]]])
    combat = torch.zeros((1, 3, len(CombatFeatureIndex)))
    combat[0, 0, CombatFeatureIndex.ATTACK] = 1
    combat[0, 0, CombatFeatureIndex.PHYSICAL] = 1
    combat[0, 0, CombatFeatureIndex.STUN] = 1
    combat[0, 0, CombatFeatureIndex.DEFENSE_SHRED] = 1
    combat[0, 0, CombatFeatureIndex.DEFENSE_IGNORE] = 1
    combat[0, 1, CombatFeatureIndex.STATUS_IMMUNITY] = 1
    combat[0, 1, CombatFeatureIndex.DEFENSE] = 0.5
    combat[0, 2, CombatFeatureIndex.PHYSICAL] = 1
    sides = torch.tensor([[0, 1, 0]])
    known = torch.ones((1, 3), dtype=torch.bool)

    relations = pairwise_relation_features(positions, combat, known, sides)

    assert relations[0, 0, 1, RelationFeatureIndex.OPPONENT].item() == 1
    assert relations[0, 0, 1, RelationFeatureIndex.PHYSICAL_EFFECTIVENESS].item() > 0
    assert relations[0, 0, 1, RelationFeatureIndex.CONTROL_PRESSURE].item() == 0
    assert relations[0, 0, 1, RelationFeatureIndex.DEFENSE_IGNORE_MATCHUP].item() == 0.5
    assert relations[0, 0, 2, RelationFeatureIndex.SAME_SIDE].item() == 1
    assert relations[0, 0, 2, RelationFeatureIndex.DEFENSE_SHRED_SYNERGY].item() == 1
    assert relations[0, 0, 0, RelationFeatureIndex.SAME_SIDE].item() == 0
    assert relations[0, 0, 0, RelationFeatureIndex.DEFENSE_SHRED_SYNERGY].item() == 0


def test_combat_and_relation_encoders_receive_gradients_from_basic_batch():
    torch.manual_seed(2)
    model = DuelTransformer(ModelConfig(num_enemy_ids=10, embedding_dim=32, heads=4, layers=1, dropout=0.0))
    batch = collate_samples(
        [sample("a" * 32, Winner.LEFT)],
        build_combat_feature_table(combat_knowledge(), num_enemy_ids=10, knowledge_sha256="a" * 64),
    )

    logits = model(
        batch.left_ids,
        batch.left_positions,
        batch.left_combat,
        batch.left_combat_known,
        batch.left_mask,
        batch.right_ids,
        batch.right_positions,
        batch.right_combat,
        batch.right_combat_known,
        batch.right_mask,
    )
    logits.sum().backward()

    assert torch.count_nonzero(model.combat_encoder[0].weight.grad).item() > 0
    assert torch.count_nonzero(model.relation_encoder[0].weight.grad).item() > 0
