from datetime import UTC, datetime

import pytest
import torch

from maa_duel.combat import AttackMode, CombatKnowledge, CombatStats, DamageType, EnemyCombatProfile, StageRules
from maa_duel.dataset import PredictorSample, PredictorUnit
from maa_duel.schema import Winner
from maa_duel.training.derived import FormationFeatureIndex, RelationFeatureIndex, RelationMaskIndex
from maa_duel.training.features import CombatFeatureIndex, build_combat_feature_table
from maa_duel.training.model import DuelTransformer, ModelConfig
from maa_duel.training.predictor import collate_samples


def sample(sample_id, winner, left_x=2.0, right_x=8.0):
    return PredictorSample(
        sample_id=sample_id,
        source_video_sha256="f" * 64,
        left_units=[PredictorUnit(enemy_id=1, x=left_x, y=4.0)],
        right_units=[
            PredictorUnit(enemy_id=2, x=right_x, y=4.0),
            PredictorUnit(enemy_id=3, x=7.0, y=5.0),
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
                enemy_id=enemy_id,
                display_name=f"测试单位-{enemy_id}",
                portrait_name=f"测试单位原型-{enemy_id}",
                page_name=f"测试单位原型-{enemy_id}",
                count_raw="1~99",
                stats=CombatStats(
                    hp=10_000,
                    attack=1_000,
                    defense=500,
                    resistance=30,
                    attack_interval=2,
                    weight=3,
                    move_speed=1,
                    attack_radius=2.5,
                ),
                attack_modes=[AttackMode.RANGED],
                damage_types=[DamageType.PHYSICAL],
            )
            for enemy_id in (1, 2, 3)
        ],
    )


def combat_table():
    return build_combat_feature_table(combat_knowledge(), num_enemy_ids=10, knowledge_sha256="a" * 64)


def test_feature_table_encodes_stats_attributes_and_explicit_unknowns():
    table = combat_table()

    assert table.features.shape == (10, len(CombatFeatureIndex))
    assert table.known.tolist() == [False, True, True, True, False, False, False, False, False, False]
    assert table.features[1, CombatFeatureIndex.HP].item() > 0
    assert table.features[1, CombatFeatureIndex.PHYSICAL].item() == 1
    assert table.raw[1, CombatFeatureIndex.HP].item() == 10_000
    assert torch.count_nonzero(table.features[4]).item() == 0
    assert table.feature_version == "combat-v2"
    assert table.knowledge_sha256 == "a" * 64


def test_collate_dynamically_pads_and_builds_derived_tensors():
    batch = collate_samples(
        [
            sample("a" * 32, Winner.LEFT),
            PredictorSample(
                sample_id="b" * 32,
                source_video_sha256="e" * 64,
                left_units=[PredictorUnit(enemy_id=1, x=1.0, y=2.0), PredictorUnit(enemy_id=2, x=3.0, y=4.0)],
                right_units=[PredictorUnit(enemy_id=3, x=9.0, y=6.0)],
                winner=Winner.RIGHT,
            ),
        ],
        combat_table(),
    )

    assert batch.left_ids.shape == (2, 2)
    assert batch.right_ids.shape == (2, 2)
    assert batch.left_mask.tolist() == [[True, False], [True, True]]
    assert batch.right_mask.tolist() == [[True, True], [True, False]]
    assert batch.right_positions[0, 0, 0].item() == pytest.approx(8.0)
    assert batch.left_formation.shape == (2, 2, len(FormationFeatureIndex))
    assert batch.relations.shape == (2, 4, 4, len(RelationFeatureIndex))
    assert batch.relation_masks.shape == (2, 4, 4, len(RelationMaskIndex))
    assert batch.labels.tolist() == [1.0, 0.0]


def test_model_is_antisymmetric_when_sides_are_swapped():
    torch.manual_seed(1)
    model = DuelTransformer(ModelConfig(num_enemy_ids=10, embedding_dim=32, heads=4, layers=1, dropout=0.0))
    model.eval()
    original = sample("a" * 32, Winner.LEFT)
    swapped_sample = original.model_copy(
        update={"left_units": original.right_units, "right_units": original.left_units, "winner": Winner.RIGHT}
    )
    batch = collate_samples([original], combat_table())
    swapped_batch = collate_samples([swapped_sample], combat_table())

    with torch.no_grad():
        forward = model(**batch.model_inputs())
        swapped = model(**swapped_batch.model_inputs())

    assert torch.allclose(forward, -swapped, atol=1e-6)
    assert torch.allclose(torch.sigmoid(forward), 1 - torch.sigmoid(swapped), atol=1e-6)


def test_unit_permutation_and_batch_padding_do_not_change_logit():
    torch.manual_seed(7)
    model = DuelTransformer(ModelConfig(num_enemy_ids=10, embedding_dim=32, heads=4, layers=1, dropout=0.0))
    model.eval()
    original = sample("a" * 32, Winner.LEFT)
    permuted = original.model_copy(update={"right_units": list(reversed(original.right_units))})
    larger = PredictorSample(
        sample_id="c" * 32,
        source_video_sha256="d" * 64,
        left_units=[
            PredictorUnit(enemy_id=1, x=1.0, y=2.0),
            PredictorUnit(enemy_id=2, x=2.0, y=3.0),
            PredictorUnit(enemy_id=3, x=3.0, y=4.0),
        ],
        right_units=[
            PredictorUnit(enemy_id=1, x=9.0, y=2.0),
            PredictorUnit(enemy_id=2, x=8.0, y=3.0),
            PredictorUnit(enemy_id=3, x=7.0, y=4.0),
        ],
        winner=Winner.RIGHT,
    )
    table = combat_table()

    with torch.no_grad():
        original_logit = model(**collate_samples([original], table).model_inputs())
        permuted_logit = model(**collate_samples([permuted], table).model_inputs())
        padded_logit = model(**collate_samples([original, larger], table).model_inputs())[:1]

    assert torch.allclose(original_logit, permuted_logit, atol=1e-6)
    assert torch.allclose(original_logit, padded_logit, atol=1e-6)


def test_collate_builds_formula_relations_in_tile_coordinates():
    batch = collate_samples([sample("a" * 32, Winner.LEFT)], combat_table())

    assert batch.relations[0, 0, 1, RelationFeatureIndex.DISTANCE].item() == pytest.approx(6.0)
    assert batch.relations[0, 0, 1, RelationFeatureIndex.PHYSICAL_DPS].item() > 0
    assert batch.relations[0, 0, 1, RelationFeatureIndex.TARGET_ALLOCATION].item() > 0
    assert batch.relation_masks[0, 0, 1, RelationMaskIndex.VALID]


def test_unit_formation_and_relation_encoders_receive_gradients_from_basic_batch():
    torch.manual_seed(2)
    model = DuelTransformer(ModelConfig(num_enemy_ids=10, embedding_dim=32, heads=4, layers=1, dropout=0.0))
    batch = collate_samples([sample("a" * 32, Winner.LEFT)], combat_table())

    logits = model(**batch.model_inputs())
    logits.sum().backward()

    assert torch.count_nonzero(model.unit_encoder[0].weight.grad).item() > 0
    assert torch.count_nonzero(model.formation_encoder[0].weight.grad).item() > 0
    assert torch.count_nonzero(model.relation_encoder[0].weight.grad).item() > 0
