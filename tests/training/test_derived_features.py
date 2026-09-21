from datetime import UTC, datetime

import pytest
import torch

from maa_duel.combat import (
    AreaShape,
    AttackAction,
    CombatKnowledge,
    CombatStats,
    ControlEffect,
    DamageType,
    EnemyCombatProfile,
    KnowledgeValueStatus,
    StageRules,
)
from maa_duel.training.derived import (
    FormationFeatureIndex,
    RelationFeatureIndex,
    build_derived_battle_features,
)
from maa_duel.training.features import PairFeatureIndex, build_combat_feature_table


def profile(enemy_id, *, hp=10_000, attack=1_000, defense=0, resistance=0, interval=1, radius=1, actions=None):
    if actions is None:
        actions = [
            AttackAction(
                name="normal_attack",
                damage_type=DamageType.PHYSICAL,
                interval_seconds=interval,
                status=KnowledgeValueStatus.CONFIRMED,
            )
        ]
    return EnemyCombatProfile(
        enemy_id=enemy_id,
        display_name=f"unit-{enemy_id}",
        portrait_name=f"unit-{enemy_id}",
        page_name=f"unit-{enemy_id}",
        count_raw="1",
        stats=CombatStats(
            hp=hp,
            attack=attack,
            defense=defense,
            resistance=resistance,
            attack_interval=interval,
            weight=1,
            move_speed=1,
            attack_radius=radius,
        ),
        opening_attacks=actions,
        damage_types=list(dict.fromkeys(action.damage_type for action in actions)),
    )


def table(*profiles):
    knowledge = CombatKnowledge(
        stage_id="VS-2",
        stage_title="争锋对决！",
        source_page="https://prts.wiki/w/VS-2",
        source_revision=1,
        fetched_at=datetime(2026, 9, 21, tzinfo=UTC),
        rules=StageRules(raw_text=""),
        enemies=list(profiles),
    )
    return build_combat_feature_table(knowledge, num_enemy_ids=16, knowledge_sha256="a" * 64)


def test_static_pair_damage_uses_raw_stats_and_deducts_defense_per_hit():
    attacker = profile(
        1,
        actions=[
            AttackAction(
                name="normal_attack",
                damage_type=DamageType.PHYSICAL,
                hits=2,
                interval_seconds=1,
                status=KnowledgeValueStatus.CONFIRMED,
            )
        ],
    )
    unarmored = profile(2, defense=0)
    armored = profile(3, defense=900)
    features = table(attacker, unarmored, armored)

    assert features.pairs[1, 2, PairFeatureIndex.PHYSICAL_DPS].item() == pytest.approx(2_000)
    assert features.pairs[1, 3, PairFeatureIndex.PHYSICAL_DPS].item() == pytest.approx(200)
    assert features.pairs[1, 3, PairFeatureIndex.NORMAL_ATTACK_DAMAGE].item() == pytest.approx(200)
    assert features.pairs[1, 3, PairFeatureIndex.NORMAL_ATTACKS_TO_KILL].item() == 50


def test_replacement_skill_and_one_shot_opening_are_not_permanent_extra_dps():
    attacker = profile(
        1,
        actions=[
            AttackAction(
                name="normal_attack",
                damage_type=DamageType.TRUE,
                interval_seconds=1,
                status=KnowledgeValueStatus.CONFIRMED,
            ),
            AttackAction(
                name="replacement",
                damage_type=DamageType.TRUE,
                attack_multiplier=2,
                cycle_seconds=2,
                replaces_normal=True,
                status=KnowledgeValueStatus.CONFIRMED,
            ),
            AttackAction(
                name="opening_only",
                damage_type=DamageType.TRUE,
                attack_multiplier=5,
                first_trigger_seconds=3,
                one_shot=True,
                status=KnowledgeValueStatus.CONFIRMED,
            ),
        ],
    )
    target = profile(2, hp=20_000)
    features = table(attacker, target)

    assert features.pairs[1, 2, PairFeatureIndex.TRUE_DPS].item() == pytest.approx(1_500)
    assert features.pairs[1, 2, PairFeatureIndex.OPENING_DAMAGE_RATIO].item() > 0.75
    assert features.pairs[1, 2, PairFeatureIndex.OPENING_DAMAGE_RATIO].item() < 1.5


def test_hard_control_immunity_is_applied_per_control_type():
    attacker = profile(1).model_copy(
        update={
            "controls": [
                ControlEffect(kind="stun", duration_seconds=4, cycle_seconds=4),
                ControlEffect(kind="freeze", duration_seconds=2, cycle_seconds=4),
            ]
        }
    )
    target = profile(2).model_copy(update={"control_immunities": ["stun"]})

    features = table(attacker, target)

    assert features.pairs[1, 2, PairFeatureIndex.HARD_CONTROL_UPTIME].item() == pytest.approx(0.5)


def test_target_allocation_is_capacity_bounded_equal_and_permutation_invariant():
    features = table(profile(1), profile(2), profile(3))
    ids = torch.tensor([[1, 2, 3]])
    positions = torch.tensor([[[0.0, 0.0], [4.0, 1.0], [4.0, -1.0]]])
    sides = torch.tensor([[0, 1, 1]])
    valid = torch.ones((1, 3), dtype=torch.bool)

    derived = build_derived_battle_features(ids, positions, sides, valid, features)
    allocation = derived.relations[0, 0, 1:, RelationFeatureIndex.TARGET_ALLOCATION]
    permuted = build_derived_battle_features(ids[:, [0, 2, 1]], positions[:, [0, 2, 1]], sides, valid, features)

    assert allocation.sum().item() == pytest.approx(1.0)
    assert allocation.tolist() == pytest.approx([0.5, 0.5])
    assert permuted.relations[0, 0, 1:, RelationFeatureIndex.TARGET_ALLOCATION].tolist() == pytest.approx([0.5, 0.5])


def test_time_to_range_stops_longer_range_target_before_short_range_source_arrives():
    features = table(profile(1, radius=1), profile(2, radius=3))
    ids = torch.tensor([[1, 2]])
    positions = torch.tensor([[[0.0, 0.0], [7.0, 0.0]]])
    sides = torch.tensor([[0, 1]])
    valid = torch.ones((1, 2), dtype=torch.bool)

    derived = build_derived_battle_features(ids, positions, sides, valid, features)

    assert derived.relations[0, 0, 1, RelationFeatureIndex.TIME_TO_RANGE].item() == pytest.approx(4.0)
    assert derived.relations[0, 1, 0, RelationFeatureIndex.TIME_TO_RANGE].item() == pytest.approx(2.0)


def test_formation_density_and_screening_respond_to_geometry_and_survival():
    weak = table(profile(1, hp=2_000), profile(2, hp=4_000), profile(3, attack=800))
    strong = table(profile(1, hp=20_000), profile(2, hp=4_000), profile(3, attack=800))
    ids = torch.tensor([[1, 2, 3]])
    sides = torch.tensor([[0, 0, 1]])
    valid = torch.ones((1, 3), dtype=torch.bool)
    aligned = torch.tensor([[[4.0, 3.0], [2.0, 3.0], [8.0, 3.0]]])
    side = torch.tensor([[[2.0, 6.0], [2.0, 3.0], [8.0, 3.0]]])

    weak_aligned = build_derived_battle_features(ids, aligned, sides, valid, weak)
    strong_aligned = build_derived_battle_features(ids, aligned, sides, valid, strong)
    moved_side = build_derived_battle_features(ids, side, sides, valid, strong)

    assert weak_aligned.relations[0, 0, 1, RelationFeatureIndex.SCREENING_SCORE] > 0.5
    assert moved_side.relations[0, 0, 1, RelationFeatureIndex.SCREENING_SCORE] < 0.05
    assert (
        strong_aligned.relations[0, 0, 1, RelationFeatureIndex.PROTECTION_SECONDS]
        > weak_aligned.relations[0, 0, 1, RelationFeatureIndex.PROTECTION_SECONDS]
    )
    assert aligned.shape[-1] == 2
    assert strong_aligned.formation.shape[-1] == len(FormationFeatureIndex) == 12


def test_grouping_changes_density_and_aoe_coverage_without_changing_roster():
    aoe = profile(
        3,
        attack=500,
        actions=[
            AttackAction(
                name="splash",
                damage_type=DamageType.ARTS,
                interval_seconds=1,
                target_count=1,
                area_shape=AreaShape.CIRCLE,
                area_radius=1.5,
                status=KnowledgeValueStatus.CONFIRMED,
            )
        ],
    )
    features = table(profile(1), profile(2), aoe)
    ids = torch.tensor([[1, 2, 3]])
    sides = torch.tensor([[0, 0, 1]])
    valid = torch.ones((1, 3), dtype=torch.bool)
    grouped = torch.tensor([[[2.0, 3.0], [2.5, 3.0], [8.0, 3.0]]])
    spread = torch.tensor([[[2.0, 1.0], [2.0, 6.0], [8.0, 3.0]]])

    grouped_features = build_derived_battle_features(ids, grouped, sides, valid, features)
    spread_features = build_derived_battle_features(ids, spread, sides, valid, features)

    assert (
        grouped_features.formation[0, 0, FormationFeatureIndex.FRIEND_DENSITY_1]
        > spread_features.formation[0, 0, FormationFeatureIndex.FRIEND_DENSITY_1]
    )
    assert grouped_features.relations[0, 2, 0, RelationFeatureIndex.POTENTIAL_TARGET_COUNT] > 1
    assert spread_features.relations[0, 2, 0, RelationFeatureIndex.POTENTIAL_TARGET_COUNT] == 1
