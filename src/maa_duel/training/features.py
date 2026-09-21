from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch

from maa_duel.combat import (
    COMBAT_FEATURE_VERSION,
    AttackMode,
    CombatKnowledge,
    DamageType,
    MechanicKind,
    MovementMode,
)


class CombatFeatureIndex(IntEnum):
    HP = 0
    ATTACK = 1
    DEFENSE = 2
    RESISTANCE = 3
    ATTACK_INTERVAL = 4
    WEIGHT = 5
    MOVE_SPEED = 6
    ATTACK_RADIUS = 7
    HP_REGEN = 8
    PHYSICAL = 9
    ARTS = 10
    TRUE = 11
    MELEE = 12
    RANGED = 13
    GROUND = 14
    AERIAL = 15
    DEFENSE_SHRED = 16
    RESISTANCE_SHRED = 17
    STUN = 18
    COLD = 19
    FREEZE = 20
    SLOW = 21
    BIND = 22
    SLEEP = 23
    SILENCE = 24
    LEVITATE = 25
    FEAR = 26
    HEAL = 27
    SHIELD = 28
    AOE = 29
    MULTI_TARGET = 30
    SUMMON = 31
    REVIVE = 32
    FORM_SWITCH = 33
    STACKING = 34
    RESIST = 35
    STATUS_IMMUNITY = 36
    INVULNERABLE = 37
    STEALTH = 38
    DODGE = 39
    EXECUTE = 40
    DISPLACEMENT = 41


COMBAT_FEATURE_DIM = len(CombatFeatureIndex)

_MECHANIC_FEATURES = {mechanic: CombatFeatureIndex[mechanic.name] for mechanic in MechanicKind}


@dataclass(frozen=True)
class CombatFeatureTable:
    features: torch.Tensor
    known: torch.Tensor
    feature_version: str
    knowledge_sha256: str


def _log_scale(value: float, maximum: float) -> float:
    return min(1.0, math.log1p(max(0.0, value)) / math.log1p(maximum))


def build_combat_feature_table(
    knowledge: CombatKnowledge,
    *,
    num_enemy_ids: int,
    knowledge_sha256: str,
) -> CombatFeatureTable:
    if num_enemy_ids < 2:
        raise ValueError("num_enemy_ids must include padding and at least one enemy")
    features = torch.zeros((num_enemy_ids, COMBAT_FEATURE_DIM), dtype=torch.float32)
    known = torch.zeros(num_enemy_ids, dtype=torch.bool)
    for profile in knowledge.enemies:
        if profile.enemy_id is None:
            continue
        if profile.enemy_id >= num_enemy_ids:
            raise ValueError(f"combat profile enemy_id {profile.enemy_id} is outside model vocabulary")
        if known[profile.enemy_id]:
            raise ValueError(f"duplicate combat profile for enemy_id {profile.enemy_id}")
        stats = profile.stats
        row = features[profile.enemy_id]
        row[CombatFeatureIndex.HP] = _log_scale(stats.hp, 1_000_000)
        row[CombatFeatureIndex.ATTACK] = _log_scale(stats.attack, 100_000)
        row[CombatFeatureIndex.DEFENSE] = _log_scale(stats.defense, 10_000)
        row[CombatFeatureIndex.RESISTANCE] = math.tanh(stats.resistance / 100.0)
        row[CombatFeatureIndex.ATTACK_INTERVAL] = min(stats.attack_interval / 10.0, 1.0)
        row[CombatFeatureIndex.WEIGHT] = min(stats.weight / 10.0, 1.0)
        row[CombatFeatureIndex.MOVE_SPEED] = min(stats.move_speed / 5.0, 1.0)
        row[CombatFeatureIndex.ATTACK_RADIUS] = min(stats.attack_radius / 5.0, 1.0)
        row[CombatFeatureIndex.HP_REGEN] = math.tanh(stats.hp_regen / 1_000.0)
        for damage_type, index in (
            (DamageType.PHYSICAL, CombatFeatureIndex.PHYSICAL),
            (DamageType.ARTS, CombatFeatureIndex.ARTS),
            (DamageType.TRUE, CombatFeatureIndex.TRUE),
        ):
            row[index] = float(damage_type in profile.damage_types)
        for attack_mode, index in (
            (AttackMode.MELEE, CombatFeatureIndex.MELEE),
            (AttackMode.RANGED, CombatFeatureIndex.RANGED),
        ):
            row[index] = float(attack_mode in profile.attack_modes)
        for movement_mode, index in (
            (MovementMode.GROUND, CombatFeatureIndex.GROUND),
            (MovementMode.AERIAL, CombatFeatureIndex.AERIAL),
        ):
            row[index] = float(movement_mode in profile.movement_modes)
        for mechanic in profile.mechanics:
            row[_MECHANIC_FEATURES[mechanic]] = 1.0
        known[profile.enemy_id] = True
    return CombatFeatureTable(
        features=features,
        known=known,
        feature_version=COMBAT_FEATURE_VERSION,
        knowledge_sha256=knowledge_sha256,
    )
