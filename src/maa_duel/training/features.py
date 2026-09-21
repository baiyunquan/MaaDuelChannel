from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import IntEnum

import torch

from maa_duel.combat import (
    COMBAT_FEATURE_VERSION,
    ActionPhase,
    AreaShape,
    AttackAction,
    AttackMode,
    CombatKnowledge,
    DamageType,
    EnemyCombatProfile,
    KnowledgeValueStatus,
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
    DEFENSE_IGNORE = 17
    RESISTANCE_SHRED = 18
    RESISTANCE_IGNORE = 19
    STUN = 20
    COLD = 21
    FREEZE = 22
    SLOW = 23
    BIND = 24
    SLEEP = 25
    SILENCE = 26
    LEVITATE = 27
    FEAR = 28
    HEAL = 29
    SHIELD = 30
    AOE = 31
    MULTI_TARGET = 32
    SUMMON = 33
    REVIVE = 34
    FORM_SWITCH = 35
    STACKING = 36
    RESIST = 37
    STATUS_IMMUNITY = 38
    INVULNERABLE = 39
    STEALTH = 40
    DODGE = 41
    EXECUTE = 42
    DISPLACEMENT = 43
    NORMAL_HITS = 44
    TARGET_COUNT = 45
    AREA_RADIUS = 46
    AREA_CIRCLE = 47
    AREA_CROSS = 48
    AREA_LINE = 49
    CONTROL_DURATION = 50
    CONTROL_CYCLE = 51
    HEAL_BURST = 52
    HPS = 53
    DEFENSE_SHRED_FLAT = 54
    DEFENSE_SHRED_RATIO = 55
    RESISTANCE_SHRED_FLAT = 56
    RESISTANCE_SHRED_RATIO = 57
    FUTURE_EXTRA_HP = 58
    REVIVE_COUNT = 59
    SUMMON_COUNT = 60
    CAN_TARGET_AIR = 61
    ACTION_KNOWN = 62


class PairFeatureIndex(IntEnum):
    PHYSICAL_DPS = 0
    ARTS_DPS = 1
    TRUE_DPS = 2
    NORMAL_ATTACK_DAMAGE = 3
    NORMAL_ATTACKS_TO_KILL = 4
    SUSTAINED_TTK = 5
    OPENING_DAMAGE_RATIO = 6
    HARD_CONTROL_UPTIME = 7


COMBAT_FEATURE_DIM = len(CombatFeatureIndex)
PAIR_FEATURE_DIM = len(PairFeatureIndex)
OPENING_WINDOW_SECONDS = 10.0

_MECHANIC_FEATURES = {
    mechanic: CombatFeatureIndex[mechanic.name]
    for mechanic in MechanicKind
    if mechanic.name in CombatFeatureIndex.__members__
}


def feature_schema_sha256() -> str:
    from maa_duel.training.derived import FormationFeatureIndex, RelationFeatureIndex, RelationMaskIndex

    payload = {
        "combat": [item.name for item in CombatFeatureIndex],
        "pair": [item.name for item in PairFeatureIndex],
        "formation": [item.name for item in FormationFeatureIndex],
        "relation": [item.name for item in RelationFeatureIndex],
        "relation_masks": [item.name for item in RelationMaskIndex],
        "feature_version": COMBAT_FEATURE_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def vocabulary_sha256(knowledge: CombatKnowledge) -> str:
    payload = [
        (profile.enemy_id, profile.display_name, profile.page_name)
        for profile in sorted(knowledge.enemies, key=lambda item: item.enemy_id or 0)
        if profile.enemy_id is not None
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CombatFeatureTable:
    features: torch.Tensor
    raw: torch.Tensor
    known: torch.Tensor
    pairs: torch.Tensor
    pair_valid: torch.Tensor
    pair_estimated: torch.Tensor
    pair_targetable: torch.Tensor
    pair_finite: torch.Tensor
    feature_version: str
    knowledge_sha256: str


def _log_scale(value: float, maximum: float) -> float:
    return min(1.0, math.log1p(max(0.0, value)) / math.log1p(maximum))


def _fallback_action(profile: EnemyCombatProfile) -> AttackAction | None:
    if profile.stats.attack <= 0:
        return None
    damage_type = profile.damage_types[0] if profile.damage_types else DamageType.PHYSICAL
    return AttackAction(
        name="normal_attack",
        damage_type=damage_type,
        interval_seconds=profile.stats.attack_interval or None,
        status=KnowledgeValueStatus.ESTIMATED,
    )


def _opening_actions(profile: EnemyCombatProfile) -> list[AttackAction]:
    actions = [action for action in profile.opening_attacks if action.phase is ActionPhase.OPENING]
    if actions:
        return actions
    fallback = _fallback_action(profile)
    return [fallback] if fallback else []


def _normal_action(profile: EnemyCombatProfile) -> AttackAction | None:
    actions = _opening_actions(profile)
    return next((action for action in actions if action.name == "normal_attack"), actions[0] if actions else None)


def _effective_damage(action: AttackAction, source: EnemyCombatProfile, target: EnemyCombatProfile) -> float:
    per_hit = source.stats.attack * action.attack_multiplier + action.fixed_damage
    if per_hit <= 0:
        return 0.0
    if action.damage_type is DamageType.PHYSICAL:
        defense = max(
            0.0,
            target.stats.defense * (1.0 - action.defense_penetration_ratio) - action.defense_penetration_flat,
        )
        damage = max(per_hit * 0.05, per_hit - defense)
    elif action.damage_type is DamageType.ARTS:
        resistance = target.stats.resistance * (1.0 - action.resistance_penetration_ratio)
        resistance = max(-100.0, min(100.0, resistance - action.resistance_penetration_flat))
        damage = max(per_hit * 0.05, per_hit * (1.0 - resistance / 100.0))
    else:
        damage = per_hit
    return damage * action.hits


def _action_cycle(action: AttackAction, source: EnemyCombatProfile) -> float | None:
    if action.cycle_seconds:
        return action.cycle_seconds
    if action.every_n_attacks and source.stats.attack_interval > 0:
        return action.every_n_attacks * source.stats.attack_interval
    if action.name == "normal_attack":
        return action.interval_seconds or source.stats.attack_interval or None
    return action.interval_seconds


def _trigger_count(action: AttackAction, source: EnemyCombatProfile) -> int:
    first = action.first_trigger_seconds
    cycle = _action_cycle(action, source)
    if action.one_shot:
        first = first or 0.0
        return int(first < OPENING_WINDOW_SECONDS)
    if action.name == "normal_attack":
        if cycle is None:
            return 0
        return max(0, math.ceil(OPENING_WINDOW_SECONDS / cycle))
    if cycle is None:
        return 0
    first = cycle if first is None else first
    if first >= OPENING_WINDOW_SECONDS:
        return 0
    return 1 + math.floor((OPENING_WINDOW_SECONDS - first - 1e-6) / cycle)


def _pair_values(source: EnemyCombatProfile, target: EnemyCombatProfile) -> tuple[list[float], bool, bool]:
    actions = _opening_actions(source)
    normal = _normal_action(source)
    normal_damage = _effective_damage(normal, source, target) if normal else 0.0
    dps = {DamageType.PHYSICAL: 0.0, DamageType.ARTS: 0.0, DamageType.TRUE: 0.0}
    opening_damage = 0.0
    if normal:
        cycle = _action_cycle(normal, source)
        if cycle:
            dps[normal.damage_type] += normal_damage / cycle
        opening_damage += normal_damage * _trigger_count(normal, source)
    for action in actions:
        if action is normal:
            continue
        damage = _effective_damage(action, source, target)
        cycle = _action_cycle(action, source)
        triggers = _trigger_count(action, source)
        if cycle and not action.one_shot:
            dps[action.damage_type] += damage / cycle
            if action.replaces_normal and normal:
                dps[normal.damage_type] = max(0.0, dps[normal.damage_type] - normal_damage / cycle)
        opening_damage += damage * triggers
        if action.replaces_normal and normal:
            opening_damage = max(0.0, opening_damage - normal_damage * triggers)
    total_dps = sum(dps.values())
    attacks_to_kill = math.ceil(target.stats.hp / normal_damage) if normal_damage > 0 else math.inf
    ttk = target.stats.hp / total_dps if total_dps > 0 else math.inf
    control_uptime = 0.0
    if MechanicKind.STATUS_IMMUNITY not in target.mechanics:
        for effect in source.controls:
            if effect.kind not in {
                MechanicKind.STUN,
                MechanicKind.FREEZE,
                MechanicKind.SLEEP,
                MechanicKind.FEAR,
            }:
                continue
            if effect.kind in target.control_immunities:
                continue
            if effect.duration_seconds is not None and effect.cycle_seconds:
                control_uptime = max(control_uptime, min(effect.duration_seconds / effect.cycle_seconds, 1.0))
    estimated = any(action.status is not KnowledgeValueStatus.CONFIRMED for action in actions)
    target_aerial = MovementMode.AERIAL in target.movement_modes
    explicit_air = [action.can_target_air for action in actions if action.can_target_air is not None]
    targetable = not target_aerial or not explicit_air or any(explicit_air)
    return (
        [
            dps[DamageType.PHYSICAL],
            dps[DamageType.ARTS],
            dps[DamageType.TRUE],
            normal_damage,
            attacks_to_kill,
            ttk,
            opening_damage / max(target.stats.hp, 1e-6),
            control_uptime,
        ],
        estimated or (target_aerial and not explicit_air),
        targetable,
    )


def _populate_unit_rows(profile: EnemyCombatProfile, raw: torch.Tensor, features: torch.Tensor) -> None:
    stats = profile.stats
    raw[CombatFeatureIndex.HP] = stats.hp
    raw[CombatFeatureIndex.ATTACK] = stats.attack
    raw[CombatFeatureIndex.DEFENSE] = stats.defense
    raw[CombatFeatureIndex.RESISTANCE] = stats.resistance
    raw[CombatFeatureIndex.ATTACK_INTERVAL] = stats.attack_interval
    raw[CombatFeatureIndex.WEIGHT] = stats.weight
    raw[CombatFeatureIndex.MOVE_SPEED] = stats.move_speed
    raw[CombatFeatureIndex.ATTACK_RADIUS] = stats.attack_radius
    raw[CombatFeatureIndex.HP_REGEN] = stats.hp_regen
    features[CombatFeatureIndex.HP] = _log_scale(stats.hp, 1_000_000)
    features[CombatFeatureIndex.ATTACK] = _log_scale(stats.attack, 100_000)
    features[CombatFeatureIndex.DEFENSE] = _log_scale(stats.defense, 10_000)
    features[CombatFeatureIndex.RESISTANCE] = math.tanh(stats.resistance / 100.0)
    features[CombatFeatureIndex.ATTACK_INTERVAL] = min(stats.attack_interval / 10.0, 1.0)
    features[CombatFeatureIndex.WEIGHT] = min(stats.weight / 10.0, 1.0)
    features[CombatFeatureIndex.MOVE_SPEED] = min(stats.move_speed / 5.0, 1.0)
    features[CombatFeatureIndex.ATTACK_RADIUS] = min(stats.attack_radius / 10.0, 1.0)
    features[CombatFeatureIndex.HP_REGEN] = math.tanh(stats.hp_regen / 1_000.0)
    for damage_type, index in (
        (DamageType.PHYSICAL, CombatFeatureIndex.PHYSICAL),
        (DamageType.ARTS, CombatFeatureIndex.ARTS),
        (DamageType.TRUE, CombatFeatureIndex.TRUE),
    ):
        value = float(
            damage_type in profile.damage_types
            or any(action.damage_type is damage_type for action in _opening_actions(profile))
        )
        raw[index] = features[index] = value
    for mode, index in ((AttackMode.MELEE, CombatFeatureIndex.MELEE), (AttackMode.RANGED, CombatFeatureIndex.RANGED)):
        raw[index] = features[index] = float(mode in profile.attack_modes)
    for mode, index in (
        (MovementMode.GROUND, CombatFeatureIndex.GROUND),
        (MovementMode.AERIAL, CombatFeatureIndex.AERIAL),
    ):
        raw[index] = features[index] = float(mode in profile.movement_modes)
    for mechanic in profile.mechanics:
        index = _MECHANIC_FEATURES.get(mechanic)
        if index is not None:
            raw[index] = features[index] = 1.0

    normal = _normal_action(profile)
    actions = _opening_actions(profile)
    representative = max(actions, key=lambda action: (action.target_count, action.area_radius), default=normal)
    if normal:
        raw[CombatFeatureIndex.NORMAL_HITS] = normal.hits
        features[CombatFeatureIndex.NORMAL_HITS] = min(normal.hits / 10.0, 1.0)
    if representative:
        raw[CombatFeatureIndex.TARGET_COUNT] = representative.target_count
        raw[CombatFeatureIndex.AREA_RADIUS] = representative.area_radius
        features[CombatFeatureIndex.TARGET_COUNT] = min(representative.target_count / 10.0, 1.0)
        features[CombatFeatureIndex.AREA_RADIUS] = min(representative.area_radius / 5.0, 1.0)
        for shape, index in (
            (AreaShape.CIRCLE, CombatFeatureIndex.AREA_CIRCLE),
            (AreaShape.CROSS, CombatFeatureIndex.AREA_CROSS),
            (AreaShape.LINE, CombatFeatureIndex.AREA_LINE),
        ):
            raw[index] = features[index] = float(representative.area_shape is shape)
        known_air = [action.can_target_air for action in actions if action.can_target_air is not None]
        if known_air:
            raw[CombatFeatureIndex.CAN_TARGET_AIR] = features[CombatFeatureIndex.CAN_TARGET_AIR] = float(any(known_air))
        raw[CombatFeatureIndex.ACTION_KNOWN] = features[CombatFeatureIndex.ACTION_KNOWN] = float(
            all(action.status is KnowledgeValueStatus.CONFIRMED for action in actions)
        )
    hard_controls = [effect for effect in profile.controls if effect.duration_seconds is not None]
    if hard_controls:
        control = max(hard_controls, key=lambda effect: effect.duration_seconds or 0.0)
        raw[CombatFeatureIndex.CONTROL_DURATION] = control.duration_seconds or 0.0
        raw[CombatFeatureIndex.CONTROL_CYCLE] = control.cycle_seconds or 0.0
        features[CombatFeatureIndex.CONTROL_DURATION] = min((control.duration_seconds or 0.0) / 10.0, 1.0)
        features[CombatFeatureIndex.CONTROL_CYCLE] = min((control.cycle_seconds or 0.0) / 30.0, 1.0)
    if profile.heals:
        heal = profile.heals[0]
        burst = stats.attack * heal.heal_multiplier + heal.fixed_heal
        hps = burst / heal.cycle_seconds if heal.cycle_seconds else 0.0
        raw[CombatFeatureIndex.HEAL_BURST] = burst
        raw[CombatFeatureIndex.HPS] = hps
        features[CombatFeatureIndex.HEAL_BURST] = _log_scale(burst, 100_000)
        features[CombatFeatureIndex.HPS] = _log_scale(hps, 100_000)
    for index, value, maximum in (
        (CombatFeatureIndex.FUTURE_EXTRA_HP, profile.future.extra_hp_ratio, 5.0),
        (CombatFeatureIndex.REVIVE_COUNT, profile.future.revive_count, 5.0),
        (CombatFeatureIndex.SUMMON_COUNT, profile.future.summon_count, 20.0),
    ):
        raw[index] = value
        features[index] = min(float(value) / maximum, 1.0)
    for action in actions:
        stacks = 1
        if action.max_stacks is not None:
            stacks = min(action.max_stacks, max(_trigger_count(action, profile), 1))
        elif MechanicKind.STACKING in profile.mechanics:
            stacks = max(_trigger_count(action, profile), 1)
        raw[CombatFeatureIndex.DEFENSE_SHRED_FLAT] = max(
            raw[CombatFeatureIndex.DEFENSE_SHRED_FLAT].item(), action.defense_shred_flat * stacks
        )
        raw[CombatFeatureIndex.DEFENSE_SHRED_RATIO] = max(
            raw[CombatFeatureIndex.DEFENSE_SHRED_RATIO].item(), min(action.defense_shred_ratio * stacks, 1.0)
        )
        raw[CombatFeatureIndex.RESISTANCE_SHRED_FLAT] = max(
            raw[CombatFeatureIndex.RESISTANCE_SHRED_FLAT].item(), action.resistance_shred_flat * stacks
        )
        raw[CombatFeatureIndex.RESISTANCE_SHRED_RATIO] = max(
            raw[CombatFeatureIndex.RESISTANCE_SHRED_RATIO].item(), min(action.resistance_shred_ratio * stacks, 1.0)
        )
    features[CombatFeatureIndex.DEFENSE_SHRED_FLAT] = _log_scale(
        raw[CombatFeatureIndex.DEFENSE_SHRED_FLAT].item(), 10_000
    )
    features[CombatFeatureIndex.DEFENSE_SHRED_RATIO] = raw[CombatFeatureIndex.DEFENSE_SHRED_RATIO]
    features[CombatFeatureIndex.RESISTANCE_SHRED_FLAT] = min(
        raw[CombatFeatureIndex.RESISTANCE_SHRED_FLAT].item() / 100.0, 1.0
    )
    features[CombatFeatureIndex.RESISTANCE_SHRED_RATIO] = raw[CombatFeatureIndex.RESISTANCE_SHRED_RATIO]


def build_combat_feature_table(
    knowledge: CombatKnowledge,
    *,
    num_enemy_ids: int,
    knowledge_sha256: str,
) -> CombatFeatureTable:
    if num_enemy_ids < 2:
        raise ValueError("num_enemy_ids must include padding and at least one enemy")
    features = torch.zeros((num_enemy_ids, COMBAT_FEATURE_DIM), dtype=torch.float32)
    raw = torch.zeros_like(features)
    known = torch.zeros(num_enemy_ids, dtype=torch.bool)
    profiles: dict[int, EnemyCombatProfile] = {}
    for profile in knowledge.enemies:
        if profile.enemy_id is None:
            continue
        if profile.enemy_id >= num_enemy_ids:
            raise ValueError(f"combat profile enemy_id {profile.enemy_id} is outside model vocabulary")
        if known[profile.enemy_id]:
            raise ValueError(f"duplicate combat profile for enemy_id {profile.enemy_id}")
        _populate_unit_rows(profile, raw[profile.enemy_id], features[profile.enemy_id])
        known[profile.enemy_id] = True
        profiles[profile.enemy_id] = profile

    pairs = torch.zeros((num_enemy_ids, num_enemy_ids, PAIR_FEATURE_DIM), dtype=torch.float32)
    pair_valid = torch.zeros((num_enemy_ids, num_enemy_ids), dtype=torch.bool)
    pair_estimated = torch.zeros_like(pair_valid)
    pair_targetable = torch.zeros_like(pair_valid)
    pair_finite = torch.zeros_like(pair_valid)
    for source_id, source in profiles.items():
        for target_id, target in profiles.items():
            values, estimated, targetable = _pair_values(source, target)
            finite = math.isfinite(values[PairFeatureIndex.NORMAL_ATTACKS_TO_KILL]) and math.isfinite(
                values[PairFeatureIndex.SUSTAINED_TTK]
            )
            values[PairFeatureIndex.NORMAL_ATTACKS_TO_KILL] = min(
                values[PairFeatureIndex.NORMAL_ATTACKS_TO_KILL], 10_000
            )
            values[PairFeatureIndex.SUSTAINED_TTK] = min(values[PairFeatureIndex.SUSTAINED_TTK], 300.0)
            pairs[source_id, target_id] = torch.tensor(values, dtype=torch.float32)
            pair_valid[source_id, target_id] = True
            pair_estimated[source_id, target_id] = estimated
            pair_targetable[source_id, target_id] = targetable
            pair_finite[source_id, target_id] = finite
    return CombatFeatureTable(
        features=features,
        raw=raw,
        known=known,
        pairs=pairs,
        pair_valid=pair_valid,
        pair_estimated=pair_estimated,
        pair_targetable=pair_targetable,
        pair_finite=pair_finite,
        feature_version=COMBAT_FEATURE_VERSION,
        knowledge_sha256=knowledge_sha256,
    )
