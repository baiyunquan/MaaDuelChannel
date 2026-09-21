from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

COMBAT_FEATURE_VERSION = "combat-v2"
DERIVED_FORMULA_VERSION = "derived-v1"


class AttackMode(StrEnum):
    MELEE = "melee"
    RANGED = "ranged"


class MovementMode(StrEnum):
    GROUND = "ground"
    AERIAL = "aerial"


class DamageType(StrEnum):
    PHYSICAL = "physical"
    ARTS = "arts"
    TRUE = "true"


class ActionPhase(StrEnum):
    OPENING = "opening"
    FUTURE = "future"


class AreaShape(StrEnum):
    SINGLE = "single"
    CIRCLE = "circle"
    CROSS = "cross"
    LINE = "line"
    UNKNOWN = "unknown"


class TargetRule(StrEnum):
    UNKNOWN = "unknown"
    EARLIEST_CONTACT = "earliest_contact"
    NEAREST = "nearest"
    FURTHEST = "furthest"
    LOWEST_DEFENSE = "lowest_defense"
    BACKLINE = "backline"


class KnowledgeValueStatus(StrEnum):
    CONFIRMED = "confirmed"
    ESTIMATED = "estimated"
    MISSING = "missing"


class MechanicKind(StrEnum):
    DEFENSE_SHRED = "defense_shred"
    DEFENSE_IGNORE = "defense_ignore"
    RESISTANCE_SHRED = "resistance_shred"
    RESISTANCE_IGNORE = "resistance_ignore"
    STUN = "stun"
    COLD = "cold"
    FREEZE = "freeze"
    SLOW = "slow"
    BIND = "bind"
    SLEEP = "sleep"
    SILENCE = "silence"
    LEVITATE = "levitate"
    FEAR = "fear"
    HEAL = "heal"
    SHIELD = "shield"
    AOE = "aoe"
    MULTI_TARGET = "multi_target"
    SUMMON = "summon"
    REVIVE = "revive"
    FORM_SWITCH = "form_switch"
    STACKING = "stacking"
    RESIST = "resist"
    STATUS_IMMUNITY = "status_immunity"
    INVULNERABLE = "invulnerable"
    STEALTH = "stealth"
    DODGE = "dodge"
    EXECUTE = "execute"
    DISPLACEMENT = "displacement"


class KnowledgeReviewStatus(StrEnum):
    AUTOMATIC = "automatic"
    REVIEWED = "reviewed"
    MISSING = "missing"


class CombatStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hp: float = Field(ge=0.0)
    attack: float = Field(ge=0.0)
    defense: float = Field(ge=0.0)
    resistance: float
    attack_interval: float = Field(ge=0.0)
    weight: float = Field(ge=0.0)
    move_speed: float = Field(ge=0.0)
    attack_radius: float = Field(ge=0.0)
    hp_regen: float = 0.0
    raw: dict[str, str] = Field(default_factory=dict)


class StageRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_text: str
    table_modifier_state: Literal["unknown", "included", "not_included"] = "unknown"
    hp_multiplier: float | None = Field(default=None, gt=0.0)
    attack_multiplier: float | None = Field(default=None, gt=0.0)
    hazard_start_seconds: float | None = Field(default=None, ge=0.0)
    hazard_cycle_seconds: float | None = Field(default=None, gt=0.0)


class AttackAction(BaseModel):
    """One independently resolved attack action; multihit damage is applied per hit."""

    model_config = ConfigDict(extra="forbid")

    name: str
    phase: ActionPhase = ActionPhase.OPENING
    damage_type: DamageType
    attack_multiplier: float = Field(default=1.0, ge=0.0)
    fixed_damage: float = Field(default=0.0, ge=0.0)
    hits: int = Field(default=1, ge=1)
    interval_seconds: float | None = Field(default=None, gt=0.0)
    first_trigger_seconds: float | None = Field(default=None, ge=0.0)
    cycle_seconds: float | None = Field(default=None, gt=0.0)
    every_n_attacks: int | None = Field(default=None, ge=1)
    replaces_normal: bool = False
    one_shot: bool = False
    defense_penetration_flat: float = Field(default=0.0, ge=0.0)
    defense_penetration_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    resistance_penetration_flat: float = Field(default=0.0, ge=0.0)
    resistance_penetration_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    defense_shred_flat: float = Field(default=0.0, ge=0.0)
    defense_shred_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    resistance_shred_flat: float = Field(default=0.0, ge=0.0)
    resistance_shred_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    max_stacks: int | None = Field(default=None, ge=1)
    target_count: int = Field(default=1, ge=1)
    area_shape: AreaShape = AreaShape.SINGLE
    area_radius: float = Field(default=0.0, ge=0.0)
    can_target_air: bool | None = None
    target_rule: TargetRule = TargetRule.UNKNOWN
    source_page: str | None = None
    source_revision: int | None = None
    status: KnowledgeValueStatus = KnowledgeValueStatus.ESTIMATED


class ControlEffect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: MechanicKind
    duration_seconds: float | None = Field(default=None, ge=0.0)
    cycle_seconds: float | None = Field(default=None, gt=0.0)
    intensity: float = Field(default=1.0, ge=0.0)
    target_count: int = Field(default=1, ge=1)
    source_page: str | None = None
    source_revision: int | None = None
    status: KnowledgeValueStatus = KnowledgeValueStatus.ESTIMATED


class HealAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    heal_multiplier: float = Field(default=0.0, ge=0.0)
    fixed_heal: float = Field(default=0.0, ge=0.0)
    first_trigger_seconds: float | None = Field(default=None, ge=0.0)
    cycle_seconds: float | None = Field(default=None, gt=0.0)
    target_count: int = Field(default=1, ge=1)
    area_radius: float = Field(default=0.0, ge=0.0)
    self_only: bool = False
    source_page: str | None = None
    source_revision: int | None = None
    status: KnowledgeValueStatus = KnowledgeValueStatus.ESTIMATED


class FutureFormSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revive_count: int = Field(default=0, ge=0)
    extra_hp_ratio: float = Field(default=0.0, ge=0.0)
    revive_pause_seconds: float = Field(default=0.0, ge=0.0)
    hp_multiplier: float = Field(default=1.0, ge=0.0)
    attack_multiplier: float = Field(default=1.0, ge=0.0)
    attack_delta: float = 0.0
    defense_multiplier: float = Field(default=1.0, ge=0.0)
    defense_delta: float = 0.0
    resistance_delta: float = 0.0
    attack_interval_delta: float = 0.0
    move_speed_delta: float = 0.0
    attack_radius: float | None = Field(default=None, ge=0.0)
    summon_count: int = Field(default=0, ge=0)
    source_page: str | None = None
    source_revision: int | None = None
    status: KnowledgeValueStatus = KnowledgeValueStatus.MISSING


class SkillKnowledge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    effect: str
    kind: Literal["skill", "talent"] = "skill"
    mechanics: list[MechanicKind] = Field(default_factory=list)
    attacks: list[AttackAction] = Field(default_factory=list)
    controls: list[ControlEffect] = Field(default_factory=list)
    heals: list[HealAction] = Field(default_factory=list)


class EnemyPageDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_name: str
    source_page: str
    source_revision: int | None = None
    attack_modes: list[AttackMode] = Field(default_factory=list)
    movement_modes: list[MovementMode] = Field(default_factory=list)
    damage_types: list[DamageType] = Field(default_factory=list)
    mechanics: list[MechanicKind] = Field(default_factory=list)
    ability_text: str = ""
    skills: list[SkillKnowledge] = Field(default_factory=list)
    attacks: list[AttackAction] = Field(default_factory=list)
    controls: list[ControlEffect] = Field(default_factory=list)
    control_immunities: list[MechanicKind] = Field(default_factory=list)
    heals: list[HealAction] = Field(default_factory=list)
    future: FutureFormSummary = Field(default_factory=FutureFormSummary)


class StageEnemyRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_index: int = Field(ge=1)
    display_name: str
    portrait_name: str
    count_raw: str
    rank_raw: str = ""
    level_raw: str = ""
    target_value_raw: str = ""
    stage_notes: str = ""
    stats: CombatStats


class ParsedStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_id: str
    title: str
    rules: StageRules
    enemies: list[StageEnemyRow]


class EnemyCombatProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enemy_id: int | None = Field(default=None, ge=1)
    display_name: str
    portrait_name: str
    page_name: str
    count_raw: str
    rank_raw: str = ""
    level_raw: str = ""
    target_value_raw: str = ""
    stage_notes: str = ""
    stats: CombatStats
    attack_modes: list[AttackMode] = Field(default_factory=list)
    movement_modes: list[MovementMode] = Field(default_factory=list)
    damage_types: list[DamageType] = Field(default_factory=list)
    damage_type_source: Literal["stage_override", "enemy_page", "default_attack_rule", "unknown"] = "unknown"
    mechanics: list[MechanicKind] = Field(default_factory=list)
    ability_text: str = ""
    skills: list[SkillKnowledge] = Field(default_factory=list)
    opening_attacks: list[AttackAction] = Field(default_factory=list)
    future_attacks: list[AttackAction] = Field(default_factory=list)
    controls: list[ControlEffect] = Field(default_factory=list)
    control_immunities: list[MechanicKind] = Field(default_factory=list)
    heals: list[HealAction] = Field(default_factory=list)
    future: FutureFormSummary = Field(default_factory=FutureFormSummary)
    source_page: str | None = None
    source_revision: int | None = None
    review_status: KnowledgeReviewStatus = KnowledgeReviewStatus.AUTOMATIC


class CombatKnowledge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    feature_version: Literal["combat-v2"] = COMBAT_FEATURE_VERSION
    formula_version: Literal["derived-v1"] = DERIVED_FORMULA_VERSION
    stage_id: str
    stage_title: str
    source_page: str
    source_revision: int
    fetched_at: datetime
    rules: StageRules
    enemies: list[EnemyCombatProfile]


def _migrate_v1(payload: dict[str, object]) -> dict[str, object]:
    migrated = dict(payload)
    migrated["schema_version"] = 2
    migrated["feature_version"] = COMBAT_FEATURE_VERSION
    migrated["formula_version"] = DERIVED_FORMULA_VERSION
    for enemy in migrated.get("enemies", []):
        if not isinstance(enemy, dict):
            continue
        stats = enemy.get("stats") if isinstance(enemy.get("stats"), dict) else {}
        damage_types = enemy.get("damage_types") or [DamageType.PHYSICAL]
        enemy.setdefault(
            "opening_attacks",
            [
                {
                    "name": "normal_attack",
                    "damage_type": str(damage_types[0]),
                    "interval_seconds": stats.get("attack_interval") or None,
                    "source_page": enemy.get("source_page"),
                    "source_revision": enemy.get("source_revision"),
                    "status": KnowledgeValueStatus.ESTIMATED,
                }
            ],
        )
        enemy.setdefault("future_attacks", [])
        enemy.setdefault("controls", [])
        enemy.setdefault("heals", [])
        enemy.setdefault("future", {})
    return migrated


def save_combat_knowledge(path: Path, knowledge: CombatKnowledge) -> str:
    payload = knowledge.model_dump_json(indent=2, exclude_none=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_combat_knowledge(path: Path) -> CombatKnowledge:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("feature_version") == "combat-v1":
        payload = _migrate_v1(payload)
    return CombatKnowledge.model_validate(payload)
