from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

COMBAT_FEATURE_VERSION = "combat-v1"


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


class MechanicKind(StrEnum):
    DEFENSE_SHRED = "defense_shred"
    RESISTANCE_SHRED = "resistance_shred"
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


class SkillKnowledge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    effect: str
    kind: Literal["skill", "talent"] = "skill"
    mechanics: list[MechanicKind] = Field(default_factory=list)


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
    mechanics: list[MechanicKind] = Field(default_factory=list)
    ability_text: str = ""
    skills: list[SkillKnowledge] = Field(default_factory=list)
    source_page: str | None = None
    source_revision: int | None = None
    review_status: KnowledgeReviewStatus = KnowledgeReviewStatus.AUTOMATIC


class CombatKnowledge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    feature_version: Literal["combat-v1"] = COMBAT_FEATURE_VERSION
    stage_id: str
    stage_title: str
    source_page: str
    source_revision: int
    fetched_at: datetime
    rules: StageRules
    enemies: list[EnemyCombatProfile]


def save_combat_knowledge(path: Path, knowledge: CombatKnowledge) -> str:
    payload = knowledge.model_dump_json(indent=2, exclude_none=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_combat_knowledge(path: Path) -> CombatKnowledge:
    return CombatKnowledge.model_validate(json.loads(path.read_text(encoding="utf-8")))
