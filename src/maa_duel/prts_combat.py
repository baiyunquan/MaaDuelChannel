from __future__ import annotations

import re
import urllib.parse
from datetime import UTC, datetime

from maa_duel.assets import AssetManifest
from maa_duel.combat import (
    AttackMode,
    CombatKnowledge,
    CombatStats,
    DamageType,
    EnemyCombatProfile,
    EnemyPageDetails,
    MechanicKind,
    MovementMode,
    ParsedStage,
    SkillKnowledge,
    StageEnemyRow,
    StageRules,
)

PRTS_STAGE_TITLE = "VS-2 争锋对决！"
PRTS_STAGE_URL = "https://prts.wiki/w/VS-2_%E4%BA%89%E9%94%8B%E5%AF%B9%E5%86%B3%EF%BC%81"

_MECHANIC_KEYWORDS: tuple[tuple[MechanicKind, tuple[str, ...]], ...] = (
    (MechanicKind.DEFENSE_SHRED, ("防御力-", "降低防御", "无视防御")),
    (MechanicKind.RESISTANCE_SHRED, ("法术抗性-", "降低法术抗性", "无视法术抗性")),
    (MechanicKind.STUN, ("晕眩",)),
    (MechanicKind.COLD, ("寒冷",)),
    (MechanicKind.FREEZE, ("冻结",)),
    (MechanicKind.SLOW, ("停顿", "减速")),
    (MechanicKind.BIND, ("束缚",)),
    (MechanicKind.SLEEP, ("沉睡",)),
    (MechanicKind.SILENCE, ("沉默",)),
    (MechanicKind.LEVITATE, ("浮空",)),
    (MechanicKind.FEAR, ("恐惧",)),
    (MechanicKind.HEAL, ("治疗", "恢复生命")),
    (MechanicKind.SHIELD, ("护盾", "屏障")),
    (MechanicKind.AOE, ("范围内", "群体", "溅射")),
    (MechanicKind.MULTI_TARGET, ("同时攻击", "多名", "多个目标")),
    (MechanicKind.SUMMON, ("召唤",)),
    (MechanicKind.REVIVE, ("重生", "复活")),
    (MechanicKind.FORM_SWITCH, ("形态", "切换")),
    (MechanicKind.STACKING, ("叠加",)),
    (MechanicKind.RESIST, ("抵抗",)),
    (MechanicKind.STATUS_IMMUNITY, ("免疫",)),
    (MechanicKind.INVULNERABLE, ("无敌", "不受伤害")),
    (MechanicKind.STEALTH, ("隐匿",)),
    (MechanicKind.DODGE, ("闪避",)),
    (MechanicKind.EXECUTE, ("直接击倒", "处决")),
    (MechanicKind.DISPLACEMENT, ("推动", "拖拽", "位移")),
)


def _field(text: str, name: str) -> str:
    match = re.search(rf"(?m)^\|{re.escape(name)}=([^\r\n]*)", text)
    return match.group(1).strip() if match else ""


def _fields(text: str, name: str) -> list[str]:
    values = [value.strip() for value in re.findall(rf"(?m)^\|{re.escape(name)}=([^\r\n]*)", text)]
    return list(dict.fromkeys(value for value in values if value))


def _number(value: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group()) if match else 0.0


def infer_mechanics(text: str) -> list[MechanicKind]:
    return [kind for kind, keywords in _MECHANIC_KEYWORDS if any(keyword in text for keyword in keywords)]


def parse_stage_wikitext(text: str) -> ParsedStage:
    stage_id = _field(text, "关卡代号")
    title = _field(text, "关卡名")
    rules = StageRules(raw_text=_field(text, "情报"))
    indices = sorted({int(value) for value in re.findall(r"(?m)^\|敌人(\d+)=", text)})
    enemies: list[StageEnemyRow] = []
    stat_fields = {
        "hp": "生命值",
        "attack": "攻击力",
        "defense": "防御力",
        "resistance": "法术抗性",
        "attack_interval": "攻击间隔",
        "weight": "重量等级",
        "move_speed": "移动速度",
        "attack_radius": "攻击范围半径",
        "hp_regen": "生命回复速度",
    }
    for index in indices:
        raw = {key: _field(text, f"敌人{index}{suffix}") for key, suffix in stat_fields.items()}
        enemies.append(
            StageEnemyRow(
                row_index=index,
                display_name=_field(text, f"敌人{index}"),
                portrait_name=_field(text, f"敌人{index}头像"),
                count_raw=_field(text, f"敌人{index}数量"),
                rank_raw=_field(text, f"敌人{index}地位"),
                level_raw=_field(text, f"敌人{index}等级"),
                target_value_raw=_field(text, f"敌人{index}目标价值"),
                stage_notes=_field(text, f"敌人{index}备注"),
                stats=CombatStats(**{key: _number(value) for key, value in raw.items()}, raw=raw),
            )
        )
    return ParsedStage(stage_id=stage_id, title=title, rules=rules, enemies=enemies)


def _page_url(page_name: str) -> str:
    return "https://prts.wiki/w/" + urllib.parse.quote(page_name.replace(" ", "_"), safe="/:()")


def parse_enemy_wikitext(page_name: str, text: str, *, revision_id: int | None = None) -> EnemyPageDetails:
    attack_text = " ".join(_fields(text, "攻击方式"))
    movement_text = " ".join(_fields(text, "行动方式"))
    attack_modes = [
        mode for label, mode in (("近战", AttackMode.MELEE), ("远程", AttackMode.RANGED)) if label in attack_text
    ]
    movement_modes = [
        mode for label, mode in (("地面", MovementMode.GROUND), ("飞行", MovementMode.AERIAL)) if label in movement_text
    ]
    ability_parts = [*_fields(text, "能力"), *_fields(text, "天赋")]
    ability_text = "<br>".join(ability_parts)
    skill_indices = sorted({int(value) for value in re.findall(r"(?m)^\|技能(\d+)=", text)})
    skills: list[SkillKnowledge] = []
    for index in skill_indices:
        names = _fields(text, f"技能{index}")
        effects = _fields(text, f"技能{index}效果")
        for position, name in enumerate(names):
            effect = effects[min(position, len(effects) - 1)] if effects else ""
            skills.append(SkillKnowledge(name=name, effect=effect, mechanics=infer_mechanics(f"{name} {effect}")))
    talents = _fields(text, "天赋")
    if talents:
        skills.append(
            SkillKnowledge(
                name="天赋",
                effect="<br>".join(talents),
                kind="talent",
                mechanics=infer_mechanics(" ".join(talents)),
            )
        )
    combined = " ".join([attack_text, ability_text, *(skill.effect for skill in skills)])
    damage_types = [
        damage_type
        for label, damage_type in (
            ("物理伤害", DamageType.PHYSICAL),
            ("法术伤害", DamageType.ARTS),
            ("真实伤害", DamageType.TRUE),
        )
        if label in combined
    ]
    return EnemyPageDetails(
        page_name=page_name,
        source_page=_page_url(page_name),
        source_revision=revision_id,
        attack_modes=attack_modes,
        movement_modes=movement_modes,
        damage_types=damage_types,
        mechanics=infer_mechanics(combined),
        ability_text=ability_text,
        skills=skills,
    )


def _normalize_name(value: str) -> str:
    return "".join(value.replace("“", "").replace("”", "").split()).casefold()


def compile_combat_knowledge(
    stage: ParsedStage,
    assets: AssetManifest,
    enemy_pages: dict[str, EnemyPageDetails],
    *,
    source_revision: int,
    fetched_at: datetime | None = None,
) -> CombatKnowledge:
    id_by_name: dict[str, int] = {}
    for enemy in assets.enemies:
        id_by_name[_normalize_name(enemy.name)] = enemy.enemy_id
        id_by_name[_normalize_name(enemy.original_name)] = enemy.enemy_id
    normalized_pages = {_normalize_name(name): details for name, details in enemy_pages.items()}
    profiles: list[EnemyCombatProfile] = []
    for row in stage.enemies:
        details = normalized_pages.get(_normalize_name(row.portrait_name))
        combined_text = " ".join(
            [row.stage_notes, details.ability_text if details else "", *(skill.effect for skill in details.skills)]
            if details
            else [row.stage_notes]
        )
        profiles.append(
            EnemyCombatProfile(
                enemy_id=id_by_name.get(_normalize_name(row.display_name))
                or id_by_name.get(_normalize_name(row.portrait_name)),
                display_name=row.display_name,
                portrait_name=row.portrait_name,
                page_name=details.page_name if details else row.portrait_name,
                count_raw=row.count_raw,
                rank_raw=row.rank_raw,
                level_raw=row.level_raw,
                target_value_raw=row.target_value_raw,
                stage_notes=row.stage_notes,
                stats=row.stats,
                attack_modes=details.attack_modes if details else [],
                movement_modes=details.movement_modes if details else [],
                damage_types=details.damage_types if details else [],
                mechanics=infer_mechanics(combined_text),
                ability_text=details.ability_text if details else "",
                skills=details.skills if details else [],
                source_page=details.source_page if details else None,
                source_revision=details.source_revision if details else None,
            )
        )
    return CombatKnowledge(
        stage_id=stage.stage_id,
        stage_title=stage.title,
        source_page=PRTS_STAGE_URL,
        source_revision=source_revision,
        fetched_at=fetched_at or datetime.now(UTC),
        rules=stage.rules,
        enemies=profiles,
    )
