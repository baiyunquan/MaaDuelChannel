from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from maa_duel.assets import AssetManifest, load_asset_manifest
from maa_duel.combat import (
    ActionPhase,
    AreaShape,
    AttackAction,
    AttackMode,
    CombatKnowledge,
    CombatStats,
    ControlEffect,
    DamageType,
    EnemyCombatProfile,
    EnemyPageDetails,
    FutureFormSummary,
    HealAction,
    KnowledgeValueStatus,
    MechanicKind,
    MovementMode,
    ParsedStage,
    SkillKnowledge,
    StageEnemyRow,
    StageRules,
    save_combat_knowledge,
)
from maa_duel.prts_assets import PrtsClient, _normalize_title, _revision_text

PRTS_STAGE_TITLE = "VS-2 争锋对决！"
PRTS_STAGE_URL = "https://prts.wiki/w/VS-2_%E4%BA%89%E9%94%8B%E5%AF%B9%E5%86%B3%EF%BC%81"

_MECHANIC_KEYWORDS: tuple[tuple[MechanicKind, tuple[str, ...]], ...] = (
    (MechanicKind.DEFENSE_SHRED, ("防御力-", "降低防御")),
    (MechanicKind.DEFENSE_IGNORE, ("无视防御", "无视目标一定的防御")),
    (MechanicKind.RESISTANCE_SHRED, ("法术抗性-", "降低法术抗性")),
    (MechanicKind.RESISTANCE_IGNORE, ("无视法术抗性",)),
    (MechanicKind.STUN, ("晕眩",)),
    (MechanicKind.COLD, ("寒冷",)),
    (MechanicKind.FREEZE, ("冻结",)),
    (MechanicKind.SLOW, ("停顿", "减速")),
    (MechanicKind.BIND, ("束缚",)),
    (MechanicKind.SLEEP, ("沉睡",)),
    (MechanicKind.SILENCE, ("沉默",)),
    (MechanicKind.LEVITATE, ("浮空",)),
    (MechanicKind.FEAR, ("恐惧",)),
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


@dataclass(frozen=True)
class PrtsCombatSyncResult:
    path: Path
    profile_count: int
    mapped_profile_count: int
    missing_page_names: tuple[str, ...]
    knowledge_sha256: str


def _field(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^\|{re.escape(name)}=(.*?)(?=^\|[^=\r\n]+=|^}}}}\s*$|\Z)",
        text,
    )
    return match.group(1).strip() if match else ""


def _fields(text: str, name: str) -> list[str]:
    values = [
        value.strip()
        for value in re.findall(
            rf"(?ms)^\|{re.escape(name)}=(.*?)(?=^\|[^=\r\n]+=|^}}}}\s*$|\Z)",
            text,
        )
    ]
    return list(dict.fromkeys(value for value in values if value))


def _number(value: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group()) if match else 0.0


def infer_mechanics(text: str) -> list[MechanicKind]:
    found = [kind for kind, keywords in _MECHANIC_KEYWORDS if any(keyword in text for keyword in keywords)]
    if re.search(r"无视[^<\r\n]{0,20}防御", text) and MechanicKind.DEFENSE_IGNORE not in found:
        found.insert(1, MechanicKind.DEFENSE_IGNORE)
    if re.search(r"无视[^<\r\n]{0,20}法术抗性", text) and MechanicKind.RESISTANCE_IGNORE not in found:
        found.insert(3, MechanicKind.RESISTANCE_IGNORE)
    if _describes_healing_action(text) and MechanicKind.HEAL not in found:
        found.append(MechanicKind.HEAL)
    return found


def _describes_healing_action(text: str) -> bool:
    without_reactive_phrases = re.sub(r"(?:受到|接受|获得)治疗", "", text)
    return bool(
        re.search(
            r"(?:治疗(?:友方|目标|其他|周围)|为[^，。；<]{0,16}(?:恢复|回复)生命|"
            r"(?:恢复|回复)(?:友方|目标|其他)[^，。；<]{0,10}生命)",
            without_reactive_phrases,
        )
    )


def _first_number(pattern: str, text: str, *, scale: float = 1.0) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) * scale if match else None


def _damage_type(text: str) -> DamageType | None:
    for labels, damage_type in (
        (("物理伤害", "物理攻击"), DamageType.PHYSICAL),
        (("法术伤害", "法术攻击"), DamageType.ARTS),
        (("真实伤害", "真实攻击"), DamageType.TRUE),
    ):
        if any(label in text for label in labels):
            return damage_type
    return None


def _debuff_values(text: str) -> dict[str, float | int | None]:
    defense_ratio = _first_number(r"防御力\s*-\s*([\d.]+)%", text, scale=0.01) or 0.0
    resistance_ratio = _first_number(r"法术抗性\s*-\s*([\d.]+)%", text, scale=0.01) or 0.0
    defense_flat = 0.0 if defense_ratio else (_first_number(r"防御力\s*-\s*([\d.]+)", text) or 0.0)
    resistance_flat = 0.0 if resistance_ratio else (_first_number(r"法术抗性\s*-\s*([\d.]+)", text) or 0.0)
    maximum_stacks = _first_number(r"最多(?:可)?叠加\s*(\d+)\s*层", text)
    return {
        "defense_shred_flat": defense_flat,
        "defense_shred_ratio": defense_ratio,
        "resistance_shred_flat": resistance_flat,
        "resistance_shred_ratio": resistance_ratio,
        "max_stacks": int(maximum_stacks) if maximum_stacks else None,
    }


def _air_targeting(text: str) -> bool | None:
    if re.search(r"不会|无法|不能", text) and re.search(r"飞行|空中", text):
        return False
    if re.search(r"可以|能够|优先", text) and re.search(r"飞行|空中", text):
        return True
    return None


def _phase(text: str) -> ActionPhase:
    return ActionPhase.FUTURE if re.search(r"第二形态|后续形态|真面目形态|重生后|复活后", text) else ActionPhase.OPENING


def _structured_attack(
    name: str,
    effect: str,
    *,
    source_page: str,
    source_revision: int | None,
) -> AttackAction | None:
    damage_type = _damage_type(effect)
    if damage_type is None or not re.search(r"伤害|攻击", effect):
        return None
    multiplier = _first_number(r"攻击力\s*([\d.]+)%", effect, scale=0.01) or 1.0
    fixed_damage = _first_number(r"(?:造成|附加)\s*([\d.]+)\s*点", effect) or 0.0
    hits = int(_first_number(r"(?:造成|连续|各造成)?\s*(\d+)\s*次", effect) or 1)
    target_count = int(_first_number(r"(?:对|攻击|选择)?\s*(\d+)\s*(?:名|个)目标", effect) or 1)
    first_trigger = _first_number(r"(?:战斗|开场)开始?\s*([\d.]+)\s*秒后", effect)
    cycle = _first_number(r"每(?:隔)?\s*([\d.]+)\s*秒", effect)
    every_n = _first_number(r"每第?\s*(\d+)\s*次攻击", effect)
    replaces_normal = bool(re.search(r"替代.*普通攻击|下一次普通攻击|普通攻击(?:变为|改为)", effect))
    defense_penetration_ratio = _first_number(r"无视[^，。；<]{0,12}?([\d.]+)%\s*防御", effect, scale=0.01) or 0.0
    resistance_penetration_ratio = (
        _first_number(r"无视[^，。；<]{0,12}?([\d.]+)%\s*法术抗性", effect, scale=0.01) or 0.0
    )
    radius = _first_number(r"半径\s*([\d.]+)", effect) or 0.0
    area_shape = AreaShape.CIRCLE if radius or re.search(r"圆形|范围内|溅射", effect) else AreaShape.SINGLE
    return AttackAction(
        name=name,
        phase=_phase(effect),
        damage_type=damage_type,
        attack_multiplier=multiplier,
        fixed_damage=fixed_damage,
        hits=hits,
        first_trigger_seconds=first_trigger,
        cycle_seconds=cycle,
        every_n_attacks=int(every_n) if every_n else None,
        replaces_normal=replaces_normal,
        one_shot=first_trigger is not None and cycle is None and every_n is None,
        defense_penetration_ratio=defense_penetration_ratio,
        resistance_penetration_ratio=resistance_penetration_ratio,
        **_debuff_values(effect),
        target_count=target_count,
        area_shape=area_shape,
        area_radius=radius,
        can_target_air=_air_targeting(effect),
        source_page=source_page,
        source_revision=source_revision,
        status=KnowledgeValueStatus.CONFIRMED,
    )


def _structured_controls(
    effect: str,
    *,
    cycle_seconds: float | None,
    source_page: str,
    source_revision: int | None,
) -> list[ControlEffect]:
    labels = {
        "晕眩": MechanicKind.STUN,
        "冻结": MechanicKind.FREEZE,
        "寒冷": MechanicKind.COLD,
        "束缚": MechanicKind.BIND,
        "沉睡": MechanicKind.SLEEP,
        "恐惧": MechanicKind.FEAR,
    }
    controls: list[ControlEffect] = []
    for label, kind in labels.items():
        match = re.search(rf"{label}[^，。；<]{{0,8}}?([\d.]+)\s*(?:秒|s)", effect)
        if not match:
            match = re.search(rf"([\d.]+)\s*(?:秒|s)[^，。；<]{{0,24}}?{label}", effect)
        if label not in effect:
            continue
        controls.append(
            ControlEffect(
                kind=kind,
                duration_seconds=float(match.group(1)) if match else None,
                cycle_seconds=cycle_seconds,
                source_page=source_page,
                source_revision=source_revision,
                status=KnowledgeValueStatus.CONFIRMED if match else KnowledgeValueStatus.MISSING,
            )
        )
    return controls


def _control_immunities(text: str) -> list[MechanicKind]:
    labels = {
        "晕眩": MechanicKind.STUN,
        "冻结": MechanicKind.FREEZE,
        "寒冷": MechanicKind.COLD,
        "束缚": MechanicKind.BIND,
        "沉睡": MechanicKind.SLEEP,
        "恐惧": MechanicKind.FEAR,
    }
    return [kind for label, kind in labels.items() if re.search(rf"免疫[^。；<]{{0,16}}{label}", text)]


def _structured_heal(
    name: str,
    effect: str,
    *,
    source_page: str,
    source_revision: int | None,
) -> HealAction | None:
    if not _describes_healing_action(effect):
        return None
    multiplier = _first_number(r"攻击力\s*([\d.]+)%", effect, scale=0.01) or 0.0
    fixed = _first_number(r"(?:恢复|回复)[^，。；<]{0,8}?([\d.]+)\s*点", effect) or 0.0
    cycle = _first_number(r"每(?:隔)?\s*([\d.]+)\s*秒", effect)
    target_count = int(_first_number(r"(\d+)\s*(?:名|个)(?:友方|目标)", effect) or 1)
    return HealAction(
        name=name,
        heal_multiplier=multiplier,
        fixed_heal=fixed,
        cycle_seconds=cycle,
        target_count=target_count,
        self_only=bool(re.search(r"自身|自己", effect)),
        source_page=source_page,
        source_revision=source_revision,
        status=KnowledgeValueStatus.CONFIRMED if cycle and (multiplier or fixed) else KnowledgeValueStatus.ESTIMATED,
    )


def _future_summary(text: str, *, source_page: str, source_revision: int | None) -> FutureFormSummary:
    has_future = bool(re.search(r"第二形态|后续形态|重生|复活", text))
    if not has_future:
        return FutureFormSummary(source_page=source_page, source_revision=source_revision)
    form_markers = list(re.finditer(r"第二形态|后续形态|真面目形态", text))
    future_text = text[form_markers[-1].start() :] if form_markers else text
    pause = (
        _first_number(r"(?:停顿|等待|经过)\s*([\d.]+)\s*(?:秒|s)[^，。；<]{0,8}(?:重生|复活|进入)", text)
        or _first_number(r"(?:持续|进行持续)\s*([\d.]+)\s*(?:秒|s)[^，。；<]{0,8}(?:重生|复活)", text)
        or 0.0
    )
    extra_hp = (
        _first_number(r"额外获得\s*([\d.]+)%\s*生命", text, scale=0.01)
        or _first_number(r"(?:重生|复活).{0,24}?恢复\s*([\d.]+)%\s*生命", text, scale=0.01)
        or _first_number(r"恢复\s*([\d.]+)%\s*生命[^，。；<]{0,16}(?:切换|进入)", text, scale=0.01)
        or 0.0
    )
    revive_count = 1 if re.search(r"重生|复活|被击倒后.*第二形态", text) else 0
    summon_count = int(_first_number(r"召唤\s*(\d+)\s*个", text) or 0)
    return FutureFormSummary(
        revive_count=revive_count,
        extra_hp_ratio=extra_hp,
        revive_pause_seconds=pause,
        attack_delta=_first_number(r"攻击力\s*\+\s*([\d.]+)", future_text) or 0.0,
        defense_delta=_first_number(r"防御力\s*\+\s*([\d.]+)", future_text) or 0.0,
        resistance_delta=_first_number(r"法术抗性\s*\+\s*([\d.]+)", future_text) or 0.0,
        attack_interval_delta=-(_first_number(r"攻击间隔\s*-\s*([\d.]+)", future_text) or 0.0),
        move_speed_delta=_first_number(r"移动速度\s*\+\s*([\d.]+)", future_text) or 0.0,
        attack_radius=_first_number(r"半径\s*([\d.]+)", future_text),
        summon_count=summon_count,
        source_page=source_page,
        source_revision=source_revision,
        status=KnowledgeValueStatus.CONFIRMED,
    )


def parse_stage_wikitext(text: str) -> ParsedStage:
    stage_id = _field(text, "关卡代号")
    title = _field(text, "关卡名")
    rules_text = _field(text, "情报")
    rules = StageRules(
        raw_text=rules_text,
        hp_multiplier=_first_number(r"生命值[^，。；<]{0,12}?(\d+(?:\.\d+)?)%", rules_text, scale=0.01),
        attack_multiplier=_first_number(r"攻击力[^，。；<]{0,12}?(\d+(?:\.\d+)?)%", rules_text, scale=0.01),
        hazard_start_seconds=_first_number(r"战斗开始\s*([\d.]+)\s*s", rules_text),
        hazard_cycle_seconds=_first_number(r"每\s*([\d.]+)\s*s", rules_text),
    )
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
    source_page = _page_url(page_name)
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
    attacks: list[AttackAction] = []
    controls: list[ControlEffect] = []
    heals: list[HealAction] = []
    for index in skill_indices:
        names = _fields(text, f"技能{index}")
        effects = _fields(text, f"技能{index}效果")
        for position, name in enumerate(names):
            effect = effects[min(position, len(effects) - 1)] if effects else ""
            attack = _structured_attack(
                name,
                effect,
                source_page=source_page,
                source_revision=revision_id,
            )
            skill_controls = _structured_controls(
                effect,
                cycle_seconds=attack.cycle_seconds if attack else None,
                source_page=source_page,
                source_revision=revision_id,
            )
            heal = _structured_heal(
                name,
                effect,
                source_page=source_page,
                source_revision=revision_id,
            )
            if attack:
                attacks.append(attack)
            controls.extend(skill_controls)
            if heal:
                heals.append(heal)
            skills.append(
                SkillKnowledge(
                    name=name,
                    effect=effect,
                    mechanics=infer_mechanics(f"{name} {effect}"),
                    attacks=[attack] if attack else [],
                    controls=skill_controls,
                    heals=[heal] if heal else [],
                )
            )
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
        source_page=source_page,
        source_revision=revision_id,
        attack_modes=attack_modes,
        movement_modes=movement_modes,
        damage_types=damage_types,
        mechanics=infer_mechanics(combined),
        ability_text=ability_text,
        skills=skills,
        attacks=attacks,
        controls=controls,
        control_immunities=_control_immunities(combined),
        heals=heals,
        future=_future_summary(combined, source_page=source_page, source_revision=revision_id),
    )


def _normalize_name(value: str) -> str:
    return "".join(value.replace("“", "").replace("”", "").split()).casefold()


def _opening_damage_type(
    stage_notes: str,
    details: EnemyPageDetails | None,
    attack: float,
) -> tuple[DamageType | None, str]:
    opening_match = re.search(r"(?:初始|第一)形态([^。；<\n]*)", stage_notes)
    if opening_match and (damage_type := _damage_type(opening_match.group(1))):
        return damage_type, "stage_override"
    if details and len(details.damage_types) == 1:
        return details.damage_types[0], "enemy_page"
    if attack > 0:
        return DamageType.PHYSICAL, "default_attack_rule"
    return None, "unknown"


def _profile_future(
    row: StageEnemyRow,
    details: EnemyPageDetails | None,
    *,
    source_revision: int,
) -> FutureFormSummary:
    page = details.source_page if details else PRTS_STAGE_URL
    revision = details.source_revision if details else None
    stage_future = _future_summary(row.stage_notes, source_page=PRTS_STAGE_URL, source_revision=source_revision)
    if stage_future.status is KnowledgeValueStatus.CONFIRMED:
        return stage_future
    return details.future if details else FutureFormSummary(source_page=page, source_revision=revision)


def _stage_structured_effects(
    row: StageEnemyRow,
    *,
    source_revision: int,
) -> tuple[list[AttackAction], list[ControlEffect], list[HealAction]]:
    normalized = row.stage_notes.replace("\n", " ").replace("<br/>", "\n").replace("<br>", "\n")
    attacks: list[AttackAction] = []
    controls: list[ControlEffect] = []
    heals: list[HealAction] = []
    phase = ActionPhase.OPENING
    for index, segment in enumerate(part.strip() for part in normalized.splitlines() if part.strip()):
        if re.search(r"第二形态|后续形态|真面目形态|重生后|复活后", segment):
            phase = ActionPhase.FUTURE
        attack = _structured_attack(
            f"stage_override_{index}",
            segment,
            source_page=PRTS_STAGE_URL,
            source_revision=source_revision,
        )
        if attack:
            attack = attack.model_copy(update={"phase": phase})
            if phase is ActionPhase.FUTURE or any(
                (
                    attack.cycle_seconds,
                    attack.every_n_attacks,
                    attack.first_trigger_seconds,
                    attack.replaces_normal,
                    attack.target_count > 1,
                    attack.hits > 1,
                )
            ):
                attacks.append(attack)
        cycle = attack.cycle_seconds if attack else None
        if attack and cycle is None and attack.every_n_attacks:
            cycle = attack.every_n_attacks * row.stats.attack_interval
        controls.extend(
            _structured_controls(
                segment,
                cycle_seconds=cycle,
                source_page=PRTS_STAGE_URL,
                source_revision=source_revision,
            )
        )
        heal = _structured_heal(
            f"stage_override_{index}",
            segment,
            source_page=PRTS_STAGE_URL,
            source_revision=source_revision,
        )
        if heal:
            heals.append(heal)
    return attacks, controls, heals


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
        opening_damage_type, damage_type_source = _opening_damage_type(row.stage_notes, details, row.stats.attack)
        damage_types = [opening_damage_type] if opening_damage_type else []
        combined_text = " ".join(
            [row.stage_notes, details.ability_text if details else "", *(skill.effect for skill in details.skills)]
            if details
            else [row.stage_notes]
        )
        normal_attack = (
            AttackAction(
                name="normal_attack",
                damage_type=opening_damage_type,
                interval_seconds=row.stats.attack_interval or None,
                source_page=PRTS_STAGE_URL,
                source_revision=source_revision,
                status=KnowledgeValueStatus.CONFIRMED,
                can_target_air=_air_targeting(combined_text),
                **_debuff_values(combined_text),
            )
            if opening_damage_type and row.stats.attack > 0
            else None
        )
        structured_attacks = details.attacks if details else []
        stage_attacks, stage_controls, stage_heals = _stage_structured_effects(row, source_revision=source_revision)
        opening_skills = [
            action for action in (*stage_attacks, *structured_attacks) if action.phase is ActionPhase.OPENING
        ]
        future_attacks = [
            action for action in (*stage_attacks, *structured_attacks) if action.phase is ActionPhase.FUTURE
        ]
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
                damage_types=damage_types,
                damage_type_source=damage_type_source,
                mechanics=infer_mechanics(combined_text),
                ability_text=details.ability_text if details else "",
                skills=details.skills if details else [],
                opening_attacks=([normal_attack] if normal_attack else []) + opening_skills,
                future_attacks=future_attacks,
                controls=stage_controls + (details.controls if details else []),
                control_immunities=_control_immunities(combined_text),
                heals=stage_heals + (details.heals if details else []),
                future=_profile_future(row, details, source_revision=source_revision),
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


def _revision_id(page: dict[str, object]) -> int | None:
    revisions = page.get("revisions")
    if not isinstance(revisions, list) or not revisions:
        return None
    revision = revisions[0]
    return revision.get("revid") if isinstance(revision, dict) else None


def sync_prts_combat_knowledge(
    workspace: Path,
    *,
    request_interval: float = 0.25,
    client: PrtsClient | None = None,
) -> PrtsCombatSyncResult:
    manifest_path = workspace / "assets" / "catalog.json"
    manifest = load_asset_manifest(manifest_path)
    api = client or PrtsClient(request_interval=request_interval)
    stage_pages = api.query_pages([PRTS_STAGE_TITLE])
    stage_page = stage_pages.get(_normalize_title(PRTS_STAGE_TITLE), {})
    stage_wikitext = _revision_text(stage_page)
    stage_revision = _revision_id(stage_page)
    if not stage_wikitext or stage_revision is None:
        raise ValueError(f"PRTS stage page is missing revision content: {PRTS_STAGE_TITLE}")
    stage = parse_stage_wikitext(stage_wikitext)

    page_names = list(dict.fromkeys(row.portrait_name for row in stage.enemies if row.portrait_name))
    pages = api.query_pages(page_names)
    details: dict[str, EnemyPageDetails] = {}
    missing: list[str] = []
    for page_name in page_names:
        page = pages.get(_normalize_title(page_name), {})
        wikitext = _revision_text(page)
        if not wikitext:
            missing.append(page_name)
            continue
        details[page_name] = parse_enemy_wikitext(
            str(page.get("title") or page_name),
            wikitext,
            revision_id=_revision_id(page),
        )

    knowledge = compile_combat_knowledge(
        stage,
        manifest,
        details,
        source_revision=stage_revision,
    )
    output_path = workspace / "assets" / "combat" / "vs2_enemy_combat.json"
    digest = save_combat_knowledge(output_path, knowledge)
    return PrtsCombatSyncResult(
        path=output_path,
        profile_count=len(knowledge.enemies),
        mapped_profile_count=sum(profile.enemy_id is not None for profile in knowledge.enemies),
        missing_page_names=tuple(missing),
        knowledge_sha256=digest,
    )
