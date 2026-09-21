from datetime import UTC, datetime

from maa_duel.assets import AssetManifest, EnemyAsset
from maa_duel.combat import (
    COMBAT_FEATURE_VERSION,
    ActionPhase,
    AttackMode,
    DamageType,
    KnowledgeValueStatus,
    MechanicKind,
    load_combat_knowledge,
    save_combat_knowledge,
)
from maa_duel.prts_combat import (
    PRTS_STAGE_TITLE,
    compile_combat_knowledge,
    parse_enemy_wikitext,
    parse_stage_wikitext,
    sync_prts_combat_knowledge,
)

STAGE_WIKITEXT = """{{普通关卡信息
|关卡代号=VS-2
|关卡名=争锋对决！
|情报=所有参赛人员的生命值降低至50%，攻击力提升至150%<br/>战斗开始60s后出现源石圈
}}
{{特殊敌方情报
|敌人1=流鼻涕虫虫
|敌人1头像=酸液源石虫·α
|敌人1数量=1~99
|敌人1生命值=2780
|敌人1攻击力=290
|敌人1防御力=0
|敌人1法术抗性=0
|敌人1攻击间隔=3.3
|敌人1重量等级=0
|敌人1移动速度=1
|敌人1攻击范围半径=2.75
|敌人1目标价值=2（+0.1）
|敌人1备注=攻击时令目标防御力-15（可无限叠加）
}}
"""


ENEMY_WIKITEXT = """{{敌人
|攻击方式=远程
|行动方式=地面
|能力=每次攻击会降低击中目标的防御<br>不会攻击飞行单位
|技能0=腐蚀喷吐
|技能0效果=对目标造成攻击力100%的法术伤害，并施加3s晕眩
|天赋=普通攻击令目标防御力-15（可无限叠加）
}}
"""


def test_stage_parser_preserves_authoritative_map_values_and_rules():
    stage = parse_stage_wikitext(STAGE_WIKITEXT)

    assert stage.stage_id == "VS-2"
    assert stage.title == "争锋对决！"
    assert "生命值降低至50%" in stage.rules.raw_text
    assert stage.rules.table_modifier_state == "unknown"
    assert len(stage.enemies) == 1
    enemy = stage.enemies[0]
    assert enemy.display_name == "流鼻涕虫虫"
    assert enemy.portrait_name == "酸液源石虫·α"
    assert enemy.stats.hp == 2780
    assert enemy.stats.attack == 290
    assert enemy.stats.attack_interval == 3.3
    assert enemy.stats.attack_radius == 2.75
    assert enemy.count_raw == "1~99"
    assert enemy.target_value_raw == "2（+0.1）"


def test_enemy_parser_structures_attack_damage_skills_and_mechanics():
    details = parse_enemy_wikitext("酸液源石虫·α", ENEMY_WIKITEXT, revision_id=123)

    assert details.attack_modes == [AttackMode.RANGED]
    assert details.damage_types == [DamageType.ARTS]
    assert details.source_revision == 123
    assert details.skills[0].name == "腐蚀喷吐"
    assert "攻击力100%" in details.skills[0].effect
    assert MechanicKind.DEFENSE_SHRED in details.mechanics
    assert MechanicKind.STUN in details.mechanics
    assert MechanicKind.STACKING in details.mechanics


def test_enemy_parser_keeps_penetration_distinct_from_team_defense_shred():
    details = parse_enemy_wikitext(
        "测试敌人",
        "|攻击方式=近战\n|技能0=装甲穿刺\n|技能0效果=造成物理伤害且无视60%防御力",
    )

    assert MechanicKind.DEFENSE_IGNORE in details.mechanics
    assert MechanicKind.DEFENSE_SHRED not in details.mechanics


def test_stage_parser_keeps_multiline_map_override_until_next_parameter():
    stage = parse_stage_wikitext(
        """{{特殊敌方情报
|关卡代号=VS-2
|关卡名=争锋对决！
|情报=地图规则第一行
地图规则第二行
|敌人1=杰斯顿
|敌人1头像=杰斯顿·威廉姆斯
|敌人1数量=1
|敌人1生命值=10000
|敌人1攻击力=900
|敌人1防御力=500
|敌人1法术抗性=30
|敌人1攻击间隔=2
|敌人1重量等级=4
|敌人1移动速度=0.8
|敌人1攻击范围半径=2.5
|敌人1备注=初始形态进行远程法术攻击。
第二形态改为近战物理攻击。
}}"""
    )

    assert stage.rules.raw_text == "地图规则第一行\n地图规则第二行"
    assert stage.enemies[0].stage_notes == "初始形态进行远程法术攻击。\n第二形态改为近战物理攻击。"


def test_structured_parser_does_not_treat_receiving_healing_as_a_heal_action():
    details = parse_enemy_wikitext(
        "食腐野兽",
        "|攻击方式=近战\n|能力=受到治疗时清除自身的流血效果\n|技能0=撕咬\n"
        "|技能0效果=下一次普通攻击造成攻击力180%的物理伤害",
        revision_id=321,
    )

    assert MechanicKind.HEAL not in details.mechanics
    assert details.heals == []
    assert len(details.attacks) == 1
    attack = details.attacks[0]
    assert attack.damage_type is DamageType.PHYSICAL
    assert attack.attack_multiplier == 1.8
    assert attack.replaces_normal is True
    assert attack.status is KnowledgeValueStatus.CONFIRMED


def test_structured_parser_records_trigger_cycle_multihit_and_control():
    details = parse_enemy_wikitext(
        "测试术师",
        "|攻击方式=远程\n|技能0=冻结齐射\n"
        "|技能0效果=战斗开始5秒后，每10秒替代一次普通攻击，对2名目标各造成3次攻击力120%的法术伤害，"
        "并使其冻结2秒",
        revision_id=7,
    )

    attack = details.attacks[0]
    assert attack.phase is ActionPhase.OPENING
    assert attack.damage_type is DamageType.ARTS
    assert attack.hits == 3
    assert attack.target_count == 2
    assert attack.first_trigger_seconds == 5
    assert attack.cycle_seconds == 10
    assert attack.replaces_normal is True
    assert details.controls[0].kind is MechanicKind.FREEZE
    assert details.controls[0].duration_seconds == 2
    assert details.controls[0].cycle_seconds == 10


def test_compiled_knowledge_maps_stage_names_to_asset_ids_and_round_trips(tmp_path):
    stage = parse_stage_wikitext(STAGE_WIKITEXT)
    details = parse_enemy_wikitext("酸液源石虫·α", ENEMY_WIKITEXT, revision_id=123)
    assets = AssetManifest(
        source_catalog="file:///catalog.csv",
        source_catalog_sha256="a" * 64,
        enemies=[EnemyAsset(enemy_id=42, name="流鼻涕虫虫", original_name="酸液源石虫·α")],
    )

    knowledge = compile_combat_knowledge(
        stage,
        assets,
        {"酸液源石虫·α": details},
        source_revision=456,
        fetched_at=datetime(2026, 9, 21, tzinfo=UTC),
    )
    path = tmp_path / "combat" / "vs2_enemy_combat.json"
    digest = save_combat_knowledge(path, knowledge)
    loaded = load_combat_knowledge(path)

    assert loaded.enemies[0].enemy_id == 42
    assert loaded.enemies[0].damage_type_source == "enemy_page"
    assert loaded.enemies[0].mechanics == [
        MechanicKind.DEFENSE_SHRED,
        MechanicKind.STUN,
        MechanicKind.STACKING,
    ]
    assert loaded.enemies[0].opening_attacks[0].defense_shred_flat == 15
    assert loaded.enemies[0].opening_attacks[0].can_target_air is False
    assert loaded.source_revision == 456
    assert len(digest) == 64


def test_compiler_marks_default_physical_attacks_when_page_has_no_damage_override():
    stage = parse_stage_wikitext(STAGE_WIKITEXT)
    details = parse_enemy_wikitext("酸液源石虫·α", "|攻击方式=远程\n|行动方式=地面", revision_id=123)
    assets = AssetManifest(
        source_catalog="file:///catalog.csv",
        source_catalog_sha256="a" * 64,
        enemies=[EnemyAsset(enemy_id=42, name="流鼻涕虫虫", original_name="酸液源石虫·α")],
    )

    profile = compile_combat_knowledge(stage, assets, {"酸液源石虫·α": details}, source_revision=456).enemies[0]

    assert profile.damage_types == [DamageType.PHYSICAL]
    assert profile.damage_type_source == "default_attack_rule"


def test_compiler_uses_opening_form_and_keeps_future_form_as_summary():
    stage = parse_stage_wikitext(
        STAGE_WIKITEXT.replace("流鼻涕虫虫", "杰斯顿")
        .replace("酸液源石虫·α", "杰斯顿·威廉姆斯")
        .replace(
            "攻击时令目标防御力-15（可无限叠加）",
            "初始形态进行远程法术攻击；被击倒后停顿3秒进入第二形态，攻击变为物理伤害并额外获得100%生命值",
        )
    )
    details = parse_enemy_wikitext(
        "杰斯顿·威廉姆斯",
        "|攻击方式=远程 近战\n|行动方式=地面\n|能力=初始形态造成法术伤害；第二形态造成物理伤害\n"
        "|技能0=第二形态重击\n|技能0效果=第二形态下一次攻击造成攻击力200%的物理伤害",
        revision_id=123,
    )
    assets = AssetManifest(
        source_catalog="file:///catalog.csv",
        source_catalog_sha256="a" * 64,
        enemies=[EnemyAsset(enemy_id=37, name="杰斯顿", original_name="杰斯顿·威廉姆斯")],
    )

    knowledge = compile_combat_knowledge(stage, assets, {"杰斯顿·威廉姆斯": details}, source_revision=456)
    profile = knowledge.enemies[0]

    assert knowledge.feature_version == COMBAT_FEATURE_VERSION == "combat-v2"
    assert knowledge.schema_version == 2
    assert [action.damage_type for action in profile.opening_attacks] == [DamageType.ARTS]
    assert all(action.phase is ActionPhase.OPENING for action in profile.opening_attacks)
    assert profile.future.revive_count == 1
    assert profile.future.extra_hp_ratio == 1
    assert profile.future.revive_pause_seconds == 3
    assert all(action.phase is ActionPhase.FUTURE for action in profile.future_attacks)


def test_sync_writes_complete_table_from_stage_and_enemy_revisions(tmp_path):
    workspace = tmp_path / "workspace"
    assets = AssetManifest(
        source_catalog="file:///catalog.csv",
        source_catalog_sha256="a" * 64,
        enemies=[EnemyAsset(enemy_id=42, name="流鼻涕虫虫", original_name="酸液源石虫·α")],
    )
    (workspace / "assets").mkdir(parents=True)
    (workspace / "assets" / "catalog.json").write_text(assets.model_dump_json(), encoding="utf-8")

    class FakeClient:
        def query_pages(self, titles):
            if titles == [PRTS_STAGE_TITLE]:
                return {
                    PRTS_STAGE_TITLE.casefold(): {
                        "title": PRTS_STAGE_TITLE,
                        "revisions": [{"revid": 456, "slots": {"main": {"content": STAGE_WIKITEXT}}}],
                    }
                }
            assert titles == ["酸液源石虫·α"]
            return {
                "酸液源石虫·α".casefold(): {
                    "title": "酸液源石虫·α",
                    "revisions": [{"revid": 123, "slots": {"main": {"content": ENEMY_WIKITEXT}}}],
                }
            }

    result = sync_prts_combat_knowledge(workspace, client=FakeClient())
    loaded = load_combat_knowledge(result.path)

    assert result.profile_count == 1
    assert result.mapped_profile_count == 1
    assert result.missing_page_names == ()
    assert loaded.enemies[0].stats.hp == 2780
    assert loaded.enemies[0].source_revision == 123
