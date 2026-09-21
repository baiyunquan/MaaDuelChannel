from datetime import UTC, datetime

from maa_duel.assets import AssetManifest, EnemyAsset
from maa_duel.combat import AttackMode, DamageType, MechanicKind, load_combat_knowledge, save_combat_knowledge
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
|能力=每次攻击会降低击中目标的防御
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
