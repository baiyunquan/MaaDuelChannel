from __future__ import annotations

from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field

from maa_duel.combat import DERIVED_FORMULA_VERSION, load_combat_knowledge
from maa_duel.contracts import ModelVersion, sha256_file
from maa_duel.dataset import BattleState, PredictorSample
from maa_duel.schema import Winner
from maa_duel.training.derived import RelationFeatureIndex, RelationMaskIndex, formula_sha256
from maa_duel.training.features import (
    build_combat_feature_table,
    feature_schema_sha256,
    vocabulary_sha256,
)
from maa_duel.training.model import DuelTransformer, ModelConfig
from maa_duel.training.predictor import collate_samples


class DuelPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left_win_probability: float = Field(ge=0.0, le=1.0)
    right_win_probability: float = Field(ge=0.0, le=1.0)
    feature_version: str
    formula_version: str
    formula_sha256: str
    knowledge_sha256: str
    calibration_sha256: str
    checkpoint: str
    explain: list[dict[str, object]] | None = None


def _validate_contract(workspace: Path, contract: ModelVersion, state: BattleState) -> tuple[Path, Path]:
    if contract.feature_version != "combat-v2" or contract.feature_schema_sha256 != feature_schema_sha256():
        raise ValueError("checkpoint feature contract does not match current combat-v2 code")
    if contract.formula_version != DERIVED_FORMULA_VERSION:
        raise ValueError("checkpoint formula contract does not match current derived formula")
    if contract.formula_sha256 != formula_sha256():
        raise ValueError("checkpoint formula hash does not match current derived formula")
    if not contract.knowledge_manifest or not contract.knowledge_sha256:
        raise ValueError("checkpoint is missing its knowledge contract")
    knowledge_path = workspace / contract.knowledge_manifest
    if not knowledge_path.is_file() or sha256_file(knowledge_path) != contract.knowledge_sha256:
        raise ValueError("checkpoint knowledge hash does not match current knowledge")
    if not contract.calibration_manifest or not contract.calibration_sha256 or not contract.calibration_id:
        raise ValueError("checkpoint is missing its calibration contract")
    calibration_path = workspace / contract.calibration_manifest
    if not calibration_path.is_file() or sha256_file(calibration_path) != contract.calibration_sha256:
        raise ValueError("checkpoint calibration hash does not match current calibration")
    if state.calibration_id != contract.calibration_id or state.calibration_sha256 != contract.calibration_sha256:
        raise ValueError("battle state calibration does not match checkpoint calibration")
    return knowledge_path, calibration_path


def _explain_rows(state: BattleState, batch) -> list[dict[str, object]]:
    units = [*state.left_units, *state.right_units]
    sides = ["left"] * len(state.left_units) + ["right"] * len(state.right_units)
    rows: list[dict[str, object]] = []
    relations = batch.relations[0]
    masks = batch.relation_masks[0]
    for source_index, source in enumerate(units):
        for target_index, target in enumerate(units):
            if source_index == target_index or not bool(masks[source_index, target_index, RelationMaskIndex.VALID]):
                continue
            relation = relations[source_index, target_index]
            rows.append(
                {
                    "source_instance_id": source.instance_id or f"{sides[source_index]}-{source_index}",
                    "target_instance_id": target.instance_id or f"{sides[target_index]}-{target_index}",
                    "source_enemy_id": source.enemy_id,
                    "target_enemy_id": target.enemy_id,
                    "source_side": sides[source_index],
                    "target_side": sides[target_index],
                    "distance": float(relation[RelationFeatureIndex.DISTANCE]),
                    "physical_dps": float(relation[RelationFeatureIndex.PHYSICAL_DPS]),
                    "arts_dps": float(relation[RelationFeatureIndex.ARTS_DPS]),
                    "true_dps": float(relation[RelationFeatureIndex.TRUE_DPS]),
                    "normal_attack_damage": float(relation[RelationFeatureIndex.NORMAL_ATTACK_DAMAGE]),
                    "normal_attacks_to_kill": float(relation[RelationFeatureIndex.NORMAL_ATTACKS_TO_KILL]),
                    "sustained_ttk": float(relation[RelationFeatureIndex.SUSTAINED_TTK]),
                    "opening_damage_ratio": float(relation[RelationFeatureIndex.OPENING_DAMAGE_RATIO]),
                    "potential_target_count": float(relation[RelationFeatureIndex.POTENTIAL_TARGET_COUNT]),
                    "target_allocation": float(relation[RelationFeatureIndex.TARGET_ALLOCATION]),
                    "screening_score": float(relation[RelationFeatureIndex.SCREENING_SCORE]),
                    "protection_seconds": float(relation[RelationFeatureIndex.PROTECTION_SECONDS]),
                    "estimated": bool(masks[source_index, target_index, RelationMaskIndex.ESTIMATED]),
                    "targetable": bool(masks[source_index, target_index, RelationMaskIndex.TARGETABLE]),
                    "finite_time": bool(masks[source_index, target_index, RelationMaskIndex.FINITE_TIME]),
                }
            )
    return rows


def predict_duel(
    workspace: Path,
    state: BattleState,
    *,
    checkpoint: Path | None = None,
    device: str | None = None,
    explain: bool = False,
) -> DuelPrediction:
    contract_path = workspace / "models" / "predictor" / "model.json"
    contract = ModelVersion.model_validate_json(contract_path.read_text(encoding="utf-8"))
    knowledge_path, _ = _validate_contract(workspace, contract, state)
    knowledge = load_combat_knowledge(knowledge_path)
    if contract.vocabulary_sha256 != vocabulary_sha256(knowledge):
        raise ValueError("checkpoint vocabulary hash does not match current knowledge")
    checkpoint_path = checkpoint or (workspace / contract.checkpoint)
    if checkpoint is None and contract.checkpoint_sha256 and sha256_file(checkpoint_path) != contract.checkpoint_sha256:
        raise ValueError("checkpoint file hash does not match model contract")
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(checkpoint_path, map_location=selected_device, weights_only=False)
    for key, expected in (
        ("feature_version", contract.feature_version),
        ("feature_schema_sha256", contract.feature_schema_sha256),
        ("formula_version", contract.formula_version),
        ("formula_sha256", contract.formula_sha256),
        ("knowledge_sha256", contract.knowledge_sha256),
        ("calibration_sha256", contract.calibration_sha256),
        ("vocabulary_sha256", contract.vocabulary_sha256),
    ):
        if payload.get(key) != expected:
            raise ValueError(f"checkpoint {key} does not match model contract")
    config = ModelConfig(**payload["model_config"])
    maximum_enemy_id = max(unit.enemy_id for unit in (*state.left_units, *state.right_units))
    if maximum_enemy_id >= config.num_enemy_ids:
        raise ValueError(f"enemy id {maximum_enemy_id} is outside checkpoint vocabulary")
    combat_table = build_combat_feature_table(
        knowledge,
        num_enemy_ids=config.num_enemy_ids,
        knowledge_sha256=contract.knowledge_sha256 or "",
    )
    sample = PredictorSample.model_validate({**state.model_dump(mode="json"), "winner": Winner.LEFT})
    batch = collate_samples([sample], combat_table).to(selected_device)
    model = DuelTransformer(config).to(selected_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        left_probability = float(torch.sigmoid(model(**batch.model_inputs()))[0].cpu())
    return DuelPrediction(
        left_win_probability=left_probability,
        right_win_probability=1.0 - left_probability,
        feature_version=contract.feature_version or "",
        formula_version=contract.formula_version or "",
        formula_sha256=contract.formula_sha256 or "",
        knowledge_sha256=contract.knowledge_sha256 or "",
        calibration_sha256=contract.calibration_sha256 or "",
        checkpoint=checkpoint_path.relative_to(workspace).as_posix(),
        explain=_explain_rows(state, batch.to(torch.device("cpu"))) if explain else None,
    )
