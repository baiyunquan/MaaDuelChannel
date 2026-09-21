from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from maa_duel.training.derived import FormationFeatureIndex, RelationFeatureIndex, RelationMaskIndex
from maa_duel.training.features import COMBAT_FEATURE_DIM


def _log_ratio(values: torch.Tensor, maximum: float) -> torch.Tensor:
    return torch.log1p(values.clamp(min=0.0)) / math.log1p(maximum)


def normalize_relation_features(values: torch.Tensor) -> torch.Tensor:
    """Transform already-computed raw formula values for stable network input."""
    output = values.clone()
    for index in (
        RelationFeatureIndex.DISTANCE,
        RelationFeatureIndex.FORWARD_DELTA,
        RelationFeatureIndex.LATERAL_DELTA,
    ):
        if index in (RelationFeatureIndex.FORWARD_DELTA, RelationFeatureIndex.LATERAL_DELTA):
            output[..., index] = torch.sign(values[..., index]) * _log_ratio(values[..., index].abs(), 20.0)
        else:
            output[..., index] = _log_ratio(values[..., index], 20.0)
    output[..., RelationFeatureIndex.RANGE_MARGIN] = torch.sign(
        values[..., RelationFeatureIndex.RANGE_MARGIN]
    ) * _log_ratio(values[..., RelationFeatureIndex.RANGE_MARGIN].abs(), 20.0)
    for index, maximum in (
        (RelationFeatureIndex.TIME_TO_RANGE, 300.0),
        (RelationFeatureIndex.PHYSICAL_DPS, 1_000_000.0),
        (RelationFeatureIndex.ARTS_DPS, 1_000_000.0),
        (RelationFeatureIndex.TRUE_DPS, 1_000_000.0),
        (RelationFeatureIndex.NORMAL_ATTACK_DAMAGE, 1_000_000.0),
        (RelationFeatureIndex.NORMAL_ATTACKS_TO_KILL, 10_000.0),
        (RelationFeatureIndex.SUSTAINED_TTK, 300.0),
        (RelationFeatureIndex.POTENTIAL_TARGET_COUNT, 100.0),
        (RelationFeatureIndex.HEAL_BURST, 1_000_000.0),
        (RelationFeatureIndex.HPS_SUPPORT, 1_000_000.0),
        (RelationFeatureIndex.DEBUFF_SYNERGY_GAIN, 1_000_000.0),
        (RelationFeatureIndex.PROTECTION_SECONDS, 10.0),
    ):
        output[..., index] = _log_ratio(values[..., index], maximum)
    output[..., RelationFeatureIndex.OPENING_DAMAGE_RATIO] = torch.tanh(
        values[..., RelationFeatureIndex.OPENING_DAMAGE_RATIO]
    )
    return output


def normalize_formation_features(values: torch.Tensor) -> torch.Tensor:
    output = values.clone()
    for index, maximum in (
        (FormationFeatureIndex.FRIEND_DENSITY_1, 20.0),
        (FormationFeatureIndex.FRIEND_DENSITY_3, 50.0),
        (FormationFeatureIndex.NEAREST_FRIEND_DISTANCE, 20.0),
        (FormationFeatureIndex.DISTANCE_TO_CENTROID, 20.0),
        (FormationFeatureIndex.FORMATION_WIDTH, 20.0),
        (FormationFeatureIndex.FORMATION_DEPTH, 20.0),
        (FormationFeatureIndex.INCOMING_DPS_PER_HP, 100.0),
        (FormationFeatureIndex.AOE_EXPOSURE, 100.0),
        (FormationFeatureIndex.MAX_PROTECTION_SECONDS, 10.0),
        (FormationFeatureIndex.INITIATIVE_WINDOW, 10.0),
    ):
        output[..., index] = _log_ratio(values[..., index], maximum)
    output[..., FormationFeatureIndex.FORWARD_PROJECTION] = torch.sign(
        values[..., FormationFeatureIndex.FORWARD_PROJECTION]
    ) * _log_ratio(values[..., FormationFeatureIndex.FORWARD_PROJECTION].abs(), 20.0)
    return output


@dataclass(frozen=True)
class ModelConfig:
    num_enemy_ids: int
    embedding_dim: int = 128
    heads: int = 4
    layers: int = 3
    dropout: float = 0.1
    combat_feature_dim: int = COMBAT_FEATURE_DIM
    formation_feature_dim: int = len(FormationFeatureIndex)
    relation_feature_dim: int = len(RelationFeatureIndex)
    relation_mask_dim: int = len(RelationMaskIndex)
    map_width: float = 15.0
    map_height: float = 11.0

    def __post_init__(self) -> None:
        if self.num_enemy_ids < 2:
            raise ValueError("num_enemy_ids must include padding and at least one enemy")
        if self.embedding_dim % self.heads:
            raise ValueError("embedding_dim must be divisible by heads")
        if self.combat_feature_dim != COMBAT_FEATURE_DIM:
            raise ValueError(f"combat_feature_dim must be {COMBAT_FEATURE_DIM}")
        if self.formation_feature_dim != len(FormationFeatureIndex):
            raise ValueError(f"formation_feature_dim must be {len(FormationFeatureIndex)}")
        if self.relation_feature_dim != len(RelationFeatureIndex):
            raise ValueError(f"relation_feature_dim must be {len(RelationFeatureIndex)}")
        if self.relation_mask_dim != len(RelationMaskIndex):
            raise ValueError(f"relation_mask_dim must be {len(RelationMaskIndex)}")


class RelationalEncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.embedding_dim,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim * 4, config.embedding_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, sequence: torch.Tensor, attention_bias: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = sequence.shape
        heads = attention_bias.shape[1]
        normalized = self.norm1(sequence)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_bias.reshape(batch_size * heads, token_count, token_count),
            need_weights=False,
        )
        sequence = sequence + self.dropout(attended)
        return sequence + self.dropout(self.feed_forward(self.norm2(sequence)))


class DuelTransformer(nn.Module):
    """Joint Relation Transformer with an exactly antisymmetric pair score."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.enemy_embedding = nn.Embedding(config.num_enemy_ids, config.embedding_dim, padding_idx=0)
        self.position_encoder = nn.Sequential(
            nn.Linear(2, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.unit_encoder = nn.Sequential(
            nn.Linear(config.combat_feature_dim + 1, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.formation_encoder = nn.Sequential(
            nn.Linear(config.formation_feature_dim, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.side_embedding = nn.Embedding(2, config.embedding_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.relation_encoder = nn.Sequential(
            nn.Linear(config.relation_feature_dim + config.relation_mask_dim, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.heads),
        )
        self.relation_context = nn.Linear(config.heads, config.embedding_dim)
        self.layers = nn.ModuleList(RelationalEncoderLayer(config) for _ in range(config.layers))
        self.output = nn.Sequential(nn.LayerNorm(config.embedding_dim), nn.Linear(config.embedding_dim, 1))

    def _ordered_score(
        self,
        first_ids: torch.Tensor,
        first_positions: torch.Tensor,
        first_combat: torch.Tensor,
        first_combat_known: torch.Tensor,
        first_formation: torch.Tensor,
        first_mask: torch.Tensor,
        second_ids: torch.Tensor,
        second_positions: torch.Tensor,
        second_combat: torch.Tensor,
        second_combat_known: torch.Tensor,
        second_formation: torch.Tensor,
        second_mask: torch.Tensor,
        relations: torch.Tensor,
        relation_masks: torch.Tensor,
    ) -> torch.Tensor:
        enemy_ids = torch.cat((first_ids, second_ids), dim=1)
        positions = torch.cat((first_positions, second_positions), dim=1)
        combat = torch.cat((first_combat, second_combat), dim=1)
        combat_known = torch.cat((first_combat_known, second_combat_known), dim=1)
        formation = torch.cat((first_formation, second_formation), dim=1)
        valid_mask = torch.cat((first_mask, second_mask), dim=1)
        sides = torch.cat((torch.zeros_like(first_ids), torch.ones_like(second_ids)), dim=1)
        position_scale = torch.tensor(
            (self.config.map_width, self.config.map_height), dtype=positions.dtype, device=positions.device
        )
        relation_input = torch.cat((normalize_relation_features(relations), relation_masks.to(relations.dtype)), dim=-1)
        encoded_relations = self.relation_encoder(relation_input)
        relation_valid = relation_masks[..., RelationMaskIndex.VALID].unsqueeze(-1).to(relations.dtype)
        encoded_relations = encoded_relations * relation_valid
        target_valid = valid_mask[:, None, :, None].to(relations.dtype)
        relation_context = (encoded_relations * target_valid).sum(dim=2) / target_valid.sum(dim=2).clamp(min=1.0)
        tokens = (
            self.enemy_embedding(enemy_ids)
            + self.position_encoder(positions / position_scale)
            + self.unit_encoder(torch.cat((combat, combat_known.unsqueeze(-1).to(combat.dtype)), dim=-1))
            + self.formation_encoder(normalize_formation_features(formation))
            + self.side_embedding(sides)
            + self.relation_context(relation_context)
        )
        batch_size, unit_count, _ = tokens.shape
        sequence = torch.cat((self.cls_token.expand(batch_size, -1, -1), tokens), dim=1)
        attention_bias = torch.zeros(
            (batch_size, self.config.heads, unit_count + 1, unit_count + 1),
            dtype=relations.dtype,
            device=relations.device,
        )
        attention_bias[:, :, 1:, 1:] = encoded_relations.permute(0, 3, 1, 2)
        cls_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=valid_mask.device)
        sequence_valid = torch.cat((cls_valid, valid_mask), dim=1)
        attention_bias = attention_bias.masked_fill(~sequence_valid[:, None, None, :], -10_000.0)
        for layer in self.layers:
            sequence = layer(sequence, attention_bias)
        return self.output(sequence[:, 0]).squeeze(-1)

    @staticmethod
    def _swap_relations(
        relations: torch.Tensor,
        relation_masks: torch.Tensor,
        left_count: int,
        right_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        order = torch.cat(
            (
                torch.arange(left_count, left_count + right_count, device=relations.device),
                torch.arange(0, left_count, device=relations.device),
            )
        )
        return (
            relations.index_select(1, order).index_select(2, order),
            relation_masks.index_select(1, order).index_select(2, order),
        )

    def forward(
        self,
        left_ids: torch.Tensor,
        left_positions: torch.Tensor,
        left_combat: torch.Tensor,
        left_combat_known: torch.Tensor,
        left_formation: torch.Tensor,
        left_mask: torch.Tensor,
        right_ids: torch.Tensor,
        right_positions: torch.Tensor,
        right_combat: torch.Tensor,
        right_combat_known: torch.Tensor,
        right_formation: torch.Tensor,
        right_mask: torch.Tensor,
        relations: torch.Tensor,
        relation_masks: torch.Tensor,
    ) -> torch.Tensor:
        left_first = self._ordered_score(
            left_ids,
            left_positions,
            left_combat,
            left_combat_known,
            left_formation,
            left_mask,
            right_ids,
            right_positions,
            right_combat,
            right_combat_known,
            right_formation,
            right_mask,
            relations,
            relation_masks,
        )
        swapped_relations, swapped_masks = self._swap_relations(
            relations, relation_masks, left_ids.shape[1], right_ids.shape[1]
        )
        right_first = self._ordered_score(
            right_ids,
            right_positions,
            right_combat,
            right_combat_known,
            right_formation,
            right_mask,
            left_ids,
            left_positions,
            left_combat,
            left_combat_known,
            left_formation,
            left_mask,
            swapped_relations,
            swapped_masks,
        )
        return left_first - right_first
