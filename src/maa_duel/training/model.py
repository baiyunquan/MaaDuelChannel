from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import nn

from maa_duel.training.features import COMBAT_FEATURE_DIM, CombatFeatureIndex


class RelationFeatureIndex(IntEnum):
    DX = 0
    DY = 1
    DISTANCE_TILES = 2
    SAME_SIDE = 3
    OPPONENT = 4
    SOURCE_IN_RANGE = 5
    TARGET_IN_RANGE = 6
    PHYSICAL_EFFECTIVENESS = 7
    ARTS_EFFECTIVENESS = 8
    TRUE_PRESSURE = 9
    CONTROL_PRESSURE = 10
    DEFENSE_SHRED_SYNERGY = 11
    RESISTANCE_SHRED_SYNERGY = 12
    HEAL_SUPPORT = 13
    SHIELD_SUPPORT = 14
    SOURCE_KNOWN = 15
    TARGET_KNOWN = 16


CONTROL_FEATURES = (
    CombatFeatureIndex.STUN,
    CombatFeatureIndex.COLD,
    CombatFeatureIndex.FREEZE,
    CombatFeatureIndex.SLOW,
    CombatFeatureIndex.BIND,
    CombatFeatureIndex.SLEEP,
    CombatFeatureIndex.SILENCE,
    CombatFeatureIndex.LEVITATE,
    CombatFeatureIndex.FEAR,
)


def _feature(values: torch.Tensor, index: CombatFeatureIndex) -> torch.Tensor:
    return values[..., int(index)]


def pairwise_relation_features(
    positions: torch.Tensor,
    combat: torch.Tensor,
    combat_known: torch.Tensor,
    sides: torch.Tensor,
    *,
    map_width: float = 15.0,
    map_height: float = 11.0,
) -> torch.Tensor:
    """Build directed source-to-target spatial, matchup, and friendly-support features."""
    source_positions = positions.unsqueeze(2)
    target_positions = positions.unsqueeze(1)
    delta = target_positions - source_positions
    distance = torch.sqrt((delta[..., 0] * map_width).square() + (delta[..., 1] * map_height).square())
    source = combat.unsqueeze(2)
    target = combat.unsqueeze(1)
    source_sides = sides.unsqueeze(2)
    target_sides = sides.unsqueeze(1)
    same_side = (source_sides == target_sides).to(combat.dtype)
    opponent = 1.0 - same_side

    source_range = _feature(source, CombatFeatureIndex.ATTACK_RADIUS) * 5.0
    target_range = _feature(target, CombatFeatureIndex.ATTACK_RADIUS) * 5.0
    source_attack = _feature(source, CombatFeatureIndex.ATTACK)
    target_defense = _feature(target, CombatFeatureIndex.DEFENSE)
    target_resistance = _feature(target, CombatFeatureIndex.RESISTANCE)
    physical = opponent * _feature(source, CombatFeatureIndex.PHYSICAL)
    arts = opponent * _feature(source, CombatFeatureIndex.ARTS)
    true_damage = opponent * _feature(source, CombatFeatureIndex.TRUE)
    source_control = torch.stack([_feature(source, index) for index in CONTROL_FEATURES], dim=-1).amax(dim=-1)
    target_immunity = torch.maximum(
        _feature(target, CombatFeatureIndex.STATUS_IMMUNITY),
        _feature(target, CombatFeatureIndex.RESIST) * 0.5,
    )

    output = torch.zeros((*distance.shape, len(RelationFeatureIndex)), dtype=combat.dtype, device=combat.device)
    output[..., RelationFeatureIndex.DX] = delta[..., 0]
    output[..., RelationFeatureIndex.DY] = delta[..., 1]
    output[..., RelationFeatureIndex.DISTANCE_TILES] = distance / max(map_width, map_height)
    output[..., RelationFeatureIndex.SAME_SIDE] = same_side
    output[..., RelationFeatureIndex.OPPONENT] = opponent
    output[..., RelationFeatureIndex.SOURCE_IN_RANGE] = opponent * (distance <= source_range).to(combat.dtype)
    output[..., RelationFeatureIndex.TARGET_IN_RANGE] = opponent * (distance <= target_range).to(combat.dtype)
    output[..., RelationFeatureIndex.PHYSICAL_EFFECTIVENESS] = (
        physical * source_attack * (1.0 - target_defense).clamp(min=0.0)
    )
    output[..., RelationFeatureIndex.ARTS_EFFECTIVENESS] = (
        arts * source_attack * (1.0 - target_resistance).clamp(min=0.0, max=2.0)
    )
    output[..., RelationFeatureIndex.TRUE_PRESSURE] = true_damage * source_attack
    output[..., RelationFeatureIndex.CONTROL_PRESSURE] = (
        opponent * source_control * (1.0 - target_immunity).clamp(min=0.0)
    )
    output[..., RelationFeatureIndex.DEFENSE_SHRED_SYNERGY] = (
        same_side * _feature(source, CombatFeatureIndex.DEFENSE_SHRED) * _feature(target, CombatFeatureIndex.PHYSICAL)
    )
    output[..., RelationFeatureIndex.RESISTANCE_SHRED_SYNERGY] = (
        same_side * _feature(source, CombatFeatureIndex.RESISTANCE_SHRED) * _feature(target, CombatFeatureIndex.ARTS)
    )
    output[..., RelationFeatureIndex.HEAL_SUPPORT] = same_side * _feature(source, CombatFeatureIndex.HEAL)
    output[..., RelationFeatureIndex.SHIELD_SUPPORT] = same_side * _feature(source, CombatFeatureIndex.SHIELD)
    output[..., RelationFeatureIndex.SOURCE_KNOWN] = combat_known.unsqueeze(2).to(combat.dtype)
    output[..., RelationFeatureIndex.TARGET_KNOWN] = combat_known.unsqueeze(1).to(combat.dtype)
    return output


@dataclass(frozen=True)
class ModelConfig:
    num_enemy_ids: int
    embedding_dim: int = 128
    heads: int = 4
    layers: int = 3
    dropout: float = 0.1
    combat_feature_dim: int = COMBAT_FEATURE_DIM
    relation_feature_dim: int = len(RelationFeatureIndex)
    map_width: float = 15.0
    map_height: float = 11.0

    def __post_init__(self) -> None:
        if self.num_enemy_ids < 2:
            raise ValueError("num_enemy_ids must include padding and at least one enemy")
        if self.embedding_dim % self.heads:
            raise ValueError("embedding_dim must be divisible by heads")
        if self.combat_feature_dim != COMBAT_FEATURE_DIM:
            raise ValueError(f"combat_feature_dim must be {COMBAT_FEATURE_DIM}")
        if self.relation_feature_dim != len(RelationFeatureIndex):
            raise ValueError(f"relation_feature_dim must be {len(RelationFeatureIndex)}")


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
    """Joint combat-relation encoder with an antisymmetric pairwise outcome logit."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.enemy_embedding = nn.Embedding(config.num_enemy_ids, config.embedding_dim, padding_idx=0)
        self.position_encoder = nn.Sequential(
            nn.Linear(2, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.combat_encoder = nn.Sequential(
            nn.Linear(config.combat_feature_dim + 1, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.side_embedding = nn.Embedding(2, config.embedding_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.relation_encoder = nn.Sequential(
            nn.Linear(config.relation_feature_dim, config.embedding_dim),
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
        first_mask: torch.Tensor,
        second_ids: torch.Tensor,
        second_positions: torch.Tensor,
        second_combat: torch.Tensor,
        second_combat_known: torch.Tensor,
        second_mask: torch.Tensor,
    ) -> torch.Tensor:
        enemy_ids = torch.cat((first_ids, second_ids), dim=1)
        positions = torch.cat((first_positions, second_positions), dim=1)
        combat = torch.cat((first_combat, second_combat), dim=1)
        combat_known = torch.cat((first_combat_known, second_combat_known), dim=1)
        valid_mask = torch.cat((first_mask, second_mask), dim=1)
        sides = torch.cat((torch.zeros_like(first_ids), torch.ones_like(second_ids)), dim=1)
        unit_relations = pairwise_relation_features(
            positions,
            combat,
            combat_known,
            sides,
            map_width=self.config.map_width,
            map_height=self.config.map_height,
        )
        encoded_unit_relations = self.relation_encoder(unit_relations)
        target_valid = valid_mask[:, None, :, None].to(combat.dtype)
        relation_context = (encoded_unit_relations * target_valid).sum(dim=2) / target_valid.sum(dim=2).clamp(min=1.0)
        tokens = (
            self.enemy_embedding(enemy_ids)
            + self.position_encoder(positions)
            + self.combat_encoder(torch.cat((combat, combat_known.unsqueeze(-1).to(combat.dtype)), dim=-1))
            + self.side_embedding(sides)
            + self.relation_context(relation_context)
        )
        batch_size, unit_count, _ = tokens.shape
        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat((cls, tokens), dim=1)

        relations = torch.zeros(
            (batch_size, unit_count + 1, unit_count + 1, self.config.relation_feature_dim),
            dtype=combat.dtype,
            device=combat.device,
        )
        relations[:, 1:, 1:] = unit_relations
        attention_bias = self.relation_encoder(relations).permute(0, 3, 1, 2)
        cls_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=valid_mask.device)
        sequence_valid = torch.cat((cls_valid, valid_mask), dim=1)
        attention_bias = attention_bias.masked_fill(~sequence_valid[:, None, None, :], -10_000.0)
        for layer in self.layers:
            sequence = layer(sequence, attention_bias)
        return self.output(sequence[:, 0]).squeeze(-1)

    def forward(
        self,
        left_ids: torch.Tensor,
        left_positions: torch.Tensor,
        left_combat: torch.Tensor,
        left_combat_known: torch.Tensor,
        left_mask: torch.Tensor,
        right_ids: torch.Tensor,
        right_positions: torch.Tensor,
        right_combat: torch.Tensor,
        right_combat_known: torch.Tensor,
        right_mask: torch.Tensor,
    ) -> torch.Tensor:
        left_first = self._ordered_score(
            left_ids,
            left_positions,
            left_combat,
            left_combat_known,
            left_mask,
            right_ids,
            right_positions,
            right_combat,
            right_combat_known,
            right_mask,
        )
        right_first = self._ordered_score(
            right_ids,
            right_positions,
            right_combat,
            right_combat_known,
            right_mask,
            left_ids,
            left_positions,
            left_combat,
            left_combat_known,
            left_mask,
        )
        return left_first - right_first
