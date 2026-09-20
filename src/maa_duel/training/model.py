from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    num_enemy_ids: int
    embedding_dim: int = 128
    heads: int = 4
    layers: int = 3
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.num_enemy_ids < 2:
            raise ValueError("num_enemy_ids must include padding and at least one enemy")
        if self.embedding_dim % self.heads:
            raise ValueError("embedding_dim must be divisible by heads")


class TeamEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.enemy_embedding = nn.Embedding(config.num_enemy_ids, config.embedding_dim, padding_idx=0)
        self.position_encoder = nn.Sequential(
            nn.Linear(2, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.heads,
            dim_feedforward=config.embedding_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, enemy_ids: torch.Tensor, positions: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.enemy_embedding(enemy_ids) + self.position_encoder(positions)
        batch_size = tokens.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat((cls, tokens), dim=1)
        cls_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=valid_mask.device)
        padding_mask = ~torch.cat((cls_valid, valid_mask), dim=1)
        encoded = self.transformer(sequence, src_key_padding_mask=padding_mask)
        return self.output_norm(encoded[:, 0])


class DuelTransformer(nn.Module):
    """Shared team encoder with an exactly antisymmetric pairwise logit."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.team_encoder = TeamEncoder(config)
        self.pair_scorer = nn.Sequential(
            nn.Linear(config.embedding_dim * 4, config.embedding_dim * 2),
            nn.GELU(),
            nn.Linear(config.embedding_dim * 2, 1),
        )

    def _ordered_score(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        features = torch.cat((first, second, first - second, first * second), dim=-1)
        return self.pair_scorer(features).squeeze(-1)

    def forward(
        self,
        left_ids: torch.Tensor,
        left_positions: torch.Tensor,
        left_mask: torch.Tensor,
        right_ids: torch.Tensor,
        right_positions: torch.Tensor,
        right_mask: torch.Tensor,
    ) -> torch.Tensor:
        left = self.team_encoder(left_ids, left_positions, left_mask)
        right = self.team_encoder(right_ids, right_positions, right_mask)
        return self._ordered_score(left, right) - self._ordered_score(right, left)

