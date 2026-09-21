from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import IntEnum

import torch

from maa_duel.combat import DERIVED_FORMULA_VERSION
from maa_duel.training.features import OPENING_WINDOW_SECONDS, CombatFeatureIndex, CombatFeatureTable, PairFeatureIndex

TIME_LIMIT_SECONDS = 300.0
ATTACK_COUNT_LIMIT = 10_000.0
SCREENING_LANE_WIDTH = 0.75
SCREENING_MOBILITY_BUFFER = 0.75
PROTECTION_WINDOW_SECONDS = 10.0


def formula_sha256() -> str:
    payload = {
        "version": DERIVED_FORMULA_VERSION,
        "physical_minimum_ratio": 0.05,
        "arts_minimum_ratio": 0.05,
        "opening_window_seconds": OPENING_WINDOW_SECONDS,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "attack_count_limit": ATTACK_COUNT_LIMIT,
        "target_softmax_temperature": 1.0,
        "screening_lane_width": SCREENING_LANE_WIDTH,
        "screening_mobility_buffer": SCREENING_MOBILITY_BUFFER,
        "screening_mobility_formula": "smoothstep(retained_initial_lead_after_projected_advance)",
        "protection_window_seconds": PROTECTION_WINDOW_SECONDS,
        "relation_features": [item.name for item in RelationFeatureIndex],
        "formation_features": [item.name for item in FormationFeatureIndex],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RelationFeatureIndex(IntEnum):
    DISTANCE = 0
    FORWARD_DELTA = 1
    LATERAL_DELTA = 2
    RANGE_MARGIN = 3
    TIME_TO_RANGE = 4
    PHYSICAL_DPS = 5
    ARTS_DPS = 6
    TRUE_DPS = 7
    NORMAL_ATTACK_DAMAGE = 8
    NORMAL_ATTACKS_TO_KILL = 9
    SUSTAINED_TTK = 10
    OPENING_DAMAGE_RATIO = 11
    POTENTIAL_TARGET_COUNT = 12
    HARD_CONTROL_UPTIME = 13
    TARGET_ALLOCATION = 14
    HEAL_BURST = 15
    HPS_SUPPORT = 16
    DEBUFF_SYNERGY_GAIN = 17
    SCREENING_SCORE = 18
    PROTECTION_SECONDS = 19


class FormationFeatureIndex(IntEnum):
    FRIEND_DENSITY_1 = 0
    FRIEND_DENSITY_3 = 1
    NEAREST_FRIEND_DISTANCE = 2
    DISTANCE_TO_CENTROID = 3
    FORWARD_PROJECTION = 4
    FORMATION_WIDTH = 5
    FORMATION_DEPTH = 6
    INCOMING_DPS_PER_HP = 7
    AOE_EXPOSURE = 8
    MAX_SCREENING_SCORE = 9
    MAX_PROTECTION_SECONDS = 10
    INITIATIVE_WINDOW = 11


class RelationMaskIndex(IntEnum):
    VALID = 0
    ESTIMATED = 1
    TARGETABLE = 2
    FINITE_TIME = 3


@dataclass(frozen=True)
class DerivedBattleFeatures:
    formation: torch.Tensor
    relations: torch.Tensor
    relation_masks: torch.Tensor


def _gather_units(values: torch.Tensor, enemy_ids: torch.Tensor) -> torch.Tensor:
    return values.to(enemy_ids.device)[enemy_ids]


def _gather_pairs(values: torch.Tensor, enemy_ids: torch.Tensor) -> torch.Tensor:
    table = values.to(enemy_ids.device)
    return table[enemy_ids.unsqueeze(2), enemy_ids.unsqueeze(1)]


def _team_directions(
    positions: torch.Tensor,
    sides: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    centroids = []
    for side in (0, 1):
        mask = valid & (sides == side)
        weights = mask.unsqueeze(-1).to(positions.dtype)
        centroid = (positions * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        centroids.append(centroid)
    axis = centroids[1] - centroids[0]
    norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    certain = norm.squeeze(-1) > 1e-6
    axis = torch.where(norm > 1e-6, axis / norm.clamp(min=1e-6), torch.zeros_like(axis))
    direction = torch.where((sides == 0).unsqueeze(-1), axis.unsqueeze(1), -axis.unsqueeze(1))
    return direction, certain


def _target_allocations(
    time_to_range: torch.Tensor,
    sides: torch.Tensor,
    valid: torch.Tensor,
    targetable: torch.Tensor,
    target_count: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    batch_size, unit_count = sides.shape
    output = torch.zeros_like(time_to_range)
    for batch in range(batch_size):
        for source in range(unit_count):
            if not valid[batch, source]:
                continue
            candidates = valid[batch] & (sides[batch] != sides[batch, source]) & targetable[batch, source]
            count = int(candidates.sum())
            if not count:
                continue
            times = time_to_range[batch, source, candidates]
            finite = torch.isfinite(times) & (times < TIME_LIMIT_SECONDS)
            if not finite.any():
                continue
            scores = torch.full_like(times, -10_000.0)
            scores[finite] = -times[finite] / temperature
            weights = torch.softmax(scores, dim=0)
            capacity = min(float(max(target_count[batch, source].item(), 1.0)), float(count))
            output[batch, source, candidates] = weights * capacity
    return output


def _potential_target_counts(
    positions: torch.Tensor,
    sides: torch.Tensor,
    valid: torch.Tensor,
    area_radius: torch.Tensor,
    target_count: torch.Tensor,
    enemy_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, unit_count = sides.shape
    output = torch.zeros((*sides.shape, unit_count), dtype=positions.dtype, device=positions.device)
    for batch in range(batch_size):
        for source in range(unit_count):
            if not valid[batch, source]:
                continue
            opponents = valid[batch] & (sides[batch] != sides[batch, source])
            for target in torch.nonzero(opponents, as_tuple=False).flatten().tolist():
                radius = float(area_radius[batch, source])
                coverage = 1
                if radius > 0:
                    distances = torch.linalg.vector_norm(positions[batch, opponents] - positions[batch, target], dim=-1)
                    coverage = int((distances <= radius + 1e-6).sum())
                direct_capacity = min(int(max(target_count[batch, source].item(), 1)), int(opponents.sum()))
                output[batch, source, target] = max(coverage, direct_capacity)
    return output * enemy_mask.to(output.dtype)


def _friendly_healing(
    raw: torch.Tensor,
    sides: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, unit_count = sides.shape
    burst = torch.zeros((batch_size, unit_count, unit_count), dtype=raw.dtype, device=raw.device)
    hps = torch.zeros_like(burst)
    for batch in range(batch_size):
        for source in range(unit_count):
            friends = valid[batch] & (sides[batch] == sides[batch, source])
            friends[source] = False
            friend_count = int(friends.sum())
            if not valid[batch, source] or not friend_count:
                continue
            source_burst = raw[batch, source, CombatFeatureIndex.HEAL_BURST]
            source_hps = raw[batch, source, CombatFeatureIndex.HPS]
            capacity = min(float(max(raw[batch, source, CombatFeatureIndex.TARGET_COUNT].item(), 1.0)), friend_count)
            allocation = capacity / friend_count
            burst[batch, source, friends] = source_burst * allocation
            hps[batch, source, friends] = source_hps * allocation
    return burst, hps


def _debuff_synergy(
    raw: torch.Tensor,
    pairs: torch.Tensor,
    allocation: torch.Tensor,
    sides: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    batch_size, unit_count = sides.shape
    output = torch.zeros((batch_size, unit_count, unit_count), dtype=raw.dtype, device=raw.device)
    for batch in range(batch_size):
        for debuffer in range(unit_count):
            if not valid[batch, debuffer]:
                continue
            defense_flat = raw[batch, debuffer, CombatFeatureIndex.DEFENSE_SHRED_FLAT]
            defense_ratio = raw[batch, debuffer, CombatFeatureIndex.DEFENSE_SHRED_RATIO]
            resistance_flat = raw[batch, debuffer, CombatFeatureIndex.RESISTANCE_SHRED_FLAT]
            resistance_ratio = raw[batch, debuffer, CombatFeatureIndex.RESISTANCE_SHRED_RATIO]
            if not any(float(value) > 0 for value in (defense_flat, defense_ratio, resistance_flat, resistance_ratio)):
                continue
            for ally in range(unit_count):
                if debuffer == ally or not valid[batch, ally] or sides[batch, ally] != sides[batch, debuffer]:
                    continue
                weighted_gain = 0.0
                for enemy in range(unit_count):
                    if not valid[batch, enemy] or sides[batch, enemy] == sides[batch, ally]:
                        continue
                    weight = float(allocation[batch, ally, enemy])
                    if weight <= 0:
                        continue
                    baseline_physical = float(pairs[batch, ally, enemy, PairFeatureIndex.PHYSICAL_DPS])
                    baseline_arts = float(pairs[batch, ally, enemy, PairFeatureIndex.ARTS_DPS])
                    defense_gain = baseline_physical * min(1.0, float(defense_ratio) + float(defense_flat) / 10_000.0)
                    resistance_gain = baseline_arts * min(2.0, float(resistance_ratio) + float(resistance_flat) / 100.0)
                    weighted_gain += weight * (defense_gain + resistance_gain)
                output[batch, debuffer, ally] = weighted_gain
    return output


def _screening(
    positions: torch.Tensor,
    raw: torch.Tensor,
    pairs: torch.Tensor,
    allocation: torch.Tensor,
    time_to_range: torch.Tensor,
    sides: torch.Tensor,
    valid: torch.Tensor,
    targetable: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, unit_count = sides.shape
    score = torch.zeros((batch_size, unit_count, unit_count), dtype=positions.dtype, device=positions.device)
    seconds = torch.zeros_like(score)
    for batch in range(batch_size):
        for protector in range(unit_count):
            if not valid[batch, protector]:
                continue
            incoming = 0.0
            for enemy in range(unit_count):
                if not valid[batch, enemy] or sides[batch, enemy] == sides[batch, protector]:
                    continue
                enemy_dps = float(pairs[batch, enemy, protector, :3].sum())
                incoming += enemy_dps * float(allocation[batch, enemy, protector])
            survival = min(
                PROTECTION_WINDOW_SECONDS,
                float(raw[batch, protector, CombatFeatureIndex.HP]) / max(incoming, 1e-6),
            )
            for protected in range(unit_count):
                if (
                    protector == protected
                    or not valid[batch, protected]
                    or sides[batch, protected] != sides[batch, protector]
                ):
                    continue
                weighted_geometry = 0.0
                threat_total = 0.0
                protected_position = positions[batch, protected]
                protector_position = positions[batch, protector]
                for enemy in range(unit_count):
                    if not valid[batch, enemy] or sides[batch, enemy] == sides[batch, protected]:
                        continue
                    if not targetable[batch, enemy, protector]:
                        continue
                    line = positions[batch, enemy] - protected_position
                    denominator = float(torch.dot(line, line))
                    if denominator <= 1e-8:
                        continue
                    relative = protector_position - protected_position
                    t = float(torch.dot(relative, line)) / denominator
                    if not 0.0 < t < 1.0:
                        continue
                    perpendicular = torch.linalg.vector_norm(relative - t * line)
                    lane = math.exp(-0.5 * (float(perpendicular) / SCREENING_LANE_WIDTH) ** 2)
                    initial_lead = t * math.sqrt(denominator)
                    threat_horizon = min(
                        float(time_to_range[batch, enemy, protected]),
                        PROTECTION_WINDOW_SECONDS,
                    )
                    protector_move_time = min(
                        threat_horizon,
                        float(time_to_range[batch, protector, enemy]),
                    )
                    protected_move_time = min(
                        threat_horizon,
                        float(time_to_range[batch, protected, enemy]),
                    )
                    protector_advance = (
                        float(raw[batch, protector, CombatFeatureIndex.MOVE_SPEED]) * protector_move_time
                    )
                    protected_advance = (
                        float(raw[batch, protected, CombatFeatureIndex.MOVE_SPEED]) * protected_move_time
                    )
                    lost_lead = max(0.0, protected_advance - protector_advance)
                    retained_lead = max(
                        0.0,
                        min(1.0, 1.0 - lost_lead / max(initial_lead, SCREENING_MOBILITY_BUFFER)),
                    )
                    mobility_offset = retained_lead * retained_lead * (3.0 - 2.0 * retained_lead)
                    contact_advantage = torch.sigmoid(
                        time_to_range[batch, enemy, protected] - time_to_range[batch, enemy, protector]
                    )
                    intercept = max(float(allocation[batch, enemy, protector]), float(contact_advantage))
                    threat = float(pairs[batch, enemy, protected, :3].sum()) * max(
                        float(allocation[batch, enemy, protected]), 1e-6
                    )
                    weighted_geometry += threat * lane * intercept * mobility_offset
                    threat_total += threat
                if threat_total > 0:
                    score[batch, protector, protected] = min(1.0, weighted_geometry / threat_total)
                    seconds[batch, protector, protected] = score[batch, protector, protected] * survival
    return score, seconds


def build_derived_battle_features(
    enemy_ids: torch.Tensor,
    positions: torch.Tensor,
    sides: torch.Tensor,
    valid: torch.Tensor,
    table: CombatFeatureTable,
) -> DerivedBattleFeatures:
    """Build FP32 formula features for one padded batch in shared tile coordinates."""
    enemy_ids = enemy_ids.long()
    positions = positions.float()
    raw = _gather_units(table.raw, enemy_ids).float()
    pairs = _gather_pairs(table.pairs, enemy_ids).float()
    pair_valid = _gather_pairs(table.pair_valid, enemy_ids)
    pair_estimated = _gather_pairs(table.pair_estimated, enemy_ids)
    targetable = _gather_pairs(table.pair_targetable, enemy_ids)
    pair_finite = _gather_pairs(table.pair_finite, enemy_ids)
    unit_pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    enemy_mask = unit_pair_valid & (sides.unsqueeze(2) != sides.unsqueeze(1))
    friendly_mask = unit_pair_valid & (sides.unsqueeze(2) == sides.unsqueeze(1))
    eye = torch.eye(sides.shape[1], dtype=torch.bool, device=sides.device).unsqueeze(0)
    friendly_mask &= ~eye

    delta = positions.unsqueeze(1) - positions.unsqueeze(2)
    distance = torch.linalg.vector_norm(delta, dim=-1)
    directions, direction_certain = _team_directions(positions, sides, valid)
    forward = (delta * directions.unsqueeze(2)).sum(dim=-1)
    lateral_axis = torch.stack((-directions[..., 1], directions[..., 0]), dim=-1)
    lateral = (delta * lateral_axis.unsqueeze(2)).sum(dim=-1)
    source_range = raw[..., CombatFeatureIndex.ATTACK_RADIUS].unsqueeze(2)
    target_range = raw[..., CombatFeatureIndex.ATTACK_RADIUS].unsqueeze(1)
    range_margin = source_range - distance
    source_speed = raw[..., CombatFeatureIndex.MOVE_SPEED].unsqueeze(2)
    target_speed = raw[..., CombatFeatureIndex.MOVE_SPEED].unsqueeze(1)
    closing_speed = source_speed + target_speed
    direct_approach = torch.where(
        closing_speed > 1e-6,
        (-range_margin).clamp(min=0.0) / closing_speed.clamp(min=1e-6),
        torch.full_like(distance, float("inf")),
    )
    until_target_stops = torch.where(
        distance > target_range,
        torch.where(
            closing_speed > 1e-6,
            (distance - target_range) / closing_speed.clamp(min=1e-6),
            torch.full_like(distance, float("inf")),
        ),
        torch.zeros_like(distance),
    )
    remaining_after_target_stops = (torch.minimum(distance, target_range) - source_range).clamp(min=0.0)
    after_target_stops = torch.where(
        remaining_after_target_stops <= 0,
        torch.zeros_like(distance),
        torch.where(
            source_speed > 1e-6,
            remaining_after_target_stops / source_speed.clamp(min=1e-6),
            torch.full_like(distance, float("inf")),
        ),
    )
    time_to_range = torch.where(
        range_margin >= 0,
        torch.zeros_like(distance),
        torch.where(source_range >= target_range, direct_approach, until_target_stops + after_target_stops),
    )
    time_to_range = torch.where(enemy_mask & targetable, time_to_range, torch.full_like(distance, float("inf")))
    target_count = raw[..., CombatFeatureIndex.TARGET_COUNT].clamp(min=1.0)
    allocation = _target_allocations(time_to_range, sides, valid, targetable, target_count)
    potential_targets = _potential_target_counts(
        positions,
        sides,
        valid,
        raw[..., CombatFeatureIndex.AREA_RADIUS],
        target_count,
        enemy_mask,
    )
    heal_burst, hps = _friendly_healing(raw, sides, valid)
    synergy = _debuff_synergy(raw, pairs, allocation, sides, valid)
    screening, protection = _screening(
        positions,
        raw,
        pairs,
        allocation,
        time_to_range,
        sides,
        valid,
        targetable,
    )

    relations = torch.zeros((*distance.shape, len(RelationFeatureIndex)), dtype=torch.float32, device=positions.device)
    relations[..., RelationFeatureIndex.DISTANCE] = distance
    relations[..., RelationFeatureIndex.FORWARD_DELTA] = forward
    relations[..., RelationFeatureIndex.LATERAL_DELTA] = lateral
    relations[..., RelationFeatureIndex.RANGE_MARGIN] = range_margin
    relations[..., RelationFeatureIndex.TIME_TO_RANGE] = time_to_range.clamp(max=TIME_LIMIT_SECONDS)
    relations[..., RelationFeatureIndex.PHYSICAL_DPS] = pairs[..., PairFeatureIndex.PHYSICAL_DPS] * enemy_mask
    relations[..., RelationFeatureIndex.ARTS_DPS] = pairs[..., PairFeatureIndex.ARTS_DPS] * enemy_mask
    relations[..., RelationFeatureIndex.TRUE_DPS] = pairs[..., PairFeatureIndex.TRUE_DPS] * enemy_mask
    relations[..., RelationFeatureIndex.NORMAL_ATTACK_DAMAGE] = (
        pairs[..., PairFeatureIndex.NORMAL_ATTACK_DAMAGE] * enemy_mask
    )
    relations[..., RelationFeatureIndex.NORMAL_ATTACKS_TO_KILL] = (
        pairs[..., PairFeatureIndex.NORMAL_ATTACKS_TO_KILL].clamp(max=ATTACK_COUNT_LIMIT) * enemy_mask
    )
    relations[..., RelationFeatureIndex.SUSTAINED_TTK] = (
        pairs[..., PairFeatureIndex.SUSTAINED_TTK].clamp(max=TIME_LIMIT_SECONDS) * enemy_mask
    )
    relations[..., RelationFeatureIndex.OPENING_DAMAGE_RATIO] = (
        pairs[..., PairFeatureIndex.OPENING_DAMAGE_RATIO] * enemy_mask
    )
    relations[..., RelationFeatureIndex.POTENTIAL_TARGET_COUNT] = potential_targets
    relations[..., RelationFeatureIndex.HARD_CONTROL_UPTIME] = (
        pairs[..., PairFeatureIndex.HARD_CONTROL_UPTIME] * enemy_mask
    )
    relations[..., RelationFeatureIndex.TARGET_ALLOCATION] = allocation
    relations[..., RelationFeatureIndex.HEAL_BURST] = heal_burst * friendly_mask
    relations[..., RelationFeatureIndex.HPS_SUPPORT] = hps * friendly_mask
    relations[..., RelationFeatureIndex.DEBUFF_SYNERGY_GAIN] = synergy * friendly_mask
    relations[..., RelationFeatureIndex.SCREENING_SCORE] = screening * friendly_mask
    relations[..., RelationFeatureIndex.PROTECTION_SECONDS] = protection * friendly_mask

    relation_masks = torch.stack(
        (
            unit_pair_valid & (pair_valid | friendly_mask),
            pair_estimated & enemy_mask,
            (targetable & enemy_mask) | friendly_mask,
            (pair_finite & enemy_mask) | friendly_mask,
        ),
        dim=-1,
    )

    batch_size, unit_count = sides.shape
    formation = torch.zeros((batch_size, unit_count, len(FormationFeatureIndex)), dtype=torch.float32)
    formation = formation.to(positions.device)
    total_dps = pairs[..., :3].sum(dim=-1)
    incoming = (total_dps * allocation * enemy_mask).sum(dim=1)
    for batch in range(batch_size):
        for side in (0, 1):
            team = valid[batch] & (sides[batch] == side)
            indices = torch.nonzero(team, as_tuple=False).flatten()
            if not len(indices):
                continue
            team_positions = positions[batch, indices]
            centroid = team_positions.mean(dim=0)
            direction = directions[batch, indices[0]]
            lateral_direction = torch.stack((-direction[1], direction[0]))
            centered = team_positions - centroid
            width = torch.std(centered @ lateral_direction, unbiased=False) if len(indices) > 1 else torch.tensor(0.0)
            depth = torch.std(centered @ direction, unbiased=False) if len(indices) > 1 else torch.tensor(0.0)
            for unit in indices.tolist():
                friends = team.clone()
                friends[unit] = False
                friend_distances = distance[batch, unit, friends]
                if len(friend_distances):
                    formation[batch, unit, FormationFeatureIndex.FRIEND_DENSITY_1] = torch.exp(
                        -0.5 * friend_distances.square()
                    ).sum()
                    formation[batch, unit, FormationFeatureIndex.FRIEND_DENSITY_3] = torch.exp(
                        -0.5 * (friend_distances / 3.0).square()
                    ).sum()
                    formation[batch, unit, FormationFeatureIndex.NEAREST_FRIEND_DISTANCE] = friend_distances.min()
                else:
                    formation[batch, unit, FormationFeatureIndex.NEAREST_FRIEND_DISTANCE] = TIME_LIMIT_SECONDS
                relative = positions[batch, unit] - centroid
                formation[batch, unit, FormationFeatureIndex.DISTANCE_TO_CENTROID] = torch.linalg.vector_norm(relative)
                formation[batch, unit, FormationFeatureIndex.FORWARD_PROJECTION] = torch.dot(relative, direction)
                formation[batch, unit, FormationFeatureIndex.FORMATION_WIDTH] = width
                formation[batch, unit, FormationFeatureIndex.FORMATION_DEPTH] = depth
                formation[batch, unit, FormationFeatureIndex.INCOMING_DPS_PER_HP] = incoming[batch, unit] / raw[
                    batch, unit, CombatFeatureIndex.HP
                ].clamp(min=1.0)
                exposure = 0.0
                for enemy in range(unit_count):
                    if not enemy_mask[batch, enemy, unit]:
                        continue
                    exposure += float(allocation[batch, enemy, unit]) * max(
                        float(potential_targets[batch, enemy, unit]) - 1.0, 0.0
                    )
                formation[batch, unit, FormationFeatureIndex.AOE_EXPOSURE] = exposure
                formation[batch, unit, FormationFeatureIndex.MAX_SCREENING_SCORE] = screening[batch, :, unit].max()
                formation[batch, unit, FormationFeatureIndex.MAX_PROTECTION_SECONDS] = protection[batch, :, unit].max()
                own_times = time_to_range[batch, unit, enemy_mask[batch, unit]]
                threat_times = time_to_range[batch, :, unit][enemy_mask[batch, :, unit]]
                own_first = own_times.min() if len(own_times) else torch.tensor(TIME_LIMIT_SECONDS)
                threat_first = threat_times.min() if len(threat_times) else torch.tensor(TIME_LIMIT_SECONDS)
                formation[batch, unit, FormationFeatureIndex.INITIATIVE_WINDOW] = torch.clamp(
                    threat_first - own_first, 0.0, PROTECTION_WINDOW_SECONDS
                )
                if not direction_certain[batch]:
                    formation[batch, unit, FormationFeatureIndex.FORWARD_PROJECTION] = 0.0
    formation *= valid.unsqueeze(-1)
    return DerivedBattleFeatures(formation=formation, relations=relations, relation_masks=relation_masks)
