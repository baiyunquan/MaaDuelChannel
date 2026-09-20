from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSignals:
    timestamp: float
    game_visible: bool
    countdown_seconds: int | None = None
    round_number: int | None = None
    layout_score: float = 0.0


@dataclass(frozen=True)
class RoundWindow:
    round_index: int
    prep_time: float | None
    layout_time: float | None
    battle_start: float | None
    battle_end: float
    complete: bool
    failure_reasons: tuple[str, ...]


@dataclass
class _ActiveRound:
    sequence_index: int
    observed_round_number: int | None = None
    prep_time: float | None = None
    zero_seen: bool = False
    last_countdown_seconds: int | None = None
    layout_time: float | None = None
    layout_score: float = 0.0
    battle_start: float | None = None


class RoundSegmenter:
    """Turn OCR and layout signals into independent round time windows."""

    def segment(self, signals: list[FrameSignals]) -> list[RoundWindow]:
        ordered = sorted(signals, key=lambda item: item.timestamp)
        rounds: list[RoundWindow] = []
        active: _ActiveRound | None = None
        last_game_timestamp: float | None = None

        for item in ordered:
            if not item.game_visible:
                continue
            last_game_timestamp = item.timestamp

            if item.countdown_seconds is not None:
                if active is not None and active.battle_start is not None:
                    rounds.append(self._finalize(active, item.timestamp))
                    active = None
                if active is None:
                    active = _ActiveRound(sequence_index=len(rounds) + 1)
                active.last_countdown_seconds = item.countdown_seconds
                if item.countdown_seconds > 0:
                    active.prep_time = item.timestamp
                else:
                    active.zero_seen = True
                continue

            if active is None:
                continue

            if item.round_number is not None:
                active.observed_round_number = item.round_number
                if active.battle_start is None:
                    active.battle_start = item.timestamp
                continue

            if (
                not active.zero_seen
                and active.last_countdown_seconds is not None
                and active.last_countdown_seconds <= 1
            ):
                active.zero_seen = True

            if active.battle_start is None and active.zero_seen and item.layout_score > active.layout_score:
                active.layout_score = item.layout_score
                active.layout_time = item.timestamp

        if active is not None and last_game_timestamp is not None:
            rounds.append(self._finalize(active, last_game_timestamp))
        return rounds

    @staticmethod
    def _finalize(active: _ActiveRound, battle_end: float) -> RoundWindow:
        reasons: list[str] = []
        if active.prep_time is None:
            reasons.append("missing_prep")
        if active.layout_time is None:
            reasons.append("missing_layout")
        if active.battle_start is None:
            reasons.append("missing_round_banner")
        if active.battle_start is not None and battle_end <= active.battle_start:
            reasons.append("missing_battle_frames")
        return RoundWindow(
            round_index=active.observed_round_number or active.sequence_index,
            prep_time=active.prep_time,
            layout_time=active.layout_time,
            battle_start=active.battle_start,
            battle_end=battle_end,
            complete=not reasons,
            failure_reasons=tuple(reasons),
        )
