from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from maa_duel.video.phases import FrameSignals, PhaseThresholds


@dataclass(frozen=True)
class TimedFrame:
    timestamp: float
    frame_index: int
    frame: np.ndarray = field(compare=False, repr=False)
    signals: FrameSignals
    encoded: bool = field(default=False, compare=False)


def select_prep_frame(
    frames: list[TimedFrame] | tuple[TimedFrame, ...],
    *,
    thresholds: PhaseThresholds | None = None,
) -> TimedFrame | None:
    limits = thresholds or PhaseThresholds()
    run = _longest_run(
        frames,
        lambda item: _battlefield(item, limits)
        and item.signals.bottom_panel_score >= limits.bottom_panel
        and item.signals.choice_buttons_score >= limits.choice_buttons,
        minimum=2,
    )
    return run[len(run) // 2] if run else None


def select_layout_frame(
    frames: list[TimedFrame] | tuple[TimedFrame, ...],
    *,
    reference: TimedFrame | None = None,
    require_corner_transition: bool = False,
    thresholds: PhaseThresholds | None = None,
) -> TimedFrame | None:
    limits = thresholds or PhaseThresholds()
    runs = _runs(
        frames,
        lambda item: _battlefield(item, limits)
        and item.signals.bottom_panel_score < limits.bottom_panel
        and item.signals.choice_buttons_score < limits.choice_buttons
        and not _has_countdown(item, limits)
        and not _has_round_banner(item, limits)
        and (
            reference is None
            or not reference.signals.corner_signature
            or not item.signals.corner_signature
            or not _corners_changed(reference, item)
        ),
    )
    valid = [run for run in runs if len(run) >= 2]
    if require_corner_transition:
        supported: list[list[TimedFrame]] = []
        for run in valid:
            later_round = any(
                frame.timestamp > run[-1].timestamp and _has_round_banner(frame, limits)
                for frame in frames
            )
            if later_round:
                supported.append(run)
                continue
            anchor = run[0]
            for index in range(1, len(run) - 1):
                if _corners_changed(anchor, run[index]) and _corners_changed(anchor, run[index + 1]):
                    prefix = run[:index]
                    if len(prefix) >= 2:
                        supported.append(prefix)
                    break
        valid = supported
    run = max(valid, key=lambda item: (len(item), item[0].timestamp), default=[])
    return run[len(run) // 2] if run else None


def select_battle_start(
    frames: list[TimedFrame] | tuple[TimedFrame, ...],
    *,
    reference: TimedFrame | None = None,
    thresholds: PhaseThresholds | None = None,
) -> TimedFrame | None:
    limits = thresholds or PhaseThresholds()
    runs = _runs(
        frames,
        lambda item: _battlefield(item, limits)
        and not _has_countdown(item, limits)
        and not _has_round_banner(item, limits)
        and (
            _corners_changed(reference, item)
            if reference is not None
            and reference.signals.corner_signature
            and item.signals.corner_signature
            else item.signals.corner_mask_score < limits.corner_mask
        ),
    )
    return next((run[0] for run in runs if len(run) >= 2), None)


def select_end_run(
    frames: list[TimedFrame] | tuple[TimedFrame, ...],
    *,
    thresholds: PhaseThresholds | None = None,
) -> tuple[TimedFrame, ...]:
    limits = thresholds or PhaseThresholds()
    runs = _runs(
        frames,
        lambda item: _battlefield(item, limits)
        and not _has_next_round_countdown(item, limits)
        and not _has_round_banner(item, limits),
    )
    valid = [run for run in runs if len(run) >= 2]
    return tuple(valid[-1][-5:]) if valid else ()


def _battlefield(frame: TimedFrame, thresholds: PhaseThresholds) -> bool:
    return frame.signals.battlefield_score >= thresholds.battlefield


def _has_countdown(frame: TimedFrame, thresholds: PhaseThresholds) -> bool:
    return (
        frame.signals.countdown_score >= thresholds.countdown
        or frame.signals.countdown_seconds is not None
    )


def _has_next_round_countdown(frame: TimedFrame, thresholds: PhaseThresholds) -> bool:
    return _has_countdown(frame, thresholds)


def _has_round_banner(frame: TimedFrame, thresholds: PhaseThresholds) -> bool:
    return (
        frame.signals.round_banner_score >= thresholds.round_banner
        or frame.signals.round_number is not None
    )


def _corners_changed(reference: TimedFrame, current: TimedFrame) -> bool:
    before = reference.signals.corner_signature
    after = current.signals.corner_signature
    if len(before) != len(after) or not before or len(before) % 4:
        return False
    corner_size = len(before) // 4
    changed = 0
    for index in range(4):
        start = index * corner_size
        end = start + corner_size
        mean_difference = (
            sum(
                abs(left - right)
                for left, right in zip(before[start:end], after[start:end], strict=True)
            )
            / corner_size
        )
        if mean_difference > 0.05:
            changed += 1
    return changed >= 3


def _runs(
    frames: list[TimedFrame] | tuple[TimedFrame, ...],
    predicate: Callable[[TimedFrame], bool],
) -> list[list[TimedFrame]]:
    ordered = sorted(frames, key=lambda item: (item.timestamp, item.frame_index))
    runs: list[list[TimedFrame]] = []
    current: list[TimedFrame] = []
    for frame in ordered:
        if predicate(frame):
            current.append(frame)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _longest_run(
    frames: list[TimedFrame] | tuple[TimedFrame, ...],
    predicate: Callable[[TimedFrame], bool],
    *,
    minimum: int,
) -> list[TimedFrame]:
    valid = [run for run in _runs(frames, predicate) if len(run) >= minimum]
    return max(valid, key=lambda run: (len(run), run[0].timestamp), default=[])
