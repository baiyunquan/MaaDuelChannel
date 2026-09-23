from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class PhaseThresholds:
    battlefield: float = 0.45
    bottom_panel: float = 0.55
    choice_buttons: float = 0.55
    countdown: float = 0.55
    round_banner: float = 0.55
    corner_mask: float = 0.45


@dataclass(frozen=True)
class FrameSignals:
    """Stateless visual evidence for one decoded video frame."""

    timestamp: float
    battlefield_score: float = 0.0
    bottom_panel_score: float = 0.0
    choice_buttons_score: float = 0.0
    countdown_score: float = 0.0
    round_banner_score: float = 0.0
    corner_mask_score: float = 0.0
    corner_signature: tuple[float, ...] = ()
    countdown_seconds: int | None = None
    round_number: int | None = None
    ocr_performed: bool = False


@dataclass(frozen=True)
class TimeSpan:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0.0 or self.end < self.start:
            raise ValueError("time span must be ordered and non-negative")


@dataclass(frozen=True)
class RoundCandidate:
    """Coarse phase spans that bound native-frame evidence searches."""

    round_index: int
    observed_round_number: int | None
    prep_span: TimeSpan | None
    round_span: TimeSpan | None
    layout_search: TimeSpan
    battle_start_search: TimeSpan
    end_search: TimeSpan
    next_phase_start: float
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoundWindow:
    """Final evidence timestamps selected from native-rate decoded frames."""

    round_index: int
    prep_time: float | None
    layout_time: float | None
    battle_start: float | None
    battle_end: float
    complete: bool
    failure_reasons: tuple[str, ...]


@dataclass
class _DraftRound:
    prep_span: TimeSpan | None = None
    round_span: TimeSpan | None = None

    @property
    def anchor(self) -> float:
        if self.prep_span is not None:
            return self.prep_span.start
        assert self.round_span is not None
        return self.round_span.start


class RoundSegmenter:
    """Group noisy per-frame signals into one coarse candidate per real round."""

    def __init__(
        self,
        *,
        thresholds: PhaseThresholds | None = None,
        merge_gap_seconds: float = 0.3,
        fallback_search_seconds: float = 3.0,
        standalone_prep_seconds: float = 0.5,
        maximum_weak_prep_lead_seconds: float = 3.0,
    ) -> None:
        self.thresholds = thresholds or PhaseThresholds()
        self.merge_gap_seconds = merge_gap_seconds
        self.fallback_search_seconds = fallback_search_seconds
        self.standalone_prep_seconds = standalone_prep_seconds
        self.maximum_weak_prep_lead_seconds = maximum_weak_prep_lead_seconds

    def segment(
        self,
        signals: list[FrameSignals],
        *,
        video_end: float | None = None,
    ) -> list[RoundCandidate]:
        ordered = sorted(signals, key=lambda item: item.timestamp)
        if not ordered:
            return []
        if video_end is None:
            video_end = ordered[-1].timestamp
        if video_end < ordered[-1].timestamp:
            raise ValueError("video_end cannot precede analyzed signals")

        sample_step = self._sample_step(ordered)
        merge_gap = max(self.merge_gap_seconds, sample_step * 1.5)
        prep_spans = self._event_spans(
            ordered,
            positive=self._is_prep,
            confirmed=self._is_confirmed_prep,
            merge_gap=merge_gap,
        )
        strong_prep_spans = self._event_spans(
            ordered,
            positive=self._is_strong_prep,
            confirmed=self._is_confirmed_strong_prep,
            merge_gap=merge_gap,
        )
        round_spans = self._event_spans(
            ordered,
            positive=self._is_round,
            confirmed=self._is_confirmed_round,
            merge_gap=merge_gap,
        )

        drafts: list[_DraftRound] = []
        used_prep_spans: list[TimeSpan] = []
        previous_round_end = 0.0
        for round_span in round_spans:
            eligible = [
                span
                for span in prep_spans
                if span not in used_prep_spans
                and span.start >= previous_round_end
                and span.start <= round_span.start
            ]
            if eligible:
                strong = [
                    span
                    for span in eligible
                    if any(self._spans_overlap(span, strong_span) for strong_span in strong_prep_spans)
                ]
                recent_weak = [
                    span
                    for span in eligible
                    if round_span.start - span.end <= self.maximum_weak_prep_lead_seconds
                ]
                if strong or recent_weak:
                    prep_span = max(strong or recent_weak, key=lambda span: span.start)
                    used_prep_spans.append(prep_span)
                    drafts.append(_DraftRound(prep_span=prep_span, round_span=round_span))
                    previous_round_end = round_span.end
                    continue
            drafts.append(_DraftRound(round_span=round_span))
            previous_round_end = round_span.end

        for strong_span in strong_prep_spans:
            if not any(
                self._spans_overlap(strong_span, used_span)
                for used_span in used_prep_spans
            ) and not any(
                self._spans_overlap(strong_span, round_span) for round_span in round_spans
            ) and strong_span.end - strong_span.start >= self.standalone_prep_seconds:
                drafts.append(_DraftRound(prep_span=strong_span))

        drafts.sort(key=lambda draft: draft.anchor)
        results: list[RoundCandidate] = []
        for index, draft in enumerate(drafts):
            next_phase_start = drafts[index + 1].anchor if index + 1 < len(drafts) else video_end
            next_phase_start = max(draft.anchor, next_phase_start)
            results.append(
                self._candidate(
                    index=index,
                    draft=draft,
                    signals=ordered,
                    next_phase_start=next_phase_start,
                    sample_step=sample_step,
                )
            )
        return results

    def _candidate(
        self,
        *,
        index: int,
        draft: _DraftRound,
        signals: list[FrameSignals],
        next_phase_start: float,
        sample_step: float,
    ) -> RoundCandidate:
        guard = max(0.1, sample_step)
        prep_span = draft.prep_span
        round_span = draft.round_span
        reasons: list[str] = []
        if prep_span is None:
            reasons.append("missing_prep")
        if round_span is None:
            reasons.append("missing_round_banner")

        if prep_span is not None and round_span is not None:
            layout_start = min(next_phase_start, max(0.0, prep_span.end - guard))
            layout_end = min(next_phase_start, round_span.start + guard)
        elif prep_span is not None:
            layout_start = min(next_phase_start, max(0.0, prep_span.end - guard))
            layout_end = min(next_phase_start, prep_span.end + self.fallback_search_seconds)
        else:
            assert round_span is not None
            layout_start = min(
                next_phase_start,
                max(0.0, round_span.start - self.fallback_search_seconds),
            )
            layout_end = min(next_phase_start, round_span.start + guard)

        if round_span is not None:
            battle_search_start = min(next_phase_start, max(layout_start, round_span.end - guard))
            battle_search_end = min(next_phase_start, round_span.end + self.fallback_search_seconds)
        else:
            assert prep_span is not None
            battle_search_start = min(next_phase_start, max(layout_start, prep_span.end - guard))
            battle_search_end = min(next_phase_start, prep_span.end + self.fallback_search_seconds)

        layout_end = max(layout_start, layout_end)
        battle_search_end = max(battle_search_start, battle_search_end)
        end_search_start = min(
            next_phase_start,
            max(battle_search_start, next_phase_start - self.fallback_search_seconds),
        )

        return RoundCandidate(
            round_index=index + 1,
            observed_round_number=self._observed_round_number(signals, round_span),
            prep_span=prep_span,
            round_span=round_span,
            layout_search=TimeSpan(layout_start, layout_end),
            battle_start_search=TimeSpan(battle_search_start, battle_search_end),
            end_search=TimeSpan(end_search_start, next_phase_start),
            next_phase_start=next_phase_start,
            failure_reasons=tuple(reasons),
        )

    def _is_battlefield(self, signal: FrameSignals) -> bool:
        return signal.battlefield_score >= self.thresholds.battlefield

    def _is_strong_prep(self, signal: FrameSignals) -> bool:
        return (
            self._is_battlefield(signal)
            and signal.bottom_panel_score >= self.thresholds.bottom_panel
            and signal.choice_buttons_score >= self.thresholds.choice_buttons
        )

    def _is_prep(self, signal: FrameSignals) -> bool:
        if not self._is_battlefield(signal):
            return False
        prep_ui = self._is_strong_prep(signal)
        countdown = (
            signal.countdown_score >= self.thresholds.countdown or signal.countdown_seconds is not None
        )
        return prep_ui or countdown

    def _is_confirmed_strong_prep(self, signal: FrameSignals) -> bool:
        return self._is_strong_prep(signal) and self._is_confirmed_prep(signal)

    def _is_confirmed_prep(self, signal: FrameSignals) -> bool:
        return (
            self._is_battlefield(signal)
            and signal.countdown_score >= self.thresholds.countdown
            and signal.countdown_seconds is not None
        )

    def _is_round(self, signal: FrameSignals) -> bool:
        return self._is_battlefield(signal) and (
            signal.round_banner_score >= self.thresholds.round_banner or signal.round_number is not None
        )

    def _is_confirmed_round(self, signal: FrameSignals) -> bool:
        return (
            self._is_battlefield(signal)
            and signal.round_banner_score >= self.thresholds.round_banner
            and signal.round_number is not None
        )

    @staticmethod
    def _spans_overlap(left: TimeSpan, right: TimeSpan) -> bool:
        return left.start <= right.end and right.start <= left.end

    @staticmethod
    def _event_spans(
        signals: list[FrameSignals],
        *,
        positive: Callable[[FrameSignals], bool],
        confirmed: Callable[[FrameSignals], bool],
        merge_gap: float,
    ) -> list[TimeSpan]:
        groups: list[list[FrameSignals]] = []
        for signal in signals:
            if not positive(signal):
                continue
            if not groups or signal.timestamp - groups[-1][-1].timestamp > merge_gap + 1e-9:
                groups.append([signal])
            else:
                groups[-1].append(signal)
        return [
            TimeSpan(group[0].timestamp, group[-1].timestamp)
            for group in groups
            if len(group) >= 2 or any(confirmed(item) for item in group)
        ]

    @staticmethod
    def _sample_step(signals: list[FrameSignals]) -> float:
        diffs = [
            right.timestamp - left.timestamp
            for left, right in zip(signals, signals[1:], strict=False)
            if right.timestamp > left.timestamp
        ]
        return float(median(diffs)) if diffs else 0.1

    @staticmethod
    def _observed_round_number(signals: list[FrameSignals], span: TimeSpan | None) -> int | None:
        if span is None:
            return None
        values = [
            signal.round_number
            for signal in signals
            if span.start <= signal.timestamp <= span.end and signal.round_number is not None
        ]
        if not values:
            return None
        counts = Counter(values)
        return min(counts, key=lambda value: (-counts[value], value))
