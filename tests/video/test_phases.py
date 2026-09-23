import pytest

from maa_duel.video.phases import FrameSignals, RoundSegmenter


def signal(
    timestamp: float,
    *,
    battlefield: float = 1.0,
    bottom_panel: float = 0.0,
    choice_buttons: float = 0.0,
    countdown: float = 0.0,
    round_banner: float = 0.0,
    corner_mask: float = 0.0,
    countdown_seconds: int | None = None,
    round_number: int | None = None,
) -> FrameSignals:
    return FrameSignals(
        timestamp=timestamp,
        battlefield_score=battlefield,
        bottom_panel_score=bottom_panel,
        choice_buttons_score=choice_buttons,
        countdown_score=countdown,
        round_banner_score=round_banner,
        corner_mask_score=corner_mask,
        countdown_seconds=countdown_seconds,
        round_number=round_number,
    )


def prep(timestamp: float, seconds: int) -> FrameSignals:
    return signal(
        timestamp,
        bottom_panel=0.9,
        choice_buttons=0.9,
        countdown=0.9,
        corner_mask=0.9,
        countdown_seconds=seconds,
    )


def banner(timestamp: float, number: int | None = None) -> FrameSignals:
    return signal(timestamp, round_banner=0.95, corner_mask=0.9, round_number=number)


def test_countdown_value_jitter_never_splits_a_round() -> None:
    signals = [
        prep(0.0, 3),
        prep(0.1, 4),
        prep(0.2, 3),
        signal(0.3, corner_mask=0.9),
        signal(0.4, corner_mask=0.9),
        banner(0.5, 1),
        banner(0.6, 1),
        signal(0.7),
        signal(4.8),
        prep(5.0, 3),
        prep(5.1, 2),
        signal(5.2, corner_mask=0.9),
        signal(5.3, corner_mask=0.9),
        banner(5.4, 2),
        banner(5.5, 2),
        signal(5.6),
        signal(7.9),
    ]

    rounds = RoundSegmenter().segment(signals, video_end=8.0)

    assert len(rounds) == 2
    assert rounds[0].round_index == 1
    assert rounds[0].observed_round_number == 1
    assert rounds[0].prep_span is not None
    assert rounds[0].round_span is not None
    assert rounds[0].next_phase_start == pytest.approx(5.0)
    assert rounds[1].round_index == 2
    assert rounds[1].next_phase_start == pytest.approx(8.0)


def test_visual_round_banner_without_ocr_still_creates_one_round() -> None:
    signals = [
        prep(1.0, 2),
        prep(1.1, 1),
        signal(1.2, corner_mask=0.9),
        signal(1.3, corner_mask=0.9),
        banner(1.4),
        banner(1.5),
        banner(1.6),
        signal(1.7),
        signal(4.0),
    ]

    rounds = RoundSegmenter().segment(signals, video_end=4.2)

    assert len(rounds) == 1
    assert rounds[0].observed_round_number is None
    assert rounds[0].round_span is not None
    assert rounds[0].round_span.start == pytest.approx(1.4)
    assert rounds[0].round_span.end == pytest.approx(1.6)


def test_round_without_prep_is_retained_as_incomplete_candidate() -> None:
    signals = [banner(2.0, 7), banner(2.1, 7), signal(2.2), signal(6.0)]

    rounds = RoundSegmenter().segment(signals, video_end=6.2)

    assert len(rounds) == 1
    assert rounds[0].prep_span is None
    assert rounds[0].round_span is not None
    assert "missing_prep" in rounds[0].failure_reasons


def test_single_frame_round_noise_does_not_create_duplicate_candidate() -> None:
    signals = [
        prep(0.0, 2),
        prep(0.1, 1),
        banner(0.4, 1),
        banner(0.5, 1),
        signal(2.0, round_banner=0.95),
        signal(4.0),
    ]

    rounds = RoundSegmenter().segment(signals, video_end=4.2)

    assert len(rounds) == 1


def test_last_round_uses_video_end_instead_of_last_detected_signal() -> None:
    signals = [prep(0.0, 1), prep(0.1, 1), banner(0.4, 1), banner(0.5, 1), signal(1.0)]

    result = RoundSegmenter().segment(signals, video_end=19.1)[0]

    assert result.next_phase_start == pytest.approx(19.1)
    assert result.end_search.end == pytest.approx(19.1)


def test_sustained_countdown_without_choice_ui_still_starts_a_prep_candidate() -> None:
    signals = [
        signal(1.0, countdown=0.9),
        signal(1.1, countdown=0.9),
        signal(1.2, corner_mask=0.9),
        banner(1.3, 2),
        banner(1.4, 2),
    ]

    rounds = RoundSegmenter().segment(signals, video_end=2.0)

    assert len(rounds) == 1
    assert rounds[0].prep_span is not None
    assert rounds[0].prep_span.start == pytest.approx(1.0)
    assert rounds[0].prep_span.end == pytest.approx(1.1)
    assert rounds[0].round_span is not None


def test_single_frame_countdown_noise_without_choice_ui_is_ignored() -> None:
    signals = [signal(1.0, countdown=0.9), signal(1.1)]

    assert RoundSegmenter().segment(signals, video_end=2.0) == []


def test_countdown_like_battle_effects_before_real_prep_do_not_create_extra_candidates() -> None:
    signals = [
        prep(0.0, 3),
        prep(0.1, 2),
        banner(0.4, 1),
        banner(0.5, 1),
        signal(2.0, countdown=0.9),
        signal(2.1, countdown=0.9),
        signal(2.5, countdown=0.9),
        signal(2.6, countdown=0.9),
        prep(4.0, 3),
        prep(4.1, 2),
        banner(4.4, 2),
        banner(4.5, 2),
    ]

    rounds = RoundSegmenter().segment(signals, video_end=6.0)

    assert len(rounds) == 2
    assert rounds[1].prep_span is not None
    assert rounds[1].prep_span.start == pytest.approx(4.0)


def test_stale_countdown_like_battle_effect_is_not_attached_to_later_round() -> None:
    signals = [
        prep(0.0, 3),
        prep(0.1, 2),
        banner(0.4, 1),
        banner(0.5, 1),
        signal(1.0, countdown=0.9),
        signal(1.1, countdown=0.9),
        banner(4.4, 2),
        banner(4.5, 2),
    ]

    rounds = RoundSegmenter().segment(signals, video_end=6.0)

    assert len(rounds) == 2
    assert rounds[0].next_phase_start == pytest.approx(4.4)
    assert rounds[1].prep_span is None
    assert "missing_prep" in rounds[1].failure_reasons


def test_prep_noise_overlapping_previous_round_banner_never_reverses_boundaries() -> None:
    signals = [
        prep(0.0, 3),
        prep(0.1, 2),
        banner(0.4, 1),
        banner(0.5, 1),
        signal(0.5, bottom_panel=0.9, choice_buttons=0.9),
        signal(0.6, bottom_panel=0.9, choice_buttons=0.9),
        banner(2.0, 2),
        banner(2.1, 2),
    ]

    rounds = RoundSegmenter().segment(signals, video_end=4.0)

    assert len(rounds) == 2
    assert all(candidate.end_search.start <= candidate.end_search.end for candidate in rounds)
    assert rounds[0].next_phase_start <= rounds[1].next_phase_start


def test_short_unpaired_prep_noise_is_ignored_but_sustained_direct_battle_is_retained() -> None:
    noise = [signal(1.0, bottom_panel=0.9, choice_buttons=0.9), signal(1.1, bottom_panel=0.9, choice_buttons=0.9)]
    direct = [
        signal(timestamp, bottom_panel=0.9, choice_buttons=0.9)
        for timestamp in (3.0, 3.1, 3.2, 3.3, 3.4, 3.5)
    ]

    rounds = RoundSegmenter().segment([*noise, *direct, signal(4.0)], video_end=5.0)

    assert len(rounds) == 1
    assert rounds[0].prep_span is not None
    assert rounds[0].prep_span.start == pytest.approx(3.0)
    assert rounds[0].round_span is None
