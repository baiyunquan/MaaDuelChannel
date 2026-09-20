import pytest

from maa_duel.video.phases import FrameSignals, RoundSegmenter


def signal(time, countdown=None, round_number=None, layout_score=0.0):
    return FrameSignals(
        timestamp=time,
        game_visible=True,
        countdown_seconds=countdown,
        round_number=round_number,
        layout_score=layout_score,
    )


def test_segmenter_extracts_multiple_rounds_and_best_layout_frame():
    signals = [
        signal(0.0, countdown=3),
        signal(0.5, countdown=1),
        signal(1.0, countdown=0),
        signal(1.1, layout_score=0.7),
        signal(1.2, layout_score=0.95),
        signal(1.4, round_number=1),
        signal(4.0),
        signal(5.0, countdown=3),
        signal(5.5, countdown=0),
        signal(5.6, layout_score=0.9),
        signal(5.8, round_number=2),
        signal(8.0),
    ]

    rounds = RoundSegmenter().segment(signals)

    assert len(rounds) == 2
    assert rounds[0].round_index == 1
    assert rounds[0].prep_time == pytest.approx(0.5)
    assert rounds[0].layout_time == pytest.approx(1.2)
    assert rounds[0].battle_start == pytest.approx(1.4)
    assert rounds[0].battle_end == pytest.approx(5.0)
    assert rounds[0].complete
    assert rounds[1].battle_end == pytest.approx(8.0)
    assert rounds[1].complete


def test_segmenter_marks_edited_round_without_layout_as_incomplete():
    signals = [
        signal(10.0, countdown=1),
        signal(10.5, round_number=7),
        signal(12.0),
    ]

    result = RoundSegmenter().segment(signals)[0]

    assert not result.complete
    assert "missing_layout" in result.failure_reasons


def test_segmenter_ignores_non_game_countdowns():
    signals = [
        FrameSignals(timestamp=0.0, game_visible=False, countdown_seconds=3),
        FrameSignals(timestamp=1.0, game_visible=False, countdown_seconds=0),
    ]

    assert RoundSegmenter().segment(signals) == []


def test_segmenter_infers_zero_window_when_one_disappears_before_round_banner():
    signals = [
        signal(0.0, countdown=2),
        signal(0.5, countdown=1),
        signal(1.0, layout_score=0.8),
        signal(1.5, round_number=1),
        signal(3.0),
    ]

    result = RoundSegmenter().segment(signals)[0]

    assert result.complete
    assert result.layout_time == pytest.approx(1.0)


def test_segmenter_restarts_after_incomplete_round_when_countdown_rises():
    signals = [
        signal(0.0, countdown=3),
        signal(0.5, countdown=1),
        signal(1.1, layout_score=0.9),
        signal(6.0, countdown=3),
        signal(6.5, countdown=0),
        signal(6.6, layout_score=0.8),
        signal(7.2, round_number=2),
        signal(8.0),
    ]

    rounds = RoundSegmenter().segment(signals)

    assert len(rounds) == 2
    assert not rounds[0].complete
    assert "missing_round_banner" in rounds[0].failure_reasons
    assert rounds[1].complete
    assert rounds[1].prep_time == pytest.approx(6.0)
    assert rounds[1].layout_time == pytest.approx(6.6)
    assert rounds[1].battle_start == pytest.approx(7.2)
