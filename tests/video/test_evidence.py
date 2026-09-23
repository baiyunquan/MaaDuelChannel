import numpy as np

from maa_duel.video.evidence import (
    TimedFrame,
    select_battle_start,
    select_end_run,
    select_layout_frame,
    select_prep_frame,
)
from maa_duel.video.phases import FrameSignals


def timed(
    timestamp: float,
    *,
    battlefield: float = 0.9,
    panel: float = 0.0,
    choices: float = 0.0,
    countdown: float = 0.0,
    banner: float = 0.0,
    masks: float = 0.0,
    corner_signature: tuple[float, ...] = (),
) -> TimedFrame:
    value = int(timestamp * 10) % 255
    return TimedFrame(
        timestamp=timestamp,
        frame_index=round(timestamp * 30),
        frame=np.full((12, 16, 3), value, dtype=np.uint8),
        signals=FrameSignals(
            timestamp=timestamp,
            battlefield_score=battlefield,
            bottom_panel_score=panel,
            choice_buttons_score=choices,
            countdown_score=countdown,
            round_banner_score=banner,
            corner_mask_score=masks,
            corner_signature=corner_signature,
        ),
    )


def test_prep_selector_uses_middle_of_longest_paired_ui_run() -> None:
    frames = [
        timed(0.0, panel=0.9, choices=0.9),
        timed(0.1, panel=0.9, choices=0.9),
        timed(0.2),
        timed(0.3, panel=0.9, choices=0.9),
        timed(0.4, panel=0.9, choices=0.9),
        timed(0.5, panel=0.9, choices=0.9),
    ]

    selected = select_prep_frame(frames)

    assert selected is not None
    assert selected.timestamp == 0.4


def test_layout_selector_uses_native_frame_gap_midpoint() -> None:
    frames = [
        timed(1.0, panel=0.9, choices=0.9, countdown=0.9, masks=0.9),
        timed(1.1, countdown=0.9, masks=0.9),
        timed(1.2, masks=0.9),
        timed(1.3, masks=0.9),
        timed(1.4, masks=0.9),
        timed(1.5, banner=0.9, masks=0.9),
    ]

    selected = select_layout_frame(frames)

    assert selected is not None
    assert selected.timestamp == 1.3


def test_layout_selector_never_falls_back_to_countdown_frame() -> None:
    frames = [timed(1.0, countdown=0.9, masks=0.9), timed(1.1, countdown=0.9, masks=0.9)]

    assert select_layout_frame(frames) is None


def test_layout_selector_does_not_depend_on_absolute_corner_brightness() -> None:
    frames = [timed(1.0), timed(1.1)]

    assert select_layout_frame(frames) is not None


def test_layout_selector_stays_before_coordinated_corner_change_when_round_is_missing() -> None:
    masked = tuple([0.2] * 256)
    unmasked = tuple([0.4] * 256)
    prep = timed(0.9, panel=0.9, choices=0.9, corner_signature=masked)
    frames = [
        timed(1.0, corner_signature=masked),
        timed(1.1, corner_signature=masked),
        timed(1.2, corner_signature=unmasked),
        timed(1.3, corner_signature=unmasked),
        timed(1.4, corner_signature=unmasked),
    ]

    selected = select_layout_frame(frames, reference=prep)

    assert selected is not None
    assert selected.timestamp == 1.1


def test_layout_selector_can_require_a_later_corner_transition_without_prep_reference() -> None:
    masked = tuple([0.2] * 256)
    unmasked = tuple([0.4] * 256)
    frames = [
        timed(1.0, corner_signature=masked),
        timed(1.1, corner_signature=masked),
        timed(1.2, corner_signature=unmasked),
        timed(1.3, corner_signature=unmasked),
    ]

    selected = select_layout_frame(frames, require_corner_transition=True)

    assert selected is not None
    assert selected.timestamp == 1.1


def test_layout_selector_rejects_missing_round_without_a_corner_transition() -> None:
    masked = tuple([0.2] * 256)
    frames = [timed(1.0, corner_signature=masked), timed(1.1, corner_signature=masked)]

    assert select_layout_frame(frames, require_corner_transition=True) is None


def test_battle_start_is_after_banner_and_corner_masks_disappear() -> None:
    frames = [
        timed(2.0, banner=0.9, masks=0.9),
        timed(2.1, masks=0.9),
        timed(2.2),
        timed(2.3),
    ]

    selected = select_battle_start(frames)

    assert selected is not None
    assert selected.timestamp == 2.2


def test_battle_start_uses_coordinated_corner_change_from_layout_reference() -> None:
    unchanged = tuple([0.2] * 256)
    changed = tuple([0.4] * 256)
    layout = timed(1.0, corner_signature=unchanged)
    frames = [
        timed(2.0, corner_signature=unchanged),
        timed(2.1, corner_signature=unchanged),
        timed(2.2, corner_signature=changed),
        timed(2.3, corner_signature=changed),
    ]

    selected = select_battle_start(frames, reference=layout)

    assert selected is not None
    assert selected.timestamp == 2.2


def test_battle_start_allows_one_static_map_corner_during_coordinated_reveal() -> None:
    layout_signature = tuple([0.2] * 256)
    revealed_signature = tuple(
        [0.21] * 64 + [0.34] * 64 + [0.26] * 64 + [0.35] * 64
    )
    layout = timed(1.0, corner_signature=layout_signature)
    frames = [
        timed(2.0, corner_signature=layout_signature),
        timed(2.1, corner_signature=revealed_signature),
        timed(2.2, corner_signature=revealed_signature),
    ]

    selected = select_battle_start(frames, reference=layout)

    assert selected is not None
    assert selected.timestamp == 2.1


def test_end_selector_uses_last_legal_run_and_allows_bottom_wait_prompt() -> None:
    frames = [
        timed(8.0),
        timed(8.1, panel=0.9),
        timed(8.2, panel=0.9),
        timed(8.3, countdown=0.9, panel=0.9, choices=0.9),
        timed(8.4, banner=0.9),
    ]

    selected = select_end_run(frames)

    assert [frame.timestamp for frame in selected] == [8.0, 8.1, 8.2]


def test_end_selector_returns_empty_when_every_frame_is_next_round() -> None:
    frames = [timed(9.0, countdown=0.9), timed(9.1, banner=0.9)]

    assert select_end_run(frames) == ()


def test_end_selector_rejects_visual_countdown_even_without_choice_ui() -> None:
    frames = [timed(9.0, countdown=0.9), timed(9.1, countdown=0.9)]

    assert select_end_run(frames) == ()
