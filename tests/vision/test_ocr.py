import pytest

from maa_duel.vision.ocr import parse_count, parse_countdown, parse_round_number


@pytest.mark.parametrize(("text", "expected"), [("×12", 12), ("x 4", 4), ("O8", 8), (" 3 ", 3)])
def test_parse_count_normalizes_common_ocr_symbols(text, expected):
    assert parse_count(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("00:04", 4), ("OO：O1", 1), ("00.00", 0), ("00-011", 1)],
)
def test_parse_countdown(text, expected):
    assert parse_countdown(text) == expected


def test_parse_round_number():
    assert parse_round_number("ROUND 06 比赛开始") == 6
    assert parse_round_number("unrelated") is None


def test_parse_count_rejects_missing_digits():
    with pytest.raises(ValueError, match="count"):
        parse_count("unknown")


def test_rapidocr_adapter_accepts_numpy_box_arrays():
    from types import SimpleNamespace

    import numpy as np

    from maa_duel.vision.ocr import RapidOcrEngine

    engine = RapidOcrEngine.__new__(RapidOcrEngine)
    engine._engine = lambda *args, **kwargs: SimpleNamespace(
        txts=["00:04"],
        scores=np.array([0.9]),
        boxes=np.array([[[0, 0], [10, 0], [10, 4], [0, 4]]], dtype=np.float32),
    )

    result = engine.recognize(np.zeros((10, 20, 3), dtype=np.uint8))

    assert result[0].text == "00:04"
    assert result[0].box[2] == (10.0, 4.0)
