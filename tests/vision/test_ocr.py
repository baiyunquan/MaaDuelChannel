import pytest

from maa_duel.vision.ocr import parse_count, parse_countdown, parse_round_number


@pytest.mark.parametrize(("text", "expected"), [("×12", 12), ("x 4", 4), ("O8", 8), (" 3 ", 3)])
def test_parse_count_normalizes_common_ocr_symbols(text, expected):
    assert parse_count(text) == expected


@pytest.mark.parametrize(("text", "expected"), [("00:04", 4), ("OO：O1", 1), ("00.00", 0)])
def test_parse_countdown(text, expected):
    assert parse_countdown(text) == expected


def test_parse_round_number():
    assert parse_round_number("ROUND 06 比赛开始") == 6
    assert parse_round_number("unrelated") is None


def test_parse_count_rejects_missing_digits():
    with pytest.raises(ValueError, match="count"):
        parse_count("unknown")
