from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class OcrText:
    text: str
    confidence: float
    box: tuple[tuple[float, float], ...] | None = None


class OcrEngine(Protocol):
    def recognize(self, image: np.ndarray, *, detect: bool = True) -> list[OcrText]: ...


_DIGIT_REPLACEMENTS = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "D": "0",
        "Q": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "：": ":",
        "．": ".",
    }
)


def _numeric_text(text: str, *, allow_separator: bool) -> str:
    candidate = text.strip().lstrip("xX×").strip()
    allowed = r"[0-9OoDQIl|\s:.：．]+" if allow_separator else r"[0-9OoDQIl|\s]+"
    if not re.fullmatch(allowed, candidate):
        digits = re.findall(r"\d+", candidate)
        if digits:
            return digits[-1]
        raise ValueError(f"count contains no digits: {text!r}")
    return candidate.translate(_DIGIT_REPLACEMENTS)


def parse_count(text: str) -> int:
    normalized = _numeric_text(text, allow_separator=False)
    matches = re.findall(r"\d+", normalized)
    if not matches:
        raise ValueError(f"count contains no digits: {text!r}")
    return int(matches[-1])


def parse_countdown(text: str) -> int | None:
    try:
        normalized = _numeric_text(text, allow_separator=True)
    except ValueError:
        return None
    match = re.search(r"(\d{1,2})\s*[:.]\s*(\d{1,2})", normalized)
    if match is None:
        return None
    minutes, seconds = (int(value) for value in match.groups())
    if seconds >= 60:
        return None
    return minutes * 60 + seconds


def parse_round_number(text: str) -> int | None:
    match = re.search(r"ROUND\s*0*(\d+)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


class RapidOcrEngine:
    """Lazy RapidOCR adapter so non-OCR commands do not require the optional dependency."""

    def __init__(self) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise RuntimeError("RapidOCR is not installed; install the ocr extra") from exc
        self._engine = RapidOCR()

    def recognize(self, image: np.ndarray, *, detect: bool = True) -> list[OcrText]:
        result = self._engine(image, use_det=detect, use_cls=detect, use_rec=True)
        texts = list(result.txts or [])
        scores = list(result.scores or [])
        boxes = list(result.boxes or []) if detect else [None] * len(texts)
        return [
            OcrText(
                text=str(text),
                confidence=float(score),
                box=tuple(tuple(float(value) for value in point) for point in box) if box is not None else None,
            )
            for text, score, box in zip(texts, scores, boxes, strict=True)
        ]
