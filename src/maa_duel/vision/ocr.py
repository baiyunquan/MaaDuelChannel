from __future__ import annotations

import contextlib
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
        "-": ":",
    }
)


def _numeric_text(text: str, *, allow_separator: bool) -> str:
    candidate = text.strip().lstrip("xX×").strip()
    allowed = r"[0-9OoDQIl|\s:.：．-]+" if allow_separator else r"[0-9OoDQIl|\s]+"
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
    total = minutes * 60 + seconds
    return total if total <= 60 else None


_ROUND_DIGIT_REPLACEMENTS = str.maketrans(
    {
        "Z": "2",
        "z": "2",
        "S": "3",
        "s": "3",
        "I": "1",
        "l": "1",
        "|": "1",
        "O": "0",
        "o": "0",
        "D": "0",
        "Q": "0",
    }
)


def parse_round_number(text: str) -> int | None:
    match = re.search(r"ROUND\s*([0-9A-Za-z|]+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    rest = match.group(1).translate(_ROUND_DIGIT_REPLACEMENTS)
    num_match = re.search(r"0*(\d+)", rest)
    return int(num_match.group(1)) if num_match else None


class RapidOcrEngine:
    """Lazy RapidOCR adapter with optional CUDA acceleration."""

    def __init__(self, *, use_cuda: bool = True) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise RuntimeError("RapidOCR is not installed; install the ocr extra") from exc

        if use_cuda:
            import ctypes
            import os
            import sys
            from pathlib import Path

            site_packages = Path(sys.prefix) / "Lib" / "site-packages"
            for rel in ("nvidia/cudnn/bin", "nvidia/cublas/bin", "nvidia/cuda_nvrtc/bin", "torch/lib"):
                dll_path = site_packages / Path(rel)
                if dll_path.is_dir():
                    with contextlib.suppress(AttributeError, OSError):
                        os.add_dll_directory(str(dll_path))
                    os.environ["PATH"] = str(dll_path) + os.pathsep + os.environ.get("PATH", "")
            for rel_dll in (
                "nvidia/cublas/bin/cublasLt64_12.dll",
                "nvidia/cublas/bin/cublas64_12.dll",
                "nvidia/cudnn/bin/cudnn64_9.dll",
            ):
                target = site_packages / rel_dll
                if target.is_file():
                    with contextlib.suppress(OSError):
                        ctypes.CDLL(str(target))

        params = None
        if use_cuda:
            params = {
                "EngineConfig.onnxruntime.use_cuda": True,
                "Det.engine_cfg.use_cuda": True,
                "Cls.engine_cfg.use_cuda": False,
                "Rec.engine_cfg.use_cuda": True,
            }
        self._RapidOCR = RapidOCR
        try:
            self._engine = RapidOCR(params=params)
            if use_cuda:
                self._engine(np.zeros((32, 32, 3), dtype=np.uint8), use_det=False, use_cls=False, use_rec=True)
        except Exception:
            self._engine = RapidOCR()

    def recognize(self, image: np.ndarray, *, detect: bool = True) -> list[OcrText]:
        try:
            result = self._engine(image, use_det=detect, use_cls=False, use_rec=True)
        except Exception:
            self._engine = self._RapidOCR()
            result = self._engine(image, use_det=detect, use_cls=False, use_rec=True)
        texts = [] if result.txts is None else list(result.txts)
        scores = [] if result.scores is None else list(result.scores)
        boxes = ([] if result.boxes is None else list(result.boxes)) if detect else [None] * len(texts)
        return [
            OcrText(
                text=str(text),
                confidence=float(score),
                box=tuple(tuple(float(value) for value in point) for point in box) if box is not None else None,
            )
            for text, score, box in zip(texts, scores, boxes, strict=True)
        ]
