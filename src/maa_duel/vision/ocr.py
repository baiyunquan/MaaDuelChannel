from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class OcrText:
    text: str
    confidence: float
    box: tuple[tuple[float, float], ...] | None = None


class OcrEngine(Protocol):
    def recognize(self, image: np.ndarray, *, detect: bool = True) -> list[OcrText]: ...


def _prepare_cuda_runtime() -> None:
    """Load CUDA/cuDNN dependencies before ONNX Runtime creates a session.

    The CUDA provider loads cuDNN lazily by the unversioned ``libcudnn.so``
    name.  Loading every shared object alphabetically (the old implementation)
    tries that symlink before its dependent cuDNN components and silently
    leaves the provider without cuDNN.  Explicit dependency order makes the
    provider usable in worker threads as well as in the main process.
    """

    import ctypes
    import os
    import sys
    from pathlib import Path

    site_packages_win = Path(sys.prefix) / "Lib" / "site-packages"
    site_packages_linux = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_packages = site_packages_linux if site_packages_linux.is_dir() else site_packages_win

    library_dirs = [
        site_packages / "nvidia" / "cuda_runtime" / "lib",
        site_packages / "nvidia" / "cuda_nvrtc" / "lib",
        site_packages / "nvidia" / "cublas" / "lib",
        site_packages / "nvidia" / "cudnn" / "lib",
        site_packages / "nvidia" / "curand" / "lib",
        site_packages / "nvidia" / "cufft" / "lib",
        site_packages / "nvidia" / "cusolver" / "lib",
        site_packages / "nvidia" / "cusparse" / "lib",
        site_packages / "nvidia" / "nvjitlink" / "lib",
        site_packages / "torch" / "lib",
    ]
    existing_dirs = [path for path in library_dirs if path.is_dir()]
    if os.name != "nt":
        current = os.environ.get("LD_LIBRARY_PATH", "")
        paths = [str(path) for path in existing_dirs]
        if current:
            paths.append(current)
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(paths)

    # Load the exact files in dependency order.  ``RTLD_GLOBAL`` is required
    # because ONNX Runtime resolves cuDNN symbols from the global process scope.
    ordered_library_names = [
        ("cuda_runtime", ("libcudart.so.12", "libcudart.so")),
        ("cuda_nvrtc", ("libnvrtc.so.12", "libnvrtc.so")),
        ("cublas", ("libcublasLt.so.12", "libcublasLt.so", "libcublas.so.12", "libcublas.so")),
        ("curand", ("libcurand.so.10", "libcurand.so")),
        ("cufft", ("libcufft.so.11", "libcufft.so")),
        ("cusolver", ("libcusolver.so.11", "libcusolver.so")),
        ("cusparse", ("libcusparse.so.12", "libcusparse.so")),
        ("nvjitlink", ("libnvJitLink.so.12", "libnvJitLink.so")),
        ("cudnn", ("libcudnn_ops.so", "libcudnn_cnn.so", "libcudnn_graph.so", "libcudnn_heuristic.so", "libcudnn.so")),
    ]
    load_errors: list[str] = []
    for directory_name, names in ordered_library_names:
        directory = site_packages / "nvidia" / directory_name / "lib"
        if not directory.is_dir():
            continue
        for name in names:
            candidate = directory / name
            if not candidate.is_file() and not candidate.is_symlink():
                continue
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                load_errors.append(f"{candidate}: {exc}")

    cudnn_dir = site_packages / "nvidia" / "cudnn" / "lib"
    cudnn = cudnn_dir / "libcudnn.so"
    if cudnn_dir.is_dir() and cudnn.exists() and load_errors:
        raise RuntimeError("CUDA cuDNN libraries could not be loaded: " + "; ".join(load_errors))


def preprocess_count_crop(image: np.ndarray, thresh: int = 160) -> np.ndarray | None:
    """
    Preprocesses a cropped count region using OpenCV color thresholding and contour bounding.
    Extracts pure white digits, eliminates background clutter, crops to minimum bounding rect,
    and pads with a black border to optimize character recognition.
    Returns None if no text contours are detected.
    """
    if image is None or image.size == 0:
        return None
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    mask = cv2.inRange(
        image,
        np.array([thresh, thresh, thresh], dtype=np.uint8),
        np.array([255, 255, 255], dtype=np.uint8),
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        _, _, bw, bh = cv2.boundingRect(c)
        if bw <= 1 or bh <= 3:
            cv2.drawContours(mask, [c], -1, 0, -1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    all_c = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_c)
    if w < 3 or h < 6:
        return None

    cropped = mask[y : y + h, x : x + w]
    padded = cv2.copyMakeBorder(cropped, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


_DIGIT_REPLACEMENTS = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "D": "0",
        "Q": "0",
        "I": "1",
        "l": "1",
        "i": "1",
        "|": "1",
        "：": ":",
        "．": ".",
        "-": ":",
    }
)


def _numeric_text(text: str, *, allow_separator: bool) -> str:
    candidate = text.strip().lstrip("xX×+*").strip()
    allowed = r"[0-9OoDQIli|\s:.：．-]+" if allow_separator else r"[0-9OoDQIli|\s]+"
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
        self._use_cuda = use_cuda
        if use_cuda:
            _prepare_cuda_runtime()
        try:
            import logging

            from rapidocr import RapidOCR
            from rapidocr.utils.log import logger as rapidocr_logger

            rapidocr_logger.setLevel(logging.ERROR)
            for h in rapidocr_logger.handlers:
                h.setLevel(logging.ERROR)
        except ImportError as exc:
            raise RuntimeError("RapidOCR is not installed; install the ocr extra") from exc

        if use_cuda:
            import onnxruntime as ort

            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError(
                    "RapidOCR CUDA requested but CUDAExecutionProvider is unavailable; "
                    f"providers={ort.get_available_providers()}"
                )

        params = None
        if use_cuda:
            params = {
                "EngineConfig.onnxruntime.use_cuda": True,
                "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                "Det.engine_cfg.use_cuda": True,
                "Cls.engine_cfg.use_cuda": False,
                "Rec.engine_cfg.use_cuda": True,
            }
        self._RapidOCR = RapidOCR
        try:
            self._engine = RapidOCR(params=params)
            if use_cuda:
                self._engine(np.zeros((32, 32, 3), dtype=np.uint8), use_det=False, use_cls=False, use_rec=True)
                self._providers = self._session_providers()
                if not self._providers or self._providers[0] != "CUDAExecutionProvider":
                    raise RuntimeError(
                        "RapidOCR CUDA requested but the OCR session did not select CUDAExecutionProvider; "
                        f"providers={self._providers}"
                    )
        except Exception as exc:
            if use_cuda:
                raise RuntimeError("RapidOCR CUDA initialization/inference failed; CPU fallback is disabled") from exc
            self._engine = RapidOCR()
        self._providers = self._session_providers()

    def _session_providers(self) -> tuple[str, ...]:
        providers: list[str] = []
        for component_name in ("text_det", "text_rec", "text_cls"):
            component = getattr(self._engine, component_name, None)
            session_wrapper = getattr(component, "session", None)
            session = getattr(session_wrapper, "session", session_wrapper)
            get_providers = getattr(session, "get_providers", None)
            if callable(get_providers):
                providers.extend(str(item) for item in get_providers())
        return tuple(dict.fromkeys(providers))

    def recognize(self, image: np.ndarray, *, detect: bool = True) -> list[OcrText]:
        try:
            result = self._engine(image, use_det=detect, use_cls=False, use_rec=True)
        except Exception as exc:
            if getattr(self, "_use_cuda", False):
                raise RuntimeError("RapidOCR CUDA inference failed; refusing a silent CPU fallback") from exc
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
