import tomllib
from pathlib import Path


def test_ocr_extra_declares_rapidocr_runtime_backend():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert any(
        requirement.startswith("onnxruntime") for requirement in pyproject["project"]["optional-dependencies"]["ocr"]
    )
