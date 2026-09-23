import pytest

from maa_duel.vision.ocr import parse_count, parse_countdown, parse_round_number


@pytest.mark.parametrize(
    ("text", "expected"),
    [("×12", 12), ("x 4", 4), ("O8", 8), (" 3 ", 3), ("+4", 4), ("*5", 5), ("xi", 1), ("x1", 1)],
)
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


def test_rapidocr_cuda_mode_does_not_silently_fallback_to_cpu():
    from types import SimpleNamespace

    import numpy as np

    from maa_duel.vision.ocr import RapidOcrEngine

    engine = RapidOcrEngine.__new__(RapidOcrEngine)
    engine._use_cuda = True
    engine._engine = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cuda failed"))
    engine._RapidOCR = lambda: SimpleNamespace()

    with pytest.raises(RuntimeError, match="refusing a silent CPU fallback"):
        engine.recognize(np.zeros((10, 20, 3), dtype=np.uint8))


def test_cpu_ocr_inference_failure_does_not_replace_configured_session():
    import numpy as np

    from maa_duel.vision.ocr import RapidOcrEngine

    engine = RapidOcrEngine.__new__(RapidOcrEngine)
    engine._use_cuda = False

    def original(*args, **kwargs):
        raise RuntimeError("cpu failed")

    engine._engine = original
    engine._RapidOCR = lambda: (_ for _ in ()).throw(AssertionError("unexpected unconfigured OCR session"))

    with pytest.raises(RuntimeError, match="CPU inference failed"):
        engine.recognize(np.zeros((10, 20, 3), dtype=np.uint8))
    assert engine._engine is original


def test_cpu_ocr_uses_requested_onnx_threads_and_cpu_provider():
    from maa_duel.vision.ocr import RapidOcrEngine

    engine = RapidOcrEngine(use_cuda=False, cpu_threads=2)

    for component_name in ("text_det", "text_rec"):
        component = getattr(engine._engine, component_name)
        session = component.session.session
        assert session.get_providers() == ["CPUExecutionProvider"]
        assert session.get_session_options().intra_op_num_threads == 2
        assert session.get_session_options().inter_op_num_threads == 1


def test_cuda_ocr_rejects_cpu_recognition_session(monkeypatch):
    from types import SimpleNamespace

    import rapidocr

    from maa_duel.vision import ocr as ocr_module

    class Session:
        def __init__(self, provider: str) -> None:
            self.provider = provider

        def get_providers(self) -> list[str]:
            return [self.provider]

    class FakeRapidOCR:
        def __init__(self, params=None) -> None:
            self.text_det = SimpleNamespace(session=SimpleNamespace(session=Session("CUDAExecutionProvider")))
            self.text_rec = SimpleNamespace(session=SimpleNamespace(session=Session("CPUExecutionProvider")))
            self.text_cls = SimpleNamespace(session=SimpleNamespace(session=Session("CPUExecutionProvider")))

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(txts=None, scores=None, boxes=None)

    monkeypatch.setattr(ocr_module, "_prepare_cuda_runtime", lambda: None)
    monkeypatch.setattr(rapidocr, "RapidOCR", FakeRapidOCR)

    with pytest.raises(RuntimeError, match="recognition.*CUDAExecutionProvider"):
        ocr_module.RapidOcrEngine(use_cuda=True)


def test_cuda_ocr_arena_can_cover_large_video_recognition_batches(monkeypatch):
    from types import SimpleNamespace

    import rapidocr

    from maa_duel.vision import ocr as ocr_module

    captured_params = {}

    class Session:
        def get_providers(self) -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    class FakeRapidOCR:
        def __init__(self, params=None) -> None:
            captured_params.update(params)
            component = SimpleNamespace(session=SimpleNamespace(session=Session()))
            self.text_det = component
            self.text_rec = component
            self.text_cls = component

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(txts=None, scores=None, boxes=None)

    monkeypatch.setattr(ocr_module, "_prepare_cuda_runtime", lambda: None)
    monkeypatch.setattr(rapidocr, "RapidOCR", FakeRapidOCR)

    ocr_module.RapidOcrEngine(use_cuda=True)

    assert captured_params["EngineConfig.onnxruntime.cuda_ep_cfg.gpu_mem_limit"] >= 4 * 1024**3


def test_preprocess_count_crop() -> None:
    import cv2
    import numpy as np

    from maa_duel.vision.ocr import preprocess_count_crop

    # Empty / None
    assert preprocess_count_crop(np.zeros((0, 0, 3), dtype=np.uint8)) is None

    # Pure black image (no white contours)
    black = np.zeros((30, 30, 3), dtype=np.uint8)
    assert preprocess_count_crop(black) is None

    # Isolated 1-pixel noise (should be filtered out)
    noise = np.zeros((30, 30, 3), dtype=np.uint8)
    noise[10, 10] = (255, 255, 255)
    assert preprocess_count_crop(noise) is None

    # Valid white text shape (e.g. 5x12 vertical bar representing digit '1')
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 8), (15, 20), (220, 220, 220), -1)
    res = preprocess_count_crop(img, thresh=160)
    assert res is not None
    # Result should be padded with 4px border: h = (20-8+1) + 8 = 21, w = (15-10+1) + 8 = 14
    assert res.shape[0] == 21
    assert res.shape[1] == 14
    assert res.shape[2] == 3
