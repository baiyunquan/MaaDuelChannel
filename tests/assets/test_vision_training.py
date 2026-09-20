from types import SimpleNamespace

import numpy as np
import pytest

from maa_duel.training.vision import (
    classification_from_result,
    detections_from_result,
    train_vision_model,
)


def test_classification_result_maps_class_index_to_enemy_id():
    result = SimpleNamespace(probs=SimpleNamespace(top1=1, top1conf=SimpleNamespace(item=lambda: 0.8)))

    classification = classification_from_result(result, {0: 10, 1: 20})

    assert classification.enemy_id == 20
    assert classification.confidence == pytest.approx(0.8)


def test_detection_results_are_normalized_and_mapped():
    boxes = SimpleNamespace(
        cls=np.array([0.0]),
        conf=np.array([0.9]),
        xyxyn=np.array([[0.1, 0.2, 0.3, 0.4]]),
    )
    result = SimpleNamespace(boxes=boxes)

    detections = detections_from_result(result, {0: 7})

    assert len(detections) == 1
    assert detections[0].enemy_id == 7
    assert detections[0].bbox.x2 == pytest.approx(0.3)


def test_training_requires_explicit_all_samples_before_loading_ultralytics(tmp_path):
    with pytest.raises(ValueError, match="--all"):
        train_vision_model("roster", tmp_path, all_samples=False)
