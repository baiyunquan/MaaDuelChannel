from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

CALIBRATION_SCHEMA_VERSION = 1
MAX_CHECK_ERROR_TILES = 0.15


class PositionQuality(StrEnum):
    CONFIRMED = "confirmed"
    ESTIMATED = "estimated"
    UNCERTAIN = "uncertain"


class CalibrationPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_x: float
    image_y: float
    ground_x: float
    ground_y: float


class SpriteGroundOffset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enemy_id: int = Field(ge=1)
    image_dx: float = 0.0
    image_dy: float = 0.0
    source: str


class BattlefieldCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = CALIBRATION_SCHEMA_VERSION
    calibration_id: str = Field(min_length=1)
    arena: Literal["green_vine"] = "green_vine"
    source_kind: Literal["video", "prts_preview"] = "video"
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    map_width: float = Field(gt=0.0)
    map_height: float = Field(gt=0.0)
    fit_points: list[CalibrationPoint]
    check_points: list[CalibrationPoint]
    sprite_offsets: list[SpriteGroundOffset] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def validate_points(self) -> BattlefieldCalibration:
        if len(self.fit_points) < 8:
            raise ValueError("battlefield calibration requires at least 8 fit points")
        if len(self.check_points) < 2:
            raise ValueError("battlefield calibration requires at least 2 independent check points")
        return self


@dataclass(frozen=True)
class FittedCalibration:
    calibration: BattlefieldCalibration
    matrix: np.ndarray
    mean_check_error_tiles: float
    max_check_error_tiles: float


def transform_point(matrix: np.ndarray, image_x: float, image_y: float) -> tuple[float, float]:
    source = np.asarray([[[image_x, image_y]]], dtype=np.float64)
    transformed = cv2.perspectiveTransform(source, matrix.astype(np.float64))[0, 0]
    return float(transformed[0]), float(transformed[1])


def fit_homography(
    calibration: BattlefieldCalibration,
    *,
    maximum_check_error_tiles: float = MAX_CHECK_ERROR_TILES,
) -> FittedCalibration:
    source = np.asarray([(point.image_x, point.image_y) for point in calibration.fit_points], dtype=np.float64)
    target = np.asarray([(point.ground_x, point.ground_y) for point in calibration.fit_points], dtype=np.float64)
    matrix, _ = cv2.findHomography(source, target, method=0)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError(f"calibration {calibration.calibration_id} could not fit a homography")
    errors = []
    for point in calibration.check_points:
        ground_x, ground_y = transform_point(matrix, point.image_x, point.image_y)
        errors.append(float(np.hypot(ground_x - point.ground_x, ground_y - point.ground_y)))
    maximum = max(errors)
    if maximum > maximum_check_error_tiles:
        raise ValueError(
            f"calibration {calibration.calibration_id} check error {maximum:.4f} tiles exceeds "
            f"{maximum_check_error_tiles:.2f} tiles"
        )
    return FittedCalibration(
        calibration=calibration,
        matrix=matrix.astype(np.float32),
        mean_check_error_tiles=float(np.mean(errors)),
        max_check_error_tiles=maximum,
    )


def calibration_sha256(calibration: BattlefieldCalibration) -> str:
    payload = calibration.model_dump_json(indent=2) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_calibration(path: Path, calibration: BattlefieldCalibration) -> str:
    fit_homography(calibration)
    payload = calibration.model_dump_json(indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return calibration_sha256(calibration)


def load_calibration(path: Path) -> BattlefieldCalibration:
    calibration = BattlefieldCalibration.model_validate_json(path.read_text(encoding="utf-8"))
    fit_homography(calibration)
    return calibration
