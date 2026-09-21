import pytest

from maa_duel.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    BattlefieldCalibration,
    CalibrationPoint,
    calibration_sha256,
    fit_homography,
    transform_point,
)


def point(image_x, image_y, ground_x, ground_y):
    return CalibrationPoint(image_x=image_x, image_y=image_y, ground_x=ground_x, ground_y=ground_y)


def exact_calibration():
    return BattlefieldCalibration(
        calibration_id="green-vine-video-v1",
        arena="green_vine",
        image_width=1500,
        image_height=1100,
        map_width=15,
        map_height=11,
        fit_points=[
            point(0, 0, 0, 0),
            point(1500, 0, 15, 0),
            point(0, 1100, 0, 11),
            point(1500, 1100, 15, 11),
            point(750, 0, 7.5, 0),
            point(750, 1100, 7.5, 11),
            point(0, 550, 0, 5.5),
            point(1500, 550, 15, 5.5),
        ],
        check_points=[
            point(300, 400, 3, 4),
            point(1200, 900, 12, 9),
        ],
    )


def test_homography_uses_eight_fit_points_and_meets_check_error_goal():
    calibration = exact_calibration()
    fitted = fit_homography(calibration)

    assert calibration.schema_version == CALIBRATION_SCHEMA_VERSION == 1
    assert fitted.max_check_error_tiles < 1e-5
    assert transform_point(fitted.matrix, 450, 700) == pytest.approx((4.5, 7), abs=1e-5)
    assert len(calibration_sha256(calibration)) == 64


def test_homography_rejects_check_points_over_point_one_five_tiles():
    calibration = exact_calibration().model_copy(
        update={"check_points": [point(300, 400, 3.3, 4), point(1200, 900, 12, 9)]}
    )

    with pytest.raises(ValueError, match="0.15"):
        fit_homography(calibration)
