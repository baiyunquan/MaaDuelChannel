import pytest

from maa_duel.video.viewport import Viewport, detect_viewport


@pytest.mark.parametrize(("width", "height", "profile"), [(1920, 864, "wide"), (1920, 1080, "hd")])
def test_supported_viewports_normalize_and_restore_points(width, height, profile):
    viewport = detect_viewport(width, height)

    assert viewport.profile == profile
    point = viewport.normalize_point(width * 0.25, height * 0.75)
    assert point == pytest.approx((0.25, 0.75))
    assert viewport.denormalize_point(*point) == pytest.approx((width * 0.25, height * 0.75))


def test_unsupported_viewport_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        detect_viewport(800, 800)


def test_viewport_rejects_points_outside_content():
    viewport = Viewport(x=10, y=20, width=100, height=50, profile="test")
    with pytest.raises(ValueError, match="outside"):
        viewport.normalize_point(5, 25)
