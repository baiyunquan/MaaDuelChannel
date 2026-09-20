from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Viewport:
    x: float
    y: float
    width: float
    height: float
    profile: str

    def normalize_point(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        normalized_x = (pixel_x - self.x) / self.width
        normalized_y = (pixel_y - self.y) / self.height
        if not (0.0 <= normalized_x <= 1.0 and 0.0 <= normalized_y <= 1.0):
            raise ValueError("point is outside the detected game viewport")
        return normalized_x, normalized_y

    def denormalize_point(self, x: float, y: float) -> tuple[float, float]:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("normalized point must be inside [0, 1]")
        return self.x + x * self.width, self.y + y * self.height


def detect_viewport(frame_width: int, frame_height: int) -> Viewport:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    ratio = frame_width / frame_height
    if abs(ratio - (16 / 9)) <= 0.03:
        profile = "hd"
    elif abs(ratio - (20 / 9)) <= 0.03:
        profile = "wide"
    else:
        raise ValueError(f"unsupported frame aspect ratio: {frame_width}x{frame_height}")
    return Viewport(x=0.0, y=0.0, width=float(frame_width), height=float(frame_height), profile=profile)
