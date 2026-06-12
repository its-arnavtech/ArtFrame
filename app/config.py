from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    tracking_width: int = 480
    tracking_height: int = 270
    mirror_camera: bool = True
    canvas_width: int = 420
    canvas_height: int = 120
    strip_height_ratio: float = 0.32
    min_strip_height: float = 80.0
    max_strip_height: float = 360.0
    smoothing_alpha: float = 0.35
    debug_hud: bool = True
    debug_finger_overlay: bool = False
