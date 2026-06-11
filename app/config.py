from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    mirror_camera: bool = True
    canvas_width: int = 640
    canvas_height: int = 180
    strip_height_ratio: float = 0.32
    min_strip_height: float = 80.0
    max_strip_height: float = 220.0
    smoothing_alpha: float = 0.35
    debug_hud: bool = True
