from dataclasses import dataclass, field, replace

from app.graphics.liquid.config import LiquidSimulationConfig
from app.compositing.hand_occlusion import HandOcclusionConfig
from app.graphics.liquid.materials.config import ArtisticLiquidConfig
from app.graphics.print.config import PrintTreatmentConfig


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    preferred_camera_name: str = "Logi Webcam C920e"
    camera_capture_width: int = 1920
    camera_capture_height: int = 1080
    camera_fps: float = 30.0
    camera_refresh_interval: float = 2.0
    camera_startup_timeout: float = 6.0
    excluded_camera_name_tokens: tuple[str, ...] = ("nvidia broadcast",)
    frame_width: int = 1280
    frame_height: int = 720
    tracking_width: int = 480
    tracking_height: int = 270
    mirror_camera: bool = True
    canvas_width: int = 420
    canvas_height: int = 120
    strip_height_ratio: float = 0.32
    min_strip_height: float = 80.0
    max_strip_height: float = 360.0
    smoothing_alpha: float = 0.35
    debug_hud: bool = False
    debug_finger_overlay: bool = False
    gpu_enabled: bool = True
    gpu_render_width: int = 1280
    gpu_render_height: int = 720
    gpu_simulation_scale: float | None = None
    gpu_vsync: bool = True
    target_display_fps: float = 60.0
    liquid: LiquidSimulationConfig = field(
        default_factory=lambda: LiquidSimulationConfig(
            simulation_scale=0.375,
            pressure_iterations=18,
            vorticity_strength=0.14,
            source_smoothing_time=0.028,
            source_prediction_time=0.055,
            source_dropout_hold=0.08,
            source_fade_time=0.38,
            dye_diffusion=0.012,
        )
    )
    debug_graphics_hud: bool = False
    hand_occlusion: HandOcclusionConfig = field(default_factory=HandOcclusionConfig)
    liquid_art: ArtisticLiquidConfig = field(default_factory=ArtisticLiquidConfig)
    print_treatment: PrintTreatmentConfig = field(default_factory=PrintTreatmentConfig)

    def effective_liquid_config(self) -> LiquidSimulationConfig:
        """Apply the legacy simulation-scale override when explicitly configured."""
        if self.gpu_simulation_scale is None:
            return self.liquid
        return replace(self.liquid, simulation_scale=self.gpu_simulation_scale)
