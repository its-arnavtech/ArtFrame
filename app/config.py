from dataclasses import dataclass, field, replace

from app.graphics.liquid.config import LiquidSimulationConfig
from app.compositing.hand_occlusion import HandOcclusionConfig
from app.graphics.liquid.materials.config import ArtisticLiquidConfig
from app.graphics.print.config import PrintTreatmentConfig


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
    gpu_enabled: bool = True
    gpu_render_width: int = 960
    gpu_render_height: int = 540
    gpu_simulation_scale: float | None = None
    gpu_vsync: bool = True
    liquid: LiquidSimulationConfig = field(default_factory=LiquidSimulationConfig)
    debug_graphics_hud: bool = False
    hand_occlusion: HandOcclusionConfig = field(default_factory=HandOcclusionConfig)
    liquid_art: ArtisticLiquidConfig = field(default_factory=ArtisticLiquidConfig)
    print_treatment: PrintTreatmentConfig = field(default_factory=PrintTreatmentConfig)

    def effective_liquid_config(self) -> LiquidSimulationConfig:
        """Apply the legacy simulation-scale override when explicitly configured."""
        if self.gpu_simulation_scale is None:
            return self.liquid
        return replace(self.liquid, simulation_scale=self.gpu_simulation_scale)
