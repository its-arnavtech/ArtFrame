from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from app.graphics.print.base import PrintTreatment, PrintTreatmentMetadata
from app.graphics.print.config import RisoConfig
from app.graphics.print.palette import RisoPalette
from app.graphics.print.quality import RisoQualityProfile
from app.graphics.print.texture import screen_parameters
from app.interaction.hand_controls import InteractionState


class RisographTreatment(PrintTreatment):
    name = "risograph"

    def __init__(
        self,
        config: RisoConfig,
        palette: RisoPalette,
        quality: RisoQualityProfile,
    ) -> None:
        self.config = config
        self.palette = palette
        self.quality = quality

    @property
    def analysis_fragment_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "shaders" / "riso_density.frag"

    @property
    def render_fragment_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "shaders" / "riso_print.frag"

    def uniforms(
        self,
        metadata: PrintTreatmentMetadata,
        interaction: InteractionState,
    ) -> dict[str, Any]:
        screen = screen_parameters(self.config, self.quality, metadata.display_size)
        active = interaction.active_hands()
        velocity = sum(math.hypot(hand.velocity.x, hand.velocity.y) for hand in active)
        pinch = sum(hand.pinch_amount for hand in active) / len(active) if active else 0.0
        openness = sum(hand.openness for hand in active) / len(active) if active else 0.0
        hand_distance = 0.0
        if interaction.left is not None and interaction.right is not None:
            hand_distance = math.hypot(
                interaction.right.position.x - interaction.left.position.x,
                interaction.right.position.y - interaction.left.position.y,
            )
        return {
            "u_display_size": metadata.display_size,
            "u_simulation_size": metadata.simulation_size,
            "u_time": metadata.elapsed_seconds,
            "u_primary_ink": self.palette.primary_ink,
            "u_secondary_ink": self.palette.secondary_ink,
            "u_paper_color": self.palette.paper,
            "u_accent_ink": self.palette.accent,
            "u_dot_period": screen.period_pixels,
            "u_primary_basis": screen.primary_basis,
            "u_secondary_basis": screen.secondary_basis,
            "u_registration_uv": screen.registration_uv,
            "u_dot_strength": self.config.dot_strength,
            "u_threshold": self.config.threshold,
            "u_density_response": self.config.density_response,
            "u_paper_strength": self.config.paper_strength,
            "u_grain_strength": self.config.grain_strength,
            "u_edge_breakup": self.config.edge_breakup,
            "u_posterization_steps": self.config.posterization_steps,
            "u_paper_detail": self.quality.paper_detail,
            "u_registration_complexity": self.quality.registration_complexity,
            "u_history_mix": self.quality.history_mix,
            "u_interaction_velocity": min(velocity, 5.0),
            "u_interaction_pinch": pinch,
            "u_interaction_openness": openness,
            "u_hand_distance": hand_distance,
        }
