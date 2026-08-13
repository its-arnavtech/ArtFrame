import math

import pytest

from app.graphics.print.config import PrintTreatmentConfig, RisoConfig
from app.graphics.print.palette import RisoPalette, RisoPaletteRegistry
from app.graphics.print.quality import RisoQuality, next_quality, quality_profile
from app.graphics.print.registry import PrintTreatmentRegistry
from app.graphics.print.riso import RisographTreatment
from app.graphics.print.texture import screen_basis, screen_parameters
from app.interaction.hand_controls import HandControl, InteractionState
from app.types import Point2D


def _treatment(config: RisoConfig | None = None) -> RisographTreatment:
    actual_config = config or RisoConfig()
    palette = RisoPaletteRegistry().get(actual_config.palette)
    return RisographTreatment(actual_config, palette, quality_profile(actual_config.quality))


def test_riso_configuration_and_print_configuration_validate_parameters():
    assert PrintTreatmentConfig().treatment == "risograph"
    assert PrintTreatmentConfig().enabled is False
    assert RisoConfig().quality is RisoQuality.STANDARD
    with pytest.raises(ValueError):
        RisoConfig(dot_scale=1.0)
    with pytest.raises(ValueError):
        RisoConfig(dot_strength=1.1)
    with pytest.raises(ValueError):
        RisoConfig(registration_offset_pixels=-0.1)
    with pytest.raises(ValueError):
        RisoConfig(posterization_steps=1)


def test_riso_palette_registry_supports_selection_and_custom_palettes():
    registry = RisoPaletteRegistry()
    custom = RisoPalette(
        "custom",
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6),
        (0.9, 0.9, 0.8),
        (0.7, 0.2, 0.1),
    )
    registry.register(custom)

    assert registry.set("custom") == custom
    assert registry.names()[-1] == "custom"


def test_quality_profiles_change_detail_without_changing_render_resolution():
    draft = quality_profile(RisoQuality.DRAFT)
    standard = quality_profile(RisoQuality.STANDARD)
    high = quality_profile(RisoQuality.HIGH)

    assert draft.dot_detail < standard.dot_detail < high.dot_detail
    assert draft.paper_detail < standard.paper_detail < high.paper_detail
    assert next_quality(RisoQuality.DRAFT) is RisoQuality.STANDARD
    assert next_quality(RisoQuality.HIGH) is RisoQuality.DRAFT


def test_print_registry_selects_risograph_and_rejects_unknown_names():
    treatment = _treatment()
    registry = PrintTreatmentRegistry((treatment,))

    assert registry.names() == ("risograph",)
    assert registry.current() is treatment
    with pytest.raises(KeyError):
        registry.set("cyanotype")


def test_screen_parameters_are_deterministic_and_convert_pixels_to_uv():
    config = RisoConfig(dot_scale=8.0, screen_angle_degrees=0.0, registration_offset_pixels=2.0)
    profile = quality_profile(RisoQuality.STANDARD)

    first = screen_parameters(config, profile, (200, 100))
    second = screen_parameters(config, profile, (200, 100))

    assert first == second
    assert first.primary_basis == pytest.approx((1.0, 0.0))
    assert first.secondary_basis == pytest.approx(screen_basis(31.0))
    assert first.registration_uv == pytest.approx((0.01, 0.02))
    assert first.period_pixels == pytest.approx(8.0)


def test_riso_uniforms_map_interaction_without_tracker_dependencies():
    left = HandControl(Point2D(0.2, 0.5), Point2D(1.0, 0.0), 0.25, 0.4)
    right = HandControl(Point2D(0.8, 0.5), Point2D(0.0, 2.0), 0.75, 0.8)
    treatment = _treatment()
    from app.graphics.print.base import PrintTreatmentMetadata

    uniforms = treatment.uniforms(
        PrintTreatmentMetadata((960, 540), (480, 270), 1.0),
        InteractionState(left, right),
    )

    assert uniforms["u_interaction_velocity"] == pytest.approx(3.0)
    assert uniforms["u_interaction_pinch"] == pytest.approx(0.5)
    assert uniforms["u_interaction_openness"] == pytest.approx(0.6)
    assert uniforms["u_hand_distance"] == pytest.approx(0.6)
    assert math.isfinite(uniforms["u_dot_period"])
