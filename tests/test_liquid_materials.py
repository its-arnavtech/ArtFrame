import pytest

from app.graphics.liquid.materials.base import LiquidMaterialInputs, material_interaction_uniforms
from app.graphics.liquid.materials.config import ArtisticLiquidConfig
from app.graphics.liquid.materials.palette import LiquidPalette, LiquidPaletteRegistry
from app.graphics.liquid.materials.registry import LiquidMaterialRegistry
from app.interaction.hand_controls import HandControl, InteractionState
from app.types import Point2D


def test_material_registry_contains_and_selects_independent_materials():
    registry = LiquidMaterialRegistry()

    assert registry.names() == ("ink", "fluid_glass", "chromatic")
    assert registry.set("fluid_glass").name == "fluid_glass"
    assert registry.next().name == "chromatic"
    assert all(material.fragment_path.exists() for material in map(registry.get, registry.names()))


def test_material_contract_exposes_all_read_only_field_inputs():
    marker = object()
    inputs = LiquidMaterialInputs(marker, marker, marker, marker, marker)

    textures = LiquidMaterialRegistry().current().textures(inputs)

    assert tuple(textures) == (
        "u_base_camera",
        "u_dye",
        "u_velocity",
        "u_curl",
        "u_pressure",
    )


def test_palette_registry_supports_defaults_and_custom_palettes():
    registry = LiquidPaletteRegistry()
    custom = LiquidPalette("custom", (0.1, 0.2, 0.3), (0.3, 0.2, 0.1), (1, 1, 1), (0, 0, 0))

    registry.register(custom)
    registry.set("custom")

    assert registry.current() == custom
    assert registry.names() == (
        "neutral_chrome",
        "cyan_blue",
        "magenta_orange",
        "blue_violet",
        "monochrome_ink",
        "custom",
    )


def test_artistic_configuration_validation():
    assert ArtisticLiquidConfig().material == "fluid_glass"
    assert ArtisticLiquidConfig().palette == "neutral_chrome"
    with pytest.raises(ValueError):
        ArtisticLiquidConfig(intensity=4.0)
    with pytest.raises(ValueError):
        ArtisticLiquidConfig(texture_strength=-0.1)


def test_material_interaction_maps_velocity_pinch_and_hand_distance():
    left = HandControl(Point2D(0.2, 0.5), Point2D(1.0, 0.0), 0.2, 0.5)
    right = HandControl(Point2D(0.8, 0.5), Point2D(0.0, 2.0), 0.8, 0.5)

    uniforms = material_interaction_uniforms(InteractionState(left, right))

    assert uniforms["u_interaction_velocity"] == pytest.approx(3.0)
    assert uniforms["u_interaction_pinch"] == pytest.approx(0.5)
    assert uniforms["u_hand_distance"] == pytest.approx(0.6)
    assert uniforms["u_left_position"] == (0.2, 0.5)
    assert uniforms["u_left_velocity"] == (1.0, -0.0)
    assert uniforms["u_right_velocity"] == (0.0, -2.0)
    assert uniforms["u_left_pinch"] == pytest.approx(0.2)
    assert uniforms["u_right_openness"] == pytest.approx(0.5)
