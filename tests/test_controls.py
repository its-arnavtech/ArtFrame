from app.styles.registry import StyleRegistry
from app.ui.controls import Controls


class _GraphicsController:
    def __init__(self) -> None:
        self.liquid_enabled = True

    def toggle_liquid_layer(self) -> bool:
        self.liquid_enabled = not self.liquid_enabled
        return self.liquid_enabled


def test_strip_and_fluid_layers_toggle_independently() -> None:
    graphics = _GraphicsController()
    controls = Controls(StyleRegistry(), graphics)

    assert controls.strip_enabled is False
    assert controls.fluid_enabled is True

    controls.handle_key(ord("S"))
    assert controls.strip_enabled is True
    assert controls.fluid_enabled is True

    controls.handle_key(ord("W"))
    assert controls.strip_enabled is True
    assert controls.fluid_enabled is False


def test_camera_cycle_request_is_consumed_once() -> None:
    controls = Controls(StyleRegistry())

    controls.handle_key(ord("C"))

    assert controls.consume_camera_cycle_request()
    assert not controls.consume_camera_cycle_request()
