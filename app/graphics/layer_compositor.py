from __future__ import annotations

from pathlib import Path

from app.graphics.gpu import GpuBackend, GpuProgram, GpuRenderTarget, GpuTexture
from app.graphics.liquid.config import LiquidDebugView
from app.graphics.liquid.gpu_resources import LiquidGpuResources
from app.graphics.shader import ShaderPass, ShaderSource, ShaderStage
from app.interaction.hand_controls import InteractionState


class LayerCompositeProgram:
    """Owns the persistent program shared across size-dependent compositors."""

    def __init__(self, backend: GpuBackend) -> None:
        shader_dir = Path(__file__).resolve().parent / "shaders"
        self.program = backend.compile(
            ShaderPass(
                "liquid_artistic_composite",
                ShaderSource(ShaderStage.VERTEX, shader_dir / "fullscreen.vert"),
                ShaderSource(ShaderStage.FRAGMENT, shader_dir / "liquid_artistic_composite.frag"),
            )
        )

    def release(self) -> None:
        self.program.release()


class LayerCompositor:
    """Composes camera, material/print output, real hands, and debug layers."""

    def __init__(
        self,
        backend: GpuBackend,
        resources: LiquidGpuResources,
        programs: LayerCompositeProgram,
    ) -> None:
        self._backend = backend
        self._resources = resources
        self._program = programs.program

    @property
    def output(self) -> GpuRenderTarget:
        return self._resources.visualization

    def render(
        self,
        base_camera: GpuTexture,
        foreground_camera: GpuTexture,
        hand_mask: GpuTexture,
        print_output: GpuTexture,
        interaction: InteractionState,
        debug_view: LiquidDebugView,
        visualization_enabled: bool,
        print_enabled: bool,
    ) -> None:
        composite_view = debug_view
        if debug_view in (
            LiquidDebugView.INK_MATERIAL,
            LiquidDebugView.GLASS_MATERIAL,
            LiquidDebugView.CHROMATIC_MATERIAL,
        ):
            composite_view = LiquidDebugView.MATERIAL_OUTPUT
        self._backend.draw_fullscreen(
            self._program,
            self._resources.visualization,
            {
                "u_base_camera": base_camera,
                "u_foreground_camera": foreground_camera,
                "u_hand_mask": hand_mask,
                "u_material": self._resources.material_output.texture,
                "u_print": print_output,
                "u_dye": self._resources.dye.read.texture,
                "u_velocity": self._resources.velocity.read.texture,
                "u_pressure": self._resources.pressure.read.texture,
                "u_divergence": self._resources.divergence.texture,
                "u_curl": self._resources.curl.texture,
            },
            {
                "u_debug_view": int(composite_view),
                "u_visualization_enabled": int(visualization_enabled),
                "u_print_enabled": int(print_enabled),
                **self._interaction_uniforms(interaction),
            },
        )

    @staticmethod
    def _interaction_uniforms(interaction: InteractionState) -> dict[str, object]:
        uniforms: dict[str, object] = {}
        for name, hand in (("left", interaction.left), ("right", interaction.right)):
            uniforms[f"u_{name}_active"] = int(hand is not None and hand.active)
            uniforms[f"u_{name}_position"] = (
                (hand.position.x, 1.0 - hand.position.y)
                if hand is not None
                else (0.5, 0.5)
            )
        return uniforms
