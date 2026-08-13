from __future__ import annotations

from pathlib import Path
from typing import Any

from app.graphics.gpu import GpuBackend, GpuRenderTarget, GpuTexture, TextureSpec
from app.graphics.liquid.gpu_resources import PingPongTargets
from app.graphics.shader import ShaderPass, ShaderSource, ShaderStage
from app.interaction.hand_controls import HandControl, InteractionState


def _hand_uniforms(prefix: str, hand: HandControl | None) -> dict[str, Any]:
    if hand is None or not hand.active:
        return {
            f"u_{prefix}_active": 0,
            f"u_{prefix}_position": (0.5, 0.5),
            f"u_{prefix}_velocity": (0.0, 0.0),
            f"u_{prefix}_pinch": 0.0,
        }
    return {
        f"u_{prefix}_active": 1,
        f"u_{prefix}_position": (hand.position.x, hand.position.y),
        f"u_{prefix}_velocity": (hand.velocity.x, hand.velocity.y),
        f"u_{prefix}_pinch": hand.pinch_amount,
    }


def interaction_uniforms(interaction: InteractionState) -> dict[str, Any]:
    uniforms = _hand_uniforms("left", interaction.left)
    uniforms.update(_hand_uniforms("right", interaction.right))
    return uniforms


class HandDistortionEffect:
    """First GPU proof: hand-driven webcam distortion with persistent feedback."""

    def __init__(
        self,
        backend: GpuBackend,
        render_size: tuple[int, int],
    ) -> None:
        self._backend = backend
        width, height = render_size
        target_spec = TextureSpec(width, height, components=4, dtype="f1")
        self._targets = PingPongTargets(
            backend.create_render_target(target_spec),
            backend.create_render_target(target_spec),
        )
        self._targets.clear(backend)
        shader_dir = Path(__file__).resolve().parents[1] / "shaders"
        self._program = backend.compile(
            ShaderPass(
                "hand_distortion",
                ShaderSource(ShaderStage.VERTEX, shader_dir / "fullscreen.vert"),
                ShaderSource(ShaderStage.FRAGMENT, shader_dir / "gpu_test.frag"),
            )
        )
        self._elapsed_seconds = 0.0

    @property
    def output(self) -> GpuRenderTarget:
        return self._targets.read

    def render(
        self,
        camera_texture: GpuTexture,
        interaction: InteractionState,
        delta_seconds: float,
    ) -> GpuRenderTarget:
        self._elapsed_seconds += max(0.0, delta_seconds)
        uniforms = interaction_uniforms(interaction)
        uniforms.update(
            {
                "u_time": self._elapsed_seconds,
                "u_resolution": self._targets.write.size,
            }
        )
        self._backend.draw_fullscreen(
            self._program,
            self._targets.write,
            textures={
                "u_camera": camera_texture,
                "u_feedback": self._targets.read.texture,
            },
            uniforms=uniforms,
        )
        self._targets.swap()
        return self._targets.read

    def release(self) -> None:
        self._program.release()
        self._targets.release()
