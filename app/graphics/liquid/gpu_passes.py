from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from app.graphics.gpu import GpuBackend, GpuProgram, GpuTexture
from app.graphics.liquid.config import LiquidSimulationConfig
from app.graphics.liquid.gpu_resources import LiquidGpuResources
from app.graphics.shader import ShaderPass, ShaderSource, ShaderStage
from app.interaction.hand_controls import HandControl, InteractionState


class LiquidPassKind(Enum):
    VELOCITY_INJECTION = "velocity_injection"
    VELOCITY_ADVECTION = "velocity_advection"
    CURL = "curl"
    VORTICITY = "vorticity"
    BOUNDARY_PRE_PROJECTION = "boundary_pre_projection"
    DIVERGENCE = "divergence"
    PRESSURE = "pressure"
    PROJECTION = "projection"
    BOUNDARY_POST_PROJECTION = "boundary_post_projection"
    DYE_INJECTION = "dye_injection"
    DYE_ADVECTION = "dye_advection"


@dataclass(frozen=True)
class LiquidGpuPass:
    kind: LiquidPassKind
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    implemented: bool = True


class LiquidGpuPassGraph:
    """Owns the stable execution order and declared resource dependencies."""

    def __init__(self, passes: tuple[LiquidGpuPass, ...]) -> None:
        kinds = tuple(gpu_pass.kind for gpu_pass in passes)
        if len(kinds) != len(set(kinds)):
            raise ValueError("liquid pass kinds must be unique")
        self._passes = passes

    @classmethod
    def simulation(cls) -> "LiquidGpuPassGraph":
        return cls(
            (
                LiquidGpuPass(
                    LiquidPassKind.VELOCITY_INJECTION,
                    reads=("velocity.previous", "interaction.sources"),
                    writes=("velocity.current",),
                ),
                LiquidGpuPass(
                    LiquidPassKind.VELOCITY_ADVECTION,
                    reads=("velocity.previous",),
                    writes=("velocity.current",),
                ),
                LiquidGpuPass(
                    LiquidPassKind.CURL,
                    reads=("velocity.current",),
                    writes=("curl",),
                ),
                LiquidGpuPass(
                    LiquidPassKind.VORTICITY,
                    reads=("velocity.current", "curl"),
                    writes=("velocity.previous",),
                ),
                LiquidGpuPass(
                    LiquidPassKind.BOUNDARY_PRE_PROJECTION,
                    reads=("velocity.previous",),
                    writes=("velocity.current",),
                ),
                LiquidGpuPass(
                    LiquidPassKind.DIVERGENCE,
                    reads=("velocity.current",),
                    writes=("divergence",),
                ),
                LiquidGpuPass(
                    LiquidPassKind.PRESSURE,
                    reads=("divergence", "pressure.previous"),
                    writes=("pressure.current",),
                ),
                LiquidGpuPass(
                    LiquidPassKind.PROJECTION,
                    reads=("velocity.current", "pressure.current"),
                    writes=("velocity.previous",),
                ),
                LiquidGpuPass(
                    LiquidPassKind.BOUNDARY_POST_PROJECTION,
                    reads=("velocity.previous",),
                    writes=("velocity.current",),
                ),
                LiquidGpuPass(
                    LiquidPassKind.DYE_INJECTION,
                    reads=("dye.previous", "interaction.sources"),
                    writes=("dye.current",),
                ),
                LiquidGpuPass(
                    LiquidPassKind.DYE_ADVECTION,
                    reads=("dye.previous", "velocity.current"),
                    writes=("dye.current",),
                ),
            )
        )

    @classmethod
    def initial(cls) -> "LiquidGpuPassGraph":
        """Compatibility alias for the pass graph introduced in the GPU scaffold."""
        return cls.simulation()

    @property
    def passes(self) -> tuple[LiquidGpuPass, ...]:
        return self._passes

    def implemented(self) -> tuple[LiquidGpuPass, ...]:
        return tuple(gpu_pass for gpu_pass in self._passes if gpu_pass.implemented)


def _source_uniforms(
    prefix: str,
    hand: HandControl | None,
    color: tuple[float, float, float],
) -> dict[str, Any]:
    if hand is None or not hand.active:
        return {
            f"u_{prefix}_active": 0,
            f"u_{prefix}_position": (0.5, 0.5),
            f"u_{prefix}_velocity": (0.0, 0.0),
            f"u_{prefix}_pinch": 0.0,
            f"u_{prefix}_openness": 0.0,
            f"u_{prefix}_color": color,
        }
    return {
        f"u_{prefix}_active": 1,
        f"u_{prefix}_position": (hand.position.x, 1.0 - hand.position.y),
        f"u_{prefix}_velocity": (hand.velocity.x, -hand.velocity.y),
        f"u_{prefix}_pinch": hand.pinch_amount,
        f"u_{prefix}_openness": hand.openness,
        f"u_{prefix}_color": color,
    }


def interaction_source_uniforms(
    interaction: InteractionState,
    config: LiquidSimulationConfig,
) -> dict[str, Any]:
    uniforms = _source_uniforms("left", interaction.left, config.left_dye_color)
    uniforms.update(_source_uniforms("right", interaction.right, config.right_dye_color))
    return uniforms


class LiquidGpuPrograms:
    """Compiles and owns one reusable shader program for each numerical pass."""

    _FRAGMENTS = {
        LiquidPassKind.VELOCITY_INJECTION: "liquid_velocity_injection.frag",
        LiquidPassKind.VELOCITY_ADVECTION: "liquid_advection.frag",
        LiquidPassKind.CURL: "liquid_curl.frag",
        LiquidPassKind.VORTICITY: "liquid_vorticity.frag",
        LiquidPassKind.BOUNDARY_PRE_PROJECTION: "liquid_velocity_boundary.frag",
        LiquidPassKind.DIVERGENCE: "liquid_divergence.frag",
        LiquidPassKind.PRESSURE: "liquid_pressure.frag",
        LiquidPassKind.PROJECTION: "liquid_projection.frag",
        LiquidPassKind.BOUNDARY_POST_PROJECTION: "liquid_velocity_boundary.frag",
        LiquidPassKind.DYE_INJECTION: "liquid_dye_injection.frag",
        LiquidPassKind.DYE_ADVECTION: "liquid_advection.frag",
    }

    def __init__(self, backend: GpuBackend) -> None:
        shader_dir = Path(__file__).resolve().parents[1] / "shaders"
        vertex = ShaderSource(ShaderStage.VERTEX, shader_dir / "fullscreen.vert")
        self._programs = {
            kind: backend.compile(
                ShaderPass(
                    kind.value,
                    vertex,
                    ShaderSource(ShaderStage.FRAGMENT, shader_dir / fragment),
                )
            )
            for kind, fragment in self._FRAGMENTS.items()
        }

    def __getitem__(self, kind: LiquidPassKind) -> GpuProgram:
        return self._programs[kind]

    def release(self) -> None:
        for program in self._programs.values():
            program.release()


class LiquidGpuPassExecutor:
    """Executes the pass graph; no solver ordering lives in the application loop."""

    def __init__(
        self,
        backend: GpuBackend,
        resources: LiquidGpuResources,
        config: LiquidSimulationConfig,
        graph: LiquidGpuPassGraph | None = None,
        programs: LiquidGpuPrograms | None = None,
    ) -> None:
        self._backend = backend
        self._resources = resources
        self._config = config
        self.graph = graph or LiquidGpuPassGraph.simulation()
        self._programs = programs or LiquidGpuPrograms(backend)
        self._owns_programs = programs is None
        self._dispatch: dict[LiquidPassKind, Callable[..., None]] = {
            LiquidPassKind.VELOCITY_INJECTION: self._inject_velocity,
            LiquidPassKind.VELOCITY_ADVECTION: self._advect_velocity,
            LiquidPassKind.CURL: self._compute_curl,
            LiquidPassKind.VORTICITY: self._apply_vorticity,
            LiquidPassKind.BOUNDARY_PRE_PROJECTION: self._apply_velocity_boundary_pre,
            LiquidPassKind.DIVERGENCE: self._compute_divergence,
            LiquidPassKind.PRESSURE: self._solve_pressure,
            LiquidPassKind.PROJECTION: self._project_velocity,
            LiquidPassKind.BOUNDARY_POST_PROJECTION: self._apply_velocity_boundary_post,
            LiquidPassKind.DYE_INJECTION: self._inject_dye,
            LiquidPassKind.DYE_ADVECTION: self._advect_dye,
        }

    def execute(
        self,
        camera_texture: GpuTexture,
        interaction: InteractionState,
        timestep: float,
    ) -> None:
        self.execute_simulation(camera_texture, interaction, timestep)

    def execute_simulation(
        self,
        camera_texture: GpuTexture,
        interaction: InteractionState,
        timestep: float,
    ) -> None:
        source_uniforms = interaction_source_uniforms(interaction, self._config)
        for gpu_pass in self.graph.implemented():
            if not self._config.enabled:
                continue
            self._dispatch[gpu_pass.kind](camera_texture, source_uniforms, timestep)

    def release(self) -> None:
        if self._owns_programs:
            self._programs.release()

    def _inject_velocity(
        self,
        camera: GpuTexture,
        sources: dict[str, Any],
        timestep: float,
    ) -> None:
        del camera
        uniforms = dict(sources)
        uniforms.update(
            {
                "u_timestep": timestep,
                "u_injection_strength": self._config.velocity_injection_strength,
                "u_velocity_scale": self._config.velocity_scale,
                "u_source_radius": self._config.source_radius,
            }
        )
        self._backend.draw_fullscreen(
            self._programs[LiquidPassKind.VELOCITY_INJECTION],
            self._resources.velocity.write,
            {"u_velocity": self._resources.velocity.read.texture},
            uniforms,
        )
        self._resources.velocity.swap()

    def _advect_velocity(
        self,
        camera: GpuTexture,
        sources: dict[str, Any],
        timestep: float,
    ) -> None:
        del camera, sources
        self._backend.draw_fullscreen(
            self._programs[LiquidPassKind.VELOCITY_ADVECTION],
            self._resources.velocity.write,
            {
                "u_quantity": self._resources.velocity.read.texture,
                "u_velocity": self._resources.velocity.read.texture,
            },
            {
                "u_timestep": timestep,
                "u_decay": self._config.frame_decay(self._config.velocity_decay, timestep),
                "u_texel_size": self._texel_size,
                "u_diffusion": 0.0,
            },
        )
        self._resources.velocity.swap()

    def _compute_curl(
        self,
        camera: GpuTexture,
        sources: dict[str, Any],
        timestep: float,
    ) -> None:
        del camera, sources, timestep
        self._backend.draw_fullscreen(
            self._programs[LiquidPassKind.CURL],
            self._resources.curl,
            {"u_velocity": self._resources.velocity.read.texture},
            {"u_texel_size": self._texel_size},
        )

    def _apply_vorticity(
        self,
        camera: GpuTexture,
        sources: dict[str, Any],
        timestep: float,
    ) -> None:
        del camera, sources
        self._backend.draw_fullscreen(
            self._programs[LiquidPassKind.VORTICITY],
            self._resources.velocity.write,
            {
                "u_velocity": self._resources.velocity.read.texture,
                "u_curl": self._resources.curl.texture,
            },
            {
                "u_texel_size": self._texel_size,
                "u_timestep": timestep,
                "u_strength": self._config.vorticity_strength,
            },
        )
        self._resources.velocity.swap()

    def _apply_velocity_boundary_pre(
        self,
        camera: GpuTexture,
        sources: dict[str, Any],
        timestep: float,
    ) -> None:
        self._apply_velocity_boundary(
            LiquidPassKind.BOUNDARY_PRE_PROJECTION, camera, sources, timestep
        )

    def _apply_velocity_boundary_post(
        self,
        camera: GpuTexture,
        sources: dict[str, Any],
        timestep: float,
    ) -> None:
        self._apply_velocity_boundary(
            LiquidPassKind.BOUNDARY_POST_PROJECTION, camera, sources, timestep
        )

    def _apply_velocity_boundary(
        self,
        kind: LiquidPassKind,
        camera: GpuTexture,
        sources: dict[str, Any],
        timestep: float,
    ) -> None:
        del camera, sources, timestep
        self._backend.draw_fullscreen(
            self._programs[kind],
            self._resources.velocity.write,
            {"u_velocity": self._resources.velocity.read.texture},
            {
                "u_texel_size": self._texel_size,
                "u_maximum_velocity": self._config.maximum_fluid_velocity,
            },
        )
        self._resources.velocity.swap()

    def _compute_divergence(
        self,
        camera: GpuTexture,
        sources: dict[str, Any],
        timestep: float,
    ) -> None:
        del camera, sources, timestep
        self._backend.draw_fullscreen(
            self._programs[LiquidPassKind.DIVERGENCE],
            self._resources.divergence,
            {"u_velocity": self._resources.velocity.read.texture},
            {"u_texel_size": self._texel_size},
        )

    def _solve_pressure(
        self,
        camera: GpuTexture,
        sources: dict[str, Any],
        timestep: float,
    ) -> None:
        del camera, sources, timestep
        for _ in range(self._config.pressure_iterations):
            self._backend.draw_fullscreen(
                self._programs[LiquidPassKind.PRESSURE],
                self._resources.pressure.write,
                {
                    "u_pressure": self._resources.pressure.read.texture,
                    "u_divergence": self._resources.divergence.texture,
                },
                {"u_texel_size": self._texel_size},
            )
            self._resources.pressure.swap()

    def _project_velocity(
        self,
        camera: GpuTexture,
        sources: dict[str, Any],
        timestep: float,
    ) -> None:
        del camera, sources, timestep
        self._backend.draw_fullscreen(
            self._programs[LiquidPassKind.PROJECTION],
            self._resources.velocity.write,
            {
                "u_velocity": self._resources.velocity.read.texture,
                "u_pressure": self._resources.pressure.read.texture,
            },
            {"u_texel_size": self._texel_size},
        )
        self._resources.velocity.swap()

    def _inject_dye(
        self,
        camera: GpuTexture,
        sources: dict[str, Any],
        timestep: float,
    ) -> None:
        del camera
        uniforms = dict(sources)
        uniforms.update(
            {
                "u_timestep": timestep,
                "u_source_radius": self._config.source_radius,
                "u_injection_strength": self._config.dye_injection_strength,
                "u_velocity_coupling": self._config.dye_velocity_coupling,
            }
        )
        self._backend.draw_fullscreen(
            self._programs[LiquidPassKind.DYE_INJECTION],
            self._resources.dye.write,
            {"u_dye": self._resources.dye.read.texture},
            uniforms,
        )
        self._resources.dye.swap()

    def _advect_dye(
        self,
        camera: GpuTexture,
        sources: dict[str, Any],
        timestep: float,
    ) -> None:
        del camera, sources
        self._backend.draw_fullscreen(
            self._programs[LiquidPassKind.DYE_ADVECTION],
            self._resources.dye.write,
            {
                "u_quantity": self._resources.dye.read.texture,
                "u_velocity": self._resources.velocity.read.texture,
            },
            {
                "u_timestep": timestep,
                "u_decay": self._config.frame_decay(self._config.dye_decay, timestep),
                "u_texel_size": self._texel_size,
                "u_diffusion": self._config.dye_diffusion,
            },
        )
        self._resources.dye.swap()

    @property
    def _texel_size(self) -> tuple[float, float]:
        width, height = self._resources.resolution.simulation_size
        return 1.0 / width, 1.0 / height
