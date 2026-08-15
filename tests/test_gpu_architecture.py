from dataclasses import dataclass

import pytest

from app.graphics.effects.hand_distortion import interaction_uniforms
from app.graphics.gpu import TextureSpec
from app.graphics.liquid.config import LiquidSimulationConfig
from app.graphics.liquid.gpu_passes import (
    LiquidGpuPassGraph,
    LiquidPassKind,
    interaction_source_uniforms,
)
from app.graphics.liquid.gpu_passes import LiquidGpuPassExecutor
from app.graphics.liquid.gpu_resources import (
    LiquidGpuResolution,
    LiquidGpuResources,
    PingPongTargets,
)
from app.graphics.liquid.liquid_effect import GpuLiquidEffect
from app.interaction.hand_controls import HandControl, InteractionState
from app.types import Point2D


@dataclass
class _FakeTarget:
    name: str
    size: tuple[int, int] = (32, 16)
    released: bool = False

    @property
    def texture(self):
        return self

    def release(self) -> None:
        self.released = True


@dataclass
class _FakeProgram:
    name: str
    released: bool = False

    def set_uniform(self, name, value):
        pass

    def release(self):
        self.released = True


@dataclass
class _FakeTexture:
    spec: TextureSpec
    released: bool = False

    def release(self):
        self.released = True


class _FakeGpuTarget:
    def __init__(self, spec: TextureSpec):
        self.texture = _FakeTexture(spec)
        self.released = False

    @property
    def size(self):
        return self.texture.spec.size

    def release(self):
        self.texture.release()
        self.released = True


class _FakeBackend:
    def __init__(self):
        self.targets = []
        self.draws = []
        self.programs = []

    def create_render_target(self, spec):
        target = _FakeGpuTarget(spec)
        self.targets.append(target)
        return target

    def clear(self, target, color):
        pass

    def compile(self, shader_pass):
        program = _FakeProgram(shader_pass.name)
        self.programs.append(program)
        return program

    def create_timer(self, label, query_lag=4):
        return _FakeTimer()

    def draw_fullscreen(self, program, target, textures, uniforms=None, viewport_size=None):
        self.draws.append(program.name)


class _FakeTimer:
    supported = False
    latest_ms = None
    average_ms = None

    def begin(self):
        pass

    def end(self):
        pass

    def release(self):
        pass


def test_texture_spec_validates_gpu_texture_shape():
    assert TextureSpec(320, 180).dtype == "f1"
    assert TextureSpec(320, 180, components=2, dtype="f2").size == (320, 180)
    with pytest.raises(ValueError):
        TextureSpec(0, 180)
    with pytest.raises(ValueError):
        TextureSpec(320, 180, components=5)


def test_ping_pong_targets_swap_without_allocating_new_targets():
    first = _FakeTarget("first")
    second = _FakeTarget("second")
    targets = PingPongTargets(first, second)

    targets.swap()

    assert targets.read is second
    assert targets.write is first


def test_liquid_resolution_scales_independently_from_display():
    resolution = LiquidGpuResolution((960, 540), simulation_scale=0.5)

    assert resolution.simulation_size == (480, 270)


def test_liquid_resource_manager_owns_all_persistent_solver_targets():
    backend = _FakeBackend()
    resolution = LiquidGpuResolution((320, 180), simulation_scale=0.5)

    resources = LiquidGpuResources.create(backend, resolution)

    assert len(backend.targets) == 10
    assert resources.velocity.read.size == (160, 90)
    assert resources.dye.read.texture.spec.components == 4
    assert resources.pressure.read.texture.spec.components == 1
    assert resources.divergence.texture.spec.dtype == "f2"
    assert resources.curl.texture.spec.components == 1
    assert resources.material_output.texture.spec.dtype == "f2"
    assert resources.visualization.size == (320, 180)

    resources.release()
    assert all(target.released for target in backend.targets)


def test_initial_liquid_pass_graph_preserves_solver_order():
    graph = LiquidGpuPassGraph.initial()

    assert tuple(gpu_pass.kind for gpu_pass in graph.passes) == tuple(LiquidPassKind)
    assert graph.implemented() == graph.passes
    dye_advection = next(
        gpu_pass for gpu_pass in graph.passes if gpu_pass.kind is LiquidPassKind.DYE_ADVECTION
    )
    assert dye_advection.reads == ("dye.previous", "velocity.current")
    assert dye_advection.writes == ("dye.current",)


def test_liquid_config_clamps_timestep_and_scales_decay_by_time():
    config = LiquidSimulationConfig(max_timestep=1.0 / 30.0)

    assert config.timestep(1.0 / 120.0) == pytest.approx(1.0 / 120.0)
    assert config.timestep(0.5) == pytest.approx(1.0 / 30.0)
    assert config.frame_decay(0.99, 1.0 / 30.0) == pytest.approx(0.99**2)


def test_liquid_source_uniforms_flip_vertical_axis_and_preserve_semantics():
    config = LiquidSimulationConfig()
    left = HandControl(
        position=Point2D(0.25, 0.75),
        velocity=Point2D(0.1, -0.2),
        pinch_amount=0.6,
        openness=0.4,
    )

    uniforms = interaction_source_uniforms(InteractionState(left=left), config)

    assert uniforms["u_left_active"] == 1
    assert uniforms["u_left_position"] == (0.25, 0.25)
    assert uniforms["u_left_velocity"] == (0.1, 0.2)
    assert uniforms["u_left_openness"] == 0.4
    assert uniforms["u_right_active"] == 0


def test_pass_executor_runs_solver_in_graph_order_with_jacobi_iterations():
    backend = _FakeBackend()
    config = LiquidSimulationConfig(simulation_scale=0.5, pressure_iterations=3)
    resources = LiquidGpuResources.create(
        backend,
        LiquidGpuResolution((320, 180), simulation_scale=config.simulation_scale),
    )
    executor = LiquidGpuPassExecutor(backend, resources, config)
    camera = _FakeTexture(TextureSpec(320, 180, components=3))

    executor.execute(camera, InteractionState(), timestep=1.0 / 60.0)

    assert backend.draws == [
        "velocity_injection",
        "velocity_advection",
        "curl",
        "vorticity",
        "boundary_pre_projection",
        "divergence",
        "pressure",
        "pressure",
        "pressure",
        "projection",
        "boundary_post_projection",
        "dye_injection",
        "dye_advection",
    ]


def test_disabled_simulation_skips_every_solver_pass():
    backend = _FakeBackend()
    config = LiquidSimulationConfig(enabled=False, pressure_iterations=2)
    resources = LiquidGpuResources.create(
        backend,
        LiquidGpuResolution((160, 90), simulation_scale=config.simulation_scale),
    )
    executor = LiquidGpuPassExecutor(backend, resources, config)
    camera = _FakeTexture(TextureSpec(160, 90, components=3))

    executor.execute(camera, InteractionState(), timestep=1.0 / 60.0)

    assert backend.draws == []


def test_liquid_reconfiguration_reallocates_targets_without_recompiling_programs():
    backend = _FakeBackend()
    effect = GpuLiquidEffect(
        backend,
        (320, 180),
        LiquidSimulationConfig(simulation_scale=0.5),
    )
    first_targets = tuple(backend.targets)
    compiled_programs = len(backend.programs)
    effect.next_material()
    effect.next_palette()
    effect.next_riso_palette()
    effect.next_riso_quality()

    effect.reconfigure(LiquidSimulationConfig(simulation_scale=1.0))

    assert all(target.released for target in first_targets)
    assert len(backend.programs) == compiled_programs
    assert effect.diagnostics["simulation_resolution"] == "320x180"
    assert effect.diagnostics["material"] == "pinch_fluid"
    assert effect.diagnostics["palette"] == "cyan_blue"
    assert effect.diagnostics["riso_palette"] == "magenta_orange"
    assert effect.diagnostics["riso_quality"] == "high"
    effect.release()
    assert all(target.released for target in backend.targets)


def test_interaction_uniforms_handle_present_and_missing_hands():
    left = HandControl(
        position=Point2D(0.25, 0.75),
        velocity=Point2D(0.1, -0.2),
        pinch_amount=0.6,
        openness=0.4,
    )

    uniforms = interaction_uniforms(InteractionState(left=left))

    assert uniforms["u_left_active"] == 1
    assert uniforms["u_left_position"] == (0.25, 0.75)
    assert uniforms["u_left_velocity"] == (0.1, -0.2)
    assert uniforms["u_right_active"] == 0
    assert uniforms["u_right_pinch"] == 0.0
