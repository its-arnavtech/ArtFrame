from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from app.camera.webcam import Webcam
from app.graphics.gpu_renderer import GpuGraphicsRenderer
from app.graphics.liquid.config import LiquidSimulationConfig
from app.graphics.liquid.stress import StressScenario, stress_interaction
from app.interaction.hand_controls import HandControl, InteractionState
from app.types import Point2D


def _synthetic_frame(width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = x[None, :]
    frame[:, :, 1] = y[:, None]
    frame[:, :, 2] = 120
    return frame


def _interaction(frame_index: int) -> InteractionState:
    phase = frame_index * 0.035
    left_x = 0.35 + 0.12 * float(np.sin(phase))
    right_y = 0.55 + 0.10 * float(np.cos(phase * 0.8))
    return InteractionState(
        left=HandControl(
            position=Point2D(left_x, 0.45),
            velocity=Point2D(0.12 * float(np.cos(phase)), 0.0),
            pinch_amount=0.2,
            openness=0.7,
        ),
        right=HandControl(
            position=Point2D(0.65, right_y),
            velocity=Point2D(0.0, -0.08 * float(np.sin(phase * 0.8))),
            pinch_amount=0.5 + 0.5 * float(np.sin(phase * 0.6)),
            openness=0.5,
        ),
    )


def _synthetic_occlusion(
    frame: np.ndarray,
    interaction: InteractionState,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a soft two-hand fixture that exercises foreground GPU compositing."""
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    radius = max(1, round(min(width, height) * 0.10))
    feather = max(1, round(radius * 0.22))
    colors = ((28, 205, 255), (245, 120, 48))
    foreground = frame.copy()
    for hand, color in zip(interaction.active_hands(), colors):
        center = (
            round(hand.position.x * (width - 1)),
            round(hand.position.y * (height - 1)),
        )
        cv2.circle(mask, center, radius, 255, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(foreground, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    kernel_size = feather * 2 + 1
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    return foreground, mask[:, :, None]


def run_smoke_test(
    frame_count: int,
    size: tuple[int, int],
    *,
    visible: bool,
    camera_index: int | None = None,
    simulation_scale: float = 0.5,
    pressure_iterations: int = 20,
    gpu_timing: bool = False,
    stress: bool = False,
    cycle_debug: bool = False,
    occlusion: bool = True,
    cycle_materials: bool = False,
    cycle_palettes: bool = False,
    cycle_riso: bool = False,
) -> tuple[dict[str, str], float]:
    liquid_config = LiquidSimulationConfig(
        simulation_scale=simulation_scale,
        pressure_iterations=pressure_iterations,
        gpu_timing_enabled=gpu_timing,
    )
    renderer = GpuGraphicsRenderer(
        size,
        title="ArtFrame GPU smoke test",
        vsync=visible,
        visible=visible,
        liquid_config=liquid_config,
    )
    webcam = None
    frame = _synthetic_frame(*size)
    if camera_index is not None:
        webcam = Webcam(camera_index=camera_index, width=size[0], height=size[1], mirror=True)
    started = time.perf_counter()
    rendered_frames = 0
    try:
        for index in range(frame_count):
            if renderer.should_close:
                break
            if webcam is not None:
                ok, camera_frame = webcam.read()
                if not ok:
                    raise RuntimeError(f"camera {camera_index} did not return a frame")
                frame = camera_frame
            if cycle_debug and index > 0 and index % max(1, frame_count // 8) == 0:
                renderer.cycle_debug_view()
            interaction = _interaction(index)
            if stress:
                scenarios = tuple(StressScenario)
                scenario = scenarios[index % len(scenarios)]
                interaction = stress_interaction(scenario, index, frame_count)
            if cycle_materials and index > 0 and index % max(1, frame_count // 3) == 0:
                renderer.next_liquid_material()
            if cycle_palettes and index > 0 and index % max(1, frame_count // 4) == 0:
                renderer.next_liquid_palette()
            if cycle_riso and index > 0 and index % max(1, frame_count // 4) == 0:
                renderer.next_riso_palette()
            if cycle_riso and index > 0 and index % max(1, frame_count // 3) == 0:
                renderer.next_riso_quality()
            foreground = None
            hand_mask = None
            if occlusion:
                foreground, hand_mask = _synthetic_occlusion(frame, interaction)
            renderer.render(
                frame,
                interaction,
                1.0 / 60.0,
                foreground_bgr=foreground,
                hand_mask=hand_mask,
            )
            rendered_frames += 1
        elapsed = time.perf_counter() - started
        fps = rendered_frames / max(elapsed, 1e-9)
        info = dict(renderer.info)
        info["pass_order"] = " -> ".join(
            gpu_pass.kind.value for gpu_pass in renderer.liquid_pass_graph.implemented()
        )
        return info, fps
    finally:
        if webcam is not None:
            webcam.release()
        renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize and exercise the ArtFrame GPU path")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--simulation-scale", type=float, default=0.5)
    parser.add_argument("--pressure-iterations", type=int, default=20)
    parser.add_argument("--gpu-timing", action="store_true")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--cycle-debug", action="store_true")
    parser.add_argument("--no-occlusion", action="store_true")
    parser.add_argument("--cycle-materials", action="store_true")
    parser.add_argument("--cycle-palettes", action="store_true")
    parser.add_argument("--cycle-riso", action="store_true")
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")

    info, fps = run_smoke_test(
        args.frames,
        (args.width, args.height),
        visible=args.visible,
        camera_index=args.camera_index,
        simulation_scale=args.simulation_scale,
        pressure_iterations=args.pressure_iterations,
        gpu_timing=args.gpu_timing,
        stress=args.stress,
        cycle_debug=args.cycle_debug,
        occlusion=not args.no_occlusion,
        cycle_materials=args.cycle_materials,
        cycle_palettes=args.cycle_palettes,
        cycle_riso=args.cycle_riso,
    )
    print(f"GPU vendor: {info['vendor']}")
    print(f"GPU renderer: {info['renderer']}")
    print(f"OpenGL version: {info['version']}")
    print(f"Liquid enabled: {info['liquid']}")
    print(f"Simulation resolution: {info['simulation_resolution']}")
    print(f"Pressure iterations: {info['pressure_iterations']}")
    print(f"Active hand sources: {info['active_sources']}")
    print(f"Debug view: {info['debug_view']}")
    print(f"Liquid material: {info['material']}")
    print(f"Liquid palette: {info['palette']}")
    print(f"Print treatment: {info['print_treatment']}")
    print(f"Riso palette: {info['riso_palette']}")
    print(f"Riso quality: {info['riso_quality']}")
    print(f"Hand occlusion: {info['hand_occlusion']}")
    print(f"GPU timing: {info['gpu_timing']}")
    print(f"GPU frame: {info['gpu_frame_ms']} ms ({info['gpu_fps']} FPS estimate)")
    print(f"GPU simulation: {info['gpu_simulation_ms']} ms")
    print(f"GPU material: {info['gpu_material_ms']} ms")
    print(f"GPU Riso: {info['gpu_riso_ms']} ms")
    print(f"GPU composition: {info['gpu_composition_ms']} ms")
    print(f"Pass order: {info['pass_order']}")
    source = "webcam" if args.camera_index is not None else "synthetic input"
    print(f"Rendered {args.frames} {source} frames without GPU readback at {fps:.1f} FPS")


if __name__ == "__main__":
    main()
