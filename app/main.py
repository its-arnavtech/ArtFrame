from __future__ import annotations

import time

from app.camera.webcam import Webcam
from app.compositing.overlay import composite_overlay
from app.compositing.hand_occlusion import HandMaskGenerator
from app.config import AppConfig
from app.geometry.strip import build_finger_section_quads
from app.geometry.warp import warp_canvas_to_quad
from app.graphics.output import create_graphics_output
from app.interaction.hand_controls import InteractionState, InteractionStateBuilder
from app.styles.registry import StyleRegistry
from app.tracking.fingers import extract_finger_control_pair, extract_finger_points
from app.tracking.hands import HandTracker
from app.tracking.smoothing import FingerControlPairSmoother
from app.ui.controls import Controls
from app.ui.hud import draw_finger_points, draw_hud
from app.utils.debug import FpsCounter


WINDOW_NAME = "hand-ar-strip"


def main() -> None:
    config = AppConfig()
    print(f"Opening physical camera index {config.camera_index}...")
    webcam = Webcam(
        camera_index=config.camera_index,
        width=config.frame_width,
        height=config.frame_height,
        mirror=config.mirror_camera,
    )
    print(f"Camera backend: index {webcam.camera_index} ({webcam.backend_name})")
    tracker = HandTracker(processing_size=(config.tracking_width, config.tracking_height))
    smoother = FingerControlPairSmoother(config.smoothing_alpha)
    registry = StyleRegistry()
    fps_counter = FpsCounter()
    interaction_builder = InteractionStateBuilder()
    hand_mask_generator = HandMaskGenerator(config.hand_occlusion)
    graphics_output = create_graphics_output(
        gpu_enabled=config.gpu_enabled,
        render_size=(config.gpu_render_width, config.gpu_render_height),
        liquid_config=config.effective_liquid_config(),
        artistic_config=config.liquid_art,
        print_config=config.print_treatment,
        vsync=config.gpu_vsync,
        window_name=WINDOW_NAME,
    )
    controls = Controls(registry, graphics_output)
    print(f"Graphics backend: {graphics_output.backend_name} ({graphics_output.info})")
    preview_ok, preview_frame = webcam.read()
    if not preview_ok or preview_frame is None:
        raise RuntimeError(f"Camera index {webcam.camera_index} stopped delivering frames")
    if Webcam.is_nearly_black(preview_frame):
        print(
            f"Camera warning: index {webcam.camera_index} is returning nearly black frames. "
            "Check the lens cover, lighting, and Windows camera permissions."
        )
    graphics_output.render(
        preview_frame,
        InteractionState(),
        1.0 / 60.0,
        foreground_bgr=preview_frame,
    )
    detections = []
    smoothed_controls = None
    tracking_ok = False
    interaction = InteractionState()
    previous_time = time.perf_counter()
    previous_camera_time = previous_time

    try:
        while not controls.should_quit and not graphics_output.should_close:
            ok, frame, camera_updated = webcam.read_latest(timeout_seconds=0.0)
            if not ok or frame is None:
                continue

            current_time = time.perf_counter()
            delta_seconds = min(current_time - previous_time, 0.1)
            previous_time = current_time
            if camera_updated:
                detections = tracker.detect(frame)
                finger_controls = extract_finger_control_pair(detections)
                smoothed_controls = smoother.update(finger_controls)
                tracking_ok = smoothed_controls is not None
                visible_hands = tuple(
                    extract_finger_points(detection) for detection in detections
                )
                interaction = interaction_builder.update(
                    visible_hands,
                    frame_size=(frame.shape[1], frame.shape[0]),
                    delta_seconds=max(current_time - previous_camera_time, 1.0e-6),
                )
                previous_camera_time = current_time

            output = frame.copy()
            hand_mask = hand_mask_generator.update(
                detections,
                frame_size=(frame.shape[1], frame.shape[0]),
                delta_seconds=delta_seconds,
            )

            if controls.strip_enabled and smoothed_controls is not None:
                quads = build_finger_section_quads(smoothed_controls)
                styles = registry.sequence(len(quads))
                canvases = {
                    style.name: style.render(frame, (config.canvas_width, config.canvas_height))
                    for style in {style.name: style for style in styles}.values()
                }

                for quad, style in zip(quads, styles):
                    canvas = canvases[style.name]
                    rendered = warp_canvas_to_quad(canvas, quad, frame.shape[1], frame.shape[0])
                    output = composite_overlay(
                        output,
                        rendered.image,
                        rendered.mask,
                        rendered.origin,
                        copy_frame=False,
                    )

            fps = fps_counter.tick()
            if config.debug_hud:
                if config.debug_finger_overlay:
                    output = draw_finger_points(output, smoothed_controls)
                diagnostics = graphics_output.diagnostics
                graphics_lines: tuple[str, ...] = (
                    f"liquid material: {diagnostics.get('material', 'n/a')}",
                )
                if config.debug_graphics_hud:
                    graphics_lines += (
                        f"cpu frame: {1000.0 / max(fps, 1e-9):.2f} ms",
                        f"gpu: {diagnostics.get('renderer', graphics_output.backend_name)}",
                        f"liquid: {diagnostics.get('liquid', 'unknown')}",
                        f"simulation: {diagnostics.get('simulation_resolution', 'n/a')}",
                        f"simulation fps: {diagnostics.get('simulation_fps', 'n/a')}",
                        f"display fps: {diagnostics.get('display_fps', 'n/a')}",
                        f"gpu frame: {diagnostics.get('gpu_frame_ms', 'disabled')} ms",
                        f"gpu sim: {diagnostics.get('gpu_simulation_ms', 'disabled')} ms",
                        f"gpu material: {diagnostics.get('gpu_material_ms', 'disabled')} ms",
                        f"gpu Riso: {diagnostics.get('gpu_riso_ms', 'disabled')} ms",
                        f"gpu compose: {diagnostics.get('gpu_composition_ms', 'disabled')} ms",
                        f"pressure: {diagnostics.get('pressure_iterations', 'n/a')} iterations",
                        f"sources: {diagnostics.get('active_sources', '0')}",
                        f"material: {diagnostics.get('material', 'n/a')}",
                        f"palette: {diagnostics.get('palette', 'n/a')}",
                        f"Riso: {diagnostics.get('riso_palette', 'n/a')} / {diagnostics.get('riso_quality', 'n/a')}",
                        f"occlusion: {diagnostics.get('hand_occlusion', 'inactive')}",
                    )
                output = draw_hud(
                    output,
                    registry.current().name,
                    tracking_ok,
                    fps,
                    graphics_lines,
                    strip_enabled=controls.strip_enabled,
                    fluid_enabled=controls.fluid_enabled,
                )

            graphics_output.render(
                output,
                interaction,
                delta_seconds,
                foreground_bgr=frame,
                hand_mask=hand_mask,
            )
            for key in graphics_output.consume_key_events():
                controls.handle_key(key)
    finally:
        graphics_output.close()
        tracker.close()
        webcam.release()


if __name__ == "__main__":
    main()
