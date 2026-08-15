from __future__ import annotations

import time

import numpy as np

from app.camera.devices import CameraCatalog, CameraDevice
from app.camera.manager import CameraCaptureSettings, CameraManager
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
from app.tracking.async_hands import AsyncHandTracker
from app.tracking.hands import HandTracker
from app.tracking.smoothing import FingerControlPairSmoother
from app.ui.controls import Controls
from app.ui.hud import draw_finger_points, draw_hud
from app.utils.debug import FpsCounter, FramePacer


WINDOW_NAME = "hand-ar-strip"


def main() -> None:
    config = AppConfig()
    camera_catalog = CameraCatalog(
        refresh_interval=config.camera_refresh_interval,
        excluded_name_tokens=config.excluded_camera_name_tokens,
    )
    camera_manager = CameraManager(
        camera_catalog,
        CameraCaptureSettings(
            output_size=(config.frame_width, config.frame_height),
            capture_size=(config.camera_capture_width, config.camera_capture_height),
            fps=config.camera_fps,
            mirror=config.mirror_camera,
        ),
        preferred_index=config.camera_index,
        preferred_name=config.preferred_camera_name,
    )
    _print_camera_catalog(camera_manager.available_devices)
    camera_manager.consume_catalog_change()
    if camera_manager.available_devices:
        print(f"Opening preferred camera {config.preferred_camera_name!r}...")
        camera_manager.wait_until_ready(config.camera_startup_timeout)
    _print_active_camera(camera_manager)
    camera_manager.consume_camera_change()
    tracker = AsyncHandTracker(
        HandTracker(processing_size=(config.tracking_width, config.tracking_height))
    )
    smoother = FingerControlPairSmoother(config.smoothing_alpha)
    registry = StyleRegistry()
    fps_counter = FpsCounter()
    frame_pacer = FramePacer(config.target_display_fps)
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
    preview_ok, preview_frame = camera_manager.read()
    if not preview_ok or preview_frame is None:
        preview_frame = np.zeros(
            (config.frame_height, config.frame_width, 3),
            dtype=np.uint8,
        )
        detail = f" ({camera_manager.last_error})" if camera_manager.last_error else ""
        print(f"No camera is active; ArtFrame will keep trying in the background{detail}.")
    elif Webcam.is_nearly_black(preview_frame):
        print(
            "Camera warning: the active camera is returning nearly black frames. "
            "Check the lens cover, lighting, and Windows camera permissions."
        )
    frame = preview_frame
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
    previous_tracking_time = previous_time
    previous_mask_time = previous_time
    tracking_ms = 0.0
    hand_mask = np.zeros(
        (config.frame_height, config.frame_width, 1),
        dtype=np.uint8,
    )

    try:
        while not controls.should_quit and not graphics_output.should_close:
            camera_changed = camera_manager.refresh()
            if controls.consume_camera_cycle_request():
                if camera_manager.request_cycle():
                    requested = camera_manager.requested_device
                    target = (
                        f"[{requested.index}] {requested.name}"
                        if requested is not None
                        else "next camera"
                    )
                    print(f"Camera switch requested: {target}...")
                else:
                    print(f"Camera switch unavailable: {camera_manager.last_error}")
            switch_failure = camera_manager.consume_switch_failure()
            if switch_failure is not None:
                print(f"Camera switch failed; keeping current camera: {switch_failure}")
            catalog_devices = camera_manager.consume_catalog_change()
            if catalog_devices is not None:
                _print_camera_catalog(catalog_devices)
            if camera_changed:
                tracker.reset()
                smoother.reset()
                interaction_builder.reset()
                hand_mask_generator.reset()
                detections = []
                smoothed_controls = None
                tracking_ok = False
                interaction = InteractionState()
                hand_mask.fill(0)
                reset_time = time.perf_counter()
                previous_tracking_time = reset_time
                previous_mask_time = reset_time
                if camera_manager.current_device is None:
                    frame.fill(0)
                _print_active_camera(camera_manager)

            ok, next_frame, camera_updated = camera_manager.read_latest(
                timeout_seconds=0.0,
                copy_frame=False,
            )
            if ok and next_frame is not None:
                frame = next_frame
            else:
                camera_updated = False

            current_time = time.perf_counter()
            delta_seconds = min(current_time - previous_time, 0.1)
            previous_time = current_time
            if camera_updated:
                tracker.submit(frame)

            tracking_result = tracker.read_latest()
            if tracking_result.updated:
                detections = list(tracking_result.detections)
                tracking_ms = tracking_result.processing_ms
                finger_controls = extract_finger_control_pair(detections)
                smoothed_controls = smoother.update(finger_controls)
                tracking_ok = smoothed_controls is not None
                visible_hands = tuple(
                    extract_finger_points(detection) for detection in detections
                )
                interaction = interaction_builder.update(
                    visible_hands,
                    frame_size=(frame.shape[1], frame.shape[0]),
                    delta_seconds=max(current_time - previous_tracking_time, 1.0e-6),
                )
                previous_tracking_time = current_time
                hand_mask = hand_mask_generator.update(
                    detections,
                    frame_size=(frame.shape[1], frame.shape[0]),
                    delta_seconds=max(current_time - previous_mask_time, 1.0e-6),
                )
                previous_mask_time = current_time

            needs_cpu_overlay = controls.strip_enabled or config.debug_hud
            output = frame.copy() if needs_cpu_overlay else frame

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
                        f"tracking: {tracking_ms:.1f} ms (async)",
                        _camera_diagnostic(camera_manager),
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
                    copy_frame=False,
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
            frame_pacer.wait()
    finally:
        graphics_output.close()
        tracker.close()
        camera_manager.close()


def _print_camera_catalog(devices: tuple[CameraDevice, ...]) -> None:
    if not devices:
        print("Cameras detected: none (waiting for hot-plug)")
        return
    summary = ", ".join(f"[{device.index}] {device.name}" for device in devices)
    print(f"Cameras detected: {summary}")


def _print_active_camera(camera_manager: CameraManager) -> None:
    device = camera_manager.current_device
    webcam = camera_manager.webcam
    if device is None or webcam is None:
        print("Active camera: none")
        return
    print(
        f"Active camera: [{device.index}] {device.name} ({webcam.backend_name}), "
        f"capture {webcam.capture_size[0]}x{webcam.capture_size[1]} "
        f"@ {webcam.capture_fps:.1f} FPS -> "
        f"display {webcam.output_size[0]}x{webcam.output_size[1]}"
    )


def _camera_diagnostic(camera_manager: CameraManager) -> str:
    device = camera_manager.current_device
    webcam = camera_manager.webcam
    if device is None or webcam is None:
        return "camera: waiting for device"
    return (
        f"camera: {device.name} / {webcam.capture_size[0]}x{webcam.capture_size[1]} "
        f"@ {webcam.capture_fps:.1f}"
    )


if __name__ == "__main__":
    main()
