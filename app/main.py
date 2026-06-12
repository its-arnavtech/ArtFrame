from __future__ import annotations

import cv2

from app.camera.webcam import Webcam
from app.compositing.overlay import composite_overlay
from app.config import AppConfig
from app.geometry.strip import build_finger_section_quads
from app.geometry.warp import warp_canvas_to_quad
from app.styles.registry import StyleRegistry
from app.tracking.fingers import extract_finger_control_pair
from app.tracking.hands import HandTracker
from app.tracking.smoothing import FingerControlPairSmoother
from app.ui.controls import Controls
from app.ui.hud import draw_finger_points, draw_hud
from app.utils.debug import FpsCounter


WINDOW_NAME = "hand-ar-strip"


def main() -> None:
    config = AppConfig()
    webcam = Webcam(
        camera_index=config.camera_index,
        width=config.frame_width,
        height=config.frame_height,
        mirror=config.mirror_camera,
    )
    tracker = HandTracker(processing_size=(config.tracking_width, config.tracking_height))
    smoother = FingerControlPairSmoother(config.smoothing_alpha)
    registry = StyleRegistry()
    controls = Controls(registry)
    fps_counter = FpsCounter()

    try:
        while not controls.should_quit:
            ok, frame = webcam.read()
            if not ok:
                break

            detections = tracker.detect(frame)
            finger_controls = extract_finger_control_pair(detections)
            smoothed_controls = smoother.update(finger_controls)
            tracking_ok = smoothed_controls is not None
            output = frame.copy()

            if smoothed_controls is not None:
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
                output = draw_hud(output, registry.current().name, tracking_ok, fps)

            cv2.imshow(WINDOW_NAME, output)
            controls.handle_key(cv2.waitKey(1))
    finally:
        tracker.close()
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
