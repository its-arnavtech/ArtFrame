from __future__ import annotations

import cv2

from app.camera.webcam import Webcam
from app.compositing.overlay import composite_overlay
from app.config import AppConfig
from app.geometry.strip import build_strip_quad
from app.geometry.warp import warp_canvas_to_quad
from app.styles.registry import StyleRegistry
from app.tracking.anchors import extract_anchor_pair
from app.tracking.hands import HandTracker
from app.tracking.smoothing import AnchorPairSmoother
from app.ui.controls import Controls
from app.ui.hud import draw_hud
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
    tracker = HandTracker()
    smoother = AnchorPairSmoother(config.smoothing_alpha)
    registry = StyleRegistry()
    controls = Controls(registry)
    fps_counter = FpsCounter()

    try:
        while not controls.should_quit:
            ok, frame = webcam.read()
            if not ok:
                break

            detections = tracker.detect(frame)
            anchors = extract_anchor_pair(detections)
            smoothed_anchors = smoother.update(anchors)
            tracking_ok = smoothed_anchors is not None
            output = frame.copy()

            if smoothed_anchors is not None:
                quad = build_strip_quad(
                    smoothed_anchors,
                    height_ratio=config.strip_height_ratio,
                    min_height=config.min_strip_height,
                    max_height=config.max_strip_height,
                )
                canvas = registry.current().render(frame, (config.canvas_width, config.canvas_height))
                rendered = warp_canvas_to_quad(canvas, quad, frame.shape[1], frame.shape[0])
                output = composite_overlay(output, rendered.image, rendered.mask)

            fps = fps_counter.tick()
            if config.debug_hud:
                output = draw_hud(output, registry.current().name, tracking_ok, fps)

            cv2.imshow(WINDOW_NAME, output)
            controls.handle_key(cv2.waitKey(1))
    finally:
        tracker.close()
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
