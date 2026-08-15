from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.types import HandDetection, Point2D


@dataclass(frozen=True)
class HandOcclusionConfig:
    enabled: bool = True
    expansion: float = 0.025
    feather_radius: float = 0.014
    confidence_threshold: float = 0.55
    temporal_response: float = 0.045
    processing_scale: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.expansion <= 0.25:
            raise ValueError("expansion must be in the range [0, 0.25]")
        if not 0.0 <= self.feather_radius <= 0.25:
            raise ValueError("feather_radius must be in the range [0, 0.25]")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in the range [0, 1]")
        if self.temporal_response < 0.0:
            raise ValueError("temporal_response must not be negative")
        if not 0.0 < self.processing_scale <= 1.0:
            raise ValueError("processing_scale must be in the range (0, 1]")


def normalized_landmark_points(
    detection: HandDetection,
    frame_size: tuple[int, int],
) -> tuple[Point2D, ...]:
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    return tuple(
        Point2D(
            max(0.0, min(1.0, float(point[0]) / width)),
            max(0.0, min(1.0, float(point[1]) / height)),
        )
        for point in detection.landmarks
    )


def mask_points(
    detection: HandDetection,
    frame_size: tuple[int, int],
) -> np.ndarray:
    width, height = frame_size
    normalized = normalized_landmark_points(detection, frame_size)
    return np.array(
        [[round(point.x * (width - 1)), round(point.y * (height - 1))] for point in normalized],
        dtype=np.int32,
    )


def composite_hand_foreground(
    base_bgr: np.ndarray,
    foreground_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    if base_bgr.shape != foreground_bgr.shape:
        raise ValueError("base and foreground frames must have matching shapes")
    if mask.shape[:2] != base_bgr.shape[:2]:
        raise ValueError("mask dimensions must match the frame")
    alpha = mask.astype(np.float32) / 255.0
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]
    return np.clip(
        foreground_bgr.astype(np.float32) * alpha
        + base_bgr.astype(np.float32) * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)


class HandMaskGenerator:
    """Builds a feathered CPU mask for GPU upload; never reads GPU state."""

    def __init__(self, config: HandOcclusionConfig = HandOcclusionConfig()) -> None:
        self.config = config
        self._smoothed: np.ndarray | None = None
        self._target: np.ndarray | None = None
        self._processing_size: tuple[int, int] | None = None
        self._output_size: tuple[int, int] | None = None

    def update(
        self,
        detections: list[HandDetection],
        frame_size: tuple[int, int],
        delta_seconds: float,
        *,
        refresh_target: bool = True,
    ) -> np.ndarray:
        if delta_seconds < 0.0:
            raise ValueError("delta_seconds must not be negative")
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        processing_size = (
            max(2, round(width * self.config.processing_scale)),
            max(2, round(height * self.config.processing_scale)),
        )
        size_changed = (
            self._processing_size != processing_size or self._output_size != frame_size
        )
        if refresh_target or self._target is None or size_changed:
            self._target = self._build_target(detections, frame_size, processing_size)
            self._processing_size = processing_size
            self._output_size = frame_size

        if self._smoothed is None or size_changed:
            self._smoothed = self._target.copy()
        else:
            alpha = 1.0
            if self.config.temporal_response > 0.0:
                alpha = 1.0 - math.exp(-delta_seconds / self.config.temporal_response)
            self._smoothed += (self._target - self._smoothed) * alpha
        mask = np.clip(self._smoothed * 255.0, 0, 255).astype(np.uint8)
        if processing_size != frame_size:
            mask = cv2.resize(mask, frame_size, interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(mask[:, :, None])

    def reset(self) -> None:
        self._smoothed = None
        self._target = None
        self._processing_size = None
        self._output_size = None

    def _build_target(
        self,
        detections: list[HandDetection],
        frame_size: tuple[int, int],
        processing_size: tuple[int, int],
    ) -> np.ndarray:
        width, height = processing_size
        target = np.zeros((height, width), dtype=np.uint8)
        if self.config.enabled:
            for detection in detections:
                if detection.score < self.config.confidence_threshold:
                    continue
                normalized = normalized_landmark_points(detection, frame_size)
                points = np.asarray(
                    [
                        [
                            round(point.x * (width - 1)),
                            round(point.y * (height - 1)),
                        ]
                        for point in normalized
                    ],
                    dtype=np.int32,
                )
                if len(points) >= 3:
                    cv2.fillConvexPoly(
                        target,
                        cv2.convexHull(points),
                        255,
                        lineType=cv2.LINE_AA,
                    )

            reference = min(width, height)
            expansion_pixels = round(self.config.expansion * reference)
            if expansion_pixels > 0:
                kernel_size = expansion_pixels * 2 + 1
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (kernel_size, kernel_size),
                )
                target = cv2.dilate(target, kernel)

            feather_pixels = round(self.config.feather_radius * reference)
            if feather_pixels > 0:
                kernel_size = feather_pixels * 2 + 1
                target = cv2.GaussianBlur(target, (kernel_size, kernel_size), 0)
        return target.astype(np.float32) / 255.0
