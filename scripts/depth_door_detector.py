#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

# OpenCV is used by the underlying functions for visualization overlays
import cv2  # noqa: F401

# Reuse the existing functional implementation under the hood
from depth_based_detection import depth_based_edge_detection


@dataclass
class DepthDetectorConfig:
    MIN_DEPTH: float = 1.0
    MAX_DEPTH: float = 4.0
    DEPTH_RANGE: Tuple[float, float] = (1.7, 2.3)
    PHYSICAL_GRADIENT_THRESHOLD: float = 0.25

    # Parameters below exist in the functional pipeline but are currently
    # internal to helper functions; kept here for future consolidation.
    MIN_LINE_LENGTH: int = 50
    cluster_x_thresh: int = 10
    glass_width_cm: int = 40
    center_frame_width_cm: int = 30


@dataclass
class DepthDetectionResult:
    roi_left: Optional[np.ndarray]
    roi_right: Optional[np.ndarray]
    door_depth_m: Optional[float]
    sobel_vis_color: Optional[np.ndarray]


class DepthDoorDetector:
    """
    OO facade wrapping the depth-based door frame detection pipeline.
    Coordinates per-frame processing and encapsulates runtime configuration.
    """

    def __init__(self, config: DepthDetectorConfig):
        self.config = config
        # Future: add temporal buffers or caches if needed

    def process_frame(self, ctx) -> DepthDetectionResult:
        """
        Process a single frame using the existing functional pipeline.
        Updates `ctx.*` fields to preserve the current contract and
        returns a structured `DepthDetectionResult` for downstream use.
        """
        roi_left, roi_right, mean_z, sobel_vis_color = depth_based_edge_detection(
            ctx.depth_frame,
            ctx.depth_image_in_meters,
            ctx.color_image_depth_based,
            MIN_DEPTH=self.config.MIN_DEPTH,
            MAX_DEPTH=self.config.MAX_DEPTH,
            DEPTH_RANGE=self.config.DEPTH_RANGE,
            PHYSICAL_GRADIENT_THRESHOLD=self.config.PHYSICAL_GRADIENT_THRESHOLD,
        )

        door_depth_m = float(mean_z) if mean_z is not None else None

        # Maintain backward-compatible context updates
        ctx.sobel_vis_color = sobel_vis_color
        if roi_left is not None and roi_right is not None:
            ctx.roi_left_depth_based = roi_left
            ctx.roi_right_depth_based = roi_right
            ctx.door_depth_m_depth_based = door_depth_m

        return DepthDetectionResult(
            roi_left=roi_left,
            roi_right=roi_right,
            door_depth_m=door_depth_m,
            sobel_vis_color=sobel_vis_color,
        )
