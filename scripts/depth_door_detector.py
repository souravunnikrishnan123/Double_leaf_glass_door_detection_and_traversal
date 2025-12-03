#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import rospy

from depth_based_detection import depth_based_edge_detection



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

    def __init__(self):
        ns = "depth_image_based_door_detector"
        self.MIN_DEPTH = rospy.get_param(f"{ns}/MIN_DEPTH", 1.0)
        self.MAX_DEPTH = rospy.get_param(f"{ns}/MAX_DEPTH", 4.0)
        self.DEPTH_RANGE = rospy.get_param(f"{ns}/DEPTH_RANGE", [1.7, 2.3])
        self.PHYSICAL_GRADIENT_THRESHOLD = rospy.get_param(f"{ns}/PHYSICAL_GRADIENT_THRESHOLD", 0.25)
        self.scale = rospy.get_param(f"{ns}/scale", 0.5)
        self.hough = {
            "threshold": rospy.get_param(f"{ns}/hough/threshold", 100),
            "min_line_length": rospy.get_param(f"{ns}/hough/min_line_length", 100),
            "max_line_gap": rospy.get_param(f"{ns}/hough/max_line_gap", 30),
        }
        self.bilateral = {
            "d": rospy.get_param(f"{ns}/bilateral/d", 5),
            "sigma_color": rospy.get_param(f"{ns}/bilateral/sigma_color", 50),
            "sigma_space": rospy.get_param(f"{ns}/bilateral/sigma_space", 75),
        }
        self.sobel = {
            "ksize": rospy.get_param(f"{ns}/sobel/ksize", 3),
        }
        self.adaptive = {
            "k_factor": rospy.get_param(f"{ns}/adaptive/k_factor", 2.0),
            "fallback_threshold": rospy.get_param(f"{ns}/adaptive/fallback_threshold", 0.1),
        }
        self.angle_threshold = rospy.get_param(f"{ns}/angle_threshold", 0.2)
        self.roi_width_for_depth_estimation = rospy.get_param(f"{ns}/roi_width_for_depth_estimation", 10)
        self.merge_lines = {
            "x_threshold_to_merge_lines": rospy.get_param(f"{ns}/merge_lines/x_threshold_to_merge_lines", 10),
            "MIN_LINE_LENGTH_after_merging": rospy.get_param(f"{ns}/merge_lines/MIN_LINE_LENGTH_after_merging", 50),
        }
        self.door_geometry = {
            "glass_width_cm": rospy.get_param(f"/door_geometry/glass_width_cm", 40),
            "center_frame_width_cm": rospy.get_param(f"/door_geometry/center_frame_width_cm", 30),
            "roi_width": rospy.get_param(f"/door_geometry/roi_width", 240),
            "min_depth": rospy.get_param(f"/door_geometry/min_depth", 1.7),
            "correction_factor": rospy.get_param(f"/door_geometry/correction_factor", 1.1),
        }


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
            MIN_DEPTH=self.MIN_DEPTH,
            MAX_DEPTH=self.MAX_DEPTH,
            DEPTH_RANGE=self.DEPTH_RANGE,
            PHYSICAL_GRADIENT_THRESHOLD=self.PHYSICAL_GRADIENT_THRESHOLD,
            scale=self.scale,
            hough_params=self.hough,
            bilateral_params=self.bilateral,
            sobel_params=self.sobel,
            adaptive_params=self.adaptive,
            roi_width_for_depth_estimation=self.roi_width_for_depth_estimation,
            merging = self.merge_lines,
            angle_threshold= self.angle_threshold,
            door_geometry = self.door_geometry
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
