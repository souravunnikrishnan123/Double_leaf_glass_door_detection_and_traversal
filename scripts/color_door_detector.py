#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import rospy



from color_image_based_frame_detection import color_image_based_frame_detection


@dataclass
class ColorDetectionResult:
    roi_left: Optional[np.ndarray]
    roi_right: Optional[np.ndarray]
    door_depth_m: Optional[float]
    edges: Optional[np.ndarray]


class ColorDoorDetector:
    """
    OO facade for the color-based door frame detection pipeline.
    Wraps existing functional code and maintains FrameContext updates.
    """

    def __init__(self):
        ns = "~color_image_based_door_detector"
        self.DEPTH_RANGE = rospy.get_param(f"{ns}/depth_range", [1.7, 2.3])
        self.scale = rospy.get_param(f"{ns}/scale", 1.0)
        # Optional tunables for image processing
        self.canny = {
            "lower_factor": rospy.get_param(f"{ns}/canny/lower_factor", 0.7),
            "upper_factor": rospy.get_param(f"{ns}/canny/upper_factor", 2.0),
        }
        self.blur = {
            "ksize": rospy.get_param(f"{ns}/blur/ksize", 3),
            "sigma": rospy.get_param(f"{ns}/blur/sigma", 0.8),
        }
        self.hough = {
            "threshold": rospy.get_param(f"{ns}/hough/threshold", 100),
            "min_line_length": rospy.get_param(f"{ns}/hough/min_line_length", 100),
            "max_line_gap": rospy.get_param(f"{ns}/hough/max_line_gap", 20),
        }
        self.angle_threshold = rospy.get_param(f"{ns}/angle_threshold", 0.2)
        self.min_num_of_valid_depths_for_depth_estimation = rospy.get_param(f"{ns}/min_num_of_valid_depths_for_depth_estimation", 5)
        self.extrapolation = {
            "gradient_threshold_for_extrapolation": rospy.get_param(f"{ns}/extrapolation/gradient_threshold_for_extrapolation", 0.1),
            "window_size_for_extrapolation": rospy.get_param(f"{ns}/extrapolation/window_size_for_extrapolation", 5),
        }
        self.door_geometry = {
            "glass_width_cm": rospy.get_param(f"/door_geometry/glass_width_cm", 40),
            "center_frame_width_cm": rospy.get_param(f"/door_geometry/center_frame_width_cm", 30),
            "roi_width": rospy.get_param(f"/door_geometry/roi_width", 240),
            "min_depth": rospy.get_param(f"/door_geometry/min_depth", 1.7),
            "correction_factor": rospy.get_param(f"/door_geometry/correction_factor", 1.1),
        }

    def process_frame(self, ctx) -> ColorDetectionResult:
        roi_left, roi_right, mean_z, edges = color_image_based_frame_detection(
            ctx.depth_image_in_meters,
            ctx.color_image_color_based,
            ctx.fx,
            DEPTH_RANGE=self.DEPTH_RANGE,
            scale=self.scale,
            canny_params=self.canny,
            blur_params=self.blur,
            hough_params=self.hough,
            min_num_of_valid_depths_for_depth_estimation=self.min_num_of_valid_depths_for_depth_estimation,
            extrapolation=self.extrapolation,
            angle_threshold=self.angle_threshold,
            door_geometry=self.door_geometry
        )

        door_depth_m = float(mean_z) if mean_z is not None else None

        # Maintain backward-compatible context updates
        ctx.edges = edges
        if roi_left is not None and roi_right is not None:
            ctx.roi_left_color_based = roi_left
            ctx.roi_right_color_based = roi_right
            ctx.door_depth_m_color_based = door_depth_m

        return ColorDetectionResult(
            roi_left=roi_left,
            roi_right=roi_right,
            door_depth_m=door_depth_m,
            edges=edges,
        )
