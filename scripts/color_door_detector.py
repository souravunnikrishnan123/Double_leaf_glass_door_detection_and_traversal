#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
import rospy
from post_processing_of_detected_vertical_lines import extrapolate_along_line_segment
from duration import get_duration_seconds
from find_glass_frame_lines import GlassFrameLineProcessor
from processing_classes import LineFilter, EdgeDetector, HoughPLineDetector, Preprocessor





@dataclass
class ColorDetectionResult:
    roi_left: Optional[np.ndarray]
    roi_right: Optional[np.ndarray]
    door_depth_m: Optional[float]
    edges: Optional[np.ndarray]



class LineExtender:
    def extend(
        self,
        depth_image_in_meters: np.ndarray,
        start_fwd: Tuple[int, int],
        start_back: Tuple[int, int],
        direction: Tuple[float, float],
        center_depth: float,
        gradient_threshold: float,
        window: int,
    ) -> Tuple[list, list, list]:
        dx, dy = direction
        extrapolated_forward = extrapolate_along_line_segment(
            depth_image_in_meters,
            start_fwd,
            (dx, dy),
            center_depth,
            gradient_threshold=gradient_threshold,
            window=window,
        )
        extrapolated_backward = extrapolate_along_line_segment(
            depth_image_in_meters,
            start_back,
            (-dx, -dy),
            center_depth,
            gradient_threshold=gradient_threshold,
            window=window,
        )
        full_segment = extrapolated_backward[::-1] + [start_back, start_fwd] + extrapolated_forward
        return extrapolated_backward, extrapolated_forward, full_segment


class ColorDoorDetector:
    """
    OO facade for the color-based door frame detection pipeline.
    Wraps existing functional code and maintains FrameContext updates.
    """

    def __init__(self):
        ns = "~color_image_based_door_detector"

        self.ransac_error = rospy.get_param(f"~plane_detector/output/ransac_error")

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
        gns = "~door_geometry"
        self.door_geometry = {
            "glass_width_cm": rospy.get_param(f"{gns}/glass_width_cm", 40),
            "center_frame_width_cm": rospy.get_param(f"{gns}/center_frame_width_cm", 30),
            "roi_width": rospy.get_param(f"{gns}/roi_width", 240),
            "correction_factor": rospy.get_param(f"{gns}/correction_factor", 1.1),
        }
        # Compose strategy components
        self.preprocessor = Preprocessor()
        self.edge_detector = EdgeDetector()
        self.line_detector = HoughPLineDetector()
        self.line_filter = LineFilter()
        self.line_extender = LineExtender()
        self.glass_frame_detector = GlassFrameLineProcessor()
        
        # duration timer
        self.timer = get_duration_seconds()
        

    def process_frame(self, ctx) -> ColorDetectionResult:
        self.timer.start("get_rgb_based_lines_using_canny_and_hough_lines")
        ransac_plane_distance = rospy.get_param(f"~plane_detector/output/ransac_plane_distance")
        DEPTH_RANGE = [ransac_plane_distance * (1 - self.ransac_error), ransac_plane_distance * (1 + self.ransac_error)]
 
        # Detect vertical lines in color image using Canny + Hough
        valid_lines = []
        depth_of_valid_lines = []
        H = ctx.color_image_color_based.shape[0]
        W = ctx.color_image_color_based.shape[1]


        color_image_scaled = self.preprocessor.resize_by_scale(ctx.color_image_color_based, self.scale)
        filtered_grey_image = self.preprocessor.gaussian_blur_filter(color_image_scaled, self.blur)
        edges_scaled = self.edge_detector.canny_edge_detection(filtered_grey_image, self.canny)
        color_lines = self.line_detector.detect(edges_scaled, self.hough, self.scale)
        edges = self.preprocessor.restore_size(edges_scaled, W, H, self.scale)
        

        self.timer.stop("get_rgb_based_lines_using_canny_and_hough_lines")
        

        self.timer.start("color_image_based_frame_detection line processing")

        if color_lines is not None:
            color_lines = self.line_filter.angle_filter(color_lines, self.angle_threshold)


            for line in color_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(ctx.color_image_color_based, (x1, y1), (x2, y2), (0, 165, 255), 2)  # All vertical lines: orange
                # Estimate Z-depth of the line robustly
                line_depth = self.line_filter.get_median_depth_along_line(ctx.depth_image_in_meters, line , self.min_num_of_valid_depths_for_depth_estimation)
                # Compute median depth
                
                if not self.line_filter.is_depth_valid(line_depth, DEPTH_RANGE):
                    continue  # skip invalid depth lines

                valid_lines.append(((x1, y1), (x2, y2)))
                depth_of_valid_lines.append(line_depth)
                 
                cv2.line(ctx.color_image_color_based, (x1, y1), (x2, y2), (255, 255, 0), 2)  # cyan

            self.timer.stop("color_image_based_frame_detection line processing")


            roi_left, roi_right, mean_z = self.glass_frame_detector.find_left_right_roi_and_door_depth(
                ctx.depth_image_in_meters,
                ctx.color_image_color_based,
                ctx.fx,
                valid_lines,
                depth_of_valid_lines,
                self.door_geometry,
                DEPTH_RANGE,
                keyword="depth"
            )


        # No lines detected; return consistent 4-tuple and preserve edges
        else:
            roi_left = None
            roi_right = None
            mean_z = None
            

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
