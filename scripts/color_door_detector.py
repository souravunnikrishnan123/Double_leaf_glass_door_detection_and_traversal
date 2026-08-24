#!/usr/bin/env python3
"""Color-edge branch of the glass-door frame detector.

The branch finds near-vertical Canny/Hough segments, rejects segments whose
depth is inconsistent with the confirmed door plane, and returns candidates
for the shared glass-frame pairing stage.
"""

import cv2
import rospy
from duration import get_duration_seconds
from processing_classes import LineFilter, EdgeDetector, HoughPLineDetector, Preprocessor


class ColorDoorDetector:
    """
    Detect depth-consistent vertical door-frame lines from a color image.

    The detector equalizes and smooths grayscale intensity, runs Canny and a
    probabilistic Hough transform, keeps nearly vertical segments, and rejects
    segments whose sampled metric depth is outside the confirmed plane band.

    Attributes:
        ransac_error:
            Fractional tolerance applied around the latest plane distance.

        scale:
            Image scale used for edge and Hough processing.

        canny:
            Median-based Canny threshold factors.

        blur:
            Gaussian kernel and sigma configuration.

        hough:
            Probabilistic Hough configuration in original-image pixels.

        angle_threshold:
            Maximum horizontal direction component accepted as vertical.

        min_num_of_valid_depths_for_depth_estimation:
            Minimum finite line samples required for a median depth.

        extrapolation:
            Depth-gradient and smoothing settings retained for line extension.

        preprocessor:
            Shared image preprocessing helper.

        edge_detector:
            Canny/Sobel edge helper.

        line_detector:
            Scale-aware Hough segment detector.

        line_filter:
            Orientation and depth validation helper.

        timer:
            Named pipeline-stage duration recorder.
    """

    def __init__(self):
        """
        Load color-branch configuration and construct processing helpers.

        Notes:
            ``~plane_detector/output/ransac_error`` is required without a
            fallback value. The latest plane distance itself is read for every
            frame because the plane state may update it at runtime.
        """
        ns = "~color_image_based_door_detector"

        self.ransac_error = rospy.get_param("~plane_detector/output/ransac_error")

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

        # Compose strategy components
        self.preprocessor = Preprocessor()
        self.edge_detector = EdgeDetector()
        self.line_detector = HoughPLineDetector()
        self.line_filter = LineFilter()
        
        # duration timer
        self.timer = get_duration_seconds()
        

    def process_frame(self, ctx):
        """
        Extract depth-consistent vertical line candidates from one frame.

        Canny and Hough operate on a scaled branch image. Hough endpoints are
        restored to original coordinates before aligned depth is sampled along
        each segment.

        Args:
            ctx:
                Shared frame context containing ``color_image_color_based``,
                aligned metric depth, and the current camera-frame artifacts.

        Returns:
            Tuple ``(valid_lines, line_depths, edges)``. Lines use
            ``((x1, y1), (x2, y2))`` coordinates at original resolution.
            When Hough finds no lines, the first two values are ``None``.

        Notes:
            The method draws all vertical candidates in orange and accepted
            depth-consistent lines in cyan on ``ctx.color_image_color_based``.
        """
        self.timer.start("get_rgb_based_lines_using_canny_and_hough_lines")
        # The plane detector is the range gate for this branch. Without it,
        # vertical edges from walls and furniture would look like frame pieces.
        ransac_plane_distance = rospy.get_param("~plane_detector/output/ransac_plane_distance")
        DEPTH_RANGE = [ransac_plane_distance * (1 - self.ransac_error), ransac_plane_distance * (1 + self.ransac_error)]
 
        # Detect vertical lines in color image using Canny + Hough
        valid_lines = []
        depth_of_valid_lines = []
        H = ctx.color_image_color_based.shape[0]
        W = ctx.color_image_color_based.shape[1]


        # Edge extraction may run on a smaller image, but all returned lines are
        # restored to full resolution before they are paired with depth pixels.
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
                # Median depth tolerates the holes and bright outliers commonly
                # produced when the sensor looks through or reflects from glass.
                # Estimate Z-depth of the line robustly
                line_depth = self.line_filter.get_median_depth_along_line(ctx.depth_image_in_meters, line , self.min_num_of_valid_depths_for_depth_estimation)
                # Compute median depth
                
                # A strong RGB edge is useful only if it belongs to the door's range band.
                if not self.line_filter.is_depth_valid(line_depth, DEPTH_RANGE):
                    continue  # skip invalid depth lines

                valid_lines.append(((x1, y1), (x2, y2)))
                depth_of_valid_lines.append(line_depth)
                 
                cv2.line(ctx.color_image_color_based, (x1, y1), (x2, y2), (255, 255, 0), 2)  # cyan

            self.timer.stop("color_image_based_frame_detection line processing")


        # No lines detected; return consistent 4-tuple and preserve edges
        else:
            valid_lines = None
            depth_of_valid_lines = None

        return valid_lines, depth_of_valid_lines, edges
