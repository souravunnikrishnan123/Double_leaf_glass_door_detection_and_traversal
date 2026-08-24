#!/usr/bin/env python3
"""Depth-gradient branch of the glass-door frame detector.

Vertical Sobel edges are restricted to the confirmed door-depth band, detected
with a probabilistic Hough transform, and merged into longer frame candidates.
"""


import numpy as np
import rospy
import cv2


from duration import get_duration_seconds
from post_processing_of_detected_vertical_lines import cluster_and_merge_lines
from processing_classes import LineFilter, EdgeDetector, HoughPLineDetector, Preprocessor


class DepthDoorDetector:
    """
    Detect vertical door-frame lines from metric depth discontinuities.

    The detector smooths the depth map, computes its horizontal Sobel response,
    restricts gradients to the confirmed plane band, applies adaptive and
    physical thresholds, and merges nearby Hough segments.

    Attributes:
        ransac_error:
            Fractional tolerance applied around the latest plane distance.

        PHYSICAL_GRADIENT_THRESHOLD:
            Minimum absolute metric Sobel response accepted as a real depth
            discontinuity.

        scale:
            Processing scale for depth filtering, Sobel, and Hough detection.

        hough:
            Probabilistic Hough configuration.

        bilateral:
            Edge-preserving depth smoothing configuration.

        sobel:
            Sobel kernel configuration.

        adaptive:
            Statistical gradient-threshold configuration.

        angle_threshold:
            Maximum horizontal component accepted as a vertical segment.

        roi_width_for_depth_estimation:
            Width sampled on each side of a detected segment.

        merge_lines:
            Horizontal clustering and minimum merged-length configuration.

        line_filter:
            Orientation and depth validation helper.

        edge_detector:
            Sobel and adaptive-threshold helper.

        line_detector:
            Scale-aware Hough detector.

        preprocessor:
            Depth resizing and bilateral-filter helper.

        timer:
            Named pipeline-stage duration recorder.
    """

    def __init__(self):
        """
        Load depth-branch parameters and construct processing helpers.

        Hough, bilateral-filter, Sobel, geometric-line, and merge settings are
        read from ``~depth_image_based_door_detector``. The resulting helpers
        are retained so each frame can be processed without rebuilding filter
        objects.

        Notes:
            ROS must be initialized before construction so private parameters
            resolve in the intended node namespace.
        """
        ns = "~depth_image_based_door_detector"
        
        self.ransac_error = rospy.get_param(f"~plane_detector/output/ransac_error", 0.02)
        
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

        self.line_filter = LineFilter()
        self.edge_detector = EdgeDetector()
        self.line_detector = HoughPLineDetector()
        self.preprocessor = Preprocessor()

        #duration timer
        self.timer = get_duration_seconds()



    def process_frame(self, ctx):
        """
        Extract and merge vertical depth-edge candidates from one frame.

        Args:
            ctx:
                Shared frame context containing aligned metric depth and
                ``color_image_depth_based`` for diagnostic overlays.

        Returns:
            Tuple ``(merged_lines, merged_depths, sobel_visualization)``.
            Merged lines use endpoint-pair coordinates at original resolution.
            If Hough detects no line, the first two values are ``None``.

        Notes:
            The adaptive threshold is computed only from gradients whose depth
            lies inside the current plane-distance band. A second fixed
            physical threshold removes weak internal frame texture.
        """
        # Ensure the color image used for depth overlays matches the depth resolution.
        # If aligned depth resolution differs from RGB, resize the color image to depth size
        # so line and ROI coordinates derived from depth map align correctly.

        self.timer.start("depth_based_edge_detection preprocessing")

        # Re-read this value because the plane search can refine it between runs.
        ransac_plane_distance = rospy.get_param(f"~plane_detector/output/ransac_plane_distance", 2.0)
        DEPTH_RANGE = [ransac_plane_distance * (1 - self.ransac_error), ransac_plane_distance * (1 + self.ransac_error)]
    
        valid_lines = []
        depth_of_valid_lines = []
        H = ctx.depth_image_in_meters.shape[0]
        W = ctx.depth_image_in_meters.shape[1]


        # Bilateral smoothing removes speckle without washing out the depth jump
        # at a real frame edge.
        depth_scaled = self.preprocessor.resize_by_scale(ctx.depth_image_in_meters, self.scale)
        filtered_depth_image = self.preprocessor.bilateral_filter(depth_scaled, self.bilateral)        
        depth_grad_x = self.edge_detector.sobel_edge_detection(filtered_depth_image, self.sobel) # Gradient along X (detect vertical edges in depth)
        
        # Apply mask (keep only valid + relevant regions)
        # After computing depth_scaled
        # Learn the adaptive threshold from the door band only. Background
        # discontinuities can otherwise dominate its mean and standard deviation.
        mask_ds = (depth_scaled > DEPTH_RANGE[0]) & (depth_scaled < DEPTH_RANGE[1])
        depth_grad_x[~mask_ds] = 0
        valid_grad_vals = depth_grad_x[mask_ds]  #Only gradients at valid depth pixels are used to compute the threshold
        depth_edges = self.edge_detector.adaptive_threshold(depth_grad_x, valid_grad_vals, self.adaptive)
        
        # Normalize for visualization (convert to 8-bit image)
        sobel_vis = cv2.normalize(depth_grad_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Convert Sobel visualization to color (BGR)
        sobel_vis_color = cv2.cvtColor(sobel_vis, cv2.COLOR_GRAY2BGR)


        #physical gradient filter, to avoid detecting depth lines within the frame( with very low depth gradient)
        # but if there are depth hole within the frame, the depth gradient will be high, that case is not covered here
        # The statistical threshold follows scene noise; this fixed metric gate
        # separately rejects weak texture within the frame itself.
        physical_mask = (depth_grad_x > self.PHYSICAL_GRADIENT_THRESHOLD).astype(np.uint8)*255
        depth_edges = cv2.bitwise_and(depth_edges, physical_mask)

        # Overlay depth edges in red
        sobel_vis_color[depth_edges > 0] = [0, 0, 255]  # Red for edge pixels

        depth_lines = self.line_detector.detect(depth_edges, self.hough, self.scale)
        sobel_vis_color = self.preprocessor.restore_size(sobel_vis_color, W, H, self.scale)


        self.timer.stop("depth_based_edge_detection preprocessing")

        self.timer.start("depth_based_edge_detection line processing")
        
        if depth_lines is not None:
            depth_lines = self.line_filter.angle_filter(depth_lines, self.angle_threshold)


            for line in depth_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(ctx.color_image_depth_based, (x1, y1), (x2, y2), (203, 192, 255), 2)  #pink
                # Estimate Z-depth of the line robustly
                # Sample both sides rather than the edge itself, where depth is
                # often missing or mixed between foreground and background.
                line_depth = self.line_filter.get_median_depth_by_roi_around(ctx.depth_image_in_meters, line , self.roi_width_for_depth_estimation)
                
                if not self.line_filter.is_depth_valid(line_depth, DEPTH_RANGE):
                    continue

                valid_lines.append(((x1, y1), (x2, y2)))
                depth_of_valid_lines.append(line_depth)
                # show the midpoint used for normals
                cv2.line(ctx.color_image_depth_based, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue for valid lines
                #cv2.circle(color_image, (x_m, y_m), 3, (255, 0, 0), -1)
                # optional annotate depth
                #cv2.putText(color_image, f"{d:.2f}m", (x_m+6, y_m-6),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

            # Hough often breaks one upright into several short pieces; merging
            # gives the later pairing stage one candidate per physical edge.
            merged_lines, merged_lines_depths = cluster_and_merge_lines(ctx.color_image_depth_based, valid_lines, depth_of_valid_lines, x_thresh=self.merge_lines["x_threshold_to_merge_lines"], min_merged_line_length = self.merge_lines["MIN_LINE_LENGTH_after_merging"])  # only merging lines that are vertical, valid, and within depth range

            self.timer.stop("depth_based_edge_detection line processing")


        else:
            merged_lines = None
            merged_lines_depths = None

        return merged_lines, merged_lines_depths, sobel_vis_color
