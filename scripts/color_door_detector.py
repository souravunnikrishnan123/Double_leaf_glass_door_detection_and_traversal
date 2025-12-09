#!/usr/bin/env python3
from dataclasses import dataclass
import math
from typing import Optional, Tuple
import cv2
import numpy as np
import rospy
from post_processing_of_detected_vertical_lines import get_median_depth_along_detected_line, extrapolate_along_line_segment
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
        # Compose strategy components
        self.preprocessor = Preprocessor()
        self.edge_detector = EdgeDetector()
        self.line_detector = HoughPLineDetector()
        self.line_filter = LineFilter()
        self.line_extender = LineExtender()
        self.glass_frame_detector = GlassFrameLineProcessor()


    def process_frame(self, ctx) -> ColorDetectionResult:
        # Detect vertical lines in color image using Canny + Hough
        vertical_lines = []
        depth_of_each_lines = []

        timer = get_duration_seconds()
        timer.start("get_rgb_based_lines_using_canny_and_hough_lines")
        H = ctx.color_image_color_based.shape[0]
        W = ctx.color_image_color_based.shape[1]

        color_image_scaled = self.preprocessor.resize_by_scale(ctx.color_image_color_based, self.scale)
        filtered_grey_image = self.preprocessor.gaussian_blur_filter(color_image_scaled, self.blur)
        edges_scaled = self.edge_detector.canny_edge_detection(filtered_grey_image, self.canny)
        lines = self.line_detector.detect(edges_scaled, self.hough, self.scale)
        edges = self.preprocessor.restore_size(edges_scaled, W, H, self.scale)
        

        timer.stop("get_rgb_based_lines_using_canny_and_hough_lines")
        

        timer = get_duration_seconds()
        timer.start("color_image_based_frame_detection line processing")

        if lines is not None:
            lines = self.line_filter.angle_filter(lines, self.angle_threshold)


            for line in lines:
                filtered_segment = []
                x1, y1, x2, y2 = line[0]
                #angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                #cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # All lines: blue
                #if 80 < abs(angle) < 100:  # near-vertical
                cv2.line(ctx.color_image_color_based, (x1, y1), (x2, y2), (0, 165, 255), 2)  # All vertical lines: orange


                #center_depth = get_median_depth_window(depth_frame, x_center, y_center, window=10)
                #if center_depth is None:
                    #center_depth = get_median_depth_along_line(depth_frame, x_center, y1, y2)

                # Extract smooth portion along detected Hough line
                pixel_length = math.hypot(x2 - x1, y2 - y1)
                num_samples = int(pixel_length)
                """
                filtered_segment,center_depth = extract_smooth_line_segment_with_moving_avg(
                    depth_frame, x1, y1, x2, y2,
                    gradient_threshold=0.1, window=10, num_samples=num_samples
                ) 
                """

                center_depth = get_median_depth_along_detected_line(ctx.depth_image_in_meters, x1, y1, x2, y2, num_samples=num_samples, min_num_of_valid_depths = self.min_num_of_valid_depths_for_depth_estimation)
                # Compute median depth and center
                
                if not self.line_filter.is_depth_valid(center_depth, (self.DEPTH_RANGE[0], self.DEPTH_RANGE[1])):
                    continue
                
                filtered_segment.append((x1, y1))
                filtered_segment.append((x2, y2))
                
                # Draw vertical_lines in cyan
                
                cv2.line(ctx.color_image_color_based, (x1, y1), (x2, y2), (255, 255, 0), 2)  # cyan

            
                # Use first and last points of filtered segment
                start_fwd = filtered_segment[-1] # take only x and y coordinate. donot take depth
                start_back = filtered_segment[0]

                # Compute direction vector of the line (normalized)
                dx = x2 - x1
                dy = y2 - y1
                norm = np.hypot(dx, dy)
                if norm == 0:
                    continue
                dx /= norm
                dy /= norm
                
                # Extrapolate forward/backward (kept for parity; result unused)
                _, _, full_line_segment = self.line_extender.extend(
                    ctx.depth_image_in_meters,
                    start_fwd,
                    start_back,
                    (dx, dy),
                    center_depth,
                    self.extrapolation["gradient_threshold_for_extrapolation"],
                    self.extrapolation["window_size_for_extrapolation"],
                )

                # Combine all
                #full_line_segment = extrapolated_backward[::-1] + filtered_segment + extrapolated_forward
                
                # For simplicity, just use filtered_segment as full_line_segment for now
                full_line_segment = filtered_segment
                

                if len(full_line_segment) >= 2:
                    
                    cv2.line(ctx.color_image_color_based, full_line_segment[0], full_line_segment[-1], (0, 255, 0), 2)  # Green
                    vertical_lines.append(full_line_segment)
                    depth_of_each_lines.append(center_depth)
                
                """
                if len(filtered_segment) >= 2:
                    pt1 = tuple(map(int, filtered_segment[0][:2]))
                    pt2 = tuple(map(int, filtered_segment[-1][:2]))
                    cv2.line(color_image, pt1, pt2, (255, 255, 0), 2)  # Cyan
                
                if len(full_line_segment) >= 2:
                    pt1 = tuple(map(int, full_line_segment[0][:2]))
                    pt2 = tuple(map(int, full_line_segment[-1][:2]))
                    cv2.line(color_image, pt1, pt2, (255, 0, 255), 2)  # Magenta
                

                # Append clipped vertical line
                MIN_LINE_LENGTH = 50  # Minimum number of points required. because otherwise a small line segment
                #on the frame ( which is clipped by the previous logic) will be still considered as valid line and cause issue with detection

                if len(full_line_segment) >= MIN_LINE_LENGTH:
                    vertical_lines.append(full_line_segment)
                    depth_of_each_lines.append(center_depth)
                    # Draw vertical_lines in green
                    pt1 = tuple(map(int, full_line_segment[0][:2]))
                    pt2 = tuple(map(int, full_line_segment[-1][:2]))
                    cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)  # Green
                    #print(f"filtered_segment depth {center_depth:.2f}m")
                """
            
            
            # there were some problem with stable_lines calculation. it was not working properly. so commenting it out for now
            # Instead, we will just use vertical_lines directly for pairing
            #update_line_history(vertical_lines, line_history, DISTANCE_THRESHOLD, MAX_LINES_TO_TRACK, color_image)


            # Pass only line points to filter function
            #stable_line_points = [line_pts for avg_x, line_pts, confidence in stable_lines]
            
            # Visualize stable_line_points (lines passed to filter_vertical_lines_glass_contact)
            #for line_pts in stable_line_points:
                #if len(line_pts) >= 2:
                    #pt1 = tuple(map(int, line_pts[0][:2]))
                    #pt2 = tuple(map(int, line_pts[-1][:2]))
                    #cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)  # Green for stable lines
            timer.stop("color_image_based_frame_detection line processing")

            roi_left, roi_right, mean_z = self.glass_frame_detector.find_left_right_roi_and_door_depth(
                ctx.depth_image_in_meters,
                ctx.color_image_color_based,
                ctx.fx,
                vertical_lines,
                depth_of_each_lines,
                self.door_geometry,
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
