#!/usr/bin/env python3
import rospy
import math
import cv2
import numpy as np



from get_color_based_lines_using_canny_and_hough_lines import get_rgb_based_lines_using_canny_and_hough_lines
from post_processing_of_detected_vertical_lines import get_median_depth_along_detected_line, extrapolate_along_line_segment
from duration import get_duration_seconds
from find_glass_frame_lines import left_right_roi_and_door_depth



def color_image_based_frame_detection(
    depth_image_in_meters,
    color_image,
    fx,
    DEPTH_RANGE,
    scale,
    canny_params,
    blur_params,
    hough_params,
    min_num_of_valid_depths_for_depth_estimation,
    extrapolation,
    angle_threshold,
    door_geometry
):
    # Detect vertical lines in color image using Canny + Hough
    vertical_lines = []
    depth_of_each_lines = []
    lines, edges = get_rgb_based_lines_using_canny_and_hough_lines(
        color_image, scale, canny_params=canny_params, blur_params=blur_params, hough_params=hough_params
    )

    timer = get_duration_seconds()
    timer.start("color_image_based_frame_detection line processing")

    if lines is not None:
        x1_np = lines[:, 0, 0]
        y1_np = lines[:, 0, 1]
        x2_np = lines[:, 0, 2]
        y2_np = lines[:, 0, 3]

        angles = np.arctan2(y2_np - y1_np, x2_np - x1_np)  # radians. 78°–102°
        vertical_mask = (np.abs(np.cos(angles)) < angle_threshold)  # near-vertical
        lines = lines[vertical_mask]  # only has near-vertical lines


        for line in lines:
            filtered_segment = []
            x1, y1, x2, y2 = line[0]
            #angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            #cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # All lines: blue
            #if 80 < abs(angle) < 100:  # near-vertical
            cv2.line(color_image, (x1, y1), (x2, y2), (0, 165, 255), 2)  # All vertical lines: orange


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

            center_depth = get_median_depth_along_detected_line(depth_image_in_meters, x1, y1, x2, y2, num_samples=num_samples, min_num_of_valid_depths=min_num_of_valid_depths_for_depth_estimation)
            # Compute median depth and center
            
            if center_depth is None:
                continue  # Skip line if no valid depth
            if not (DEPTH_RANGE[0] <= center_depth <= DEPTH_RANGE[1]):
                continue  # Still skip if out of expected depth range
            
            filtered_segment.append((x1, y1))
            filtered_segment.append((x2, y2))
            
            # Draw vertical_lines in cyan
            
            cv2.line(color_image, (x1, y1), (x2, y2), (255, 255, 0), 2)  # cyan

        
            # Use first and last points of filtered segment
            start_fwd = filtered_segment[-1] # take only x and y coordinate. donot take depth
            start_back = filtered_segment[0]

            # Compute direction vector of the line (normalized)
            dx = x2 - x1
            dy = y2 - y1
            norm = np.hypot(dx, dy)
            dx /= norm
            dy /= norm
            
            # Extrapolate forward
            
            extrapolated_forward = extrapolate_along_line_segment(
                depth_image_in_meters, start_fwd, (dx, dy), center_depth, gradient_threshold=extrapolation["gradient_threshold_for_extrapolation"], window=extrapolation["window_size_for_extrapolation"]
            )

            # Extrapolate backward
            extrapolated_backward = extrapolate_along_line_segment(
                depth_image_in_meters, start_back, (-dx, -dy), center_depth, gradient_threshold=extrapolation["gradient_threshold_for_extrapolation"], window=extrapolation["window_size_for_extrapolation"]
            )

            # Combine all
            full_line_segment = extrapolated_backward[::-1] + filtered_segment + extrapolated_forward
            
            # For simplicity, just use filtered_segment as full_line_segment for now
            full_line_segment = filtered_segment
            

            if len(full_line_segment) >= 2:
                
                cv2.line(color_image, full_line_segment[0], full_line_segment[-1], (0, 255, 0), 2)  # Green
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

        roi_left, roi_right, mean_z = left_right_roi_and_door_depth(
            depth_image_in_meters,
            color_image,
            fx,
            vertical_lines,
            depth_of_each_lines,
            door_geometry,
            keyword="color"
        )
        return roi_left, roi_right, mean_z, edges
    # No lines detected; return consistent 4-tuple and preserve edges
    return None, None, None, edges

            
            