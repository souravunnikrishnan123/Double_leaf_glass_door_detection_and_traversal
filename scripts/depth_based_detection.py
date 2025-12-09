#!/usr/bin/env python3
import rospy
from collections import deque
import time
import cv2  
import numpy as np 
import pyrealsense2 as rs


from cluster_and_merge_depth_based_lines import cluster_and_merge_lines
from get_depth_based_lines_using_sobel_and_hough_lines import get_depth_based_lines_using_sobel_and_hough_lines 
from find_glass_frame_lines import left_right_roi_and_door_depth

from duration import get_duration_seconds

confidence_history = deque()



def robust_line_z_roi(depth_image_in_meters, x1, y1, x2, y2, roi_width=10):
    """
    Estimates Z-depth of a detected line using rectangular ROIs to the left and right,
    using get_z_depth for accurate Z-axis depth.
    """
    if abs(x1 - x2) > abs(y1 - y2):
        print("Warning: Line is not primarily vertical. This method assumes vertical lines.")
        return None
    
    H, W = depth_image_in_meters.shape

    y_start, y_end = sorted((y1, y2))
    y_start= max(0, y_start)
    y_end = min(H, y_end)

    if y_end <= y_start:
        return None
    
    ys = np.arange(y_start, y_end)

    x_center = int((x1 + x2) / 2)

    x_left_roi_start = max(0, x_center - roi_width)
    x_left_roi_end = min(W, x_center)

    x_right_roi_start = max(0, x_center + 1)
    x_right_roi_end = min(W, x_center + roi_width + 1)

    # List to store valid Z-depths for each ROI
    z_depths1 = depth_image_in_meters[ys, x_left_roi_start:x_left_roi_end] if x_left_roi_end > x_left_roi_start else np.empty((0, 0), dtype=np.float32)
    z_depths2 = depth_image_in_meters[ys, x_right_roi_start:x_right_roi_end] if x_right_roi_end > x_right_roi_start else np.empty((0, 0), dtype=np.float32)   

    z_depths1 = z_depths1.ravel() #to convert to 1D array
    z_depths2 = z_depths2.ravel() #to convert to 1D array

    z_depths1 = z_depths1[np.isfinite(z_depths1) & (z_depths1 > 0)]
    z_depths2 = z_depths2[np.isfinite(z_depths2) & (z_depths2 > 0)]


    # Compute medians
    med_z_depth1 = np.median(z_depths1) if len(z_depths1) > 0 else 0
    med_z_depth2 = np.median(z_depths2) if len(z_depths2) > 0 else 0

    # Apply your filtering logic
    if med_z_depth1 == 0 and med_z_depth2 == 0:
        return None
    elif med_z_depth1 == 0:
        return med_z_depth2
    elif med_z_depth2 == 0:
        return med_z_depth1
    else:
        return min(med_z_depth1, med_z_depth2)




# -------------------- STEP 2: DEPTH GRADIENT + HOUGH (Z-Depth) --------------------
def depth_based_edge_detection( depth_image_in_meters, color_image, fx, MIN_DEPTH, MAX_DEPTH, DEPTH_RANGE , PHYSICAL_GRADIENT_THRESHOLD,
                               scale, hough_params, bilateral_params, sobel_params, adaptive_params, roi_width_for_depth_estimation, merging ,angle_threshold, door_geometry):

    timer = get_duration_seconds()
    timer.start("depth_based_edge_detection preprocessing")

    valid_lines = []
    depth_of_valid_lines = []

    depth_lines, sobel_vis_color = get_depth_based_lines_using_sobel_and_hough_lines(
        depth_image_in_meters,
        MIN_DEPTH,
        MAX_DEPTH,
        PHYSICAL_GRADIENT_THRESHOLD,
        scale,
        hough_params,
        bilateral_params,
        sobel_params,
        adaptive_params
    )

    timer.stop("depth_based_edge_detection preprocessing")

    timer.start("depth_based_edge_detection line processing")
    
    if depth_lines is not None:
        x1_np = depth_lines[:, 0, 0]
        y1_np = depth_lines[:, 0, 1]
        x2_np = depth_lines[:, 0, 2]
        y2_np = depth_lines[:, 0, 3]

        angles = np.arctan2(y2_np - y1_np, x2_np - x1_np)  # radians. 78°–102°
        vertical_mask = (np.abs(np.cos(angles)) < angle_threshold)  # near-vertical
        depth_lines = depth_lines[vertical_mask]  # only has near-vertical lines


        for line in depth_lines:
            x1, y1, x2, y2 = line[0]

            # Compute line angle
            #angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Keep only near-vertical lines
            #if 80 < abs(angle) < 100:
            # Estimate Z-depth of the line robustly
            d = robust_line_z_roi(depth_image_in_meters, x1, y1, x2, y2, roi_width=roi_width_for_depth_estimation)
            cv2.line(color_image, (x1, y1), (x2, y2), (203, 192, 255), 2)  #pink
            #x_m = int((x1 + x2) / 2)
            #y_m = int((y1 + y2) / 2)
            
            # Limit area for left and right
            #left_condition = (left_x - margin_to_the_glass_side <= x_avg <= left_x + margin_to_the_frame_side)
            #right_condition = (right_x - margin_to_the_frame_side <= x_avg <= right_x + margin_to_the_glass_side)


            # Draw only if within specified depth range
            #if (d is not None and DEPTH_RANGE[0] <= d <= DEPTH_RANGE[1] and (left_condition or right_condition)):
            if (d is not None and DEPTH_RANGE[0] <= d <= DEPTH_RANGE[1]):
                
                valid_lines.append((x1, y1, x2, y2))
                depth_of_valid_lines.append(d)
                # show the midpoint used for normals
                cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue for valid lines
                #cv2.circle(color_image, (x_m, y_m), 3, (255, 0, 0), -1)
                # optional annotate depth
                #cv2.putText(color_image, f"{d:.2f}m", (x_m+6, y_m-6),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    
        merged_lines, merged_lines_depths = cluster_and_merge_lines(valid_lines, depth_of_valid_lines, x_thresh=merging["x_threshold_to_merge_lines"])  # only merging lines that are vertical, valid, and within depth range

        
        filtered_merged_lines = []
        filtered_merged_lines_depths = []
        
        for line, depth in zip(merged_lines, merged_lines_depths):
            x1, y1, x2, y2 = line
            if abs(y2 - y1) >= merging["MIN_LINE_LENGTH_after_merging"]:
                filtered_merged_lines.append(((x1, y1), (x2, y2)))
                cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green for filtered merged lines
                filtered_merged_lines_depths.append(depth)
        
        

        timer.stop("depth_based_edge_detection line processing")


        roi_left, roi_right, mean_z = left_right_roi_and_door_depth( 
            depth_image_in_meters,
            color_image,
            fx,
            filtered_merged_lines,
            filtered_merged_lines_depths,
            door_geometry,
            keyword="depth"
        )
        return roi_left, roi_right, mean_z, sobel_vis_color
    
    return None, None, 0, None