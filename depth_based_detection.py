from collections import deque
import time
import cv2  
import numpy as np 
import pyrealsense2 as rs

from confidence_score_calculation import calculate_confidence_scores 
from cluster_and_merge_depth_based_lines import cluster_and_merge_lines
from find_frame_line_pair import filter_vertical_lines_glass_contact
from roi import process_filtered_lines
from sort_lines_by_y import sort_lines_by_y
from get_depth_based_lines_using_sobel_and_hough_lines import get_depth_based_lines_using_sobel_and_hough_lines 

from duration import get_duration_seconds

confidence_history = deque()



def robust_line_z_roi(depth_frame, x1, y1, x2, y2, roi_width=10):
    """
    Estimates Z-depth of a detected line using rectangular ROIs to the left and right,
    using get_z_depth for accurate Z-axis depth.
    """
    if abs(x1 - x2) > abs(y1 - y2):
        print("Warning: Line is not primarily vertical. This method assumes vertical lines.")
        return None
    depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0
    H, W = depth_m.shape

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
    z_depths1 = depth_m[ys, x_left_roi_start:x_left_roi_end] if x_left_roi_end > x_left_roi_start else np.empty((0, 0), dtype=np.float32)
    z_depths2 = depth_m[ys, x_right_roi_start:x_right_roi_end] if x_right_roi_end > x_right_roi_start else np.empty((0, 0), dtype=np.float32)   

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
def depth_based_edge_detection(depth_frame, depth_image_in_meters, color_image, MIN_DEPTH, MAX_DEPTH, DEPTH_RANGE , PHYSICAL_GRADIENT_THRESHOLD=0.25):

    timer1 = get_duration_seconds()

    valid_lines = []
    depth_of_valid_lines = []

    depth_lines, sobel_vis_color = get_depth_based_lines_using_sobel_and_hough_lines(depth_frame, MIN_DEPTH, MAX_DEPTH, PHYSICAL_GRADIENT_THRESHOLD)

    timer1.get_duration("depth_based_edge_detection preprocessing")
    timer2 = get_duration_seconds()

    if depth_lines is not None:
        x1_np = depth_lines[:, 0, 0]
        y1_np = depth_lines[:, 0, 1]
        x2_np = depth_lines[:, 0, 2]
        y2_np = depth_lines[:, 0, 3]

        angles = np.arctan2(y2_np - y1_np, x2_np - x1_np)  # radians. 78°–102°
        vertical_mask = (np.abs(np.cos(angles)) < 0.2)  # near-vertical
        depth_lines = depth_lines[vertical_mask]  # only has near-vertical lines


        for line in depth_lines:
            x1, y1, x2, y2 = line[0]

            # Compute line angle
            #angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Keep only near-vertical lines
            #if 80 < abs(angle) < 100:
            # Estimate Z-depth of the line robustly
            d = robust_line_z_roi(depth_frame, x1, y1, x2, y2, roi_width=20)
            cv2.line(color_image, (x1, y1), (x2, y2), (203, 192, 255), 2)  #pink
            x_m = int((x1 + x2) / 2)
            y_m = int((y1 + y2) / 2)
            
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
                cv2.circle(color_image, (x_m, y_m), 3, (255, 0, 0), -1)
                # optional annotate depth
                #cv2.putText(color_image, f"{d:.2f}m", (x_m+6, y_m-6),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    
    timer2.get_duration("depth_based_edge_detection line processing")
    timer3 = get_duration_seconds()
    merged_lines, merged_lines_depths = cluster_and_merge_lines(valid_lines, depth_of_valid_lines, x_thresh=10)  # only merging lines that are vertical, valid, and within depth range

    MIN_LINE_LENGTH = 50
    filtered_merged_lines = []
    filtered_merged_lines_depths = []

    for line, depth in zip(merged_lines, merged_lines_depths):
        x1, y1, x2, y2 = line
        if abs(y2 - y1) >= MIN_LINE_LENGTH:
            filtered_merged_lines.append(((x1, y1), (x2, y2)))
            filtered_merged_lines_depths.append(depth)

    paired_lines = filter_vertical_lines_glass_contact(
                        filtered_merged_lines, filtered_merged_lines_depths, depth_frame, glass_width_cm=40, center_frame_width_cm=30
                    )

    # Adjust all pairs so each line's points are sorted by y. so that gradient can be calculated correctly
    paired_lines_sorted = [
        (sort_lines_by_y(left_line), sort_lines_by_y(right_line), left_depth, right_depth)
        for left_line, right_line, left_depth, right_depth in paired_lines]
    

    # Visualize paired lines (glass frame candidates)
    for left_line, right_line, left_depth, right_depth in paired_lines_sorted:
        # Draw left line in red
        if len(left_line) >= 2:
            pt1 = tuple(map(int, left_line[0][:2]))
            pt2 = tuple(map(int, left_line[-1][:2]))
            cv2.line(color_image, pt1, pt2, (0, 0, 255), 2)
        # Draw right line in cyan
        if len(right_line) >= 2:
            pt1 = tuple(map(int, right_line[0][:2]))
            pt2 = tuple(map(int, right_line[-1][:2]))
            cv2.line(color_image, pt1, pt2, (0, 255, 255), 2)

    roi_polygon_left_list, roi_polygon_right_list, filtered_pairs , mean_z_depth_along_frame_lines_list = process_filtered_lines(paired_lines_sorted, depth_image_in_meters, color_image)


    if filtered_pairs: 
        if len(filtered_pairs) > 1:# more than one pair detected as glass frame. 
            #in this case we need to filter out the correct frame line at the center. if we are getting more than one glass -frame candidate means, mostly it is due to the glass area in the  inward opening door 
            #so in this case chances are high that ransac detected the whole door plane. hence we can use the width of detected ransac plane to filter out the correct frame line pair
            #correct frame line pair will be close to the center of detected ransac door plane
            # 1. Compute the center x of the detected plane (average of its 4 corners)
            #plane_center_x = np.mean([pt[0] for pt in detected_plane]) if detected_plane is not None else color_image.shape[1] // 2
            plane_center_x = color_image.shape[1] // 2 #since wer are stopping about 2m far from glass door plane. the center of entire view would be almost same as the center of glass door plane
            #this is to avoid dependancy to glass detection algorithm
            # 2. Find the pair whose center is closest to the plane center
            min_dist = float('inf')
            best_pair = None
            best_idx = None
            for i, (left_line, right_line, left_depth, right_depth) in enumerate(filtered_pairs):
                # Compute mean x of left and right line
                left_x = np.mean([pt[0] for pt in left_line])
                right_x = np.mean([pt[0] for pt in right_line])
                pair_center_x = (left_x + right_x) / 2
                dist = abs(pair_center_x - plane_center_x)
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (left_line, right_line)
                    best_idx = i
            
            idx = best_idx

        else:# filtered pairs has only one pair.
            best_pair = filtered_pairs[0]
            idx = 0


        final_left_frame_line = best_pair[0]
        final_right_frame_line = best_pair[1]
        roi_polygon_left = roi_polygon_left_list[idx]
        roi_polygon_right = roi_polygon_right_list[idx]
        mean_z_depth_along_frame_lines = mean_z_depth_along_frame_lines_list[idx]

        timer3.get_duration("depth_based_edge_detection pairing and roi processing")
        return roi_polygon_left, roi_polygon_right, mean_z_depth_along_frame_lines, sobel_vis_color
    else:
        return None, None, 0, None