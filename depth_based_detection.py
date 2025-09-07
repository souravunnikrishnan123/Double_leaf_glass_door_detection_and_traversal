from collections import deque
import time
import cv2  
import numpy as np 
import pyrealsense2 as rs

from confidence_score_calculation import calculate_confidence_scores 
from cluster_and_merge_depth_based_lines import cluster_and_merge_lines
from get_z_depth import get_z_depth


confidence_history = deque()


def robust_line_z_roi(depth_frame, x1, y1, x2, y2, roi_width=10):
    """
    Estimates Z-depth of a detected line using rectangular ROIs to the left and right,
    using get_z_depth for accurate Z-axis depth.
    """
    if abs(x1 - x2) > abs(y1 - y2):
        print("Warning: Line is not primarily vertical. This method assumes vertical lines.")
        return None

    y_start = min(y1, y2)
    y_end = max(y1, y2)
    x_center = int((x1 + x2) / 2)


    # List to store valid Z-depths for each ROI
    z_depths1, z_depths2 = [], []

    # Iterate through the height of the line to sample points for ROI 1
    for y in range(y_start, y_end):
        for x_offset in range(-roi_width, 0):
            x_pixel = x_center + x_offset
            z = get_z_depth(depth_frame, x_pixel, y)
            if z is not None:
                z_depths1.append(z)

    # Iterate through the height of the line to sample points for ROI 2
    for y in range(y_start, y_end):
        for x_offset in range(1, roi_width + 1):
            x_pixel = x_center + x_offset
            z = get_z_depth(depth_frame, x_pixel, y)
            if z is not None:
                z_depths2.append(z)

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

    


def robust_line_z_roi_new(z_depth_map, x1, y1, x2, y2, roi_width=20, min_valid=0):
    """
    Estimates Z-depth of a detected line using rectangular ROIs on both sides,
    sampling from z_depth_map instead of expensive get_z_depth().
    
    Args:
        z_depth_map: precomputed Z-depth (H x W) array in meters
        x1, y1, x2, y2: line endpoints
        roi_width: how many pixels to sample to the left/right of line
        min_valid: minimum number of valid samples needed to compute median
    
    Returns:
        median Z-depth (float) or None if not enough valid samples
    """
    depth_height, depth_width = z_depth_map.shape

    # Only makes sense for vertical-ish lines
    if abs(x1 - x2) > abs(y1 - y2):
        print("Warning: Line is not primarily vertical.")
        return None

    y_start, y_end = min(y1, y2), max(y1, y2)
    x_center = int((x1 + x2) / 2)

    z_depths1, z_depths2 = [], []

    # ROI 1: left side
    for y in range(y_start, y_end):
        for x_offset in range(-roi_width, 0):
            x_pixel = x_center + x_offset
            if 0 <= x_pixel < depth_width and 0 <= y < depth_height:
                z = z_depth_map[y, x_pixel]
                if np.isfinite(z):
                    z_depths1.append(z)

    # ROI 2: right side
    for y in range(y_start, y_end):
        for x_offset in range(1, roi_width + 1):
            x_pixel = x_center + x_offset
            if 0 <= x_pixel < depth_width and 0 <= y < depth_height:
                z = z_depth_map[y, x_pixel]
                if np.isfinite(z):
                    z_depths2.append(z)

    # Compute medians (only if enough valid points)
    med_z1 = np.median(z_depths1) if len(z_depths1) >= min_valid else None
    med_z2 = np.median(z_depths2) if len(z_depths2) >= min_valid else None

    # Filtering logic
    if med_z1 is None and med_z2 is None:
        return None
    elif med_z1 is None:
        return med_z2
    elif med_z2 is None:
        return med_z1
    else:
        return min(med_z1, med_z2)  # always take nearer surface




# -------------------- STEP 2: DEPTH GRADIENT + HOUGH (Z-Depth) --------------------
def depth_based_edge_detection(depth_frame, color_image, MIN_DEPTH, MAX_DEPTH, DEPTH_RANGE , PHYSICAL_GRADIENT_THRESHOLD=0.25, final_left_frame_line=None, final_right_frame_line=None):

    depth_height, depth_width = depth_frame.height, depth_frame.width


    # Build Z-depth map
    z_depth_map = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0


    # Global validity check
    # Filter out sensor invalid values (too close, too far, nan/0)
    valid_mask = (z_depth_map > MIN_DEPTH) & (z_depth_map < MAX_DEPTH)

    # Detect strong edges between ~door distance (1.8–2.2 m) and background (0 or >2.5 m)
    roi_mask = ((z_depth_map > 1.8) & (z_depth_map < 2.2)) | (z_depth_map == 0) | (z_depth_map > 2.5)

    # Combine masks
    final_mask = valid_mask & roi_mask

    # Smooth Z-depth (bilateral preserves edges better than Gaussian)
    z_depth_filtered = cv2.bilateralFilter(z_depth_map, d=7, sigmaColor=50, sigmaSpace=75)

    # Gradient along X (detect vertical edges in depth)
    depth_grad_x = cv2.Sobel(z_depth_filtered, cv2.CV_32F, 1, 0, ksize=5)
    depth_grad_x = np.abs(depth_grad_x)
    # Apply mask (keep only valid + relevant regions)
    depth_grad_x[~final_mask] = 0 

    # Normalize for visualization (convert to 8-bit image)
    sobel_vis = cv2.normalize(depth_grad_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    # Convert Sobel visualization to color (BGR)
    sobel_vis_color = cv2.cvtColor(sobel_vis, cv2.COLOR_GRAY2BGR)


    # Adaptive threshold: mean + k*std of valid gradients
    valid_grad_vals = depth_grad_x[final_mask]  #Only gradients at valid depth pixels are used to compute the threshold
    if len(valid_grad_vals) > 0:
        mean_val = np.mean(valid_grad_vals)
        std_val = np.std(valid_grad_vals)
        k = 2.0  # tune sensitivity
        thresh_val = mean_val + k * std_val
    else:
        thresh_val = 0.1  # fallback threshold

    _, depth_edges = cv2.threshold(depth_grad_x, thresh_val, 255, cv2.THRESH_BINARY)
    depth_edges = depth_edges.astype(np.uint8)

    #physical gradient filter, to avoid detecting depth lines within the frame( with very low depth gradient)
    physical_mask = (depth_grad_x > PHYSICAL_GRADIENT_THRESHOLD).astype(np.uint8)*255
    depth_edges = cv2.bitwise_and(depth_edges, physical_mask)

    # Overlay depth edges in red
    sobel_vis_color[depth_edges > 0] = [0, 0, 255]  # Red for edge pixels
    cv2.imshow("Sobel + Depth Edges", sobel_vis_color)

    # Hough Transform on depth-based vertical edges
    depth_lines = cv2.HoughLinesP(depth_edges, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=30)

    # Compute average x for frame lines
    left_x = np.mean([pt[0] for pt in final_left_frame_line])
    right_x = np.mean([pt[0] for pt in final_right_frame_line])
    margin_to_the_frame_side = 5  # pixels
    margin_to_the_glass_side = 20  # pixels

    valid_lines = []
    if depth_lines is not None:
        for line in depth_lines:
            x1, y1, x2, y2 = line[0]

            x_avg = (x1 + x2) / 2

            # Compute line angle
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Keep only near-vertical lines
            if 80 < abs(angle) < 100:
                # Estimate Z-depth of the line robustly
                d = robust_line_z_roi(depth_frame, x1, y1, x2, y2, roi_width=20)
                #cv2.line(color_image, (x1, y1), (x2, y2), (203, 192, 255), 2)  #pink
                x_m = int((x1 + x2) / 2)
                y_m = int((y1 + y2) / 2)
                
                # Limit area for left and right
                left_condition = (left_x - margin_to_the_glass_side <= x_avg <= left_x + margin_to_the_frame_side)
                right_condition = (right_x - margin_to_the_frame_side <= x_avg <= right_x + margin_to_the_glass_side)


                # Draw only if within specified depth range
                if (d is not None and DEPTH_RANGE[0] <= d <= DEPTH_RANGE[1] and (left_condition or right_condition)):
                    #cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 255), 2)  # magenta
                    valid_lines.append((x1, y1, x2, y2))
                    # show the midpoint used for normals

                    cv2.circle(color_image, (x_m, y_m), 3, (255, 0, 0), -1)
                    # optional annotate depth
                    #cv2.putText(color_image, f"{d:.2f}m", (x_m+6, y_m-6),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    
    merged_lines = cluster_and_merge_lines(valid_lines)  # only merging lines that are vertical, valid, and within depth range
    if merged_lines is not None and len(merged_lines) > 0:
        scored_lines = []
        for line in merged_lines:
            x1, y1, x2, y2 = line
            cv2.line(color_image, (x1, y1), (x2, y2), (0, 165, 255), 2)#orange for merged lines
            num_samples = 10
            scored_lines.append(calculate_confidence_scores(line,depth_grad_x, depth_height, depth_width, color_image, num_samples))

            top_lines = []
            if len(scored_lines) > 0:
                # Sort lines by confidence score (descending)
                top_lines = sorted(scored_lines, key=lambda x: x[1], reverse=True)[:2]
                for line_info in top_lines:
                    x1, y1, x2, y2 = line_info[0]
                    cv2.line(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for top 2 lines

                # (Optional) Update confidence history with the highest score
                now = time.time()
                confidence_history.append((now, top_lines[0][1]))
                while confidence_history and now - confidence_history[0][0] > 10:
                    confidence_history.popleft()
                if confidence_history:
                    avg_conf = np.mean([score for _, score in confidence_history])
                    #print(f"[DEBUG] Avg confidence score (last 10s): {avg_conf:.3f}")
                else:
                    print("[DEBUG] No confidence scores in last 10 seconds.")
    else:
        print("[DEBUG] No depth-based Hough lines found.")
    
    cv2.imshow("Main Output", color_image)






def depth_based_edge_detection_within_rgb_based_frame_lines_roi(depth_np_array, color_image, MIN_DEPTH, MAX_DEPTH, DEPTH_RANGE, PHYSICAL_GRADIENT_THRESHOLD=0.25):

    depth_height, depth_width = depth_np_array.shape[:2]


    # Build Z-depth map
    z_depth_map = depth_np_array.astype(np.float32) / 1000.0
    

    # Global validity check
    # Filter out sensor invalid values (too close, too far, nan/0)
    valid_mask = (z_depth_map > MIN_DEPTH) & (z_depth_map < MAX_DEPTH)

    # Detect strong edges between ~door distance (1.8–2.2 m) and background (0 or >2.5 m)
    roi_mask = ((z_depth_map > 1.8) & (z_depth_map < 2.2)) | (z_depth_map == 0) | (z_depth_map > 2.5)

    # Combine masks
    final_mask = valid_mask & roi_mask

    # Smooth Z-depth (bilateral preserves edges better than Gaussian)
    z_depth_filtered = cv2.bilateralFilter(z_depth_map, d=7, sigmaColor=50, sigmaSpace=75)

    # Gradient along X (detect vertical edges in depth)
    depth_grad_x = cv2.Sobel(z_depth_filtered, cv2.CV_32F, 1, 0, ksize=5)
    depth_grad_x = np.abs(depth_grad_x)
    depth_grad_x[~final_mask] = 0  #Only gradients at valid depth pixels are used to compute the threshold

    # Normalize for visualization (convert to 8-bit image)
    sobel_vis = cv2.normalize(depth_grad_x, None, 0, 255, cv2.NORM_MINMAX)
    sobel_vis = sobel_vis.astype(np.uint8)

    # Convert Sobel visualization to color (BGR)
    sobel_vis_color = cv2.cvtColor(sobel_vis, cv2.COLOR_GRAY2BGR)


    # Adaptive threshold: mean + k*std of valid gradients
    valid_grad_vals = depth_grad_x[final_mask]  #Only gradients at valid depth pixels are used to compute the threshold
    if len(valid_grad_vals) > 0:
        mean_val = np.mean(valid_grad_vals)
        std_val = np.std(valid_grad_vals)
        k = 1.0  # tune sensitivity
        thresh_val = mean_val + k * std_val
    else:
        thresh_val = 0.1  # fallback threshold

    _, depth_edges = cv2.threshold(depth_grad_x, thresh_val, 255, cv2.THRESH_BINARY)
    depth_edges = depth_edges.astype(np.uint8)
    
    #physical gradient filter, to avoid detecting depth lines within the frame( with very low depth gradient)
    physical_mask = (depth_grad_x > PHYSICAL_GRADIENT_THRESHOLD).astype(np.uint8)*255
    depth_edges = cv2.bitwise_and(depth_edges, physical_mask)

    # Overlay depth edges in red
    sobel_vis_color[depth_edges > 0] = [0, 0, 255]  # Red for edge pixels
    cv2.imshow("Sobel + Depth Edges", sobel_vis_color)

    # Hough Transform on depth-based vertical edges
    depth_lines = cv2.HoughLinesP(depth_edges, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=30)


    valid_lines = []
    if depth_lines is not None:
        for line in depth_lines:
            x1, y1, x2, y2 = line[0]

            # Compute line angle
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Keep only near-vertical lines
            if 80 < abs(angle) < 100:
                # Estimate Z-depth of the line robustly
                d = robust_line_z_roi_new(z_depth_map, x1, y1, x2, y2, roi_width=20)
                #cv2.line(color_image, (x1, y1), (x2, y2), (203, 192, 255), 2)  #pink
                x_m = int((x1 + x2) / 2)
                y_m = int((y1 + y2) / 2)
                

                # Draw only if within specified depth range
                if d is not None and DEPTH_RANGE[0] <= d <= DEPTH_RANGE[1]:
                    cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 255), 2)  # magenta
                    valid_lines.append((x1, y1, x2, y2))
                    # show the midpoint used for normals

                    cv2.circle(color_image, (x_m, y_m), 3, (255, 0, 0), -1)
                    # optional annotate depth
                    #cv2.putText(color_image, f"{d:.2f}m", (x_m+6, y_m-6),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    
    merged_lines = cluster_and_merge_lines(valid_lines)  # only merging lines that are vertical, valid, and within depth range
    if merged_lines is not None and len(merged_lines) > 0:
        scored_lines = []
        for line in merged_lines:
            x1, y1, x2, y2 = line
            cv2.line(color_image, (x1, y1), (x2, y2), (0, 165, 255), 2)#orange for merged lines
            num_samples = 10
            scored_lines.append(calculate_confidence_scores(line,depth_grad_x, depth_height, depth_width, color_image, num_samples))

            top_lines = []
            if len(scored_lines) > 0:
                # Sort lines by confidence score (descending)
                top_lines = sorted(scored_lines, key=lambda x: x[1], reverse=True)[:2]
                for line_info in top_lines:
                    x1, y1, x2, y2 = line_info[0]
                    cv2.line(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for top 2 lines

                # (Optional) Update confidence history with the highest score
                now = time.time()
                confidence_history.append((now, top_lines[0][1]))
                while confidence_history and now - confidence_history[0][0] > 10:
                    confidence_history.popleft()
                if confidence_history:
                    avg_conf = np.mean([score for _, score in confidence_history])
                    #print(f"[DEBUG] Avg confidence score (last 10s): {avg_conf:.3f}")
                else:
                    print("[DEBUG] No confidence scores in last 10 seconds.")
    else:
        print("[DEBUG] No depth-based Hough lines found.")
    
    cv2.imshow("Main Output", color_image)