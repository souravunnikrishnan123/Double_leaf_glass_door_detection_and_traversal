#!/usr/bin/env python3
import rospy
import cv2
import numpy as np 

def get_depth_based_lines_using_sobel_and_hough_lines(depth_frame, MIN_DEPTH=1.0, MAX_DEPTH=4.0, PHYSICAL_GRADIENT_THRESHOLD=0.02):

    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy

    # Build Z-depth map
    z_depth_map = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0


    # Global validity check
    # Filter out sensor invalid values (too close, too far, nan/0)
    final_mask = (z_depth_map > MIN_DEPTH) & (z_depth_map < MAX_DEPTH)

    # Detect strong edges between ~door distance (1.8–2.2 m) and background (0 or >2.5 m)
    #roi_mask = ((z_depth_map > 1.8) & (z_depth_map < 2.2)) | (z_depth_map == 0) | (z_depth_map > 2.5)

    # Combine masks
    #final_mask = valid_mask & roi_mask

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
    # but if there are depth hole within the frame, the depth gradient will be high, that case is not covered here
    physical_mask = (depth_grad_x > PHYSICAL_GRADIENT_THRESHOLD).astype(np.uint8)*255
    depth_edges = cv2.bitwise_and(depth_edges, physical_mask)

    # Overlay depth edges in red
    sobel_vis_color[depth_edges > 0] = [0, 0, 255]  # Red for edge pixels


    # Hough Transform on depth-based vertical edges
    depth_lines = cv2.HoughLinesP(depth_edges, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=30)
    
    return depth_lines, sobel_vis_color