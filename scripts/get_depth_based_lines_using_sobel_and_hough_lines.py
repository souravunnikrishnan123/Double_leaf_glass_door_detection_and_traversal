#!/usr/bin/env python3
import rospy
import cv2
import numpy as np 

def get_depth_based_lines_using_sobel_and_hough_lines(
    depth_image_in_meters,
    MIN_DEPTH,
    MAX_DEPTH,
    PHYSICAL_GRADIENT_THRESHOLD,
    scale,
    hough_params,
    bilateral_params,
    sobel_params,
    adaptive_params,
):


    # Global validity check
    # Filter out sensor invalid values (too close, too far, nan/0)
    final_mask = (depth_image_in_meters > MIN_DEPTH) & (depth_image_in_meters < MAX_DEPTH)

    # Detect strong edges between ~door distance (1.8–2.2 m) and background (0 or >2.5 m)
    #roi_mask = ((z_depth_map > 1.8) & (z_depth_map < 2.2)) | (z_depth_map == 0) | (z_depth_map > 2.5)

    # Combine masks
    #final_mask = valid_mask & roi_mask
    depth_image_in_meters = depth_image_in_meters.astype(np.float32, copy=False)

    print(f"scale: {scale}")
    orig_h, orig_w = depth_image_in_meters.shape
    # 3) Downsample depth image and mask
    if scale != 1.0:
        ds_w = max(1, int(round(orig_w * scale)))
        ds_h = max(1, int(round(orig_h * scale)))
        depth_ds = cv2.resize(depth_image_in_meters, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
        # resize mask with nearest to keep boolean semantics
        mask_ds = cv2.resize((final_mask.astype(np.uint8) * 255), (ds_w, ds_h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        depth_ds = depth_image_in_meters
        mask_ds = final_mask

    # Smooth Z-depth (bilateral preserves edges better than Gaussian)
    d = int(bilateral_params.get("d", 5))
    sigma_color = float(bilateral_params.get("sigma_color", 50))
    sigma_space = float(bilateral_params.get("sigma_space", 75))

    z_depth_filtered = cv2.bilateralFilter(depth_ds, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
    # Gradient along X (detect vertical edges in depth)

    ksize = int(sobel_params.get("ksize", 3))

    depth_grad_x = cv2.Sobel(z_depth_filtered, cv2.CV_32F, 1, 0, ksize=ksize)
    depth_grad_x = np.abs(depth_grad_x)
    # Apply mask (keep only valid + relevant regions)
    depth_grad_x[~mask_ds] = 0 

    # Normalize for visualization (convert to 8-bit image)
    sobel_vis = cv2.normalize(depth_grad_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    # Convert Sobel visualization to color (BGR)
    sobel_vis_color = cv2.cvtColor(sobel_vis, cv2.COLOR_GRAY2BGR)


    # Adaptive threshold: mean + k*std of valid gradients
    valid_grad_vals = depth_grad_x[mask_ds]  #Only gradients at valid depth pixels are used to compute the threshold
    k_factor = float(adaptive_params.get("k_factor", 2.0))
    fallback = float(adaptive_params.get("fallback_threshold", 0.1))

    if len(valid_grad_vals) > 0:
        mean_val = np.mean(valid_grad_vals)
        std_val = np.std(valid_grad_vals)
        thresh_val = mean_val + k_factor * std_val
    else:
        thresh_val = fallback  # fallback threshold

    _, depth_edges = cv2.threshold(depth_grad_x, thresh_val, 255, cv2.THRESH_BINARY)
    depth_edges = depth_edges.astype(np.uint8)

    #physical gradient filter, to avoid detecting depth lines within the frame( with very low depth gradient)
    # but if there are depth hole within the frame, the depth gradient will be high, that case is not covered here
    physical_mask = (depth_grad_x > PHYSICAL_GRADIENT_THRESHOLD).astype(np.uint8)*255
    depth_edges = cv2.bitwise_and(depth_edges, physical_mask)

    # Overlay depth edges in red
    sobel_vis_color[depth_edges > 0] = [0, 0, 255]  # Red for edge pixels
    if scale != 1.0:
        sobel_vis_color_new = cv2.resize(sobel_vis_color, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    else:
        sobel_vis_color_new = sobel_vis_color

    # Hough Transform on depth-based vertical edges
    hough_threshold = int(hough_params.get("threshold", 100))
    min_line_length = int(hough_params.get("min_line_length", 100))
    max_line_gap = int(hough_params.get("max_line_gap", 30))

    depth_lines = cv2.HoughLinesP(
        depth_edges,
        1,
        np.pi / 180,
        threshold=int(hough_threshold * scale),
        minLineLength=int(min_line_length * scale),
        maxLineGap=int(max_line_gap * scale),
    )
    # 9) Map Hough lines back to original coordinates (if any)
    # Draw detected vertical lines on image
    if depth_lines is not None and len(depth_lines) > 0 and scale != 1.0:
        depth_lines[..., :4] = np.rint(depth_lines[..., :4] / scale).astype(np.int32)
    
    return depth_lines, sobel_vis_color_new