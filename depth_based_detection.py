import cv2  
import numpy as np 
import pyrealsense2 as rs 



def get_z_depth(depth_frame, x, y):
    depth = depth_frame.get_distance(x, y)
    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    _, _, z = rs.rs2_deproject_pixel_to_point(intr, [x, y], depth)
    return z

# -------------------- STEP 2: DEPTH GRADIENT + HOUGH (Z-Depth) --------------------
def depth_based_edge_detection(depth_frame, color_image, MIN_DEPTH, MAX_DEPTH, DEPTH_RANGE):
    
    depth_height, depth_width = depth_frame.height, depth_frame.width

    # Build Z-depth map (forward depth in meters from camera)
    z_depth_map = np.zeros((depth_height, depth_width), dtype=np.float32)

    for y in range(depth_height):
        for x in range(depth_width):
            z_val = get_z_depth(depth_frame, x, y)  # Must return Z in meters
            if z_val is not None and np.isfinite(z_val) and z_val > 0:
                z_depth_map[y, x] = z_val
            else:
                z_depth_map[y, x] = 0.0

    # Valid mask (ignore too near/far or invalid points)
    valid_mask = (z_depth_map > MIN_DEPTH) & (z_depth_map < MAX_DEPTH)

    # Smooth Z-depth (bilateral preserves edges better than Gaussian)
    z_depth_filtered = cv2.bilateralFilter(z_depth_map, d=7, sigmaColor=50, sigmaSpace=75)

    # Gradient along X (detect vertical edges in depth)
    depth_grad_x = cv2.Sobel(z_depth_filtered, cv2.CV_32F, 1, 0, ksize=5)
    depth_grad_x = np.abs(depth_grad_x)
    depth_grad_x[~valid_mask] = 0

    # Adaptive threshold: mean + k*std of valid gradients
    valid_grad_vals = depth_grad_x[valid_mask]
    if len(valid_grad_vals) > 0:
        mean_val = np.mean(valid_grad_vals)
        std_val = np.std(valid_grad_vals)
        k = 1.0  # tune sensitivity
        thresh_val = mean_val + k * std_val
    else:
        thresh_val = 0.05  # fallback threshold

    _, depth_edges = cv2.threshold(depth_grad_x, thresh_val, 255, cv2.THRESH_BINARY)
    depth_edges = depth_edges.astype(np.uint8)
    cv2.imshow("Depth Edges (Z)", depth_edges)

    # Hough Transform on depth-based vertical edges
    depth_lines = cv2.HoughLinesP(depth_edges, 1, np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=20)

    debug_copy = color_image.copy()
    if depth_lines is not None:
        for line in depth_lines:
            x1, y1, x2, y2 = line[0]

            # Compute line angle
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Keep only near-vertical lines
            if 70 < abs(angle) < 110:
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                d = get_z_depth(depth_frame, x_center, y_center)

                if d is not None and DEPTH_RANGE[0] <= d <= DEPTH_RANGE[1]:
                    cv2.line(debug_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green
    else:
        print("[DEBUG] No depth-based Hough lines found.")

    cv2.imshow("Depth Hough Lines (Z-based)", debug_copy)
