from collections import deque
import time
import cv2  
import numpy as np 
import pyrealsense2 as rs

from confidence_score_calculation import calculate_confidence_scores 


confidence_history = deque()

def get_z_depth(depth_frame, x, y):
    depth = depth_frame.get_distance(x, y)
    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    _, _, z = rs.rs2_deproject_pixel_to_point(intr, [x, y], depth)
    return z


def robust_line_z(depth_frame, x1, y1, x2, y2,
                  num_steps=5, step_size=2):
    """
    Estimate Z-depth of a detected line by sampling pixels perpendicular to it.
    Always pick the side with *smaller* median Z (the nearer frame).
    """
    # Line vector
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length < 1e-3:
        return None

    # Unit normal (perpendicular to the line)
    nx, ny = -dy / length, dx / length

    zs_neg, zs_pos = [], []

    for i in range(num_steps):
        offset = (i + 1) * step_size

        # sample on negative normal side
        px_neg = int((x1 + x2) / 2 - nx * offset)
        py_neg = int((y1 + y2) / 2 - ny * offset)
        z_neg = get_z_depth(depth_frame, px_neg, py_neg)
        if z_neg is not None:
            zs_neg.append(z_neg)

        # sample on positive normal side
        px_pos = int((x1 + x2) / 2 + nx * offset)
        py_pos = int((y1 + y2) / 2 + ny * offset)
        z_pos = get_z_depth(depth_frame, px_pos, py_pos)
        if z_pos is not None:
            zs_pos.append(z_pos)

    # compute medians safely
    med_neg = np.median(zs_neg) if len(zs_neg) > 0 else np.inf
    med_pos = np.median(zs_pos) if len(zs_pos) > 0 else np.inf

    # choose nearer side (smaller Z)
    if med_neg < med_pos and med_neg < np.inf:
        return med_neg
    elif med_pos < np.inf:
        return med_pos
    else:
        return None
    

# -------------------- STEP 2: DEPTH GRADIENT + HOUGH (Z-Depth) --------------------
def depth_based_edge_detection(depth_frame, color_image, MIN_DEPTH, MAX_DEPTH, DEPTH_RANGE):

    depth_height, depth_width = depth_frame.height, depth_frame.width

    """
    # Build Z-depth map (forward depth in meters from camera)
    z_depth_map = np.zeros((depth_height, depth_width), dtype=np.float32)

    for y in range(depth_height):
        for x in range(depth_width):
            z_val = get_z_depth(depth_frame, x, y)  # Must return Z in meters
            if z_val is not None and np.isfinite(z_val) and z_val > 0:
                z_depth_map[y, x] = z_val
            else:
                z_depth_map[y, x] = 0.0
    """
    # More efficient way to build Z-depth map
    z_depth_map = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0

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
        k = 2.0  # tune sensitivity
        thresh_val = mean_val + k * std_val
    else:
        thresh_val = 0.05  # fallback threshold

    _, depth_edges = cv2.threshold(depth_grad_x, thresh_val, 255, cv2.THRESH_BINARY)
    depth_edges = depth_edges.astype(np.uint8)
    #cv2.imshow("Depth Edges (Z)", depth_edges)

    # Hough Transform on depth-based vertical edges
    depth_lines = cv2.HoughLinesP(depth_edges, 1, np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=20)

    scored_lines = []
    if depth_lines is not None:
        for line in depth_lines:
            x1, y1, x2, y2 = line[0]

            # Compute line angle
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            # Keep only near-vertical lines
            if 80 < abs(angle) < 100:
                # Estimate Z-depth of the line robustly
                #d = robust_line_z(depth_frame, x1, y1, x2, y2,num_steps=5, step_size=2)

                x_m = int((x1 + x2) / 2)
                y_m = int((y1 + y2) / 2)
                d = get_z_depth(depth_frame, x_m, y_m)

                # Draw only if within specified depth range
                if d is not None and DEPTH_RANGE[0] <= d <= DEPTH_RANGE[1]:
                    cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 255), 2)  # magenta

                    # show the midpoint used for normals

                    cv2.circle(color_image, (x_m, y_m), 3, (255, 0, 0), -1)
                    # optional annotate depth
                    #cv2.putText(color_image, f"{d:.2f}m", (x_m+6, y_m-6),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

                    num_samples = 10
                    scored_lines.append(calculate_confidence_scores(line[0],depth_grad_x, depth_height, depth_width, color_image, num_samples))

            if len(scored_lines) > 0:
            # Find the line with the highest confidence score
                max_line = max(scored_lines, key=lambda x: x[1])
                x1, y1, x2, y2 = max_line[0]

                
                now = time.time()
                confidence_history.append((now, max_line[1]))

                # Remove entries older than 10 seconds
                while confidence_history and now - confidence_history[0][0] > 10:
                    confidence_history.popleft()

                # Compute average of scores in last 10 seconds
                if confidence_history:
                    avg_conf = np.mean([score for _, score in confidence_history])
                    print(f"[DEBUG] Avg confidence score (last 10s): {avg_conf:.3f}")
                else:
                    print("[DEBUG] No confidence scores in last 10 seconds.")
                cv2.line(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for highest score line
    else:
        print("[DEBUG] No depth-based Hough lines found.")
    
    cv2.imshow("Main Output", color_image)
