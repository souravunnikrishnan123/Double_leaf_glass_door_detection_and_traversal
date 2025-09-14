from get_z_depth import get_z_depth
import numpy as np



def extract_smooth_line_segment_with_moving_avg(depth_frame, x1, y1, x2, y2, gradient_threshold=0.1, window=5, min_valid_points=6, num_samples=100):
    """
    Traverse the line from (x1, y1) to (x2, y2) and extract the longest segment
    with smoothed depth gradient below a threshold.
    Returns: [(x, y, depth)] filtered segment or empty list.
    a line in RGB can be a composite of lines with different depths. this can cause inaccuracy. 
    hence we need to take majority part of line which has a constant depth and take that depth as the depth of complete line
    """
    

    points = []
    for i in range(num_samples + 1):
        t = i / num_samples
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        if 0 <= x < depth_frame.width and 0 <= y < depth_frame.height:
            d = get_z_depth(depth_frame, x, y)
            if d > 0 and np.isfinite(d):
                points.append((x, y, d))

    if len(points) < min_valid_points:
        return [], 0

    max_segment = []
    center_depth = 0
    current_segment = []
    depth_window = []

    for i, (x, y, d) in enumerate(points):
        depth_window.append(d)
        if len(depth_window) > window:
            depth_window.pop(0)

        if len(depth_window) < window:
            current_segment.append((x, y, d))
            continue

        smoothed = np.mean(depth_window)
        if abs(smoothed - d) < gradient_threshold:
            current_segment.append((x, y, d))
        else:
            depths_from_current_segment = [d for _, _, d in current_segment]
            center_depth_of_current_segment = np.median(depths_from_current_segment) 
            if len(current_segment) > len(max_segment):
                max_segment = current_segment
                center_depth = center_depth_of_current_segment
            current_segment = []  # Reset when gradient too high
            depth_window = []     # Also reset smoothing window

    # Final check
    if len(current_segment) > len(max_segment): # to cover the case where,there is no depth variation in the entire line detected by houglines and canny. that is if abs(smoothed - d) < gradient_threshold: never became false
        depths_from_current_segment = [d for _, _, d in current_segment]
        center_depth_of_current_segment = np.median(depths_from_current_segment)
        max_segment = current_segment
        center_depth = center_depth_of_current_segment

    return max_segment, center_depth


def extrapolate_along_line_segment(depth_frame, start_point, direction_vector, center_depth, gradient_threshold=0.1, window=5, max_steps=100):
    """
    Extrapolate along a direction (unit vector) from a starting point until depth gradient exceeds threshold.
    Returns a list of (x, y, depth) tuples.
    """


    x, y = start_point
    dx, dy = direction_vector
    extrapolated = []
    depth_history = []

    for step in range(max_steps):
        x += dx
        y += dy
        xi, yi = int(round(x)), int(round(y))

        if not (0 <= xi < depth_frame.width and 0 <= yi < depth_frame.height):
            break

        d = get_z_depth(depth_frame, xi, yi)
        if d == 0 or not np.isfinite(d):
            continue

        depth_history.append(d)
        if len(depth_history) > window:
            depth_history.pop(0)

        smoothed = np.mean(depth_history)
        if abs(smoothed - center_depth) > gradient_threshold:
            break

        extrapolated.append((xi, yi, center_depth))

    return extrapolated



def get_median_depth_window(depth_frame, x, y, window=10):
    """Get the median depth in a square window around (x, y)."""

    
    half = window // 2
    depths = []
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            px = x + dx
            py = y + dy
            if 0 <= px < depth_frame.width and 0 <= py < depth_frame.height:
                d =  get_z_depth(depth_frame,px, py)
                if DEPTH_RANGE[0] <= d <= DEPTH_RANGE[1]:
                    depths.append(d)
    return np.median(depths) if len(depths) >= 6 else None  # Require minimum valid values




def get_median_depth_along_detected_line(depth_frame, x1, y1, x2, y2, num_samples=20):
    """Samples depth values along the line segment from (x1, y1) to (x2, y2) and returns the median."""

    depths = []

    for i in range(num_samples):
        x = int(round(x1 + (x2 - x1) * i / (num_samples - 1)))
        y = int(round(y1 + (y2 - y1) * i / (num_samples - 1)))

        if 0 <= x < depth_frame.width and 0 <= y < depth_frame.height: 
            d = get_z_depth(depth_frame, x, y)
            if DEPTH_RANGE[0] <= d <= DEPTH_RANGE[1]:
                depths.append(d)

    return np.median(depths) if len(depths) >= 6 else None  # Require minimum valid points