from get_z_depth import get_z_depth
import numpy as np




def extract_smooth_line_segment_with_moving_avg(depth_frame, x1, y1, x2, y2, gradient_threshold=0.1, window=10, min_valid_points=30, num_samples=100):
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
            if d is not None and d > 0 and np.isfinite(d):
                points.append((x, y, d))

    if len(points) < min_valid_points:
        return [], 0


    max_consecutive_outliers = 2
    consecutive_outliers = 0    
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
            consecutive_outliers = 0
            continue

        smoothed = np.median(depth_window)
        if abs(smoothed - d) < gradient_threshold:
            current_segment.append((x, y, d))
            consecutive_outliers = 0
        else:
            consecutive_outliers += 1
            if consecutive_outliers <= max_consecutive_outliers:
                current_segment.append((x, y, d))
            else:
                depths_from_current_segment = [d for _, _, d in current_segment]
                center_depth_of_current_segment = np.median(depths_from_current_segment) 
                if len(current_segment) > len(max_segment):
                    max_segment = current_segment
                    center_depth = center_depth_of_current_segment
                current_segment = []  # Reset when gradient too high
                depth_window = []     # Also reset smoothing window
                consecutive_outliers = 0

    # Final check
    if len(current_segment) > len(max_segment): # to cover the case where,there is no depth variation in the entire line detected by houglines and canny. that is if abs(smoothed - d) < gradient_threshold: never became false
        depths_from_current_segment = [d for _, _, d in current_segment]
        center_depth_of_current_segment = np.median(depths_from_current_segment)
        max_segment = current_segment
        center_depth = center_depth_of_current_segment

    return max_segment, center_depth


"""
def extrapolate_along_line_segment(depth_frame, start_point, direction_vector, center_depth, gradient_threshold=0.1, window=5, max_steps=300):
    
    #Extrapolate along a direction (unit vector) from a starting point until depth gradient exceeds threshold.
    #Returns a list of (x, y, depth) tuples.
    


    x0, y0 = start_point
    dx, dy = direction_vector
    extrapolated = []
    depth_history = []

    for step in range(max_steps):
        x0 += dx
        y0 += dy
        xi, yi = int(round(x0)), int(round(y0))

        if not (0 <= xi < depth_frame.width and 0 <= yi < depth_frame.height):
            break

        d = get_z_depth(depth_frame, xi, yi)
        
        if d is None or d == 0 or not np.isfinite(d):
            continue
   

        depth_history.append(d)
        if len(depth_history) > window:
            depth_history.pop(0)

        smoothed = np.mean(depth_history)
        if abs(smoothed - center_depth) > gradient_threshold:
            break

        extrapolated.append((xi, yi))

    return extrapolated
"""

def extrapolate_along_line_segment(depth_image_in_meters, start_point, direction_vector, center_depth,
                                   gradient_threshold=0.1, window=5, max_steps=100):
    """
    Fully vectorized extrapolation along a line in a depth frame.
    Skips invalid depths entirely (no zero fill).
    Stops when smoothed depth deviates from center_depth beyond threshold.

    Returns: list of (x, y, depth) tuples.
    """
    


    H, W = depth_image_in_meters.shape

    # --- Define ray coordinates ---
    x0, y0 = start_point
    dx, dy = direction_vector
    t = np.arange(1, max_steps + 1, dtype=np.float32)
    xs = np.rint(x0 + dx * t).astype(int)
    ys = np.rint(y0 + dy * t).astype(int)

    # --- Filter out-of-bounds ---
    inb = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if not inb.any():
        return []

    xs, ys = xs[inb], ys[inb]
    d = depth_image_in_meters[ys, xs]

    # --- Keep only valid depths (finite & positive) ---
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return []

    xs, ys, d = xs[valid], ys[valid], d[valid]
    n = len(d)
    if n == 0:
        return []

    # --- Compute trailing moving average using convolution ---
    # Kernel for a simple mean over `window` samples
    kernel = np.ones(window, dtype=np.float32) / float(window)

    # Perform convolution with 'same' mode so output aligns with input
    smoothed = np.convolve(d, kernel, mode='same')


    #NumPy implicitly uses zero-padding, so edge values are influenced by partial windows.
    #Finally, the result is prepended with window−1 samples using edge padding, which repeats the first smoothed value at the front.
    #This creates extra leading samples without changing the computed smoothing
    smoothed = np.pad(smoothed, (window - 1, 0), mode='edge')


    # --- Find first deviation beyond threshold ---
    deviation = np.abs(smoothed - float(center_depth))
    cond_break = deviation > float(gradient_threshold)

    if np.any(cond_break):
        brk = np.argmax(cond_break)
        xs, ys, d = xs[:brk], ys[:brk], d[:brk]

    # --- Return list of tuples ---
    return [(int(x), int(y)) for x, y, z in zip(xs, ys, d)]



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
                if d is None:
                    continue
                if DEPTH_RANGE[0] <= d <= DEPTH_RANGE[1]:
                    depths.append(d)
    return np.median(depths) if len(depths) >= 6 else None  # Require minimum valid values



"""
def get_median_depth_along_detected_line(depth_frame, x1, y1, x2, y2, num_samples, min_num_of_valid_depths=10):
    #Samples depth values along the line segment from (x1, y1) to (x2, y2) and returns the median.

    depths = []

    for i in range(num_samples):
        x = int(round(x1 + (x2 - x1) * i / (num_samples - 1)))
        y = int(round(y1 + (y2 - y1) * i / (num_samples - 1)))

        if 0 <= x < depth_frame.width and 0 <= y < depth_frame.height: 
            d = get_z_depth(depth_frame, x, y)
            if d is None:
                continue
            #if DEPTH_RANGE[0] <= d <= DEPTH_RANGE[1]:
            if d > 0 and np.isfinite(d):
                depths.append(d)

    return np.median(depths) if len(depths) >= min_num_of_valid_depths else None  # Require minimum valid points
"""

def get_median_depth_along_detected_line(depth_image_in_meters, x1, y1, x2, y2, num_samples, min_num_of_valid_depths=10):
    """Samples depth values along the line segment from (x1, y1) to (x2, y2) and returns the median."""

    H, W = depth_image_in_meters.shape


    t = np.linspace(0, 1, num_samples, dtype=np.float32)
    xs = np.rint(x1 + (x2 - x1) * t).astype(int)
    ys = np.rint(y1 + (y2 - y1) * t).astype(int)

    # --- Filter out-of-bounds ---
    inb = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if not inb.any():
        return None

    xs, ys = xs[inb], ys[inb]
    d = depth_image_in_meters[ys, xs]

    # --- Keep only valid depths (finite & positive) ---
    valid = np.isfinite(d) & (d > 0)

    d_valid = d[valid]

    return float(np.median(d_valid)) if len(d_valid) >= min_num_of_valid_depths else None