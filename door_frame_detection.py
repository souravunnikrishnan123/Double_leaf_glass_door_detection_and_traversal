# Import necessary libraries
import pyrealsense2 as rs   # RealSense SDK for Python
import numpy as np          # For array and matrix operations
import cv2                  # OpenCV for image processing
from collections import deque, Counter

from depth_based_detection import depth_based_edge_detection, depth_based_edge_detection_within_rgb_based_frame_lines_roi
from roi import process_filtered_lines
from door_status import detect_door_state
from get_z_depth import get_z_depth
from get_roi_bounding_box_for_depth_line_detection import get_roi_bounding_box_from_frame_lines_for_depth_lines_detection

# -------------------------------
# Initialize RealSense Pipeline
# -------------------------------
pipeline = rs.pipeline()
config = rs.config()

# Enable depth stream (z16 = 16-bit grayscale)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
# Enable color stream (bgr8 = standard OpenCV format)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

# Start streaming
pipeline.start(config)

# Align depth to color stream so depth and color pixels correspond
align = rs.align(rs.stream.color)

# -------------------------------
# Constants for Depth Display
# -------------------------------
MIN_DEPTH = 0.3  # Minimum depth (in meters)
MAX_DEPTH = 6  # Maximum depth (in meters)

MIN_DEPTH_DEPTH_EDGE_DETECTION = 0.3  # Minimum depth for edge detection (in meters)
MAX_DEPTH_DEPTH_EDGE_DETECTION = 3.0  # Maximum depth for edge detection (in meters). Because the algo works best in this range. if the glass is open, or closed if the object is beyonod 3m, then its okay we get the depth as zero. anyway we want to find large depth gradient

DEPTH_RANGE = (1.7, 2.2)  # in meters

ROI_WIDTH = 60            # width of ROI in pixels

MAX_CLIP_DEPTH = 0.1
DOOR_MIN_DEPTH = 1.9  # e.g., 1 meter away
DOOR_MAX_DEPTH = 2.1

# -------------------------------
# Convert raw depth image to color for visualization
# -------------------------------
def depth_to_colormap(depth_image):
    # Clip depth image to desired range
    depth_scaled = np.clip(depth_image, MIN_DEPTH * 1000, MAX_DEPTH * 1000)
    # Convert depth to 8-bit for color mapping
    depth_scaled = cv2.convertScaleAbs(depth_scaled, alpha=0.03)
    # Apply colormap (Jet: Blue → Red gradient)
    return cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)

# -------------------------------
# Mouse click event for checking depth at pixel
# -------------------------------
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_frame, scale_factor = param  # unpack parameters
        # Map coordinates back to original resolution
        orig_x = int(x / scale_factor)
        orig_y = int(y / scale_factor)

        depth = get_z_depth(depth_frame, orig_x, orig_y)
        print(f"Clicked at ({orig_x}, {orig_y}) → Depth: {depth:.2f} m")


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

def extract_smooth_line_segment_with_moving_avg(depth_frame, x1, y1, x2, y2, gradient_threshold=0.1, window=5, min_valid_points=6, num_samples=100):
    """
    Traverse the line from (x1, y1) to (x2, y2) and extract the longest segment
    with smoothed depth gradient below a threshold.
    Returns: [(x, y, depth)] filtered segment or empty list
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
            if center_depth_of_current_segment and (DEPTH_RANGE[0] <= center_depth_of_current_segment <= DEPTH_RANGE[1]):  
                if len(current_segment) > len(max_segment):
                    max_segment = current_segment
                    center_depth = center_depth_of_current_segment
                current_segment = []  # Reset when gradient too high
                depth_window = []     # Also reset smoothing window

    # Final check
    if len(current_segment) > len(max_segment): # to vover the case where,there is no depth variation in the entire line detected by houglines and canny. that is if abs(smoothed - d) < gradient_threshold: never became false
        depths_from_current_segment = [d for _, _, d in current_segment]
        center_depth_of_current_segment = np.median(depths_from_current_segment)
        if center_depth_of_current_segment and (DEPTH_RANGE[0] <= center_depth_of_current_segment <= DEPTH_RANGE[1]):
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


def filter_vertical_lines_glass_contact(lines, depth_frame, fx, glass_width_cm,center_frame_width_cm ):
    """
    Filters vertical lines that are likely in contact with a glass pane.

    Keeps lines that:
    - Have no neighbor on one side (left or right), OR
    - The neighbor is farther than `glass_width_cm` (in centimeters)

    Args:
        lines: list of tuples (x_center, y_top, y_bottom, center_depth)
        depth_frame: RealSense depth frame for accurate distance lookup
        fx: focal length in pixels (from intrinsics)
        glass_width_cm: minimum distance (in cm) we expect between the frame and its neighbor
        
    Returns:
        filtered: list of lines that likely represent frame-glass boundary
    """

    if not lines:
        return []
    
    
    # Sort lines left to right based on x coordinate avg. as there can be a a lot of same coordinate for a line, it can cause bias. hence it is better to take mean to sort the line
    lines = sorted(lines, key=lambda line: sum(point[0] for point in line) / len(line))
    filtered = []
    # find the mean x-coordinate for each sorted line
    # and store it in a new list.
    mean_x_coords = []
    for line in lines:
        if len(line) > 0:

            # Calculate the mean x-coordinate 
            mean_x = sum(point[0] for point in line) / len(line) #as there can be a a lot of same coordinate for a line, it can cause bias. hence it is better to take mean to sort the line
            # Append it to the new list
            mean_x_coords.append(mean_x)
        else:
            # Handle empty lines
            mean_x_coords.append(None) # Or some other placeholder

    paired_lines = []

    for i, line in enumerate(lines):
        depths = [point[2] for point in line]
        median_depth = np.median(depths) # for depth median is more reliable as the values are different from one to another. chances of repetition is less in the dataset
        # Get depth at the center of this vertical line in meters
        

        # Convert depth to centimeters
        depth_cm = median_depth * 100

        # Compute how many pixels `glass_width_cm` maps to at this depth using focal length
        required_pixel_gap = (glass_width_cm / depth_cm) * fx
        frame_pixel_gap = (center_frame_width_cm / depth_cm) * fx 
        #print(fx)
        #print(frame_pixel_gap)

        
        left_exists = i > 0
        right_exists = i < len(lines) - 1

        x0 = mean_x_coords[i]
        
        # --- Case 1: Both left and right neighbors exist ---
        if left_exists and right_exists:
            x_left = mean_x_coords[i - 1]
            x_right = mean_x_coords[i + 1]
            dist_left = abs(x0 - x_left)
            dist_right = abs(x0 - x_right)

            # One must be wide, one must be narrow
            if ((dist_left >= required_pixel_gap and dist_right <= frame_pixel_gap) or
                (dist_right >= required_pixel_gap and dist_left <= frame_pixel_gap)):
                filtered.append(line)
                


         # --- Case 2: Only one neighbor exists ---
        elif left_exists:
            dist_left = abs(x0 - mean_x_coords[i - 1])
            if dist_left <= frame_pixel_gap:
                filtered.append(line)
                


        elif right_exists:
            dist_right = abs(x0 - mean_x_coords[i + 1])
            if dist_right <= frame_pixel_gap:
                filtered.append(line)
                

        
    paired_lines = get_paired_lines(filtered, frame_pixel_gap)

    #print("Filtered lines:", filtered)
    #print("Paired lines:", paired_lines)

    return paired_lines


def get_paired_lines(filtered, frame_pixel_gap):
    """
    Given a sorted list of filtered lines, return a list of tuples (line, neighbor_line)
    where each pair is within frame_pixel_gap.
    """
    paired_lines = []

    filtered = sorted(filtered, key=lambda line: sum(point[0] for point in line) / len(line))

    mean_x_coords = [np.mean([pt[0] for pt in line]) for line in filtered]

    for i, line in enumerate(filtered):
        x0 = mean_x_coords[i]
        left_exists = i > 0
        right_exists = i < len(filtered) - 1

        # Both neighbors exist
        if left_exists and right_exists:
            x_left = mean_x_coords[i - 1]
            x_right = mean_x_coords[i + 1]
            dist_left = abs(x0 - x_left)
            dist_right = abs(x0 - x_right)
            # Pair with the neighbor within frame_pixel_gap
            if dist_left <= frame_pixel_gap and dist_right > frame_pixel_gap:
                paired_lines.append(( filtered[i - 1], line))
            elif dist_right <= frame_pixel_gap and dist_left > frame_pixel_gap:
                paired_lines.append((line, filtered[i + 1]))
            # If both are within gap, you can choose one or both (here, choose left)
            elif dist_left <= frame_pixel_gap and dist_right <= frame_pixel_gap:
                paired_lines.append(( filtered[i - 1], line))
        # Only left neighbor
        elif left_exists:
            x_left = mean_x_coords[i - 1]
            dist_left = abs(x0 - x_left)
            if dist_left <= frame_pixel_gap:
                paired_lines.append((filtered[i - 1], line))
        # Only right neighbor
        elif right_exists:
            x_right = mean_x_coords[i + 1]
            dist_right = abs(x0 - x_right)
            if dist_right <= frame_pixel_gap:
                paired_lines.append((line, filtered[i + 1]))


    # Remove duplicate pairs (order-insensitive)
    unique_pairs = []
    seen = set()
    for l1, l2 in paired_lines:
        # Use tuple of sorted ids to avoid (A,B) and (B,A) duplicates
        key = tuple(sorted([id(l1), id(l2)]))
        if key not in seen:
            unique_pairs.append((l1, l2))
            seen.add(key)
    #print("Unique pairs found:", unique_pairs)
    return unique_pairs



# Parameters
HISTORY_LENGTH = 30  # Frames to track
MAX_LINES_TO_TRACK = 5  # Keep top N stable lines
DISTANCE_THRESHOLD = 15  # Pixels for grouping similar lines

# History: store list of detected lines (each as a tuple: (avg_x, points))
line_history = deque(maxlen=HISTORY_LENGTH)


def update_line_history(filtered_lines):
    """
    Add detected lines from current frame to history.
    Args:
        filtered_lines: list of lines (each is a list of (x,y) tuples)
    """
    frame_lines = []
    for pts in filtered_lines:
        pts = np.array(pts)  # Convert list of tuples to Nx2 array
        avg_x = int(np.mean(pts[:, 0]))
        frame_lines.append((avg_x, pts))
    line_history.append(frame_lines)


def get_stable_lines():
    """
    Analyze history and return the most stable lines.
    Returns:
        List of tuples (avg_x, line_points, confidence)
    """
    if len(line_history) < 5:
        return []  # Not enough history

    # Flatten all historical lines into one list
    all_lines = []
    for frame_lines in line_history:
        all_lines.extend(frame_lines)

    # Group lines by proximity in X
    clusters = []
    for avg_x, pts in all_lines:
        placed = False
        for cluster in clusters:
            if abs(cluster["center_x"] - avg_x) < DISTANCE_THRESHOLD:
                cluster["lines"].append((avg_x, pts))
                cluster["center_x"] = np.mean([l[0] for l in cluster["lines"]])
                placed = True
                break
        if not placed:
            clusters.append({"center_x": avg_x, "lines": [(avg_x, pts)]})

    # Compute stability (frequency) for each cluster
    stable_lines = []
    for cluster in clusters:
        # Count how many frames this cluster appeared in
        frame_counts = set()
        for avg_x, _ in cluster["lines"]:
            frame_counts.add(avg_x)  # Rough frame-based uniqueness
        confidence = len(cluster["lines"]) / (len(line_history))  # normalized
        # Take the most recent line points from this cluster
        recent_line = cluster["lines"][-1][1]
        stable_lines.append((int(cluster["center_x"]), recent_line, confidence))

    # Sort clusters by confidence (most stable first)
    stable_lines.sort(key=lambda x: x[2], reverse=True)

    # Keep top N stable lines
    return stable_lines[:MAX_LINES_TO_TRACK]



def draw_stable_lines(image, stable_lines, color=(0, 255, 255)):
    """
    Draws stable lines on the image.
    stable_lines: list of (avg_x, pts_array, score)
    """
    for avg_x, pts, score in stable_lines:
        # Ensure pts is a numpy array of points
        pts = np.array(pts)
        if pts.ndim != 2 or pts.shape[1] < 2:
            continue
        if len(pts) < 2:
            continue
        x1, y1 = int(pts[0][0]), int(pts[0][1])
        x2, y2 = int(pts[-1][0]), int(pts[-1][1])
        #cv2.line(image, (x1, y1), (x2, y2), color, 2)
        #cv2.putText(image, f"x={avg_x}, s={score:.2f}", (x1, y1 - 10),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
def sort_line_by_y(line_points):
    # Sort points by y-coordinate (ascending)
    return sorted(line_points, key=lambda pt: pt[1])

# -------------------------------
# Main loop for live streaming
# -------------------------------
try:
    while True:
        # Wait for next set of frames
        frames = pipeline.wait_for_frames()

        # Align depth to color
        aligned = align.process(frames)

        # Extract depth and color frames
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        # If either frame is not available, skip this loop
        if not depth_frame or not color_frame:
            continue

        # Convert to numpy arrays for OpenCV
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert depth to colored display image
        depth_colormap = depth_to_colormap(depth_image)


        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(16, 16))
        contrast = clahe.apply(gray)

        #filtered = cv2.bilateralFilter(contrast, d=9, sigmaColor=150, sigmaSpace=100)
        filtered = cv2.GaussianBlur(contrast, (3, 3), 0.5)

        median_val = np.median(filtered)
        lower = int(max(0, 0.5 * median_val))
        upper = int(min(255, 1.5 * median_val))

        edges = cv2.Canny(filtered, lower, upper)

        #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        #edges = cv2.Canny(blurred, 50, 150)  # Canny edge detector
        
        # Hough Line Transform to detect lines
        # Detect lines using Probabilistic Hough Transform
        # Parameters:
        #  - 1: pixel resolution of the Hough grid
        #  - np.pi / 180: angle resolution in radians (1 degree)
        #  - threshold=100: minimum number of intersections to detect a line
        #  - minLineLength=100: minimum length of line in pixels to be considered
        #  - maxLineGap=10: maximum allowed gap between line segments to link them
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=30)  #maxlingap of 30px is needed to detect door handle
        
        # Draw detected vertical lines on image
                # If any lines are detected
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                #cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # All lines: blue
                if 80 < abs(angle) < 100:  # near-vertical
                    #cv2.line(color_image, (x1, y1), (x2, y2), (0, 165, 255), 2)  # All vertical lines: orange
                    image_height = depth_image.shape[0]
                    y_top = 0
                    y_bottom = image_height - 1

                    #center_depth = get_median_depth_window(depth_frame, x_center, y_center, window=10)
                    #if center_depth is None:
                        #center_depth = get_median_depth_along_line(depth_frame, x_center, y1, y2)

                   # Extract smooth portion along detected Hough line
                    filtered_segment,center_depth = extract_smooth_line_segment_with_moving_avg(
                        depth_frame, x1, y1, x2, y2,
                        gradient_threshold=0.1, window=5, num_samples=100
                    ) 

                    #center_depth = get_median_depth_along_detected_line(depth_frame, x1, y1, x2, y2)
                    # Compute median depth and center

                    if center_depth is None:
                        continue  # Skip line if no valid depth
                    if not (DEPTH_RANGE[0] <= center_depth <= DEPTH_RANGE[1]):
                        continue  # Still skip if out of expected depth range


                    # Use first and last points of filtered segment
                    start_fwd = filtered_segment[-1][:2] # take only x and y coordinate. donot take depth
                    start_back = filtered_segment[0][:2]

                    # Compute direction vector of the line (normalized)
                    dx = x2 - x1
                    dy = y2 - y1
                    norm = np.hypot(dx, dy)
                    dx /= norm
                    dy /= norm

                    # Extrapolate forward
                    extrapolated_forward = extrapolate_along_line_segment(
                        depth_frame, start_fwd, (dx, dy), center_depth, gradient_threshold=0.1, window=5
                    )

                    # Extrapolate backward
                    extrapolated_backward = extrapolate_along_line_segment(
                        depth_frame, start_back, (-dx, -dy), center_depth, gradient_threshold=0.1, window=5
                    )

                    # Combine all
                    full_line_segment = extrapolated_backward[::-1] + filtered_segment + extrapolated_forward
                    
                    
                    if len(filtered_segment) >= 2:
                        pt1 = tuple(map(int, filtered_segment[0][:2]))
                        pt2 = tuple(map(int, filtered_segment[-1][:2]))
                        #cv2.line(color_image, pt1, pt2, (255, 255, 0), 2)  # Cyan

                    if len(full_line_segment) >= 2:
                        pt1 = tuple(map(int, full_line_segment[0][:2]))
                        pt2 = tuple(map(int, full_line_segment[-1][:2]))
                        #cv2.line(color_image, pt1, pt2, (255, 0, 255), 2)  # Magenta
                    

                    # Append clipped vertical line
                    vertical_lines.append(full_line_segment)
                    
                    #cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 255), 1)  # Yellow, thin
                    #cv2.line(color_image, (x_center, y_top), (x_center, y_bottom), (0, 255, 0), 2)  # Green, thicker
            
            intr = depth_frame.profile.as_video_stream_profile().intrinsics
            fx = intr.fx  # in pixels
            

            update_line_history(vertical_lines)
            stable_lines = get_stable_lines()
            draw_stable_lines(color_image, stable_lines)

            # Pass only line points to filter function
            stable_line_points = [line_pts for avg_x, line_pts, confidence in stable_lines]
            
            # Visualize stable_line_points (lines passed to filter_vertical_lines_glass_contact)
            for line_pts in stable_line_points:
                if len(line_pts) >= 2:
                    pt1 = tuple(map(int, line_pts[0][:2]))
                    pt2 = tuple(map(int, line_pts[-1][:2]))
                    #cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)  # Green for stable lines
                    

            paired_lines = filter_vertical_lines_glass_contact(
                stable_line_points, depth_frame, fx, glass_width_cm=40, center_frame_width_cm=30
            )
            
            # Adjust all pairs so each line's points are sorted by y
            paired_lines_sorted = [
                (sort_line_by_y(left_line), sort_line_by_y(right_line))
                for left_line, right_line in paired_lines]
            

            # Visualize paired lines (glass frame candidates)
            for left_line, right_line in paired_lines_sorted:
                # Draw left line in red
                if len(left_line) >= 2:
                    pt1 = tuple(map(int, left_line[0][:2]))
                    pt2 = tuple(map(int, left_line[-1][:2]))
                    #cv2.line(color_image, pt1, pt2, (0, 0, 255), 2)
                # Draw right line in cyan
                if len(right_line) >= 2:
                    pt1 = tuple(map(int, right_line[0][:2]))
                    pt2 = tuple(map(int, right_line[-1][:2]))
                    #cv2.line(color_image, pt1, pt2, (0, 255, 255), 2)



            # Example: Extract full coordinates of each stable line
            #for avg_x, line_pts, confidence in stable_lines:
                # Print the number of points instead of shape
                #print(f"Stable Line X={avg_x}, Confidence={confidence:.2f}, NumPoints={len(line_pts)}")

            
            avg_z_left, avg_z_right, filtered_pairs , mean_z_depth_along_frame_lines = process_filtered_lines(paired_lines_sorted, depth_frame, color_image)

            #Filter for the leftmost pair (lowest average x of left line). this is temporary logic to avoid getting the lines near to the tv in the PC lab being detected as door frame lines. need to improve it
            if filtered_pairs:
                leftmost_idx = np.argmin([np.mean([pt[0] for pt in pair[0]]) for pair in filtered_pairs])
                final_left_frame_line = filtered_pairs[leftmost_idx][0]
                final_right_frame_line = filtered_pairs[leftmost_idx][1]


                #door_state = detect_door_state(depth_frame, color_image, final_left_frame_line, final_right_frame_line,
                      #roi_width=240, margin=10, threshold=0.3, z_door_depth = mean_z_depth_along_frame_lines)


                #print(f"Door is {door_state}")


                # Compute ROI bounding box
                min_x, max_x, min_y, max_y = get_roi_bounding_box_from_frame_lines_for_depth_lines_detection(final_left_frame_line, final_right_frame_line, margin=40, img_shape=color_image.shape)
                # Crop color and depth images
                roi_color = color_image[min_y:max_y, min_x:max_x]
                roi_depth_np = np.asanyarray(depth_frame.get_data())[min_y:max_y, min_x:max_x]

                # need to adapt depth_based_edge_detection to accept numpy arrays for depth
                depth_based_edge_detection_within_rgb_based_frame_lines_roi(roi_depth_np, roi_color, MIN_DEPTH_DEPTH_EDGE_DETECTION, MAX_DEPTH_DEPTH_EDGE_DETECTION, DEPTH_RANGE)
                

        
        #depth_based_edge_detection(depth_frame, color_image, MIN_DEPTH_DEPTH_EDGE_DETECTION, MAX_DEPTH_DEPTH_EDGE_DETECTION, DEPTH_RANGE)

        # Process if enough vertical lines detected

        # Draw depth-refined vertical line
            
        # Stack visualizations horizontally:
        # [Color Image | Depth Map | Edge Map]
        target_height, target_width = color_image.shape[:2]
        edges_resized = cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (target_width, target_height))

        stacked = np.hstack((
            color_image,
            depth_colormap,
            edges_resized
        ))

        # Show the result in one window
        # --- Resize stacked output to fit the display ---
        screen_width = 1920   # adjust if needed
        screen_height = 1080   # adjust if needed

        scale_w = screen_width / stacked.shape[1]
        scale_h = screen_height / stacked.shape[0]
        scale_factor = min(scale_w, scale_h)  # preserve aspect ratio

        stacked_resized = cv2.resize(stacked, None, fx=scale_factor, fy=scale_factor)
        cv2.imshow("Color | Depth | Edges+ Lines", stacked_resized)


        # Attach mouse callback to print depth when clicking
        cv2.setMouseCallback("Color | Depth | Edges+ Lines", click_event, param=(depth_frame, scale_factor))


        # Exit when ESC key is pressed
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

# ------------------------------------
# Stop pipeline and clean up on exit
# ------------------------------------
finally:
    pipeline.stop()
    cv2.destroyAllWindows()