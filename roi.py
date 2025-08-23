import pyrealsense2 as rs   # RealSense SDK for Python
import numpy as np
import cv2


def get_z_depth(depth_frame, x, y):
    depth = depth_frame.get_distance(x, y)
    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    _, _, z = rs.rs2_deproject_pixel_to_point(intr, [x, y], depth)
    return z


def get_strip_avg_z(depth_frame, line_points, side="left", roi_width=20, min_depth=1.7):
    """
    Calculate the average Z coordinate in a strip (ROI) parallel to a given line.
    """
    # Take only x, y (ignore any extra fields like depth)
    pts = np.array([(p[0], p[1]) for p in line_points], dtype=np.int32)

    # Compute direction of the line
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return None, None

    dx /= length
    dy /= length

    if side == "right":
        offset_vec = np.array([-dy, dx])
    else:
        offset_vec = np.array([dy, -dx])

    offset_vec = offset_vec / np.linalg.norm(offset_vec) * roi_width

    # Shift points
    pts_offset = pts + offset_vec
    roi_polygon = np.vstack((pts, pts_offset[::-1]))

    # Create mask for ROI
    depth_height, depth_width = depth_frame.height, depth_frame.width
    mask = np.zeros((depth_height, depth_width), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon.astype(np.int32)], 255)

    # Collect Z values
    z_values = []
    ys, xs = np.where(mask == 255)
    for (x, y) in zip(xs, ys):
        if 0 <= x < depth_width and 0 <= y < depth_height:
            z = get_z_depth(depth_frame, x, y)
            if z >= min_depth and z != 0 and not np.isnan(z):
                z_values.append(z)

    avg_z = float(np.mean(z_values)) if len(z_values) > 0 else None
    return avg_z, roi_polygon


def process_filtered_lines(filtered_lines, depth_frame, color_image):
    """
    Process leftmost and rightmost lines:
    - Create ROIs on left of leftmost and right of rightmost lines
    - Compute average Z depth inside each ROI
    - Draw the ROIs on the color image

    Args:
        filtered_lines: list of arrays, each array is line points [(x, y), (x, y), ...]
        depth_frame: RealSense depth frame
        color_image: BGR image for visualization

    Returns:
        (avg_z_left, avg_z_right)
    """
    if len(filtered_lines) < 2:
        return None, None  # Need at least 2 lines

    # Sort lines by their average x position
    filtered_lines.sort(key=lambda line: np.mean([p[0] for p in line]))

    left_line_points = filtered_lines[0]
    right_line_points = filtered_lines[-1]

    # Compute average Z for left ROI
    avg_z_left, roi_polygon_left = get_strip_avg_z(depth_frame, left_line_points, side="left", roi_width=60, min_depth=1.7)

    # Compute average Z for right ROI
    avg_z_right, roi_polygon_right = get_strip_avg_z(depth_frame, right_line_points, side="right", roi_width=60, min_depth=1.7)

    # Draw ROIs if available
    if roi_polygon_left is not None:
        cv2.polylines(color_image, [roi_polygon_left.astype(np.int32)], isClosed=True, color=(255, 0, 255), thickness=2)
    if roi_polygon_right is not None:
        cv2.polylines(color_image, [roi_polygon_right.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)

    return avg_z_left, avg_z_right


if __name__ == "__main__":
    main()
