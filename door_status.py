import cv2
import numpy as np
import pyrealsense2 as rs 


from create_3d_points_and_detect_ransac_plane import backproject_depth_to_points, ransac_plane_from_points
from get_z_depth import get_z_depth



def offset_roi_polygon(roi_polygon, side="left", margin=10):
    """
    Offset the ROI polygon horizontally by margin.
    For left ROI, shift left; for right ROI, shift right.
    """
    offset = -margin if side == "left" else margin
    roi_polygon_offset = roi_polygon.copy()
    roi_polygon_offset[:, 0] += offset  # Shift x-coordinates
    return roi_polygon_offset



def build_side_rect_roi(line_points, side="left", roi_width=40, margin=10, image_height=None):
    """
    Build rectangular ROI offset from a vertical line.
    - margin: how many pixels to leave empty next to the line
    - roi_width: width of ROI strip
    """
    xs = [p[0] for p in line_points]
    ys = [p[1] for p in line_points]
    x_min, x_max = int(min(xs)), int(max(xs))
    #y_min = int(min(ys))
    #y_min = 0  # start from top of image
    # Use image_height if provided, else use max y from line
    #y_max = image_height - 1 if image_height is not None else int(max(ys))
    y_min, y_max = int(min(ys)), int(max(ys))


    if side == "left":
        x1 = x_min - margin - roi_width
        x2 = x_min - margin
    else:  # right
        x1 = x_max + margin
        x2 = x_max + margin + roi_width

    roi_polygon = np.array([
        [x1, y_min],
        [x2, y_min],
        [x2, y_max],
        [x1, y_max]
    ], dtype=np.int32)

    return roi_polygon

def check_side_roi_against_door(depth_image_in_meters, fx, fy, cx, cy, color_image, roi_polygon, door_depth, color = (0, 255, 0)):
    """
    Check how much of ROI depth matches door reference depth.
    """

    valid_points, uv,_ = backproject_depth_to_points(depth_image_in_meters, fx, fy, cx, cy, max_depth=5.0, subsample=1,roi_polygon = roi_polygon)

    if len(valid_points) == 0:
        return 0.0


    # Filter: only consider values within [-20%, +20%] of door_depth
    lower = door_depth * 0.8
    upper = door_depth * 2.5
    filtered_points = [(x, y, z) for (x, y, z) in valid_points if lower <= z <= upper]
    filtered_indices = [i for i, (x, y, z) in enumerate(valid_points) if lower <= z <= upper]
    if len(filtered_points) == 0:
        return 0.0
    


    # Count only those within ±10% of door_depth
    close_points = [(x, y, z) for (x, y, z) in filtered_points if abs(z - door_depth) <= door_depth * 0.1]
    
    # Convert close_points to numpy array
    close_points_np = np.array(close_points) # shape (N, 3)
    plane_model, inlier_indices = ransac_plane_from_points(
    close_points_np,
    distance_threshold=0.05,  # 5cm, adjust as needed
    ransac_n=3,
    num_iterations=1000
    )

    coplanar_points = close_points_np[inlier_indices]
    non_coplanar_indices = list(set(range(len(close_points_np))) - set(inlier_indices))
    non_coplanar_points = close_points_np[non_coplanar_indices]

    H, W = color_image.shape[:2]

    for idx in inlier_indices:
        u, v = uv[filtered_indices[idx]]
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(color_image, (int(u), int(v)), 2, (0, 255, 0), -1)  # green for coplanar
    for idx in non_coplanar_indices:
        u, v = uv[filtered_indices[idx]]
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(color_image, (int(u), int(v)), 2, (0, 255, 255), -1)  # cyan for non-coplanar

    print(len(close_points))
    fraction_close = len(close_points) / len(filtered_points)

    
    # Visualization
    if color_image is not None:
        cv2.polylines(color_image, [roi_polygon.astype(np.int32)], isClosed=True, color=color, thickness=2)
        for i in filtered_indices:
            u, v = uv[i]
            z = valid_points[i][2]
            x, y = int(u), int(v)
            """
            if 0 <= x < W and 0 <= y < H:
                if abs(z - door_depth) <= door_depth * 0.1:
                    cv2.circle(color_image, (x, y), 1, (0, 0, 255), -1)  # red = matches door depth
                else:
                    cv2.circle(color_image, (x, y), 1, (255, 0, 0), -1)  # blue = valid but not close
            """
    return fraction_close



def detect_door_state(depth_image_in_meters, fx, fy, cx, cy,color_image, roi_polygon_left, roi_polygon_right,
                      roi_width=40, margin=10, threshold=0.3, z_door_depth=None):
    """
    Decide OPEN/CLOSED based on ROIs and door depth reference.
    """


    # Step 2: build ROIs

    #reuse roi_polygon_left and roi_polygon_right from door_frame_detection.py but with a margin offset. becuase we dont want to
    #include the vertical door frame line pixels in the ROI for depth checking to know the status of door( especially when the detected RGB houghline are not at the frame end but slightly inward)
    roi_left_offset = offset_roi_polygon(roi_polygon_left, side="left", margin=10)
    roi_right_offset = offset_roi_polygon(roi_polygon_right, side="right", margin=10)


    # Step 3: check depth consistency
    left_match = check_side_roi_against_door(depth_image_in_meters, fx, fy, cx, cy, color_image, roi_left_offset, z_door_depth, color=(255, 0, 255))
    right_match = check_side_roi_against_door(depth_image_in_meters, fx, fy, cx, cy, color_image, roi_right_offset, z_door_depth, color=(0, 255, 255))

    print(f"Left match: {left_match:.2f}, Right match: {right_match:.2f}")

    # Step 4: decision
    left_consistent = left_match > threshold
    right_consistent = right_match > threshold

    if left_consistent and right_consistent:
        door_state = "Closed"
    elif left_consistent and not right_consistent:
        door_state = "Open (on right side)"
    elif not left_consistent and right_consistent:
        door_state = "Open (on left side)"
    else:
        door_state = "Open or Unknown"

    # Overlay decision text
    cv2.putText(color_image, f"Door State: {door_state}", (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)

    return door_state


