import cv2
import numpy as np
import pyrealsense2 as rs 


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

def check_side_roi_against_door(depth_frame, color_image, roi_polygon, door_depth, color = (0, 255, 0)):
    """
    Check how much of ROI depth matches door reference depth.
    """


    mask = np.zeros((depth_frame.height, depth_frame.width), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon.astype(np.int32)], 255)

    valid_points = []
    ys, xs = np.where(mask == 255)
    for (x, y) in zip(xs, ys):
        z = get_z_depth(depth_frame, x, y)
        if z > 0:
            valid_points.append((x, y, z))

    if len(valid_points) == 0:
        return 0.0


    # Filter: only consider values within [-20%, +20%] of door_depth
    lower = door_depth * 0.8
    upper = door_depth * 2.5
    filtered_points = [(x, y, z) for (x, y, z) in valid_points if lower <= z <= upper]
    if len(filtered_points) == 0:
        return 0.0



    # Count only those within ±10% of door_depth
    close_points = [(x, y, z) for (x, y, z) in filtered_points if abs(z - door_depth) <= door_depth * 0.1]

    fraction_close = len(close_points) / len(filtered_points)

    # Visualization
    if color_image is not None:
        cv2.polylines(color_image, [roi_polygon.astype(np.int32)], isClosed=True, color=color, thickness=2)
        for (x, y, z) in filtered_points:
            if abs(z - door_depth) <= door_depth * 0.1:
                color_image = cv2.circle(color_image, (x, y), 1, (0, 0, 255), -1)  # red = matches door depth
            else:
                color_image = cv2.circle(color_image, (x, y), 1, (255, 0, 0), -1)  # blue = valid but not close

    return fraction_close

def detect_door_state(depth_frame,color_image, roi_polygon_left, roi_polygon_right,
                      roi_width=40, margin=10, threshold=0.3, z_door_depth=None):
    """
    Decide OPEN/CLOSED based on ROIs and door depth reference.
    """


    # Step 2: build ROIs
    #roi_left = build_side_rect_roi(left_line_points, side="left", roi_width=roi_width, margin=margin, image_height=depth_frame.height)
    #roi_right = build_side_rect_roi(right_line_points, side="right", roi_width=roi_width, margin=margin, image_height=depth_frame.height)

    #reuse roi_polygon_left and roi_polygon_right from door_frame_detection.py but with a margin offset. becuase we dont want to
    #include the vertical door frame line pixels in the ROI for depth checking to know the status of door( especially when the detected RGB houghline are not at the frame end but slightly inward)
    roi_left_offset = offset_roi_polygon(roi_polygon_left, side="left", margin=10)
    roi_right_offset = offset_roi_polygon(roi_polygon_right, side="right", margin=10)


    # Step 3: check depth consistency
    left_match = check_side_roi_against_door(depth_frame, color_image, roi_left_offset, z_door_depth, color=(255, 0, 255))
    right_match = check_side_roi_against_door(depth_frame, color_image, roi_right_offset, z_door_depth, color=(0, 255, 255))

    print(f"Left match: {left_match:.2f}, Right match: {right_match:.2f}")

    # Step 4: decision
    left_consistent = left_match > threshold
    right_consistent = right_match > threshold

    if left_consistent and right_consistent:
        door_state = "Closed"
    elif left_consistent and not right_consistent:
        door_state = "Open (on left side)"
    elif not left_consistent and right_consistent:
        door_state = "Open (on right side)"
    else:
        door_state = "Open or Unknown"

    # Overlay decision text
    cv2.putText(color_image, f"Door State: {door_state}", (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)

    return door_state


