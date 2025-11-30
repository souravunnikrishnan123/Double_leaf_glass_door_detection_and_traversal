#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import pyrealsense2 as rs 
import open3d as o3d
from duration import get_duration_seconds
from check_if_passable import check_if_passable

from create_3d_points_and_detect_ransac_plane import backproject_depth_to_points, ransac_plane_from_points





def offset_roi_polygon(roi_polygon, side="left", margin=10):
    """
    Offset the ROI polygon horizontally by margin.
    For left ROI, shift left; for right ROI, shift right.
    """
    if roi_polygon is None:
        return None
    offset = -margin if side == "left" else margin
    roi_polygon_offset = roi_polygon.copy()
    roi_polygon_offset[:, 0] += offset  # Shift x-coordinates
    return roi_polygon_offset


def extract_roi_corners(roi_polygon):
    """
    Extract the 4 corner points of a parallelogram-shaped ROI.
    Works even if roi_polygon has many intermediate points.
    """
    roi = np.asarray(roi_polygon, dtype=np.float32).reshape(-1, 2)
    if roi.shape[0] < 4:
        return roi.astype(np.int32)

    # Step 1: Get convex hull (robust to noisy edges)
    hull = cv2.convexHull(roi)
    hull = hull.reshape(-1, 2)

    # Step 2: Approximate hull to 4-point polygon
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True).reshape(-1, 2)

    # Fallback: if we didn’t get 4 corners, pick extreme ones
    if approx.shape[0] != 4:
        x, y = hull[:, 0], hull[:, 1]
        pts = np.array([
            [np.min(x), np.min(y)],
            [np.max(x), np.min(y)],
            [np.max(x), np.max(y)],
            [np.min(x), np.max(y)],
        ], dtype=np.float32)
    else:
        pts = approx

    # Step 3: Order points: top-left, top-right, bottom-right, bottom-left
    # sort by y (top to bottom)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top2, bottom2 = y_sorted[:2], y_sorted[2:]

    # left/right among top and bottom
    top_left = top2[np.argmin(top2[:, 0])]
    top_right = top2[np.argmax(top2[:, 0])]
    bottom_left = bottom2[np.argmin(bottom2[:, 0])]
    bottom_right = bottom2[np.argmax(bottom2[:, 0])]

    ordered = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
    return ordered


def extend_roi_polygon_to_full_height(roi_polygon, image_height):
    """
    Robustly extend a 4-point ROI (parallelogram) to full image height.
    - Handles arbitrary ordering of the 4 points.
    - Preserves the left/right slant by using x of the top pair for y=0
      and x of the bottom pair for y=image_height-1.
    """
    roi = np.asarray(roi_polygon).reshape(-1, 2)

    # sort points by y (top -> bottom)
    sorted_idx = np.argsort(roi[:, 1])
    top2 = roi[sorted_idx[:2]] # selects the two points with the smallest y values
    bot2 = roi[sorted_idx[2:]] #selects the two points with the largest y values

    # determine left/right for top and bottom pairs by x
    top_left_x = int(round(np.min(top2[:, 0])))
    top_right_x = int(round(np.max(top2[:, 0])))
    bot_left_x = int(round(np.min(bot2[:, 0])))
    bot_right_x = int(round(np.max(bot2[:, 0])))

    extended = np.array([
        [top_left_x, 0],
        [top_right_x, 0],
        [bot_right_x, image_height - 1],
        [bot_left_x, image_height - 1]
    ], dtype=np.int32)

    return extended









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

def check_side_roi_against_door(depth_image_in_meters, fx, fy, cx, cy, color_image, roi_polygon, door_depth, color = (0, 255, 0), keyword="color"):
    """
    Check how much of ROI depth matches door reference depth.
    """
    timer = get_duration_seconds()
    timer.start(f"check_side_roi_against_door {keyword}")
    H, W = depth_image_in_meters.shape
    
    filtered_points, filtered_uv,_ = backproject_depth_to_points(depth_image_in_meters, fx, fy, cx, cy, max_depth=door_depth * 20, min_depth=door_depth * 0.9, subsample=4, roi_polygon = roi_polygon)
    
    if len(filtered_points) == 0:
        return 0.0

    filtered_points = np.asarray(filtered_points,dtype=np.float32)
    filtered_uv = np.asarray(filtered_uv,dtype=np.float32)

    close_mask = np.abs(filtered_points[:, 2] - door_depth) <= door_depth * 0.05
    close_points = filtered_points[close_mask]
    close_uv = filtered_uv[close_mask]

    #filter out the points from floor using normal calculation. because we only have few points in close_points
    pc_nf = o3d.geometry.PointCloud()
    pc_nf.points = o3d.utility.Vector3dVector(close_points)
    try:
        pc_nf.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=40))
        pc_nf.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
        normals = np.asarray(pc_nf.normals)

    except Exception:
        normals = None

    if normals is not None:
        ny_thr = 0.7 
        vertical_mask = np.abs(normals[:, 1]) < ny_thr #(cos^-1(0.7) ≈ 45°)
        close_points = close_points[vertical_mask]
        close_uv = close_uv[vertical_mask]


        
    #print(len(bottom_points))
    fraction_close = close_points.shape[0] / filtered_points.shape[0]

    #print(f"Fraction close: {fraction_close:.3f}")
    timer.stop(f"check_side_roi_against_door {keyword}")
    
    timer.start(f"check_side_roi_against_door--> visualization {keyword}")
    # Visualization
    if color_image is not None:
        # Draw ROI polygon
        cv2.polylines(color_image, [roi_polygon.astype(np.int32)], isClosed=True, color=color, thickness=2)
        
        # Prepare coordinate arrays
        uv_filtered = filtered_uv.astype(np.int32)
        uv_close = close_uv.astype(np.int32)
        #uv_removed = removed_close_uv.astype(np.int32)

        # Validity masks to avoid out-of-bounds
        valid_filtered = (uv_filtered[:, 0] >= 0) & (uv_filtered[:, 0] < W) & (uv_filtered[:, 1] >= 0) & (uv_filtered[:, 1] < H)
        valid_close = (uv_close[:, 0] >= 0) & (uv_close[:, 0] < W) & (uv_close[:, 1] >= 0) & (uv_close[:, 1] < H)
        #valid_removed = (uv_removed[:, 0] >= 0) & (uv_removed[:, 0] < W) & (uv_removed[:, 1] >= 0) & (uv_removed[:, 1] < H)

        # Draw "all filtered points" as blue
        color_image[uv_filtered[valid_filtered][:, 1], uv_filtered[valid_filtered][:, 0]] = (255, 0, 0)
        # Draw "close to door depth" points as cyan
        color_image[uv_close[valid_close][:, 1], uv_close[valid_close][:, 0]] = (0, 255, 255)
        # Draw "removed (coplanar/floor)" points as green
        #color_image[uv_removed[valid_removed][:, 1], uv_removed[valid_removed][:, 0]] = (0, 255, 0)


    timer.stop(f"check_side_roi_against_door--> visualization {keyword}")                 
    return fraction_close





def detect_door_state(depth_image_in_meters, fx, fy, cx, cy,color_image, roi_polygon_left, roi_polygon_right,
                      roi_width=40, margin=10, threshold=0.05, z_door_depth=None, plotname = "door_state_based_on_color_image"):
    """
    Decide OPEN/CLOSED based on ROIs and door depth reference.
    """
    color_image_for_passable_check = color_image.copy()

    if "color" in plotname:
        keyword = "color"
    elif "depth" in plotname:
        keyword = "depth"
    else:
        keyword = "unknown"

    # Step 2: build ROIs

    #reuse roi_polygon_left and roi_polygon_right from door_frame_detection.py but with a margin offset. becuase we dont want to
    #include the vertical door frame line pixels in the ROI for depth checking to know the status of door( especially when the detected RGB houghline are not at the frame end but slightly inward)
    roi_left_offset = offset_roi_polygon(roi_polygon_left, side="left", margin=10)
    roi_right_offset = offset_roi_polygon(roi_polygon_right, side="right", margin=10)


    # Step 3: check depth consistency
    left_match = check_side_roi_against_door(depth_image_in_meters, fx, fy, cx, cy, color_image, roi_left_offset, z_door_depth, color=(255, 0, 255),keyword=keyword)
    right_match = check_side_roi_against_door(depth_image_in_meters, fx, fy, cx, cy, color_image, roi_right_offset, z_door_depth, color=(0, 255, 255),keyword=keyword)

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


    
    color_image_roi = None
    bird_eye_view = None
    if door_state == "Open (on left side)":
        roi_polygon_left_corners = extract_roi_corners(roi_left_offset)
        roi_left_offset_full_height = extend_roi_polygon_to_full_height(roi_polygon_left_corners, color_image.shape[0])
        passable_fraction, color_image_roi, bird_eye_view = check_if_passable(depth_image_in_meters, fx, fy, cx, cy, color_image_for_passable_check, roi_left_offset_full_height, z_door_depth, keyword)
    
    
    elif door_state == "Open (on right side)":
        roi_polygon_right_corners = extract_roi_corners(roi_right_offset)
        roi_right_offset_full_height = extend_roi_polygon_to_full_height(roi_polygon_right_corners, color_image.shape[0])
        passable_fraction, color_image_roi, bird_eye_view = check_if_passable(depth_image_in_meters, fx, fy, cx, cy, color_image_for_passable_check, roi_right_offset_full_height, z_door_depth, keyword)

    # Overlay decision text
    cv2.putText(color_image, f"Door State: {door_state}", (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
    
    cv2.putText(color_image, f"Left match: {left_match:.2f}", (30, 170),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.putText(color_image, f"Right match: {right_match:.2f}", (30, 210),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return door_state, color_image_roi, bird_eye_view


