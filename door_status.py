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

def check_side_roi_against_door(depth_image_in_meters, fx, fy, cx, cy, color_image, roi_polygon, door_depth, found_vertical_planes, color = (0, 255, 0)):
    """
    Check how much of ROI depth matches door reference depth.
    """
    H, W = depth_image_in_meters.shape
    
    valid_points, uv,_ = backproject_depth_to_points(depth_image_in_meters, fx, fy, cx, cy, max_depth=5.0, subsample=1,roi_polygon = roi_polygon)

    if len(valid_points) == 0:
        return 0.0


    # Filter: only consider values within [-20%, +20%] of door_depth
    lower = door_depth * 0.8
    upper = door_depth * 2.0
    filtered_points = [(x, y, z) for (x, y, z) in valid_points if lower <= z <= upper]
    filtered_indices = [i for i, (x, y, z) in enumerate(valid_points) if lower <= z <= upper]
    if len(filtered_points) == 0:
        return 0.0
    


    # Count only those within ±10% of door_depth
    close_points = [(x, y, z) for (x, y, z) in filtered_points if abs(z - door_depth) <= door_depth * 0.1]
    # Build a set for fast lookup
    original_close_points_set = set(tuple(pt) for pt in close_points)


    #because we only interested in the floor plane, we only process the first detected horizontal plane
    #because the horizontal planes are sorted based on the number of inliers, the first one should be the floor
    #because of holes in the floor which heavily depends on the texture and lighting, we cant be sure of the inlier density and the area of the plane. 
    
    # Eliminate points that are on the horizontal plane (e.g., floor)
    if found_vertical_planes is not None and len(found_vertical_planes) == 1:
        vertical_plane_model, vertical_inlier_indices, vertical_inlier_points = found_vertical_planes[0]
         
         # here we dont want to use the more reliable detected_plane( which is the 97% percentile of detected door plane).
         #This is because our intention is to remove points that are close to the horizontal plane (e.g., floor). hence in the 
         #detected_plane, these points may not be available. thats why we use the vertical_plane_model which is estimated from all inlier points
         # Remove points that are close to the horizontal plane
        a, b, c, d = vertical_plane_model
        plane_norm = np.linalg.norm([a, b, c])
        # Set a tight threshold for coplanarity (2cm)
        coplanar_thresh = 0.05
        close_points = [
            (x, y, z)
            for (x, y, z) in close_points
            if abs(a * x + b * y + c * z + d) / plane_norm < coplanar_thresh
        ]
        
    close_points_set = set(tuple(pt) for pt in close_points)
    removed_close_points_set = original_close_points_set - close_points_set


    close_indices = [i for i, pt in enumerate(filtered_points) if pt in close_points_set]
    bottom_points = [
    filtered_points[i]
    for i in close_indices
    if uv[filtered_indices[i]][1] > H / 2
]
    #print(len(bottom_points))
    fraction_close = len(close_points) / len(filtered_points)
    #bottom_ration = len(bottom_points) / len(filtered_points)
    #print(f"Fraction close: {fraction_close:.3f}, bottom ration: {bottom_ration:.3f}")
    
    # Visualization
    if color_image is not None:
        cv2.polylines(color_image, [roi_polygon.astype(np.int32)], isClosed=True, color=color, thickness=2)
       
        for i, (x, y, z) in enumerate(filtered_points):
            u, v = uv[filtered_indices[i]]
            px, py = int(u), int(v)
            if 0 <= px < W and 0 <= py < H:
                pt_tuple = (x, y, z)
                
                if pt_tuple in close_points_set:
                    cv2.circle(color_image, (px, py), 1, (0, 255, 255), -1)  # cyan for close_points
                elif pt_tuple in removed_close_points_set:
                    cv2.circle(color_image, (px, py), 1, (0, 255, 0), -1)  # green for removed close points
                else:
                    cv2.circle(color_image, (px, py), 1, (255, 0, 0), -1)  # blue for others
                     
    return fraction_close



def detect_door_state(depth_image_in_meters, fx, fy, cx, cy,color_image, roi_polygon_left, roi_polygon_right,found_vertical_planes,
                      roi_width=40, margin=10, threshold=0.1, z_door_depth=None):
    """
    Decide OPEN/CLOSED based on ROIs and door depth reference.
    """


    # Step 2: build ROIs

    #reuse roi_polygon_left and roi_polygon_right from door_frame_detection.py but with a margin offset. becuase we dont want to
    #include the vertical door frame line pixels in the ROI for depth checking to know the status of door( especially when the detected RGB houghline are not at the frame end but slightly inward)
    roi_left_offset = offset_roi_polygon(roi_polygon_left, side="left", margin=10)
    roi_right_offset = offset_roi_polygon(roi_polygon_right, side="right", margin=10)


    # Step 3: check depth consistency
    left_match = check_side_roi_against_door(depth_image_in_meters, fx, fy, cx, cy, color_image, roi_left_offset, z_door_depth, found_vertical_planes, color=(255, 0, 255))
    right_match = check_side_roi_against_door(depth_image_in_meters, fx, fy, cx, cy, color_image, roi_right_offset, z_door_depth, found_vertical_planes, color=(0, 255, 255))

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


