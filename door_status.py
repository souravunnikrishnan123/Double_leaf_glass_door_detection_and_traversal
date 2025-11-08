import cv2
import numpy as np
import pyrealsense2 as rs 
import open3d as o3d
from duration import get_duration_seconds

from create_3d_points_and_detect_ransac_plane import backproject_depth_to_points, ransac_plane_from_points

from bird_eye_view import check_passable_birdeye



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

def check_side_roi_against_door(depth_image_in_meters, fx, fy, cx, cy, color_image, roi_polygon, door_depth, found_vertical_planes, color = (0, 255, 0)):
    """
    Check how much of ROI depth matches door reference depth.
    """
    timer = get_duration_seconds()
    H, W = depth_image_in_meters.shape
    
    valid_points, uv,_ = backproject_depth_to_points(depth_image_in_meters, fx, fy, cx, cy, max_depth=30, subsample=1,roi_polygon = roi_polygon)

    if len(valid_points) == 0:
        return 0.0

    valid_points = np.asarray(valid_points,dtype=np.float32)
    uv = np.asarray(uv,dtype=np.float32)

    # Filter: only consider values within [-10%, +200%] of door_depth
    lower = door_depth * 0.9
    upper = door_depth * 20

    valid_z_points = valid_points[:, 2]
    filtered_mask = (valid_z_points >= lower) & (valid_z_points <= upper)

    if not np.any(filtered_mask):
        return 0.0
    
    filtered_points = valid_points[filtered_mask]
    filtered_uv = uv[filtered_mask]

    close_mask = np.abs(filtered_points[:, 2] - door_depth) <= door_depth * 0.05
    close_points = filtered_points[close_mask]
    close_uv = filtered_uv[close_mask]

    removed_close_points = np.empty((0, 3), dtype=np.float32)
    removed_close_uv = np.empty((0, 2), dtype=np.float32)


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
        #vectorized planarity check
        plane_distances =  np.abs((a*close_points[:,0] + b*close_points[:,1] + c*close_points[:,2] + d) / plane_norm)

        coplanar_mask = plane_distances < coplanar_thresh

        # Save removed points before filtering
        removed_close_points = close_points[~coplanar_mask]
        removed_close_uv = close_uv[~coplanar_mask]

        close_points = close_points[coplanar_mask]
        close_uv = close_uv[coplanar_mask]
        
        
    #print(len(bottom_points))
    fraction_close = close_points.shape[0] / filtered_points.shape[0]

    #print(f"Fraction close: {fraction_close:.3f}")
    timer.get_duration("check_side_roi_against_door")
    
    timer2 = get_duration_seconds()
    # Visualization
    if color_image is not None:
        # Draw ROI polygon
        cv2.polylines(color_image, [roi_polygon.astype(np.int32)], isClosed=True, color=color, thickness=2)
        
        # Prepare coordinate arrays
        uv_filtered = filtered_uv.astype(np.int32)
        uv_close = close_uv.astype(np.int32)
        uv_removed = removed_close_uv.astype(np.int32)

        # Validity masks to avoid out-of-bounds
        valid_filtered = (uv_filtered[:, 0] >= 0) & (uv_filtered[:, 0] < W) & (uv_filtered[:, 1] >= 0) & (uv_filtered[:, 1] < H)
        valid_close = (uv_close[:, 0] >= 0) & (uv_close[:, 0] < W) & (uv_close[:, 1] >= 0) & (uv_close[:, 1] < H)
        valid_removed = (uv_removed[:, 0] >= 0) & (uv_removed[:, 0] < W) & (uv_removed[:, 1] >= 0) & (uv_removed[:, 1] < H)

        # Draw "all filtered points" as blue
        color_image[uv_filtered[valid_filtered][:, 1], uv_filtered[valid_filtered][:, 0]] = (255, 0, 0)
        # Draw "close to door depth" points as cyan
        color_image[uv_close[valid_close][:, 1], uv_close[valid_close][:, 0]] = (0, 255, 255)
        # Draw "removed (coplanar/floor)" points as green
        color_image[uv_removed[valid_removed][:, 1], uv_removed[valid_removed][:, 0]] = (0, 255, 0)


    timer2.get_duration("check_side_roi_against_door: visualization")                 
    return fraction_close


def check_if_passable(depth_image_in_meters, fx, fy, cx, cy, color_image, roi_polygon, door_depth, plotname="passable"):
    """
    Check how much of ROI depth matches door reference depth.
    """
    timer = get_duration_seconds()
    H, W = depth_image_in_meters.shape

    timer1 = get_duration_seconds()
    # Visualization: translucent ROI fill + outline
    try:
        if color_image is not None:
            overlay = color_image.copy()
            fill_color = (50, 50, 200)  # BGR (reddish-blue)
            cv2.fillPoly(overlay, [roi_polygon.astype(np.int32)], fill_color)
            alpha = 0.25
            cv2.addWeighted(overlay, alpha, color_image, 1.0 - alpha, 0, color_image)
            cv2.polylines(color_image, [roi_polygon.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    except Exception:
        pass

    valid_points, uv,_ = backproject_depth_to_points(depth_image_in_meters, fx, fy, cx, cy, max_depth=4, subsample=2,roi_polygon = roi_polygon)

    if len(valid_points) == 0:
        return 0.0
    
    # Ensure numpy arrays
    points_np = np.asarray(valid_points)
    uv = np.asarray(uv)


    # 1) Image-local denoise + hole-fix: close tiny holes in validity, then median
    try:
        depth_img = depth_image_in_meters.astype(np.float32)

        # Close small holes in the valid-depth mask
        #Inpaint only tiny holes (by area), not big gaps.
        valid = (depth_img > 0).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        valid_closed = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Inpaint only newly "filled" hole pixels (prevents biasing valid depths)
        hole_mask = ((valid_closed > 0) & (valid == 0)).astype(np.uint8) * 255
        if np.any(hole_mask):
            # Inpaint supports 8U/32F; we use 32F meters
            depth_img = cv2.inpaint(depth_img, hole_mask, 3.0, cv2.INPAINT_NS)

        # Median filter to suppress spikes while preserving edges
        depth_med = cv2.medianBlur(depth_img, 5)
    except Exception:
        depth_med = depth_image_in_meters

    # Image-local median filter to reject specular / spike depths
    depth_dev_thresh = 0.12  # meters; tune as needed

    # Round UVs, clip to valid image bounds
    uv_int = np.round(uv).astype(int)
    valid_mask = (
        (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) &
        (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
    )

    u = uv_int[:, 0][valid_mask]
    v = uv_int[:, 1][valid_mask]

    d_img = depth_image_in_meters[v, u]
    d_med = depth_med[v, u]

    valid_depth_mask = (d_img > 0) & (~np.isnan(d_img)) & (~np.isnan(d_med))
    diff_mask = np.abs(d_img - d_med) <= depth_dev_thresh
    combined_mask = np.zeros(len(points_np), dtype=bool)
    combined_mask[valid_mask] = valid_depth_mask & diff_mask

    if not combined_mask.any():
        return 0.0

    points_1_depth_gradient = points_np[combined_mask]
    uv_1_depth_gradient = uv[combined_mask]

    timer1.get_duration("check_if_passable: depth gradient filter")
    # 2) Statistical 3D outlier removal (keeps mapping by applying indices to uv)
    timer2 = get_duration_seconds()
    try:
        pc_clean = o3d.geometry.PointCloud()
        pc_clean.points = o3d.utility.Vector3dVector(points_1_depth_gradient)
        pc_filtered, ind = pc_clean.remove_statistical_outlier(nb_neighbors=16, std_ratio=1.5)
        ind = np.array(ind, dtype=int)
        if ind.size == 0:
            return 0.0
        points_2_3d_outlier_removal = points_1_depth_gradient[ind]
        uv_2_3d_outlier_removal = uv_1_depth_gradient[ind]
    except Exception:
        points_2_3d_outlier_removal = points_1_depth_gradient
        uv_2_3d_outlier_removal = uv_1_depth_gradient
        pass

    if points_2_3d_outlier_removal.shape[0] == 0:
        return 0.0
    timer2.get_duration("check_if_passable: 3D outlier removal")

    timer3 = get_duration_seconds()
    # 3) Remove floor by percentile (top 98% in camera-frame Y)
    # If your vertical axis is not column 1, change the index accordingly.
    floor_height = float(np.percentile(points_2_3d_outlier_removal[:, 1], 98))
    margin = 0.02  # meters above floor
    non_floor_mask = points_2_3d_outlier_removal[:, 1] < (floor_height - margin)
    print(f"Detected floor height at y={floor_height:.3f} meters")
    points_3_nofloor = points_2_3d_outlier_removal[non_floor_mask]
    uv_3_nofloor = uv_2_3d_outlier_removal[non_floor_mask]
    if points_3_nofloor.shape[0] == 0:
        return 0.0
    
    timer3.get_duration("check_if_passable: floor removal")

    timer4 = get_duration_seconds()
    # 4) Keep points whose normals are close to camera Z axis (door plane direction)
    pc_nf = o3d.geometry.PointCloud()
    pc_nf.points = o3d.utility.Vector3dVector(points_3_nofloor)
    try:
        pc_nf.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
        pc_nf.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
        normals = np.asarray(pc_nf.normals)

    except Exception:
        normals = None

    if normals is None:
        points_4_normal = points_3_nofloor
        uv_4_normal = uv_3_nofloor
    else:
        ny_thr = 0.7
        vertical_mask = np.abs(normals[:, 1]) < ny_thr
        keep_idx = np.where(vertical_mask)[0]
        if keep_idx.size == 0:
            return 0.0
        points_4_normal = points_3_nofloor[keep_idx]
        uv_4_normal = uv_3_nofloor[keep_idx]    

    timer4.get_duration("check_if_passable: normal filtering")

        # --- Remove small patches in image space (connected components) ---
    timer5 = get_duration_seconds()
    try:
        # Defaults so visualization doesn't disappear if nothing is filtered
        uv_5_remove_patches = uv_4_normal
        points_5_remove_patches = points_4_normal

        # Depth-aware params
        base_area_at_1m = 140  # px required at ~1m; tune 80–140
        min_z = 0.1           # clamp near
        max_z = 4.0            # clamp far
        near_z = 0.8           # anything closer is "near field"
        min_area_near = 8      # px required for near-field small obstacles

        # Build 1px mask (no morphology and vectorized)

        uv_int = np.round(uv_4_normal).astype(int)
        valid_uv = (
            (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) &
            (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
        )
        mask = np.zeros((H, W), np.uint8)
        mask[uv_int[valid_uv, 1], uv_int[valid_uv, 0]] = 255

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num <= 1:
            return 0.0  # only background
        
# Vectorized mapping: label lookup per UV pixel
        lbl_values = labels[
            np.clip(uv_int[valid_uv, 1], 0, H - 1),
            np.clip(uv_int[valid_uv, 0], 0, W - 1)
        ]

        # Filter out background (label 0)
        valid_labels = lbl_values > 0
        lbl_values = lbl_values[valid_labels]
        uv_valid = uv_4_normal[valid_uv][valid_labels]
        pts_valid = points_4_normal[valid_uv][valid_labels]

        # Precompute per-component area and median depth
        unique_lbls, inverse_idx = np.unique(lbl_values, return_inverse=True)
        areas = stats[unique_lbls, cv2.CC_STAT_AREA]

        keep_mask = np.zeros(len(lbl_values), dtype=bool)
        for i, lbl in enumerate(unique_lbls):
            label_mask = inverse_idx == i
            area_px = areas[i]
            z_vals = pts_valid[label_mask, 2]
            z_vals = z_vals[~np.isnan(z_vals)]
            if z_vals.size == 0:
                continue
            z_med = np.median(z_vals)
            z_eff = np.clip(z_med, min_z, max_z)
            area_thresh = base_area_at_1m / (z_eff * z_eff)
            if (z_eff < near_z and area_px >= min_area_near) or (area_px >= area_thresh):
                keep_mask[label_mask] = True

        if np.any(keep_mask):
            uv_5_remove_patches = uv_valid[keep_mask]
            points_5_remove_patches = pts_valid[keep_mask]
    except Exception:
        # fall back without filtering on any error
        uv_5_remove_patches = uv_4_normal
        points_5_remove_patches = points_4_normal    
    timer5.get_duration("check_if_passable: remove small patches")

    x_vals = points_np[:, 0]
    # Robust percentiles to reject outliers, then pad slightly
    x_min = float(np.percentile(x_vals, 5))
    x_max = float(np.percentile(x_vals, 95))

    timer.get_duration("check_if_passable: main passability check")
    timer6 = get_duration_seconds()
    # Check passability using bird-eye view grid
    clearance_m, passable, occ_map = check_passable_birdeye(
    points_5_remove_patches,
    door_depth,
    x_min,
    x_max,  
    uv_pts=uv_5_remove_patches,         # matching uv coordinates for the 3D points
    grid_res=0.02,
    required_clearance=0.45,
    max_obstacle_height=-2.0,      # tune for your robot / camera mounting
    plotname=plotname

)
    timer6.get_duration("check_if_passable: BEV passability check")
    
    timer7 = get_duration_seconds()
    #  mark kept points on the image for debugging
    try:
        if color_image is not None:
            """
            for uv_set, color in [
                (uv, (128, 128, 128)),     # raw points: gray
                (uv_1_depth_gradient, (0, 0, 255)), # after depth dev filter: red
                (uv_2_3d_outlier_removal, (255, 0, 255)), # after statistical outlier removal: magenta
                (uv_3_nofloor, (0, 165, 255)), # floor removed: orange
                (uv_4_normal, (255, 0, 0)),    # normals filtered: blue
                (uv_5_remove_patches, (0, 255, 255))    # final kept (CC + size): yellow
            ]:
                if uv_set.size > 0:
                    px = np.round(uv_set[:, 0]).astype(int)
                    py = np.round(uv_set[:, 1]).astype(int)
                    valid_mask = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                    color_image[py[valid_mask], px[valid_mask]] = color
            """
            for uv_set, color in [
                (uv, (0, 255, 0)),     # raw points: green
                (uv_1_depth_gradient, (0, 0, 255)), # after depth dev filter: red
                (uv_2_3d_outlier_removal, (255, 0, 255)), # after statistical outlier removal: magenta
                (uv_3_nofloor, (0, 165, 255)), # floor removed: orange
                (uv_4_normal, (255, 0, 0)),    # normals filtered: blue
                (uv_5_remove_patches, (0, 255, 255))    # final kept (CC + size): yellow
            ]:
                if uv_set.size > 0:
                    px = np.round(uv_set[:, 0]).astype(int)
                    py = np.round(uv_set[:, 1]).astype(int)
                    valid_mask = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                    color_image[py[valid_mask], px[valid_mask]] = color



        
    except Exception:
        pass
    timer7.get_duration("check_if_passable: visualization of final points")

    point_cloud_vertical_ratio = len(points_5_remove_patches)/len(valid_points)
    cv2.putText(color_image, f"Passable ratio: {point_cloud_vertical_ratio:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    cv2.imshow(plotname, color_image)


    
    print(f"Point cloud vertical ratio: {point_cloud_vertical_ratio:.3f}")
    return point_cloud_vertical_ratio

    



def detect_door_state(depth_image_in_meters, fx, fy, cx, cy,color_image, roi_polygon_left, roi_polygon_right,found_vertical_planes,
                      roi_width=40, margin=10, threshold=0.05, z_door_depth=None, plotname = "door_state_based_on_color_image"):
    """
    Decide OPEN/CLOSED based on ROIs and door depth reference.
    """
    color_image_for_passable_check = color_image.copy()


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


    

    if door_state == "Open (on left side)":
        roi_polygon_left_corners = extract_roi_corners(roi_left_offset)
        roi_left_offset_full_height = extend_roi_polygon_to_full_height(roi_polygon_left_corners, color_image.shape[0])
        passable_fraction  = check_if_passable(depth_image_in_meters, fx, fy, cx, cy, color_image_for_passable_check, roi_left_offset_full_height, z_door_depth, plotname=plotname)
    
    
    elif door_state == "Open (on right side)":
        roi_polygon_right_corners = extract_roi_corners(roi_right_offset)
        roi_right_offset_full_height = extend_roi_polygon_to_full_height(roi_polygon_right_corners, color_image.shape[0])
        passable_fraction = check_if_passable(depth_image_in_meters, fx, fy, cx, cy, color_image_for_passable_check, roi_right_offset_full_height, z_door_depth, plotname=plotname)

    # Overlay decision text
    cv2.putText(color_image, f"Door State: {door_state}", (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
    
    cv2.putText(color_image, f"Left match: {left_match:.2f}", (30, 170),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.putText(color_image, f"Right match: {right_match:.2f}", (30, 210),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return door_state


