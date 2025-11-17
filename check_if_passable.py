import cv2
import numpy as np
import open3d as o3d
from create_3d_points_and_detect_ransac_plane import backproject_depth_to_points
from duration import get_duration_seconds
from bird_eye_view import check_passable_birdeye



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
    subsample =2 # to speed up processing. but donot set to high value like 4. 
    #because for for door status checking it was okay as we were trying to find points on the door frame which are big objects. but now for passabiity we need to 
    #detect small objects on the ground which may get missed if we use high subsampling value.
    valid_points, uv,_ = backproject_depth_to_points(depth_image_in_meters, fx, fy, cx, cy, max_depth=5, min_depth = 0.1, subsample=subsample,roi_polygon = roi_polygon)

    if len(valid_points) == 0:
        return 0.0
    
    # Ensure numpy arrays
    points_np = np.asarray(valid_points)
    uv = np.asarray(uv)

    """
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
    """
    points_1_depth_gradient = valid_points
    uv_1_depth_gradient = uv

    timer3 = get_duration_seconds()
    #ransac
    max_planes = 3
    ransac_n=3
    distance_threshold=0.02
    num_iterations = 50
    min_inliers =50
    horizontal_tol = 0.3
    floor_inliers = None
    for _ in range(max_planes):
        if len(points_1_depth_gradient) < ransac_n:
            break

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points_1_depth_gradient)
        plane_model, inliers = pc.segment_plane(distance_threshold=distance_threshold,
                                                ransac_n=ransac_n,
                                                num_iterations=num_iterations)
        if len(inliers) < min_inliers:
            break

        a, b, c, d = plane_model
        # Check if the plane is horizontal
        normal = np.array(plane_model[:3])
        normal = normal / np.linalg.norm(normal)
        if abs(abs(normal[1]) - 1.0) < horizontal_tol:
            floor_inliers = np.array(inliers, dtype=int)
            break
    
    
    if floor_inliers is not None and floor_inliers.size > 0:
        #take the largest horizontal plane as floor
        keep_mask = np.ones(points_1_depth_gradient.shape[0], dtype=bool)
        keep_mask[floor_inliers] = False
        points_3_nofloor = points_1_depth_gradient[keep_mask]
        uv_3_nofloor = uv_1_depth_gradient[keep_mask]
    else:
        points_3_nofloor = points_1_depth_gradient
        uv_3_nofloor = uv_1_depth_gradient


    timer3.get_duration("check_if_passable: floor removal")
    
    """
    mask_below_robot_eye_level_for_floor_points_removal = points_1_depth_gradient[:,1] > -0.1  # keep points above -0.1m (assuming camera is mounted at ~0.5-0.6m height)
    
    points_for_percentile_calculation = points_1_depth_gradient[mask_below_robot_eye_level_for_floor_points_removal]
    # 3) Remove floor by percentile (top 98% in camera-frame Y)
    floor_height = float(np.percentile(points_for_percentile_calculation[:, 1], 98))
    margin = 0.03  # meters above floor
    non_floor_mask = points_1_depth_gradient[:, 1] < (floor_height - margin)
    print(f"Detected floor height at y={floor_height:.3f} meters")
    points_3_nofloor = points_1_depth_gradient[non_floor_mask]
    uv_3_nofloor = uv_1_depth_gradient[non_floor_mask]
    if points_3_nofloor.shape[0] == 0:
        return 0.0
    """

    
    

    mask_below_robot_eye_level = points_3_nofloor[:,1] > 0.0  # keep points above -0.1m (assuming camera is mounted at ~0.5-0.6m height)



    

    timer4 = get_duration_seconds()
    # 4) Keep points whose normals are close to camera Z axis (door plane direction)
    pc_nf = o3d.geometry.PointCloud()
    pc_nf.points = o3d.utility.Vector3dVector(points_3_nofloor[mask_below_robot_eye_level])
    try:
        pc_nf.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=40))
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

        idx_above = np.where(~mask_below_robot_eye_level)[0]
        idx_below = np.where(mask_below_robot_eye_level)[0] # where normal filtering was applied

        filtered_idx_below = idx_below[vertical_mask]
        
        # keep all idx_above plus filtered subset of idx_below
        if filtered_idx_below.size == 0 and idx_above.size == 0:
            return 0.0

        keep_idx = (
            filtered_idx_below if idx_above.size == 0
            else (idx_above if filtered_idx_below.size == 0
                  else np.concatenate([idx_above, filtered_idx_below]))
        )

        # Preserve original order (optional)
        keep_idx = np.sort(keep_idx)
        points_4_normal = points_3_nofloor[keep_idx]
        uv_4_normal = uv_3_nofloor[keep_idx]    

    timer4.get_duration("check_if_passable: normal filtering")

        # 2) Statistical 3D outlier removal (keeps mapping by applying indices to uv)
    timer2 = get_duration_seconds()
    try:
        pc_clean = o3d.geometry.PointCloud()
        pc_clean.points = o3d.utility.Vector3dVector(points_4_normal)
        pc_filtered, ind = pc_clean.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.3)
        ind = np.array(ind, dtype=int)
        if ind.size == 0:
            return 0.0
        points_2_3d_outlier_removal = points_4_normal[ind]
        uv_2_3d_outlier_removal = uv_4_normal[ind]
    except Exception:
        points_2_3d_outlier_removal = points_4_normal
        uv_2_3d_outlier_removal = uv_4_normal
        pass

    if points_2_3d_outlier_removal.shape[0] == 0:
        return 0.0
    timer2.get_duration("check_if_passable: 3D outlier removal")


        # --- Remove small patches in image space (connected components) ---
    timer5 = get_duration_seconds()
    try:
        # Defaults so visualization doesn't disappear if nothing is filtered
        uv_5_remove_patches = uv_2_3d_outlier_removal
        points_5_remove_patches = points_2_3d_outlier_removal 

        area_scale = subsample * subsample  # compensate for subsampling
        # Depth-aware params
        base_area_at_1m = 560  # px required at ~1m; tune 80–140
        base_area_at_1m /= area_scale
        min_z = 0.1           # clamp near
        max_z = 4.0            # clamp far
        near_z = 0.8           # anything closer is "near field"
        min_area_near = 80     # px required for near-field small obstacles
        min_area_near = max(3, int(min_area_near / area_scale))

        # Build 1px mask (no morphology and vectorized)

        uv_int = np.round(uv_2_3d_outlier_removal).astype(int)
        valid_uv = (
            (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) &
            (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
        )
        mask = np.zeros((H, W), np.uint8)
        mask[uv_int[valid_uv, 1], uv_int[valid_uv, 0]] = 255
        #“fill” the sparse mask a bit to restore local connectivity  due to subsampling
        #Use subsample itself as the kernel size so it scales automatically
        mask = cv2.dilate(mask, np.ones((subsample+1, subsample+1), np.uint8), iterations=1)

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
        uv_valid = uv_2_3d_outlier_removal[valid_uv][valid_labels]
        pts_valid = points_2_3d_outlier_removal [valid_uv][valid_labels]

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
        uv_5_remove_patches = uv_2_3d_outlier_removal
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
                (uv, (0, 255, 0)),     # raw points: black
                (uv_3_nofloor, (255, 0, 255)), # after statistical outlier removal: magenta
                (uv_4_normal, (0, 165, 255)), # floor removed: orange
                (uv_2_3d_outlier_removal, (255, 0, 0)),    # normals filtered: blue
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

    