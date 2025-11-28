#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import open3d as o3d

from create_3d_points_and_detect_ransac_plane import backproject_depth_to_points, draw_plane_outline_on_image, find_vertical_planes, ransac_plane_from_points, highlight_planes_on_image
from evaluate_detected_ransac_planes import evaluate_plane_candidate
from plots import debug_visualize



# ---------------------------------------------------------
# Main detector: find glass-door plane (if present) and compute distance
# ---------------------------------------------------------
def detect_glass_door_plane(color_image, depth_image_in_meters,
                            fx, fy, cx, cy,
                            hole_fraction_threshold=0.05,       # fraction of holes expected in glass
                            max_inlier_density=1.0,            # inliers / ROI nonzero pixels
                            max_depth_consider=4.0,
                            min_depth_consider=1.0,
                            ):
    """
    Top-level function:
    - Attempts to find a vertical plane in the ROI using RANSAC.
    - Computes statistics on inliers (holes, connectedness, physical size).
    - Uses a small heuristic rule set to decide whether the plane is likely a glass door.
    Returns:
      dict with keys:
        'plane_model' -> [a,b,c,d]
        'distance_m' -> distance camera->plane
        'is_door_candidate' -> True/False
      detected_plane -> list of 4 (x,y) tuples in image pixel coords outlining the detected plane (or None)
    """
    found_vertical_planes = []
    planes = []
    fraction_of_holes_in_plane = 0.0
    inlier_density = 0.0
    H, W = depth_image_in_meters.shape


    # Step 1: Backproject depth -> points, uv coords (only non-zero points get returned)
    points, uv, valid_mask = backproject_depth_to_points(depth_image_in_meters,
                                                         fx, fy, cx, cy,
                                                         max_depth = max_depth_consider,min_depth = min_depth_consider,
                                                         subsample = 1)
    # If there are no valid points in ROI, nothing to do
    if points.shape[0] == 0:
        return {"plane_model": None, "distance_m": None, "is_door_candidate": False}, None, found_vertical_planes


    # ---------------------------------------------------------
    # 💡 Step 2: Voxel Downsampling (AFTER backprojection)
    # ---------------------------------------------------------
    # Setting: voxel_size = 0.01 to 0.015 meters (1–1.5 cm)
    # Why: Reduces redundant neighboring points while preserving
    #      metal frame edges and door boundaries. This balances
    #      point density for RANSAC without losing structure.
    # Use a numpy voxel-grid so we can apply the same index selection to uv.
    voxel_size = 0.008  # meters (0.8 cm)
    # compute voxel coordinates
    vox_coords = np.floor(points / voxel_size).astype(np.int64)
    # find unique voxels and keep one representative point per voxel
    _, unique_idx = np.unique(vox_coords, axis=0, return_index=True)
    unique_idx = np.array(unique_idx, dtype=np.int64)
    # sort so ordering is deterministic (optional)
    unique_idx = np.sort(unique_idx)
    points = points[unique_idx]
    uv = uv[unique_idx]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    

    # ---------------------------------------------------------
    # 💡 Step 3: Pre-filter by surface normal
    # ---------------------------------------------------------
    # Setting: estimate normals and remove points with near-horizontal normals.
    # Why: RANSAC is sensitive to dominant surfaces (like floor/ceiling).
    #      Pre-filter keeps only points whose normals are likely vertical,
    #      i.e., |normal_y| < 0.8 and |normal_z| < 0.8, since we expect
    #      glass door planes to stand roughly vertical to the ground.
    pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
    )
    normals = np.asarray(pc.normals)

    # Filter out points whose normals are too horizontal
    # (optional threshold, adjust if doors tilt slightly)
    #vertical_mask = np.abs(normals[:, 1]) < 0.7 #(cos^-1(0.7) ≈ 45°)

    # Keep points whose normals are close to the Z axis (nx ~ 0, ny ~ 0, |nz| ~ 1)
    # Tune thresholds as needed for your sensor / mounting:
    nx_thr = 0.30   # allow small x component
    ny_thr = 0.30   # allow small y component (reduces horizontal surfaces)
    nz_min = 0.85   # require nz to be fairly large (cos angle ≳ 31.8°)

    z_axis_mask = (np.abs(normals[:, 0]) < nx_thr) & (np.abs(normals[:, 1]) < ny_thr) & (np.abs(normals[:, 2]) >= nz_min)
    
    keep_idx = np.where(z_axis_mask)[0]
    if keep_idx.size == 0:
        return {"plane_model": None, "distance_m": None, "is_door_candidate": False}, None, found_vertical_planes
    pc = pc.select_by_index(keep_idx)
    points = np.asarray(pc.points)
    uv = uv[keep_idx]
    # ---------------------------------------------------------






    found_vertical_planes , found_horizontal_planes, all_planes = find_vertical_planes(points,
                         distance_threshold=0.03,
                         ransac_n=3,
                         num_iterations=100,
                         vertical_tol=0.1,
                         horizontal_tol = 0.2,
                         min_inliers=2000,
                         max_planes=1)
    #print(f"Found {len(found_vertical_planes)} vertical planes and {len(found_horizontal_planes)} horizontal planes")
    
    #highlight_planes_on_image(color_image, uv, found_vertical_planes)
    # now uv corresponds to the 'points' used by RANSAC — pass uv here so inlier indices map correctly
    highlight_planes_on_image(color_image, uv, all_planes)
    
    for i, (plane_model, inlier_indices, inlier_points) in enumerate(all_planes):
        planes.append(draw_plane_outline_on_image(color_image, plane_model, inlier_points, fx, fy, cx, cy, color=(0,255,255), thickness=2))

    for i, (plane_model, inlier_indices, inlier_points) in enumerate(found_vertical_planes):
        detected_plane = draw_plane_outline_on_image(color_image, plane_model, inlier_points, fx, fy, cx, cy, color=(0,255,255), thickness=2)
        mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
        if detected_plane:
            pts = np.array(detected_plane, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
            depths_in_plane = depth_image_in_meters[mask == 255]
            depths_in_plane = depths_in_plane[np.isfinite(depths_in_plane) & (depths_in_plane > 0)]
            fraction_of_holes_in_plane = (np.sum((mask == 255) & ((depth_image_in_meters == 0) | ~np.isfinite(depth_image_in_meters)))) / (np.sum(mask == 255) + 1e-9)
            inlier_density = len(inlier_indices) / (np.sum(mask == 255) + 1e-9)
            if depths_in_plane.size > 0:
                depth_mean = np.mean(depths_in_plane)
                depth_std = np.std(depths_in_plane)
                depth_min = np.min(depths_in_plane)
                depth_max = np.max(depths_in_plane)
                print(f"Depth mean: {depth_mean:.3f} m, std: {depth_std:.3f} m, min: {depth_min:.3f} m, max: {depth_max:.3f} m, hole fraction: {fraction_of_holes_in_plane:.3f}, inlier density: {inlier_density:.3f}")
            #debug_visualize(points, uv, plane_model, inlier_indices)

    if not found_vertical_planes or len(found_vertical_planes[0]) < 2:
        return {"plane_model": None, "distance_m": None, "is_door_candidate": False}, None, found_vertical_planes


    #print(found_vertical_planes)
    plane_model, inlier_indices = found_vertical_planes[0][:2] # get the dominant plane


    # If RANSAC found no plane (rare if points exist), return
    if len(inlier_indices) == 0:
        return {"plane_model": None, "distance_m": None, "is_door_candidate": False}, None, found_vertical_planes

    # convert plane params and compute distance
    a, b, c, d = plane_model  # plane: a*x + b*y + c*z + d = 0
    # distance from camera origin (0,0,0) to plane: |d| / norm(n)
    plane_norm = np.sqrt(a*a + b*b + c*c)
    distance_m = abs(d) / (plane_norm + 1e-12)


    # Step 5: heuristic rules to decide if this plane is likely a glass door (not a wall)
    # Conditions we favor for glass-door:
    #  - plane roughly facing camera (orientation_flag)
    #  - plane distance within sensible range (e.g., < max_depth_consider)
    #  - plane inlier pixel count above a small floor (so RANSAC wasn't from tiny speck)
    #  - inlier density may be low for glass (we accept low density), but the bbox/physical width must be in door range
    #  - hole_fraction inside bbox should be non-trivial for glass (holes due to transparent returns)
    
    is_door = False

    if fraction_of_holes_in_plane >= hole_fraction_threshold or inlier_density <= max_inlier_density:
        is_door = True

    result = {
        "plane_model": plane_model,
        "distance_m": distance_m,
        "is_door_candidate": is_door,
    }

    #processing of detected horizontal planes. our intention is to detect the floor and use it for door state detection
    #if len(found_horizontal_planes) == 1:
        #because we only interested in the floor plane, we only process the first detected horizontal plane
        #because the horizontal planes are sorted based on the number of inliers, the first one should be the floor
        #horizontal_plane_model, horizontal_inlier_indices, horizontal_inlier_points = found_horizontal_planes[0]
        #because of holes in the floor which heavily depends on the texture and lighting, we cant be sure of the inlier density and the area of the plane. 
        

        
    return result, detected_plane, found_vertical_planes

