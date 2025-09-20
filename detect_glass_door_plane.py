import numpy as np

from create_3d_points_and_detect_ransac_plane import backproject_depth_to_points, ransac_plane_from_points
from evaluate_detected_ransac_planes import evaluate_plane_candidate
from plots import debug_visualize



# ---------------------------------------------------------
# Main detector: find glass-door plane (if present) and compute distance
# ---------------------------------------------------------
def detect_glass_door_plane(depth_image_in_meters,
                            fx, fy, cx, cy,
                            roi=None,
                            # RANSAC / geometric params
                            distance_threshold=0.03,   # meters; larger for glass noisy returns
                            ransac_n=3,
                            num_iterations=1500,
                            # heuristics for deciding door vs wall
                            min_inlier_pixels=200,     # scoping threshold - change for resolution
                            min_largest_comp_fraction=0.05,
                            max_wall_width_m=4.0,
                            expected_door_width_m=(0.6, 1.4),   # plausible door widths
                            expected_door_height_m=(1.8, 2.3),  # plausible door heights
                            hole_fraction_threshold=0.20,       # fraction of holes expected in glass
                            min_inlier_density=0.01,            # inliers / ROI nonzero pixels
                            max_depth_consider=5.0,
                            debug=False):
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
        'stats' -> dictionary returned from evaluate_plane_candidate
    """

    H, W = depth_image_in_meters.shape

    # Step 1: Backproject depth -> points, uv coords (only non-zero points get returned)
    points, uv, valid_mask = backproject_depth_to_points(depth_image_in_meters,
                                                         fx, fy, cx, cy,
                                                         max_depth = max_depth_consider,
                                                         roi = roi,
                                                         subsample = 1)
    # If there are no valid points in ROI, nothing to do
    if points.shape[0] == 0:
        return {"plane_model": None, "distance_m": None, "is_door_candidate": False, "stats": None, "reason": "no_valid_points"}

    # Step 2: Run RANSAC plane fit on points (robust to outliers)
    plane_model, inlier_indices = ransac_plane_from_points(points,
                                                           distance_threshold=distance_threshold,
                                                           ransac_n=ransac_n,
                                                           num_iterations=num_iterations)


    debug_visualize(points, uv, plane_model, inlier_indices)


    # If RANSAC found no plane (rare if points exist), return
    if len(inlier_indices) == 0:
        return {"plane_model": None, "distance_m": None, "is_door_candidate": False, "stats": None, "reason": "no_plane_found"}

    # convert plane params and compute distance
    a, b, c, d = plane_model  # plane: a*x + b*y + c*z + d = 0
    # distance from camera origin (0,0,0) to plane: |d| / norm(n)
    plane_norm = np.sqrt(a*a + b*b + c*c)
    distance_m = abs(d) / (plane_norm + 1e-12)

    # Step 3: geometric orientation check: we want planes roughly facing camera
    # In camera coordinates: Z axis points forward, Y axis points down. A closed glass door facing camera
    # will have a normal vector with a large Z component (|c| close to 1) and small Y component.
    # So require the plane normal to have significant c (z-component).
    z_comp = abs(c) / (plane_norm + 1e-12)
    y_comp = abs(b) / (plane_norm + 1e-12)

    # quick normal orientation test (tunable)
    if z_comp < 0.6:
        # plane not sufficiently facing camera -> likely side wall or angled object
        # still continue to compute stats (maybe door is slightly angled) but mark as low confidence
        orientation_flag = False
    else:
        orientation_flag = True

    # Step 4: gather inlier points and uv for image-space evaluation
    inlier_indices = np.array(inlier_indices, dtype=np.int32)
    points_inliers = points[inlier_indices, :]
    uv_inliers = uv[inlier_indices, :]

    # Evaluate image-space heuristics (hole fraction, bbox extents, connectedness)
    stats = evaluate_plane_candidate(depth_image_in_meters, uv, uv_inliers, points_inliers, roi, H, W)

    # compute inlier density = inliers / nonzero pixels inside ROI
    # number of non-zero depth pixels in ROI:
    if roi is None:
        roi_y0, roi_y1, roi_x0, roi_x1 = 0, H, 0, W
    else:
        roi_y0, roi_y1, roi_x0, roi_x1 = roi
    roi_nonzero = np.count_nonzero((depth_image_in_meters[roi_y0:roi_y1, roi_x0:roi_x1] > 0.0) & np.isfinite(depth_image_in_meters[roi_y0:roi_y1, roi_x0:roi_x1]))
    if roi_nonzero == 0:
        inlier_density = 0.0
    else:
        inlier_density = float(stats["inlier_pixel_count"]) / float(roi_nonzero + 1e-9)

    # Step 5: heuristic rules to decide if this plane is likely a glass door (not a wall)
    # Conditions we favor for glass-door:
    #  - plane roughly facing camera (orientation_flag)
    #  - plane distance within sensible range (e.g., < max_depth_consider)
    #  - plane inlier pixel count above a small floor (so RANSAC wasn't from tiny speck)
    #  - inlier density may be low for glass (we accept low density), but the bbox/physical width must be in door range
    #  - hole_fraction inside bbox should be non-trivial for glass (holes due to transparent returns)
    #  - largest connected inlier component should occupy a reasonable fraction of bbox (not utterly scattered)
    is_door = False
    reasons = []

    # require orientation roughly facing camera
    if not orientation_flag:
        reasons.append("plane_not_facing_camera")
    # require at least some inliers
    if stats["inlier_pixel_count"] < min_inlier_pixels:
        reasons.append("too_few_inliers")
    # physical width test: door width plausible
    if not (expected_door_width_m[0] <= stats["physical_width_m"] <= expected_door_width_m[1]):
        # if width too big, likely wall; if too small, could be object or partial
        reasons.append("width_out_of_expected_range")
    # physical height test: doors have about expected height
    if not (expected_door_height_m[0] <= stats["physical_height_m"] <= expected_door_height_m[1]):
        reasons.append("height_out_of_expected_range")
    # hole fraction: glass often yields holes -> expect larger hole fraction than wall
    if stats["hole_fraction"] < hole_fraction_threshold:
        reasons.append("not_enough_holes_for_glass")
    # largest connected region: we expect some contiguous area of inliers (the door plane fragments)
    if stats["largest_comp_fraction"] < min_largest_comp_fraction:
        reasons.append("inliers_too_fragmented")
    # inlier density check: we accept low density but if density is extremely low it might be spurious
    if inlier_density < min_inlier_density:
        reasons.append("inlier_density_too_low")

    # Final decision:
    # If many failure reasons, it's not a door. We consider it a door if most checks pass.
    # A conservative scoring approach:
    fail_count = len(reasons)
    # allow at most 2 failing soft checks (tunes: you can be stricter/looser)
    if fail_count <= 2 and orientation_flag:
        is_door = True
    else:
        is_door = False

    result = {
        "plane_model": plane_model,
        "distance_m": distance_m,
        "is_door_candidate": is_door,
        "stats": stats,
        "inlier_density": inlier_density,
        "orientation_z_comp": z_comp,
        "orientation_y_comp": y_comp,
        "reasons": reasons
    }

    if debug:
        result["debug_points_count"] = points.shape[0]
        result["debug_inliers_count"] = int(len(inlier_indices))

    return result

