import numpy as np
import cv2
import matplotlib.pyplot as plt

def check_passable_birdeye(points_above_floor,
                           door_depth,
                            x_min,
                           x_max,
                           uv_pts=None,
                           grid_res=0.02,
                           required_clearance=0.45,
                           max_obstacle_height=-1.5,
                           visualize=True):
    """
    BEV passability check that (a) computes x_min/x_max from the provided ROI
    polygon (in image pixel coordinates) if available, and (b) ignores obstacles
    whose height (Y coordinate in camera frame) is above max_obstacle_height.

    Args:
        points_above_floor (np.ndarray): Nx3 points in camera frame (already floor-removed).
        door_depth (float): Z (depth) of door plane in same camera frame units (meters).
        roi_polygon_uv (np.ndarray or list, optional): polygon in image pixel coords (Nx2).
            If provided together with uv_pts, the function will compute x_min/x_max from the
            3D points that project inside this polygon.
        uv_pts (np.ndarray, optional): Nx2 array of (u,v) pixel coordinates mapping one-to-one
            with points_above_floor. Required if roi_polygon_uv is passed.
        roi_width (float): fallback ROI width (meters) used if roi_polygon_uv not supplied.
        grid_res (float): BEV cell resolution in meters.
        required_clearance (float): required horizontal corridor width (m) for passability.
        max_obstacle_height (float): ignore points with Y > max_obstacle_height (meters). set to 1.5m. because robot will only go through 
              the place where human can go through.
            - Camera-frame convention assumed: Y is vertical (positive upward).
            - This value should be the maximum height at which an obstacle would block the robot.
              Example: if robot body top is at 0.35 m from floor and camera origin is approximately
              at floor-level + camera_mount_height, pick accordingly. Tune as needed.
        visualize (bool): show BEV image when True.

    Returns:
        max_clearance_m (float), passable (bool), occ_map (np.ndarray or None)
    """

    # 1) quick exits
    if points_above_floor is None or len(points_above_floor) == 0:
        return 0.0, False, None

    # 2) apply height filter: ignore high points (they are above robot height -> not blocking)
    #    points_above_floor[:,1] is vertical (Y). Keep points whose Y <= max_obstacle_height.
    #    NOTE: The meaning of 'max_obstacle_height' depends on camera-to-floor reference.
    low_height_mask = points_above_floor[:, 1] >= max_obstacle_height
    points_filtered = points_above_floor[low_height_mask]

    # If uv_pts provided, filter them in the same way so indices match
    if uv_pts is not None:
        uv_pts = np.asarray(uv_pts)
        if uv_pts.shape[0] == len(points_above_floor):
            uv_filtered = uv_pts[low_height_mask]
        else:
            # mismatch: can't reliably use uv for ROI extraction; null out uv_filtered
            uv_filtered = None
    else:
        uv_filtered = None

    if len(points_filtered) == 0:
        return 0.0, False, None


    # 4) Define Z extents relative to door
    z_min, z_max = 0.1, door_depth + 1.5  # meters

    # 5) Filter points to the horizontal (X) and depth (Z) window we will consider
    mask_roi_space = (
        (points_filtered[:, 2] > z_min) &
        (points_filtered[:, 2] < z_max) &
        (points_filtered[:, 0] > x_min) &
        (points_filtered[:, 0] < x_max)
    )
    roi_points = points_filtered[mask_roi_space]

    if len(roi_points) == 0:
        return 0.0, False, None

    # 6) Prepare BEV grid sizes
    x = roi_points[:, 0]  # X-axis = horizontal axis (left-right direction relative to camera)
    z = roi_points[:, 2]  # Z-axis = forward direction (depth away from camera)

    # Compute the number of grid cells in X and Z based on desired resolution
    # e.g. if x_min=-0.5, x_max=0.5 and grid_res=0.02 → x_bins = 50
    x_bins = max(1, int(np.ceil((x_max - x_min) / grid_res)))
    z_bins = max(1, int(np.ceil((z_max - z_min) / grid_res)))

    # 7) Initialize occupancy grid (z_bins rows, x_bins columns)
    #  Initialize state grid (0=unknown, 1=free, 2=occupied)
    occ_map = np.zeros((z_bins, x_bins), dtype=np.uint8)

    # 8) Rasterize points into grid cells
    # Convert each 3D point’s X,Z position into a BEV grid index
    occ_mask = np.zeros((z_bins, x_bins), dtype=np.uint8)
    xi = np.clip(((x - x_min) / grid_res).astype(int), 0, x_bins - 1)
    zi = np.clip(((z - z_min) / grid_res).astype(int), 0, z_bins - 1)

    # Mark those grid cells as occupied (1)
    occ_mask[zi, xi] = 1

    # 9) Morphological cleanup (remove tiny holes/noise)
    occ_mask = cv2.morphologyEx(occ_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    occ_map[occ_mask == 1] = 2  # occupied

    # 9) Raycast-style free space: for each X column, mark rows from sensor (near) up to nearest occupied as free
    for c in range(x_bins):
        rows = np.flatnonzero(occ_map[:, c] == 2)
        if rows.size > 0:
            z_near = int(rows.min())
            if z_near > 0:
                occ_map[0:z_near, c] = np.maximum(occ_map[0:z_near, c], 1)  # set to free (don’t overwrite occupied)

    no_obstacle_cols = np.where(np.sum(occ_map == 2, axis=0) == 0)[0]
    if no_obstacle_cols.size > 0:
        occ_map[:, no_obstacle_cols] = 1  # mark entire column as free if no occupied cells


    # 10) Compute free fraction per column (X direction)
    #Compute how much of each column (X direction) is free
    free_cols = (occ_map == 1).astype(np.uint8)
    known_cols = ((occ_map == 1) | (occ_map == 2)).astype(np.uint8)

    free_counts = np.sum(free_cols, axis=0).astype(np.float32)
    known_counts = np.sum(known_cols, axis=0).astype(np.float32)
    free_fraction = np.divide(free_counts, np.maximum(known_counts, 1.0))  # avoid /0

    coverage = np.sum(known_cols, axis=0) / float(z_bins)

    min_free_frac = 0.90   # ≥90% of column must be free
    min_coverage = 0.60    # ≥60% of column must be observed (not unknown)

    free_widths = (free_fraction >= min_free_frac).astype(np.uint8)


    # 11) find largest continuous free column run
    max_clear_cells = 0
    current = 0
    for val in free_widths:
        if val == 1:
            current += 1
            if current > max_clear_cells:
                max_clear_cells = current
        else:
            current = 0

    # 12) convert cells -> meters, decide passability
    max_clearance_m = max_clear_cells * grid_res
    passable = max_clearance_m >= required_clearance

    # 13) visualization (optional)
    if visualize:
        bev_vis = np.zeros((z_bins, x_bins, 3), dtype=np.uint8)
        bev_vis[occ_map == 2] = (0, 0, 255)     # red occupied
        bev_vis[occ_map == 1] = (0, 255, 0)     # green free
        bev_vis[occ_map == 0] = (80, 80, 80)    # gray unknown

        # highlight free columns (those that met the >90% free criterion)
        free_col_indices = np.where(free_widths == 1)[0]
        for col in free_col_indices:
            bev_vis[:, col] = (0, 200, 255)     # orange for clear corridor

        # resize for display - keep aspect ratio but make it readable
        if bev_vis.shape[1] > 0:
            scale = 400.0 / bev_vis.shape[1]
        else:
            scale = 1.0
        bev_vis_resized = cv2.resize(bev_vis,
                                     (int(bev_vis.shape[1] * scale), int(bev_vis.shape[0] * scale)),
                                     interpolation=cv2.INTER_NEAREST)
        
        # display with near at bottom, far at top (flip vertically for OpenCV)
        bev_vis_display = cv2.flip(bev_vis_resized, 0)
        # display with axes in meters (extent = [x_min, x_max, z_min, z_max])
        #Near/far are inverted because OpenCV shows row 0 at the top. Flip the BEV image vertically before imshow.
        cv2.namedWindow("BEV occupancy", cv2.WINDOW_NORMAL)
        cv2.imshow("BEV occupancy", bev_vis_display)
        cv2.waitKey(1)

    return max_clearance_m, passable, occ_map
