"""Geometry and point-cloud helpers retained for detector experiments."""

import cv2
import numpy as np
import open3d as o3d

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


def find_planes(points,
                         distance_threshold=0.05,
                         ransac_n=3,
                         num_iterations=1000,
                         vertical_tol=0.3,
                         horizontal_tol = 0.3,
                         min_inliers=10000,
                         max_planes=4):
  """
  Iteratively find up to max_planes vertical planes in the point cloud.
  Returns a list of (plane_model, inlier_indices) for vertical planes.
  """

  remaining_points = points.copy()
  remaining_indices = np.arange(points.shape[0])  # Track original indices
  found_vertical_planes = []
  found_horizontal_planes = []
  all_planes = []

  for _ in range(max_planes):
    if len(remaining_points) < ransac_n:
          break

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(remaining_points)
    plane_model, inliers = pc.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=ransac_n,
                                            num_iterations=num_iterations)
    
    all_planes.append((plane_model, remaining_indices[inliers], remaining_points[inliers]))
    
    if len(inliers) < min_inliers:
        break
    
    
    #to get edge points for the next iteration of RANSAC plane detection
    #It prevents dominant planes (like the floor) from "stealing" all points near boundaries
    inlier_points = remaining_points[inliers]
    a, b, c, d = plane_model
    distances = np.abs(a*inlier_points[:,0] + b*inlier_points[:,1] + c*inlier_points[:,2] + d) / np.linalg.norm([a,b,c])
    #  a tighter threshold is defined for "core" inliers. this is to get edge points for the next iteration of RANSAC plane detection
    core_threshold = 0.02     # 2cm

    # Indices of core inliers (to remove)
    core_inlier_mask = distances < core_threshold
    core_inlier_indices = np.array(inliers)[core_inlier_mask]

    # Indices of edge inliers (to keep for next iteration)
    edge_inlier_mask = ~core_inlier_mask
    edge_inlier_indices = np.array(inliers)[edge_inlier_mask]

    mask = np.ones(len(remaining_points), dtype=bool)
    # Remove inliers from remaining_points for next iteration. we are only removig the core inliers to get edge points for the next iteration of RANSAC plane detection.
    #because, for slim door planes, floor is taking lot of points near the door boundary
    
    mask[core_inlier_indices] = False  # Remove core inliers


    # Check if the plane is vertical
    normal = np.array(plane_model[:3])
    normal = normal / np.linalg.norm(normal)
    #print("Detected plane normal:", normal)
    if abs(abs(normal[2]) - 1.0) < vertical_tol:
        # Map inliers to original indices
        orig_inlier_indices = remaining_indices[inliers]  
        # Save vertical plane
        found_vertical_planes.append((plane_model, orig_inlier_indices, remaining_points[inliers]))
        print(f"Found vertical plane with {len(inliers)} inliers.")
        mask[edge_inlier_indices] = False  # remove the edge inliers for next iteration
        #here because we detected a potential door plane, we remove all inliers (core + edge) to get to avoid getting the same plane again
    
    elif abs(abs(normal[1]) - 1.0) < horizontal_tol:
        # Map inliers to original indices
        orig_inlier_indices = remaining_indices[inliers]  
        found_horizontal_planes.append((plane_model, core_inlier_indices, remaining_points[inliers]))
        print(f"Found horizontal plane with {len(inliers)} inliers.")
    

    remaining_points = remaining_points[mask]
    remaining_indices = remaining_indices[mask]

  return found_vertical_planes, found_horizontal_planes, all_planes


# ---------------------------------------------------------
# Utility: run Open3D RANSAC plane segmentation on point cloud
# RANSAC works by randomly sampling minimal sets of points (here, 3 points per sample)
# and iteratively searching for the plane that has the most inliers—points that lie within a specified distance from the plane.

# ---------------------------------------------------------

def ransac_plane_from_points(points,
                             distance_threshold=0.05,
                             ransac_n=3,
                             num_iterations=1000):
    """
    Fit a plane to a set of 3D points using Open3D's RANSAC.
    - points: (N,3) numpy array
    - distance_threshold: in meters, maximum distance from model to be considered an inlier
    - ransac_n: minimal points to estimate plane (3)
    - num_iterations: RANSAC iterations
    Returns:
      plane_model: [a,b,c,d] as floats for plane ax+by+cz+d=0
      inlier_indices: list of indices into the input points that are inliers
    """
    # convert numpy points to Open3D point cloud
    if points is None or len(points) == 0 or points.shape[1] != 3:
        return None, []
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    # segment_plane returns plane_model and list of inlier indices
    plane_model, inliers = pc.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=ransac_n,
                                            num_iterations=num_iterations)
    return plane_model, inliers

def check_passable_birdeye(self, points_above_floor,
                            x_min,
                        x_max,
                        z_min,
                        z_max):
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
    low_height_mask = points_above_floor[:, 1] >= self.max_obstacle_height
    points_filtered = points_above_floor[low_height_mask]


    if len(points_filtered) == 0:
        return 0.0, False, None


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
    x_bins = max(1, int(np.ceil((x_max - x_min) / self.grid_res)))
    z_bins = max(1, int(np.ceil((z_max - z_min) / self.grid_res)))

    # 7) Initialize occupancy grid (z_bins rows, x_bins columns)
    #  Initialize state grid (0=unknown, 1=free, 2=occupied)
    occ_map = np.zeros((z_bins, x_bins), dtype=np.uint8)

    # 8) Rasterize points into grid cells
    # Convert each 3D point’s X,Z position into a BEV grid index
    occ_mask = np.zeros((z_bins, x_bins), dtype=np.uint8)
    #xi = np.clip(((x - x_min) / grid_res).astype(int), 0, x_bins - 1)
    #zi = np.clip(((z - z_min) / grid_res).astype(int), 0, z_bins - 1)

    xi = ((x - x_min) / self.grid_res)
    zi = ((z - z_min) / self.grid_res)

    # floor manually to avoid rounding-up distortions
    xi = np.floor(xi).astype(int)
    zi = np.floor(zi).astype(int)

    # clamp
    xi = np.clip(xi, 0, x_bins - 1)
    zi = np.clip(zi, 0, z_bins - 1)

    # Mark those grid cells as occupied (1)
    occ_mask[zi, xi] = 1

    # 9) Morphological cleanup (remove tiny holes/noise)
    
    occ_mask = cv2.morphologyEx(occ_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    #occ_mask = cv2.dilate(occ_mask, np.ones((3, 3), np.uint8), iterations=1)
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

    # 10) Columns considered clear only if they contain no occupied cells anywhere
    free_widths = (np.sum(occ_map == 2, axis=0) == 0).astype(np.uint8)

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
    max_clearance_m = max_clear_cells * self.grid_res
    passable = max_clearance_m >= self.required_clearance


    # 13) visualization (optional)

    bev_vis = np.zeros((z_bins, x_bins, 3), dtype=np.uint8)
    bev_vis[occ_map == 2] = (0, 0, 255)     # red occupied
    bev_vis[occ_map == 1] = (0, 255, 0)     # green free
    bev_vis[occ_map == 0] = (80, 80, 80)    # gray unknown

    # highlight free columns (those that met the >99.9% free criterion)
    
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
    #bev_vis_display = bev_vis_resized

    # display with axes in meters (extent = [x_min, x_max, z_min, z_max])
    #Near/far are inverted because OpenCV shows row 0 at the top. Flip the BEV image vertically before imshow.
    #cv2.namedWindow(plotname+"BEV", cv2.WINDOW_NORMAL)
    #cv2.imshow(plotname+"BEV", bev_vis_display)

        

    return max_clearance_m, passable, bev_vis_display

