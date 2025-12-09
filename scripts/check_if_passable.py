#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import open3d as o3d
from create_3d_points_and_detect_ransac_plane import backproject_depth_to_points
from duration import get_duration_seconds




class BirdsEyePassabilityPipeline:
    """
    Orchestrates the passability check as an object-oriented pipeline.
    Each step mirrors the existing procedural blocks and preserves comments/behavior.
    """

    def __init__(self, keyword):
        self.keyword = keyword
        ns = "~door_status_detector"


        # Grouped detector params 
        self.bev_params = {
                "grid_res": rospy.get_param(f"{ns}/bev_params/grid_res", 0.01),
                "required_clearance": rospy.get_param(f"{ns}/bev_params/required_clearance", 0.45),
                "max_obstacle_height": rospy.get_param(f"{ns}/bev_params/max_obstacle_height", -1.5),
                    }
        self.back_proj_params = {
                "subsample": rospy.get_param(f"{ns}/back_proj_params/subsample", 2),
            }
        self.filter_points_params = {
            "ransac": {
                "max_planes": rospy.get_param(f"{ns}/filter_points_params/ransac/max_planes", 3),
                "n": rospy.get_param(f"{ns}/filter_points_params/ransac/n", 3),
                "distance_threshold": rospy.get_param(f"{ns}/filter_points_params/ransac/distance_threshold", 0.02),
                "num_iterations": rospy.get_param(f"{ns}/filter_points_params/ransac/num_iterations", 50),
                "min_inliers": rospy.get_param(f"{ns}/filter_points_params/ransac/min_inliers", 50),
                "horizontal_tol": rospy.get_param(f"{ns}/filter_points_params/ransac/horizontal_tol", 0.3),
            },
            "normal_filter": {
                "radius": rospy.get_param(f"{ns}/filter_points_params/normal_filter/radius", 0.15),
                "max_nn": rospy.get_param(f"{ns}/filter_points_params/normal_filter/max_nn", 40),
                "ny_thr": rospy.get_param(f"{ns}/filter_points_params/normal_filter/ny_thr", 0.7),
            },
            "outlier": {
                "nb_neighbors": rospy.get_param(f"{ns}/filter_points_params/outlier/nb_neighbors", 50),
                "std_ratio": rospy.get_param(f"{ns}/filter_points_params/outlier/std_ratio", 1.0),
            },
            "cc_filter": {
                "near_z": rospy.get_param(f"{ns}/filter_points_params/cc_filter/near_z", 0.6),
                "min_area_near": rospy.get_param(f"{ns}/filter_points_params/cc_filter/min_area_near", 640),
                "minimum_area_far": rospy.get_param(f"{ns}/filter_points_params/cc_filter/minimum_area_far", 600),
                "base_area_at_1m": rospy.get_param(f"{ns}/filter_points_params/cc_filter/base_area_at_1m", 1400),
                "min_z": rospy.get_param(f"{ns}/filter_points_params/cc_filter/min_z", 0.1),
                "max_z": rospy.get_param(f"{ns}/filter_points_params/cc_filter/max_z", 4.0),
            },
        }

        self.subsample = self.back_proj_params.get("subsample", 2)

        self.max_planes = self.filter_points_params["ransac"].get("max_planes", 3)
        self.ransac_n = self.filter_points_params["ransac"].get("n", 3)
        self.distance_threshold = self.filter_points_params["ransac"].get("distance_threshold", 0.02)
        self.num_iterations = self.filter_points_params["ransac"].get("num_iterations", 50)
        self.min_inliers = self.filter_points_params["ransac"].get("min_inliers", 50)
        self.horizontal_tol = self.filter_points_params["ransac"].get("horizontal_tol", 0.3)   
        
        self.radius = self.filter_points_params["normal_filter"].get("radius", 0.15)
        self.max_nn = self.filter_points_params["normal_filter"].get("max_nn", 40)
        self.ny_thr = self.filter_points_params["normal_filter"].get("ny_thr", 0.7)

        self.nb_neighbors = self.filter_points_params["outlier"].get("nb_neighbors", 50)
        self.std_ratio = self.filter_points_params["outlier"].get("std_ratio", 1.0)
        # anything closer is "near field"   
        self.near_z = self.filter_points_params["cc_filter"].get("near_z", 0.6)
        # px required for near-field small obstacles
        self.min_area_near = self.filter_points_params["cc_filter"].get("min_area_near", 640)
        
        self.minimum_area_far = self.filter_points_params["cc_filter"].get("minimum_area_far", 600)
        self.base_area_at_1m = self.filter_points_params["cc_filter"].get("base_area_at_1m", 1400)
        self.min_z = self.filter_points_params["cc_filter"].get("min_z", 0.1)
        self.max_z = self.filter_points_params["cc_filter"].get("max_z", 4.0)

        self.grid_res = self.bev_params.get("grid_res", 0.01)
        self.required_clearance = self.bev_params.get("required_clearance", 0.45)
        self.max_obstacle_height = self.bev_params.get("max_obstacle_height", -1.5)
        


    def visualize_roi(self, color_image, roi_polygon):
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
        return color_image

    def backproject(self, depth_image_in_meters, fx, fy, cx, cy, roi_polygon, door_depth, subsample):
        #  Define Z extents relative to door
        z_min, z_max = 0.05, door_depth + 1.5  # meters
        valid_points, uv, _ = backproject_depth_to_points(
            depth_image_in_meters, fx, fy, cx, cy,
            max_depth=z_max, min_depth=z_min,
            subsample=subsample, roi_polygon=roi_polygon
        )
        return valid_points, uv, z_min, z_max

    def split_eye_level(self, valid_points, uv):
        mask_below_robot_eye_level = valid_points[:, 1] > 0.0
        points_above_robot_eye_level = valid_points[~mask_below_robot_eye_level]
        uv_above_robot_eye_level = uv[~mask_below_robot_eye_level]
        points_1_depth_gradient = valid_points[mask_below_robot_eye_level]
        uv_1_depth_gradient = uv[mask_below_robot_eye_level]
        return (
            mask_below_robot_eye_level,
            points_above_robot_eye_level,
            uv_above_robot_eye_level,
            points_1_depth_gradient,
            uv_1_depth_gradient,
        )

    def remove_floor_ransac(self, points_1_depth_gradient, uv_1_depth_gradient):
        #ransac
        floor_inliers = None
        for _ in range(self.max_planes):
            if len(points_1_depth_gradient) < self.ransac_n:
                break

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points_1_depth_gradient)
            plane_model, inliers = pc.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.num_iterations,
            )
            if len(inliers) < self.min_inliers:
                break

            a, b, c, d = plane_model
            # Check if the plane is horizontal
            normal = np.array(plane_model[:3])
            normal = normal / np.linalg.norm(normal)
            if abs(abs(normal[1]) - 1.0) < self.horizontal_tol:
                floor_inliers = np.array(inliers, dtype=int)
                break

        floor_height = None
        if floor_inliers is not None and floor_inliers.size > 0:
            #take the largest horizontal plane as floor
            keep_mask = np.ones(points_1_depth_gradient.shape[0], dtype=bool)
            keep_mask[floor_inliers] = False
            points_3_nofloor = points_1_depth_gradient[keep_mask]
            uv_3_nofloor = uv_1_depth_gradient[keep_mask]
            floor_height = float(np.mean(points_1_depth_gradient[floor_inliers, 1]))
        else:
            points_3_nofloor = points_1_depth_gradient
            uv_3_nofloor = uv_1_depth_gradient

        print(f"Detected floor height at y={floor_height:.3f} meters")
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
            return 0.0, None, None
        """
        return points_3_nofloor, uv_3_nofloor, floor_height

    def normal_filtering(self, points_3_nofloor, uv_3_nofloor):
        # 4) Keep points whose normals are close to camera Z axis (door plane direction)
        pc_nf = o3d.geometry.PointCloud()
        pc_nf.points = o3d.utility.Vector3dVector(points_3_nofloor)
        try:
            pc_nf.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius, max_nn=self.max_nn))
            pc_nf.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
            normals = np.asarray(pc_nf.normals)
        except Exception:
            normals = None

        if normals is None:
            points_4_normal = points_3_nofloor
            uv_4_normal = uv_3_nofloor
        else:
            vertical_mask = np.abs(normals[:, 1]) < self.ny_thr
            points_4_normal = points_3_nofloor[vertical_mask]
            uv_4_normal = uv_3_nofloor[vertical_mask]
        return points_4_normal, uv_4_normal

    def outlier_removal(self, points_4_normal, uv_4_normal):
        # 2) Statistical 3D outlier removal (keeps mapping by applying indices to uv)
        print(f"number of points before S3O {len(points_4_normal)}")
        try:
            pc_clean = o3d.geometry.PointCloud()
            pc_clean.points = o3d.utility.Vector3dVector(points_4_normal)
            pc_filtered, ind = pc_clean.remove_statistical_outlier(nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio)
            ind = np.array(ind, dtype=int)
            if ind.size == 0:
                return np.empty((0, 3)), np.empty((0, 2))
            points_2_3d_outlier_removal = points_4_normal[ind]
            uv_2_3d_outlier_removal = uv_4_normal[ind]
        except Exception:
            points_2_3d_outlier_removal = points_4_normal
            uv_2_3d_outlier_removal = uv_4_normal
        if points_2_3d_outlier_removal.shape[0] == 0:
            return np.empty((0, 3)), np.empty((0, 2))
        return points_2_3d_outlier_removal, uv_2_3d_outlier_removal

    def connected_components_filter(self, points_2_3d_outlier_removal, uv_2_3d_outlier_removal, W, H, floor_height):
        print(f"number of points before CC {len(points_2_3d_outlier_removal)}")
        # --- Remove small patches in image space (connected components) ---
        try:
            # Defaults so visualization doesn't disappear if nothing is filtered
            uv_5_remove_patches = uv_2_3d_outlier_removal
            points_5_remove_patches = points_2_3d_outlier_removal

            area_scale = self.subsample * self.subsample  # compensate for subsampling
            # Depth-aware params
            base_area_at_1m = self.base_area_at_1m / area_scale 
            min_area_near = max(6, int(self.min_area_near / area_scale))
            minimum_area_far = max(6, int(self.minimum_area_far / area_scale))

            # Build 1px mask (no morphology and vectorized)
            uv_int = np.round(uv_2_3d_outlier_removal).astype(int)
            valid_uv = (
                (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) &
                (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
            )
            mask = np.zeros((H, W), np.uint8)
            mask[uv_int[valid_uv, 1], uv_int[valid_uv, 0]] = 255
            #kernel_size = max(7, subsample*2+1)
            kernel_size = self.subsample
            #“fill” the sparse mask a bit to restore local connectivity  due to subsampling
            #Use subsample itself as the kernel size so it scales automatically
            mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

            if num <= 1:
                return np.empty((0, 3)), np.empty((0, 2))

            # Vectorized mapping: label lookup per UV pixel
            lbl_values = labels[
                np.clip(uv_int[valid_uv, 1], 0, H - 1),
                np.clip(uv_int[valid_uv, 0], 0, W - 1)
            ]

            # Filter out background (label 0)
            valid_labels = lbl_values > 0
            lbl_values = lbl_values[valid_labels]
            uv_valid = uv_2_3d_outlier_removal[valid_uv][valid_labels]
            pts_valid = points_2_3d_outlier_removal[valid_uv][valid_labels]

            # Precompute per-component area and median depth
            unique_lbls, inverse_idx = np.unique(lbl_values, return_inverse=True)
            areas = stats[unique_lbls, cv2.CC_STAT_AREA]

            keep_mask = np.zeros(len(lbl_values), dtype=bool)

            for i, lbl in enumerate(unique_lbls):
                """
                ####        visualization of connected components   #####
                # Create mask for this component
                comp_mask = (labels == lbl).astype(np.uint8)

                # Create a visualization copy
                vis = color_image.copy()

                # Colorize the component (green overlay)
                vis[comp_mask == 1] = (0, 0, 255)   # set component pixels to red

                # Optionally, dim background to highlight the component better
                background_mask = (comp_mask == 0)
                vis[background_mask] = (vis[background_mask] * 0.3).astype(np.uint8)

                # Add label text
                x, y, w, h, area = stats[lbl]
                cv2.putText(vis, f"ID:{lbl}, area:{area}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                cv2.imshow(f"Component {lbl}", vis)
                ####    visualization of connected components  end  ####    
                """

                label_mask = inverse_idx == i
                area_px = areas[i]
                z_vals = pts_valid[label_mask, 2]
                z_vals = z_vals[~np.isnan(z_vals)]
                if z_vals.size == 0:
                    continue
                z_med = np.median(z_vals)
                z_eff = np.clip(z_med, self.min_z, self.max_z)

                y_vals = pts_valid[label_mask, 1]
                y_vals = y_vals[~np.isnan(y_vals)]
                if y_vals.size == 0:
                    continue
                y_med = np.median(y_vals)
                if floor_height is not None and y_med > floor_height + 0.1:
                    continue

                #there could be floor noisy points due to reflection which have high z value but small area. so these should be removed.
                #depths above 3.5m are removed during back projection itself.but there can be noisy floor points with depth less than 3.5m but higher than actual floor depth.
                z_mad = np.median(np.abs(z_vals - np.median(z_vals)))
                if z_mad > (0.30 + 0.05 * (area_px / 100)):
                    continue  # skip component with wildly varying depth
                #area_thresh = base_area_at_1m / (z_eff * z_eff)
                area_thresh = minimum_area_far
                if (z_eff < self.near_z and area_px >= min_area_near) or (area_px >= area_thresh):
                    keep_mask[label_mask] = True

            if np.any(keep_mask):
                uv_5_remove_patches = uv_valid[keep_mask]
                points_5_remove_patches = pts_valid[keep_mask]
        except Exception:
            # fall back without filtering on any error
            uv_5_remove_patches = uv_2_3d_outlier_removal
            points_5_remove_patches = points_2_3d_outlier_removal

        print(f"number of points after CC {len(points_5_remove_patches)}")
        return points_5_remove_patches, uv_5_remove_patches

    def visualize_points(self, color_image, W, H, uv_sets_with_color):
        #  mark kept points on the image for debugging
        try:
            if color_image is not None:
                for uv_set, color in uv_sets_with_color:
                    if uv_set.size > 0:
                        px = np.round(uv_set[:, 0]).astype(int)
                        py = np.round(uv_set[:, 1]).astype(int)
                        valid_mask = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                        color_image[py[valid_mask], px[valid_mask]] = color
        except Exception:
            pass
        return color_image


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


    def run(self, depth_image_in_meters, fx, fy, cx, cy, color_image, roi_polygon, door_depth):
        timer_full = get_duration_seconds()
        H, W = depth_image_in_meters.shape
        timer_full.start(f"check_if_passable--> main passability check {self.keyword}")

        timer = get_duration_seconds()
        timer.start(f"check_if_passable--> backprojection {self.keyword}")
        color_image = self.visualize_roi(color_image, roi_polygon)

        valid_points, uv, z_min, z_max = self.backproject(
            depth_image_in_meters, fx, fy, cx, cy, roi_polygon, door_depth, self.subsample 
        )

        if len(valid_points) == 0:
            return 0.0, None, None

        (
            mask_below_robot_eye_level,
            points_above_robot_eye_level,
            uv_above_robot_eye_level,
            points_1_depth_gradient,
            uv_1_depth_gradient,
        ) = self.split_eye_level(valid_points, uv)

        timer.stop(f"check_if_passable--> backprojection {self.keyword}")

        timer.start(f"check_if_passable--> floor removal {self.keyword}")
        points_3_nofloor, uv_3_nofloor, floor_height = self.remove_floor_ransac(
            points_1_depth_gradient, uv_1_depth_gradient
        )
        timer.stop(f"check_if_passable--> floor removal {self.keyword}")

        timer.start(f"check_if_passable--> normal filtering {self.keyword}")
        points_4_normal, uv_4_normal = self.normal_filtering(points_3_nofloor, uv_3_nofloor)
        timer.stop(f"check_if_passable--> normal filtering {self.keyword}")

        timer.start(f"check_if_passable--> 3D outlier removal {self.keyword}")
        points_2_3d_outlier_removal, uv_2_3d_outlier_removal = self.outlier_removal(
            points_4_normal, uv_4_normal
        )
        if points_2_3d_outlier_removal.shape[0] == 0:
            return 0.0, None, None
        timer.stop(f"check_if_passable--> 3D outlier removal {self.keyword}")

        timer.start(f"check_if_passable--> connected components {self.keyword}")
        points_5_remove_patches, uv_5_remove_patches = self.connected_components_filter(
            points_2_3d_outlier_removal, uv_2_3d_outlier_removal, W, H, floor_height
        )
        timer.stop(f"check_if_passable--> connected components {self.keyword}")

        final_points = np.vstack([points_5_remove_patches, points_above_robot_eye_level])
        final_uv = np.vstack([uv_5_remove_patches, uv_above_robot_eye_level])
        """
        #will preserve order. for plottin and bev order doesnt matter
        idx_above = np.where(~mask_below_robot_eye_level)[0]
        idx_below = np.where(mask_below_robot_eye_level)[0] # where normal filtering was applied

        after_filter_mask = np.isin(idx_below, np.where(np.isin(valid_points, points_5_remove_patches).all(axis=1))[0])

        filtered_idx_below = idx_below[after_filter_mask]
        
        # keep all idx_above plus filtered subset of idx_below
        if filtered_idx_below.size == 0 and idx_above.size == 0:
            return 0.0, None, None

        keep_idx = (
            filtered_idx_below if idx_above.size == 0
            else (idx_above if filtered_idx_below.size == 0
                    else np.concatenate([idx_above, filtered_idx_below]))
        )

        # Preserve original order (optional)
        keep_idx = np.sort(keep_idx)
        final_points = valid_points[keep_idx]
        final_uv = uv[keep_idx] 
        """


        x_vals = valid_points[:, 0]
        x_min = float(np.percentile(x_vals, 5))
        x_max = float(np.percentile(x_vals, 95))

        timer_full.stop(f"check_if_passable--> main passability check {self.keyword}")

        timer.start(f"check_if_passable--> BEV passability check {self.keyword}")
        clearance_m, passable, bird_eye_view = self.check_passable_birdeye(
            final_points,
            x_min,
            x_max,
            z_min,
            z_max
        )
        timer.stop(f"check_if_passable--> BEV passability check {self.keyword}")

        timer.start(f"check_if_passable--> visualization of final points {self.keyword}")
        color_image = self.visualize_points(
            color_image,
            W,
            H,
            [
                (uv, (0, 255, 0)),
                (uv_3_nofloor, (255, 0, 255)),
                (uv_4_normal, (0, 165, 255)),
                (uv_2_3d_outlier_removal, (255, 0, 0)),
                (uv_5_remove_patches, (0, 0, 255)),
                (final_uv, (0, 255, 255)),
            ],
        )
        timer.stop(f"check_if_passable--> visualization of final points {self.keyword}")

        point_cloud_vertical_ratio = len(final_points) / len(valid_points)
        cv2.putText(
            color_image,
            f"Passable ratio: {point_cloud_vertical_ratio:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 0),
            2,
        )

        print(f"Point cloud vertical ratio: {point_cloud_vertical_ratio:.3f}")
        return point_cloud_vertical_ratio, color_image, bird_eye_view
