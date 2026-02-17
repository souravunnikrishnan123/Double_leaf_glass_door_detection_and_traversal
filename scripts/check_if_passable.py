#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import open3d as o3d
from duration import get_duration_seconds



class Passability_checker:
    """
    Orchestrates the passability check as an object-oriented pipeline.
    Each step mirrors the existing procedural blocks and preserves comments/behavior.
    """

    def __init__(self):
        ns = "~passabilility_check"
        # -----------------------------
        # Backprojection parameters
        # -----------------------------
        self.subsample = rospy.get_param(f"{ns}/back_proj_params/subsample", 2)

        # -----------------------------
        # split eye level params
        # -----------------------------
        
        #self.robot_eye_level_y = rospy.get_param(f"{ns}/filter_points_params/split_eye_level/robot_eye_level_y", -0.1)
        
        # -----------------------------
        # RANSAC floor removal params
        # -----------------------------
        self.max_planes = rospy.get_param(f"{ns}/filter_points_params/ransac/max_planes", 3)
        self.ransac_n = rospy.get_param(f"{ns}/filter_points_params/ransac/n", 3)
        self.distance_threshold = rospy.get_param(f"{ns}/filter_points_params/ransac/distance_threshold", 0.02)
        self.num_iterations = rospy.get_param(f"{ns}/filter_points_params/ransac/num_iterations", 50)
        self.min_inliers = rospy.get_param(f"{ns}/filter_points_params/ransac/min_inliers", 50)
        self.horizontal_tol = rospy.get_param(f"{ns}/filter_points_params/ransac/horizontal_tol", 0.3)

        # -----------------------------
        # Normal filtering params
        # -----------------------------
        self.radius = rospy.get_param(f"{ns}/filter_points_params/normal_filter/radius", 0.15)
        self.max_nn = rospy.get_param(f"{ns}/filter_points_params/normal_filter/max_nn", 40)
        self.ny_thr = rospy.get_param(f"{ns}/filter_points_params/normal_filter/ny_thr", 0.7)

        # -----------------------------
        # Outlier removal params
        # -----------------------------
        self.nb_neighbors = rospy.get_param(f"{ns}/filter_points_params/outlier/nb_neighbors", 50)
        self.std_ratio = rospy.get_param(f"{ns}/filter_points_params/outlier/std_ratio", 1.0)

        # -----------------------------
        # Connected component params
        # -----------------------------
        self.near_z = rospy.get_param(f"{ns}/filter_points_params/cc_filter/near_z", 0.6)
        self.min_area_near = rospy.get_param(f"{ns}/filter_points_params/cc_filter/min_area_near", 640)
        self.minimum_area_far = rospy.get_param(f"{ns}/filter_points_params/cc_filter/minimum_area_far", 600)
        self.min_z = rospy.get_param(f"{ns}/filter_points_params/cc_filter/min_z", 0.1)
        self.max_z = rospy.get_param(f"{ns}/filter_points_params/cc_filter/max_z", 4.0)
        #self.min_z = rospy.get_param(f"{ns}/filter_points_params/cc_filter/min_z", 0.1)
        #self.max_z = rospy.get_param(f"{ns}/filter_points_params/cc_filter/max_z", 4.0)

        # -----------------------------
        # Traversal safety thresholds
        # -----------------------------
        
        self.minimum_depth_points_after_filtering = rospy.get_param(f"{ns}/traversal_params/minimum_depth_points_after_filtering", 20)  # points

        # duration timer
        self.timer = get_duration_seconds()


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

    """
    def split_eye_level(self, points, uv):
        mask_below_robot_eye_level = points[:, 1] > self.robot_eye_level_y
        points_above_robot_eye_level = points[~mask_below_robot_eye_level]
        uv_above_robot_eye_level = uv[~mask_below_robot_eye_level]
        points_below_robot_eye_level = points[mask_below_robot_eye_level]
        uv_below_robot_eye_level = uv[mask_below_robot_eye_level]
        return (
            points_above_robot_eye_level,
            uv_above_robot_eye_level,
            points_below_robot_eye_level,
            uv_below_robot_eye_level,
        )
    """

    def remove_floor_ransac(self, points, uv):
        #ransac
        floor_inliers = None
        for _ in range(self.max_planes):
            if len(points) < self.ransac_n:
                break

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points)
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
            #take the largest horizontal plane as floor. this is an assumption.
            keep_mask = np.ones(points.shape[0], dtype=bool)
            keep_mask[floor_inliers] = False
            points_3_nofloor = points[keep_mask]
            uv_3_nofloor = uv[keep_mask]
            floor_height = float(np.mean(points[floor_inliers, 1]))
        else:
            points_3_nofloor = points
            uv_3_nofloor = uv

        if floor_height is None:
            print("Detected floor height: None (RANSAC failed)")
        else:
            print(f"Detected floor height at y={floor_height:.3f} meters")
            pass
        """
        mask_below_robot_eye_level_for_floor_points_removal = points[:,1] > self.robot_eye_level_y  # keep points above -0.1m (assuming camera is mounted at ~0.5-0.6m height)
        
        points_for_percentile_calculation = points[mask_below_robot_eye_level_for_floor_points_removal]
        # 3) Remove floor by percentile (top 98% in camera-frame Y)
        floor_height = float(np.percentile(points_for_percentile_calculation[:, 1], 98))
        margin = 0.03  # meters above floor
        non_floor_mask = points[:, 1] < (floor_height - margin)
        print(f"Detected floor height at y={floor_height:.3f} meters")
        points_3_nofloor = points[non_floor_mask]
        uv_3_nofloor = uv[non_floor_mask]
        if points_3_nofloor.shape[0] == 0:
            return 0.0, None, None
        """
        return points_3_nofloor, uv_3_nofloor, floor_height

    def normal_filtering(self, points, uv):
        # 4) Keep points whose normals are close to camera Z axis (door plane direction)
        pc_nf = o3d.geometry.PointCloud()
        pc_nf.points = o3d.utility.Vector3dVector(points)
        try:
            pc_nf.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius, max_nn=self.max_nn))
            pc_nf.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
            normals = np.asarray(pc_nf.normals)
        except Exception:
            normals = None

        if normals is None:
            points_4_normal = points
            uv_4_normal = uv
        else:
            vertical_mask = np.abs(normals[:, 1]) < self.ny_thr
            points_4_normal = points[vertical_mask]
            uv_4_normal = uv[vertical_mask]
        return points_4_normal, uv_4_normal

    def outlier_removal(self, points, uv):
        # 2) Statistical 3D outlier removal (keeps mapping by applying indices to uv)
        #print(f"number of points before S3O {len(points)}")
        try:
            pc_clean = o3d.geometry.PointCloud()
            pc_clean.points = o3d.utility.Vector3dVector(points)
            pc_filtered, ind = pc_clean.remove_statistical_outlier(nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio)
            ind = np.array(ind, dtype=int)
            if ind.size == 0:
                return np.empty((0, 3)), np.empty((0, 2))
            points_2_3d_outlier_removal = points[ind]
            uv_2_3d_outlier_removal = uv[ind]
        except Exception:
            points_2_3d_outlier_removal = points
            uv_2_3d_outlier_removal = uv
        if points_2_3d_outlier_removal.shape[0] == 0:
            return np.empty((0, 3)), np.empty((0, 2))
        return points_2_3d_outlier_removal, uv_2_3d_outlier_removal

    def connected_components_filter(self, points, uv, W, H, floor_height):
        #print(f"number of points before CC {len(points)}")
        # --- Remove small patches in image space (connected components) ---
        try:
            # Defaults so visualization doesn't disappear if nothing is filtered
            uv_5_remove_patches = uv
            points_5_remove_patches = points

            area_scale = self.subsample * self.subsample  # compensate for subsampling
            # Depth-aware params
            # be it corridor or local passability check, the depth range is small (like 0.5m for local and 2-3m for corridor). so using min_z and max_z for area scaling.
            # because setting min_area_near and minimum_area_far has nothing to do with depth of points in corridor or local passability check. these are just area thresholds to remove small patches.
            # because points that are about 0.5m to robot is considered as near points( always )
            min_area_near = max(6, int(self.min_area_near / area_scale))
            minimum_area_far = max(6, int(self.minimum_area_far / area_scale))

            # Build 1px mask (no morphology and vectorized)
            uv_int = np.round(uv).astype(int)
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
            uv_valid = uv[valid_uv][valid_labels]
            pts_valid = points[valid_uv][valid_labels]

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
                
                #z_eff = np.clip(z_med, self.min_z, self.max_z)

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
                # need to review if below code is needed or not. because remving points based on  depth variation can cause removal of valid points also.
                if z_mad > (0.30 + 0.05 * (area_px / 100)):
                    continue  # skip component with wildly varying depth
                #area_thresh = base_area_at_1m / (z_eff * z_eff)
                #using above line will not remove patches due to floor reflection. the points in floor near to robot will have high z_eff which will make area_thresh small and hence these patches will be kept instead of removing them.
                #so use minimum_area_far as threshold for far points.

                if (z_med < self.near_z and area_px >= min_area_near) or (area_px >= minimum_area_far):
                    keep_mask[label_mask] = True

            if np.any(keep_mask):
                uv_5_remove_patches = uv_valid[keep_mask]
                points_5_remove_patches = pts_valid[keep_mask]
        except Exception:
            # fall back without filtering on any error
            uv_5_remove_patches = uv
            points_5_remove_patches = points

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



    def run(self, depth_image_in_meters, color_image, corridor_pts_input, corridor_uv_input, z_max):

        H, W = depth_image_in_meters.shape
        self.timer.start(f"check_if_passable--> main passability check")

        self.timer.start(f"check_if_passable--> floor removal")
        points_3_nofloor, uv_3_nofloor, floor_height = self.remove_floor_ransac(
            corridor_pts_input, corridor_uv_input
        )
        self.timer.stop(f"check_if_passable--> floor removal")

        self.timer.start(f"check_if_passable--> normal filtering")
        points_4_normal, uv_4_normal = self.normal_filtering(points_3_nofloor, uv_3_nofloor)
        self.timer.stop(f"check_if_passable--> normal filtering")

        self.timer.start(f"check_if_passable--> 3D outlier removal")
        points_2_3d_outlier_removal, uv_2_3d_outlier_removal = self.outlier_removal(
            points_4_normal, uv_4_normal
        )

        self.timer.stop(f"check_if_passable--> 3D outlier removal")

        self.timer.start(f"check_if_passable--> connected components")
        points_5_remove_patches, uv_5_remove_patches = self.connected_components_filter(
            points_2_3d_outlier_removal, uv_2_3d_outlier_removal, W, H, floor_height
        )
        self.timer.stop(f"check_if_passable--> connected components")

        final_points = points_5_remove_patches
        final_uv = uv_5_remove_patches

        

        if len(final_points) > self.minimum_depth_points_after_filtering:
            front_clearance = np.min(final_points[:, 2])

        else: # no enough points in corridor after filtering
            # means the the corridor is free of obstacles.because lcoal passabiity and even for corridor passaability check we may not be getting any points in small roi around corridor center.
            # that doesnt mean passabilty is not there. it can be there. so we assume front clearance to be large value like z_max.
            front_clearance = z_max

        self.timer.stop(f"check_if_passable--> main passability check")

        self.timer.start(f"check_if_passable--> visualization of final points")
        color_image = self.visualize_points(
            color_image,
            W,
            H,
            [
                (corridor_uv_input, (0, 255, 0)), # green
                (uv_3_nofloor, (255, 0, 255)), # magenta
                (uv_4_normal, (0, 165, 255)), # orange
                (uv_2_3d_outlier_removal, (255, 0, 0)), # blue
                (uv_5_remove_patches, (0, 0, 255)),# red
                (final_uv, (0, 255, 255)) # yellow
            ],
        )

        self.timer.stop(f"check_if_passable--> visualization of final points")


        return front_clearance, color_image, final_points

