#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import open3d as o3d
from create_3d_points_and_detect_ransac_plane import backproject_depth_to_points
from duration import get_duration_seconds
from bird_eye_view import check_passable_birdeye




class BirdsEyePassabilityPipeline:
    """
    Orchestrates the passability check as an object-oriented pipeline.
    Each step mirrors the existing procedural blocks and preserves comments/behavior.
    """

    def __init__(self, keyword="color"):
        self.keyword = keyword

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
        max_planes = 3
        ransac_n = 3
        distance_threshold = 0.02
        num_iterations = 50
        min_inliers = 50
        horizontal_tol = 0.3
        floor_inliers = None
        for _ in range(max_planes):
            if len(points_1_depth_gradient) < ransac_n:
                break

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points_1_depth_gradient)
            plane_model, inliers = pc.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations,
            )
            if len(inliers) < min_inliers:
                break

            a, b, c, d = plane_model
            # Check if the plane is horizontal
            normal = np.array(plane_model[:3])
            normal = normal / np.linalg.norm(normal)
            if abs(abs(normal[1]) - 1.0) < horizontal_tol:
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
            points_4_normal = points_3_nofloor[vertical_mask]
            uv_4_normal = uv_3_nofloor[vertical_mask]
        return points_4_normal, uv_4_normal

    def outlier_removal(self, points_4_normal, uv_4_normal):
        # 2) Statistical 3D outlier removal (keeps mapping by applying indices to uv)
        print(f"number of points before S3O {len(points_4_normal)}")
        try:
            pc_clean = o3d.geometry.PointCloud()
            pc_clean.points = o3d.utility.Vector3dVector(points_4_normal)
            pc_filtered, ind = pc_clean.remove_statistical_outlier(nb_neighbors=50, std_ratio=1)
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

    def connected_components_filter(self, points_2_3d_outlier_removal, uv_2_3d_outlier_removal, W, H, subsample, color_image, floor_height):
        print(f"number of points before CC {len(points_2_3d_outlier_removal)}")
        # --- Remove small patches in image space (connected components) ---
        try:
            # Defaults so visualization doesn't disappear if nothing is filtered
            uv_5_remove_patches = uv_2_3d_outlier_removal
            points_5_remove_patches = points_2_3d_outlier_removal

            area_scale = subsample * subsample  # compensate for subsampling
            # Depth-aware params
            base_area_at_1m = 1400  # px required at ~1m; tune 80–140
            base_area_at_1m /= area_scale
            min_z = 0.1           # clamp near
            max_z = 4.0            # clamp far. 
            near_z = 0.6           # anything closer is "near field"
            min_area_near = 640     # px required for near-field small obstacles
            min_area_near = max(6, int(min_area_near / area_scale))
            minimum_area_far = 600 # absolute minimum area to filter noise
            minimum_area_far = max(6, int(minimum_area_far / area_scale))

            # Build 1px mask (no morphology and vectorized)
            uv_int = np.round(uv_2_3d_outlier_removal).astype(int)
            valid_uv = (
                (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) &
                (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H)
            )
            mask = np.zeros((H, W), np.uint8)
            mask[uv_int[valid_uv, 1], uv_int[valid_uv, 0]] = 255
            #kernel_size = max(7, subsample*2+1)
            kernel_size = subsample
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
                z_eff = np.clip(z_med, min_z, max_z)

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
                if (z_eff < near_z and area_px >= min_area_near) or (area_px >= area_thresh):
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

    def run(self, depth_image_in_meters, fx, fy, cx, cy, color_image, roi_polygon, door_depth):
        timer_full = get_duration_seconds()
        H, W = depth_image_in_meters.shape
        timer_full.start(f"check_if_passable--> main passability check {self.keyword}")

        timer = get_duration_seconds()
        timer.start(f"check_if_passable--> backprojection {self.keyword}")
        color_image = self.visualize_roi(color_image, roi_polygon)

        subsample = 2  # to speed up processing; keep small to not miss small obstacles
        valid_points, uv, z_min, z_max = self.backproject(
            depth_image_in_meters, fx, fy, cx, cy, roi_polygon, door_depth, subsample
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
            points_2_3d_outlier_removal, uv_2_3d_outlier_removal, W, H, subsample, color_image, floor_height
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
        clearance_m, passable, bird_eye_view = check_passable_birdeye(
            final_points,
            door_depth,
            x_min,
            x_max,
            z_min,
            z_max,
            grid_res=0.01,
            required_clearance=0.45,
            max_obstacle_height=-1.5,
            plotname=self.keyword,
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
