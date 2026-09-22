#!/usr/bin/env python3
"""Point-cloud filtering pipeline used to measure forward clearance."""

import rospy
import cv2
import numpy as np
import open3d as o3d
from duration import get_duration_seconds


def grouped_median(values, group_index, num_groups):
    """
    Median of ``values`` within each group, ignoring NaN entries.

    Computing medians with one pass per group costs one full-length scan per
    group. Sorting by ``(group, value)`` once instead makes every group a
    contiguous slice, so all medians come from a single ordering.

    Args:
        values:
            Flat array of samples.

        group_index:
            Group label per sample, in ``[0, num_groups)``.

        num_groups:
            Total number of groups.

    Returns:
        Tuple ``(medians, counts)``. ``counts`` gives the number of non-NaN
        samples per group; ``medians`` is NaN wherever that count is zero.

    Notes:
        Matches ``numpy.median`` semantics, averaging the two central samples
        when a group has an even number of values.
    """
    finite = ~np.isnan(values)
    vals = values[finite]
    groups = group_index[finite]

    counts = np.bincount(groups, minlength=num_groups)
    medians = np.full(num_groups, np.nan, dtype=np.float64)
    if vals.size == 0:
        return medians, counts

    # Primary key is the group, secondary key the value, so each group occupies
    # a sorted contiguous run.
    order = np.lexsort((vals, groups))
    sorted_vals = vals[order]

    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    non_empty = counts > 0
    # For an even count the median averages the two central samples; for an odd
    # count both indices coincide on the single central sample.
    lo = starts[non_empty] + (counts[non_empty] - 1) // 2
    hi = starts[non_empty] + counts[non_empty] // 2
    medians[non_empty] = 0.5 * (sorted_vals[lo] + sorted_vals[hi])
    return medians, counts


class Passability_checker:
    """
    Filter corridor points and report the nearest remaining obstacle.

    The stages remove the floor, reject unsuitable normals and statistical
    outliers, then discard small disconnected image-space patches. If too few
    obstacle points survive, the requested range is treated as clear.

    Attributes:
        subsample:
            Depth backprojection stride used to scale connected-component area
            thresholds.

        max_planes:
            Maximum number of RANSAC plane attempts during floor removal.

        ransac_n:
            Number of points sampled for each plane hypothesis.

        distance_threshold:
            Maximum RANSAC point-to-plane distance in meters.

        num_iterations:
            RANSAC iterations per plane attempt.

        min_inliers:
            Minimum inlier count for accepting a plane.

        horizontal_tol:
            Tolerance for classifying a plane normal as camera-Y aligned.

        radius:
            Neighborhood radius used for normal estimation.

        max_nn:
            Maximum neighbors used for normal estimation.

        ny_thr:
            Maximum absolute normal Y component retained as obstacle-like.

        nb_neighbors:
            Neighbor count used by statistical outlier removal.

        std_ratio:
            Standard-deviation multiplier used by outlier removal.

        near_z:
            Depth separating near and far connected-component thresholds.

        min_area_near:
            Minimum image-space area for near components.

        minimum_area_far:
            Minimum image-space area for farther components.

        min_z:
            Configured lower depth bound reserved for connected-component
            threshold scaling.

        max_z:
            Configured upper depth bound reserved for connected-component
            threshold scaling.

        minimum_depth_points_after_filtering:
            Minimum surviving point count required to report a measured
            obstacle rather than the far range.

        timer:
            Shared named-stage duration recorder.
    """

    def __init__(self):
        """
        Load passability-filter parameters from the ROS parameter server.

        All parameters live below the historical
        ``~passabilility_check`` namespace. Defaults are chosen for the
        configured camera resolution and point-cloud subsampling.
        """
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

        # Diagnostic overlays are six scatter writes over the full point set, so
        # they are skipped entirely unless diagnostics are requested.
        self.enable_visualization = rospy.get_param("~enable_visualization", True)

        # duration timer
        self.timer = get_duration_seconds()


    def visualize_roi(self, color_image, roi_polygon):
        """
        Draw a translucent polygon showing the passability input region.

        Args:
            color_image:
                BGR image modified in place. ``None`` is accepted.

            roi_polygon:
                Polygon convertible to an int32 OpenCV point array.

        Returns:
            The same image reference, including any successfully drawn overlay.

        Notes:
            Drawing failures are deliberately ignored so diagnostics cannot
            interrupt the safety pipeline.
        """
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
        """
        Remove the first sufficiently large horizontal RANSAC plane.

        Plane fitting stops when a camera-Y-aligned normal is found. Inlier
        points belonging to that plane are removed while UV correspondence is
        preserved.

        Args:
            points:
                ``N x 3`` camera-frame point array.

            uv:
                ``N x 2`` image coordinates paired with ``points``.

        Returns:
            Tuple ``(non_floor_points, non_floor_uv, floor_height)``.
            ``floor_height`` is the mean camera-frame Y of floor inliers, or
            ``None`` when no horizontal plane is found.

        Notes:
            The method assumes the first accepted horizontal plane is the
            floor. When fitting fails, all input points are retained.
        """
        #ransac
        # Floor is usually the largest horizontal surface in the corridor ROI,
        # so stop at the first sufficiently supported horizontal fit.
        floor_inliers = None
        # Index of each remaining point in the original array, so inliers found
        # after peeling can still be reported in the caller's coordinates.
        remaining_idx = np.arange(points.shape[0], dtype=np.intp)
        remaining_points = points
        for _ in range(self.max_planes):
            if remaining_points.shape[0] < self.ransac_n:
                break

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(remaining_points)
            plane_model, inliers = pc.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.num_iterations,
            )
            if len(inliers) < self.min_inliers:
                break

            inliers = np.asarray(inliers, dtype=np.intp)
            # Check if the plane is horizontal
            normal = np.asarray(plane_model[:3], dtype=np.float64)
            normal = normal / np.linalg.norm(normal)
            if abs(abs(normal[1]) - 1.0) < self.horizontal_tol:
                floor_inliers = remaining_idx[inliers]
                break

            # The dominant plane was not the floor. Remove it before retrying;
            # without this the next pass would re-fit the identical cloud and
            # simply re-find the same plane at full RANSAC cost.
            keep = np.ones(remaining_points.shape[0], dtype=bool)
            keep[inliers] = False
            remaining_points = remaining_points[keep]
            remaining_idx = remaining_idx[keep]

        floor_height = None
        if floor_inliers is not None and floor_inliers.size > 0:
            #take the largest horizontal plane as floor. this is an assumption.
            # Apply the same mask to XYZ and UV; later visual checks depend on
            # that one-to-one correspondence.
            keep_mask = np.ones(points.shape[0], dtype=bool)
            keep_mask[floor_inliers] = False
            points_3_nofloor = points[keep_mask]
            uv_3_nofloor = uv[keep_mask]
            floor_height = float(np.mean(points[floor_inliers, 1]))
        else:
            points_3_nofloor = points
            uv_3_nofloor = uv

        if floor_height is None:
            rospy.logdebug("Detected floor height: None (RANSAC failed)")
        else:
            rospy.logdebug("Detected floor height at y=%.3f meters", floor_height)
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
        """
        Keep obstacle-like surfaces and preserve their pixel correspondence.

        Args:
            points:
                ``N x 3`` floor-filtered point array.

            uv:
                ``N x 2`` pixel coordinates paired with ``points``.

        Returns:
            Tuple of points and UV coordinates whose estimated normals satisfy
            ``abs(normal_y) < ny_thr``.

        Notes:
            If Open3D normal estimation fails, the original arrays are returned
            unchanged.
        """
        # 4) Keep points whose normals are close to camera Z axis (door plane direction)
        pc_nf = o3d.geometry.PointCloud()
        pc_nf.points = o3d.utility.Vector3dVector(points)
        try:
            pc_nf.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius, max_nn=self.max_nn))
            pc_nf.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
            normals = np.asarray(pc_nf.normals)
        except Exception:
            normals = None

        # On an estimation failure, retaining points is the conservative choice:
        # it may shorten clearance but will not hide a possible obstacle.
        if normals is None:
            points_4_normal = points
            uv_4_normal = uv
        else:
            vertical_mask = np.abs(normals[:, 1]) < self.ny_thr
            points_4_normal = points[vertical_mask]
            uv_4_normal = uv[vertical_mask]
        return points_4_normal, uv_4_normal

    def outlier_removal(self, points, uv):
        """
        Apply Open3D statistical outlier removal to paired points and pixels.

        Args:
            points:
                ``N x 3`` point array.

            uv:
                ``N x 2`` pixel array aligned with ``points``.

        Returns:
            Filtered ``(points, uv)`` tuple. If Open3D reports no inliers, two
            empty arrays are returned.

        Notes:
            Exceptions from Open3D fall back to the unfiltered inputs.
        """
        # 2) Statistical 3D outlier removal (keeps mapping by applying indices to uv)
        #print(f"number of points before S3O {len(points)}")
        try:
            pc_clean = o3d.geometry.PointCloud()
            pc_clean.points = o3d.utility.Vector3dVector(points)
            # Open3D returns source indices, which lets us filter the paired UVs
            # without a costly nearest-neighbor rematch.
            _, ind = pc_clean.remove_statistical_outlier(nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio)
            # asarray avoids the guaranteed copy np.array makes, and intp is the
            # native index width (int is only 32-bit on 32-bit ARM builds).
            ind = np.asarray(ind, dtype=np.intp)
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
        """
        Reject small or depth-incoherent point patches in image space.

        The sparse UV samples are dilated to restore connectivity lost through
        subsampling. Each connected component is gated by area, median depth,
        floor-relative height, and median absolute depth deviation.

        Args:
            points:
                ``N x 3`` camera-frame point array.

            uv:
                ``N x 2`` pixel coordinates paired with ``points``.

            W:
                Image width in pixels.

            H:
                Image height in pixels.

            floor_height:
                Estimated camera-frame floor Y coordinate, or ``None``.

        Returns:
            Tuple of retained points and their UV coordinates.

        Notes:
            On an unexpected processing error, the method returns the original
            input arrays. When the generated mask has no foreground component,
            it returns empty arrays.
        """
        #print(f"number of points before CC {len(points)}")
        # --- Remove small patches in image space (connected components) ---
        try:
            # Defaults so visualization doesn't disappear if nothing is filtered
            uv_5_remove_patches = uv
            points_5_remove_patches = points

            # Image components lose roughly stride squared pixels during
            # subsampling, so scale area thresholds by the same amount.
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
            # Restore local connectivity only to the size removed by subsampling;
            # a larger dilation could merge two separate obstacles.
            mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

            if num <= 1:
                return np.empty((0, 3)), np.empty((0, 2))

            # Vectorized mapping: label lookup per UV pixel. valid_uv already
            # restricted the indices to the image, so no clipping is required.
            lbl_values = labels[uv_int[valid_uv, 1], uv_int[valid_uv, 0]]

            # Filter out background (label 0). Combining the two selections into a
            # single index array avoids materializing an intermediate copy of both
            # uv and points.
            valid_labels = lbl_values > 0
            lbl_values = lbl_values[valid_labels]
            selected = np.flatnonzero(valid_uv)[valid_labels]
            uv_valid = uv[selected]
            pts_valid = points[selected]

            # Precompute per-component area and median depth
            unique_lbls, inverse_idx = np.unique(lbl_values, return_inverse=True)
            areas = stats[unique_lbls, cv2.CC_STAT_AREA]

            # Component statistics are evaluated for every component at once.
            # Testing them one component at a time costs a full-length scan per
            # component, which dominates on frames with many reflection patches.
            num_groups = unique_lbls.size
            z_all = pts_valid[:, 2]
            y_all = pts_valid[:, 1]

            z_med, z_count = grouped_median(z_all, inverse_idx, num_groups)
            y_med, y_count = grouped_median(y_all, inverse_idx, num_groups)

            #there could be floor noisy points due to reflection which have high z value but small area. so these should be removed.
            #depths above 3.5m are removed during back projection itself.but there can be noisy floor points with depth less than 3.5m but higher than actual floor depth.
            # Median absolute deviation is robust to a few reflected depth
            # samples and exposes components spread implausibly far in Z.
            # Deviations inherit NaN from the samples, so they drop out exactly as
            # they do in the median above.
            z_deviation = np.abs(z_all - z_med[inverse_idx])
            z_mad, _ = grouped_median(z_deviation, inverse_idx, num_groups)

            # A component survives only if it has usable depth and height samples,
            # sits at or below the floor-relative height limit, has coherent depth,
            # and is large enough for its range band.
            keep_group = (z_count > 0) & (y_count > 0)
            if floor_height is not None:
                keep_group &= ~(y_med > floor_height + 0.1)
            # need to review if below code is needed or not. because remving points based on  depth variation can cause removal of valid points also.
            keep_group &= ~(z_mad > (0.30 + 0.05 * (areas / 100)))
            #area_thresh = base_area_at_1m / (z_eff * z_eff)
            #using above line will not remove patches due to floor reflection. the points in floor near to robot will have high z_eff which will make area_thresh small and hence these patches will be kept instead of removing them.
            #so use minimum_area_far as threshold for far points.
            keep_group &= ((z_med < self.near_z) & (areas >= min_area_near)) | (areas >= minimum_area_far)

            keep_mask = keep_group[inverse_idx]

            if np.any(keep_mask):
                uv_5_remove_patches = uv_valid[keep_mask]
                points_5_remove_patches = pts_valid[keep_mask]
        except Exception:
            # fall back without filtering on any error
            uv_5_remove_patches = uv
            points_5_remove_patches = points

        rospy.logdebug("number of points after CC %d", len(points_5_remove_patches))
        return points_5_remove_patches, uv_5_remove_patches

    def visualize_points(self, color_image, W, H, uv_sets_with_color):
        """
        Overlay several UV point sets using caller-supplied BGR colors.

        Args:
            color_image:
                BGR image modified in place. ``None`` is accepted.

            W:
                Image width used for bounds checking.

            H:
                Image height used for bounds checking.

            uv_sets_with_color:
                Iterable of ``(uv_array, bgr_color)`` pairs.

        Returns:
            The input image after all valid point sets have been drawn.

        Notes:
            Visualization errors are ignored because the overlay is diagnostic.
        """
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
        """
        Run the complete obstacle-filtering and clearance pipeline.

        Stages execute in the following order: floor removal, normal filtering,
        statistical outlier rejection, and image-space component filtering.
        The nearest surviving Z value becomes the front clearance.

        Args:
            depth_image_in_meters:
                Aligned ``H x W`` metric depth image.

            color_image:
                BGR image modified with filter-stage diagnostics.

            corridor_pts_input:
                Camera-frame 3D points already restricted to the corridor.

            corridor_uv_input:
                Pixels corresponding one-to-one with the input points.

            z_max:
                Far limit reported when no meaningful obstacle survives.

        Returns:
            Tuple ``(front_clearance, visualization, final_points)`` where
            clearance is in meters and ``final_points`` contains the surviving
            obstacle cloud.

        Notes:
            If no more than ``minimum_depth_points_after_filtering`` points
            survive, the region is treated as clear up to ``z_max``.
        """

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

        

        # A real obstacle should leave a small cluster, not one or two isolated
        # returns. Once that support exists, the nearest point is the safe bound.
        if len(final_points) > self.minimum_depth_points_after_filtering:
            front_clearance = np.min(final_points[:, 2])

        else: # no enough points in corridor after filtering
            # means the the corridor is free of obstacles.because lcoal passabiity and even for corridor passaability check we may not be getting any points in small roi around corridor center.
            # that doesnt mean passabilty is not there. it can be there. so we assume front clearance to be large value like z_max.
            # A nearly empty, pre-cropped corridor is interpreted as observed clear
            # space up to the requested range rather than as a zero-distance block.
            front_clearance = z_max

        self.timer.stop(f"check_if_passable--> main passability check")

        self.timer.start(f"check_if_passable--> visualization of final points")
        # These overlays are plain NumPy scatter writes, so they are not disabled
        # by the no-op cv2 patching and must be skipped explicitly. They also write
        # into the caller's cached colour frame in place, which would otherwise
        # accumulate debug pixels on a frame that is reused.
        if self.enable_visualization:
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
