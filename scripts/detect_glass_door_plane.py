#!/usr/bin/env python3
"""RANSAC and temporal filtering for a front-facing glass-door plane."""

import rospy
import cv2
import numpy as np
import open3d as o3d

from processing_classes import backproject_depth_to_points
from resolution_scaling import ResolutionScaler





class TemporalPlaneTracker:
    """
    Stabilize a single plane hypothesis across consecutive frames.

    New planes are associated with the smoothed hypothesis using normal-angle
    and distance gates. Associated measurements update an exponential moving
    average, while a bounded Boolean history applies separate confirmation and
    drop thresholds.

    Attributes:
        alpha:
            Exponential moving-average weight of the newest measurement.

        max_jump_deg:
            Maximum normal-angle change allowed for association.

        max_jump_m:
            Maximum distance change allowed for association.

        window:
            Maximum number of association decisions retained.

        confirm_k:
            Associated-frame count required to confirm an unconfirmed track.

        drop_k:
            Associated-frame count at or below which a confirmed track drops.

        n_s:
            Smoothed unit plane normal, or ``None`` before initialization.

        d_s:
            Smoothed plane distance in meters.

        history:
            Recent Boolean association decisions.

        confirmed:
            Current hysteretic confirmation state.
    """

    def __init__(self):
        """
        Initialize temporal thresholds and an empty plane track.

        Configuration is loaded from
        ``~plane_detector/temporal_smoothing``. No plane is considered
        confirmed until :meth:`update` has associated enough observations
        within the configured angular and distance jump limits.

        Notes:
            The tracker starts without a smoothed normal or distance. Its
            history contains only association decisions made after the first
            candidate initializes the track.
        """
        ns = "~plane_detector/temporal_smoothing"
        self.alpha = rospy.get_param(f"{ns}/alpha", 0.3)
        self.max_jump_deg = rospy.get_param(f"{ns}/max_jump_deg", 12.0)
        self.max_jump_m = rospy.get_param(f"{ns}/max_jump_m", 0.4)
        self.window = int(rospy.get_param(f"{ns}/window", 10))
        self.confirm_k = int(rospy.get_param(f"{ns}/confirm_k", 6))
        self.drop_k = int(rospy.get_param(f"{ns}/drop_k", 2))
        self.n_s = None
        self.d_s = None
        self.history = []  # bool list of associated frames
        self.confirmed = False

    def _normalize(self, n):
        """
        Normalize a vector without dividing by exact zero.

        Args:
            n:
                Array-like vector.

        Returns:
            NumPy vector divided by its norm plus a small numerical epsilon.
        """
        norm = np.linalg.norm(n)
        return n / (norm + 1e-12)

    def _angle_deg(self, n1, n2):
        """
        Compute the unsigned angle between two unit vectors.

        Args:
            n1:
                First unit vector, or ``None``.

            n2:
                Second unit vector, or ``None``.

        Returns:
            Angle in degrees. Missing vectors return ``180.0``.
        """
        if n1 is None or n2 is None:
            return 180.0

        # Clip round-off before arccos; nearly identical unit vectors can dot
        # to a value a few ulps above one.
        c = np.clip(float(np.dot(n1, n2)), -1.0, 1.0)
        return np.degrees(np.arccos(c))

    def _push(self, val: bool):
        """
        Append one association result to the bounded history.

        Args:
            val:
                Whether the current measurement matched the tracked plane.
        """
        self.history.append(bool(val))
        if len(self.history) > self.window:
            self.history.pop(0)

    def _update_confirmed(self):
        """
        Recompute confirmation state using hysteresis thresholds.

        Notes:
            An unconfirmed track needs at least ``confirm_k`` true entries,
            while a confirmed track is retained until that count falls to
            ``drop_k`` or below.
        """
        #counts the number of True in history. If >= confirm_k, set confirmed to True. If <= drop_k, set confirmed to False.
        #self.history is a list of booleans. np.sum of False is 0, True is 1.
        # Two thresholds create hysteresis: confirmation is slow, while a brief
        # run of missing depth does not immediately erase a good track.
        count_true = int(np.sum(self.history))
        #hysteresis
        if not self.confirmed and count_true >= self.confirm_k:
            self.confirmed = True
        elif self.confirmed and count_true <= self.drop_k:
            self.confirmed = False

    def update(self, n_t, d_t):
        """
        Associate and smooth one plane measurement.

        Args:
            n_t:
                Current plane normal. It is normalized internally; ``None``
                records a missed association.

            d_t:
                Current plane distance in meters. Nonfinite values are treated
                as a missed association.

        Returns:
            Tuple ``(confirmed, smoothed_normal, smoothed_distance)``.

        Notes:
            A valid first measurement initializes the smoothed hypothesis.
            Later measurements update it only when both association gates pass.
        """
        if n_t is None or not np.isfinite(d_t):
            self._push(False)
            self._update_confirmed()
            return self.confirmed, self.n_s, self.d_s

        n_t = self._normalize(n_t)
        d_t = float(d_t)

        # initialize
        if self.n_s is None:
            self.n_s = n_t
            self.d_s = d_t
            self._push(True)
            self._update_confirmed()
            return self.confirmed, self.n_s, self.d_s

        # Association checks orientation and range together so a nearby wall
        # cannot quietly take over an established door track.
        angle = self._angle_deg(self.n_s, n_t)
        jump = abs(d_t - self.d_s)
        associated = (angle <= self.max_jump_deg) and (jump <= self.max_jump_m)
        self._push(associated)
        if associated:
            # Blend only associated measurements; rejected planes should not
            # drag the stored normal toward clutter.
            # EMA smoothing
            n_blend = (1.0 - self.alpha) * self.n_s + self.alpha * n_t
            self.n_s = self._normalize(n_blend)
            self.d_s = (1.0 - self.alpha) * self.d_s + self.alpha * d_t
        self._update_confirmed()
        return self.confirmed, self.n_s, self.d_s



class PlaneDetector:
    """
    Detect and temporally confirm a front-facing door-sized plane.

    Metric depth is backprojected near the expected door distance, voxel
    downsampled, filtered by surface normal, and segmented with Open3D RANSAC.
    Candidates must satisfy orientation, size, and distance gates before the
    temporal tracker can confirm them.

    Attributes:
        reference_door_distance_m:
            Nominal door distance supplied by the global map.

        global_map_distance_accuracy_to_door_plane:
            Fractional uncertainty applied to the nominal map distance.

        distance_range_m:
            Accepted metric deviation around the nominal distance.

        max_depth_backprojection:
            Exclusive far limit for point generation.

        min_depth_backprojection:
            Exclusive near limit for point generation.

        subsample:
            Pixel stride used during backprojection.

        voxel_size:
            Voxel edge length in meters.

        normal_radius:
            Open3D normal-estimation search radius.

        normal_max_nn:
            Maximum neighbors used for normal estimation.

        nx_thr:
            Maximum absolute normal X component retained.

        ny_thr:
            Maximum absolute normal Y component retained.

        nz_min:
            Minimum absolute normal Z component retained.

        ransac_distance_threshold:
            Maximum point-to-plane inlier distance in meters.

        ransac_n:
            Number of sampled points per RANSAC hypothesis.

        ransac_num_iterations:
            Number of RANSAC iterations.

        vertical_tol:
            Tolerance for accepting a camera-facing plane normal.

        min_inliers:
            Minimum RANSAC inlier count.

        max_inlier_density:
            Configured upper inlier-density limit retained for candidate
            evaluation.

        max_planes:
            Maximum planes segmented from one frame.

        lower_pct:
            Lower percentile used for robust plane bounds.

        upper_pct:
            Upper percentile used for robust plane bounds.

        min_width_m:
            Minimum accepted physical plane width.

        min_height_m:
            Minimum accepted physical plane height.

        tracker:
            :class:`TemporalPlaneTracker` used for cross-frame confirmation.
    """

    def __init__(self):
        """
        Load plane-detection parameters and create the temporal tracker.

        Notes:
            Parameters are read once. The resulting object should be reused
            between frames so temporal confirmation history is preserved.
        """
        ns = "~plane_detector"
        self.reference_door_distance_m = rospy.get_param("~reference_door_distance_m", 2.0)
        self.global_map_distance_accuracy_to_door_plane = rospy.get_param(f"{ns}/global_map_distance_accuracy_to_door_plane", 0.15)
        self.distance_range_m = self.reference_door_distance_m * self.global_map_distance_accuracy_to_door_plane
        #backprojection
        # Crop in depth before building the cloud. This is both faster and a
        # strong guard against fitting one of the corridor walls behind the door.
        self.max_depth_backprojection = self.reference_door_distance_m + self.distance_range_m
        self.min_depth_backprojection = self.reference_door_distance_m - self.distance_range_m
        # The stride is taken over the list of valid pixels, whose length scales
        # with image area, so it is converted by the square of the linear factor.
        # This holds the back-projected point count roughly constant, keeping
        # min_inliers below meaningful at any resolution.
        self.subsample = ResolutionScaler().stride(
            rospy.get_param(f"{ns}/backproject/subsample", 1)
        )

        #filter points
        # Downsampling
        self.voxel_size = rospy.get_param(f"{ns}/filter_points/voxel_size", 0.008)
        # Normal estimation
        self.normal_radius = rospy.get_param(f"{ns}/filter_points/normal_radius", 0.03)
        self.normal_max_nn = rospy.get_param(f"{ns}/filter_points/normal_max_nn", 30)
        # Normal filtering thresholds
        self.nx_thr = rospy.get_param(f"{ns}/filter_points/nx_thr", 0.30)
        self.ny_thr = rospy.get_param(f"{ns}/filter_points/ny_thr", 0.30)
        self.nz_min = rospy.get_param(f"{ns}/filter_points/nz_min", 0.85)

        # RANSAC
        self.ransac_distance_threshold = rospy.get_param(f"{ns}/ransac/distance_threshold", 0.03)
        self.ransac_n = rospy.get_param(f"{ns}/ransac/n", 3)
        self.ransac_num_iterations = rospy.get_param(f"{ns}/ransac/num_iterations", 100)
        self.vertical_tol = rospy.get_param(f"{ns}/ransac/vertical_tol", 0.1)
        self.min_inliers = rospy.get_param(f"{ns}/ransac/min_inliers", 2000)
        self.max_planes = rospy.get_param(f"{ns}/ransac/max_planes", 1)
        # Outline drawing
        self.lower_pct = rospy.get_param(f"{ns}/outline/lower_pct", 5)
        self.upper_pct = rospy.get_param(f"{ns}/outline/upper_pct", 95)
        self.min_width_m = rospy.get_param(f"{ns}/outline/min_width_m", 0.8)
        self.min_height_m = rospy.get_param(f"{ns}/outline/min_height_m", 0.5)
        #
        self.max_inlier_density = rospy.get_param(f"{ns}/max_inlier_density", 1.0)
        # Plane inlier overlays are diagnostics only.
        self.enable_visualization = rospy.get_param("~enable_visualization", True)
        # Temporal smoothing tracker
        self.tracker = TemporalPlaneTracker()


    def find_vertical_planes(self,points):
        """
        Iteratively segment front-facing planes from a point cloud.

        Args:
            points:
                ``N x 3`` filtered camera-frame point array.

        Returns:
            List of ``(plane_model, original_indices, inlier_points)`` tuples
            for accepted planes.

        Notes:
            In this camera convention, a door plane faces the optical axis and
            therefore has ``abs(normal_z)`` close to one. Inliers are removed
            after every fit so later iterations can find another plane.
        """

        remaining_points = points.copy()
        remaining_indices = np.arange(points.shape[0])  # Track original indices
        found_vertical_planes = []



        for _ in range(self.max_planes):
            if len(remaining_points) < self.ransac_n:
                break

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(remaining_points)
            # RANSAC handles the missing and reflected samples that make ordinary
            # least-squares fitting unreliable on glass.
            plane_model, inliers = pc.segment_plane(distance_threshold=self.ransac_distance_threshold,
                                                    ransac_n=self.ransac_n,
                                                    num_iterations=self.ransac_num_iterations)

            inliers = np.array(inliers, dtype=np.int64)
            if len(inliers) < self.min_inliers:
                break

            orig_inlier_indices = remaining_indices[inliers]
            orig_inlier_points = remaining_points[inliers]


            # In the optical frame, a plane standing in front of the robot has
            # a normal along Z. "Vertical" here describes the physical surface,
            # not the direction of its normal in camera coordinates.
            # Check if the plane is vertical
            normal = np.array(plane_model[:3])
            normal = normal / np.linalg.norm(normal)
            #print("Detected plane normal:", normal)
            if abs(abs(normal[2]) - 1.0) < self.vertical_tol:
                # Save vertical plane
                found_vertical_planes.append((plane_model, orig_inlier_indices, orig_inlier_points))
                rospy.logdebug("Found vertical plane with %d inliers.", len(inliers))
            else:
                rospy.logdebug("Detected plane is not vertical; skipping.")


            mask = np.ones(len(remaining_points), dtype=bool)
            mask[inliers] = False

            remaining_points = remaining_points[mask]
            remaining_indices = remaining_indices[mask]
        return found_vertical_planes


    def draw_plane_outline_on_image(self, color_image, corners_3d, fx, fy, cx, cy):
        """
        Project 3D plane corners and draw their polygon on a color image.

        Args:
            color_image:
                BGR image modified in place.

            corners_3d:
                Iterable of camera-frame ``(X, Y, Z)`` corners.

            fx:
                Horizontal focal length in pixels.

            fy:
                Vertical focal length in pixels.

            cx:
                Horizontal principal point in pixels.

            cy:
                Vertical principal point in pixels.

        Returns:
            List of projected ``(u, v)`` integer coordinates. Corners with
            nonpositive Z are skipped.

        Notes:
            The polygon is drawn in red and closed by OpenCV.
        """

        corners_uv = []
        for x, y, z in corners_3d:
            if z <= 0:
                continue
            u = int(round(x * fx / z + cx))
            v = int(round(y * fy / z + cy))
            corners_uv.append((u, v))


        pts = np.array(corners_uv, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(color_image, [pts], isClosed=True, color=(0,0,255), thickness=2)
        return corners_uv  # Return the list of (u, v) tuples



    def voxel_downsample(self, points, uv):
        """
        Keep one point per voxel while preserving point-to-pixel indices.

        Args:
            points:
                ``N x 3`` point array.

            uv:
                ``N x 2`` matching pixel array.

        Returns:
            Tuple of downsampled points and corresponding pixels.

        Notes:
            The first source index found in each voxel is retained.
        """
        # Voxel downsampling
        # Reduces redundant neighboring points while preserving
        # metal frame edges and door boundaries. This balances
        # point density for RANSAC without losing structure.
        #add size heuristics and inliner density based on high depth change(glass)

        # Retain source indices so every surviving 3D point still maps to the
        # correct color pixel after downsampling.
        vox_coords = np.floor(points / self.voxel_size).astype(np.int64)
        # np.unique(..., axis=0) builds a structured view and lexsorts three
        # columns. Packing the three voxel indices into one integer key lets the
        # much cheaper 1-D unique do the same job. The packing is only valid while
        # each axis fits in its bit field, so the extent is checked first and the
        # row-wise form is kept as an exact fallback.
        vox_min = vox_coords.min(axis=0)
        vox_extent = vox_coords.max(axis=0) - vox_min
        BITS_PER_AXIS = 21
        if vox_coords.size and np.all(vox_extent < (1 << BITS_PER_AXIS)):
            shifted = vox_coords - vox_min
            keys = (
                (shifted[:, 0] << (2 * BITS_PER_AXIS))
                | (shifted[:, 1] << BITS_PER_AXIS)
                | shifted[:, 2]
            )
            _, unique_idx = np.unique(keys, return_index=True)
        else:
            _, unique_idx = np.unique(vox_coords, axis=0, return_index=True)
        unique_idx = np.sort(np.asarray(unique_idx, dtype=np.intp))
        points = points[unique_idx]
        uv = uv[unique_idx]
        return points, uv

    def normal_filter(self, points, uv):
        """
        Keep surfaces whose normals face roughly along the camera Z axis.

        Args:
            points:
                Voxel-downsampled ``N x 3`` point array.

            uv:
                ``N x 2`` pixels paired with ``points``.

        Returns:
            Tuple of normal-filtered points and matching pixels.

        Notes:
            Accepted normals satisfy all configured X, Y, and Z component
            thresholds.
        """
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)

        # Normal estimation
        pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=self.normal_max_nn)
        )
        normals = np.asarray(pc.normals)

        # Reject floors and side walls before RANSAC; only camera-facing patches
        # can plausibly be the door plane at this stage.
        # Normal filtering
        z_axis_mask = (np.abs(normals[:, 0]) < self.nx_thr) & (np.abs(normals[:, 1]) < self.ny_thr) & (np.abs(normals[:, 2]) >= self.nz_min)
        keep_idx = np.where(z_axis_mask)[0]

        pc = pc.select_by_index(keep_idx)
        points = np.asarray(pc.points)
        uv = uv[keep_idx]
        return points, uv


    def check_plane_size(self, inlier_points, plane_model):
        """
        Check whether a plane has door-like physical width and height.

        Args:
            inlier_points:
                ``N x 3`` RANSAC inlier points.

            plane_model:
                Coefficients ``(a, b, c, d)`` for
                ``a*x + b*y + c*z + d = 0``.

        Returns:
            Four robust 3D corners when the plane satisfies the configured
            dimensions, otherwise ``None``.

        Notes:
            X and Y bounds use configured percentiles rather than raw extrema.
            Z for each corner is solved from the fitted plane equation.
        """
        if inlier_points.shape[0] == 0:
            return None
        # Percentiles ignore stray inliers that would otherwise inflate the
        # apparent width or height of the plane.
        x_min = np.percentile(inlier_points[:, 0], self.lower_pct)
        x_max = np.percentile(inlier_points[:, 0], self.upper_pct)
        y_min = np.percentile(inlier_points[:, 1], self.lower_pct)
        y_max = np.percentile(inlier_points[:, 1], self.upper_pct)

        width = x_max - x_min
        height = y_max - y_min
        if width < self.min_width_m or height < self.min_height_m:
            return None  # Plane too small to consider

        corners_3d = np.array([
            [x_min, y_min, 0],
            [x_max, y_min, 0],
            [x_max, y_max, 0],
            [x_min, y_max, 0]
        ])


        a, b, c, d = plane_model
        for i in range(4):
            x, y, _ = corners_3d[i]
            z = -(a*x + b*y + d) / c
            corners_3d[i, 2] = z

        return corners_3d

    def find_plane_metrics(self, plane_model, depth_image_in_meters, inlier_indices, plane_corners_2d, mask):
        """
        Measure depth spread, holes, density, distance, and plane normal.

        Args:
            plane_model:
                Plane coefficients ``(a, b, c, d)``.

            depth_image_in_meters:
                Aligned metric depth image.

            inlier_indices:
                Source indices belonging to the selected plane.

            plane_corners_2d:
                Projected image-space plane polygon.

            mask:
                Caller-provided uint8 image mask modified in place.

        Returns:
            Dictionary containing robust depth statistics, hole fraction,
            inlier density, origin-to-plane distance, and a unit normal.

        Notes:
            Depth values outside the 5th-to-95th percentile interval are
            discarded before calculating mean and standard deviation.
        """
        pts = np.array(plane_corners_2d, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

        # Holes and sparse inliers are retained as diagnostics because they are
        # common on glass, even though geometry is the main acceptance gate.
        fraction_of_holes_in_plane = (np.sum((mask == 255) & ((depth_image_in_meters == 0) | ~np.isfinite(depth_image_in_meters)))) / (np.sum(mask == 255) + 1e-9)
        inlier_density = len(inlier_indices) / (np.sum(mask == 255) + 1e-9)

        depths_in_plane = depth_image_in_meters[mask == 255]
        depths_in_plane = depths_in_plane[np.isfinite(depths_in_plane) & (depths_in_plane > 0)]
        depth_min_p_5 = np.percentile(depths_in_plane, 5)
        depth_max_p_95 = np.percentile(depths_in_plane, 95)
        depths_in_plane = depths_in_plane[(depths_in_plane >= depth_min_p_5) & (depths_in_plane <= depth_max_p_95)]


        a, b, c, d = plane_model
        plane_norm = np.sqrt(a*a + b*b + c*c)
        distance_m = abs(d) / (plane_norm + 1e-12)
        plane_norm_vector = np.array([a, b, c], dtype=np.float32) / plane_norm

        depth_mean = np.mean(depths_in_plane)
        depth_std = np.std(depths_in_plane)
        normalized_std = depth_std / (depth_mean + 1e-9)

        return {
            "depth_mean": depth_mean,
            "depth_std": depth_std,
            "normalized_std": normalized_std,
            "depth_min": depth_min_p_5,
            "depth_max": depth_max_p_95,
            "fraction_of_holes_in_plane": fraction_of_holes_in_plane,
            "inlier_density": inlier_density,
            "distance_m": distance_m,
            "plane_norm_vector" : plane_norm_vector

        }


    def highlight_planes_on_image(self, color_image, uv, found_vertical_planes):
        """
        Overlay every detected plane's inlier pixels with a distinct color.

        Args:
            color_image:
                ``H x W x 3`` BGR image modified in place.

            uv:
                Pixels corresponding to the filtered point cloud used by
                RANSAC.

            found_vertical_planes:
                Plane tuples returned by :meth:`find_vertical_planes`.

        Returns:
            ``None``.

        Notes:
            Colors repeat when more planes are provided than entries in the
            fixed palette. The overlay is skipped entirely when diagnostics are
            disabled, since the pixel writes below are plain NumPy and are not
            removed by the no-op drawing patches.
        """
        if not self.enable_visualization:
            return
        # Define a list of distinct colors (BGR for OpenCV)
        plane_colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (128, 128, 255),# Pinkish
            (0, 128, 255),  # Orange
        ]
        # Marking inliers one at a time meant thousands of Python iterations per
        # frame, and the index lookup and bounds test ran even when the drawing
        # call itself was disabled. A single-pixel dot is a direct pixel write, so
        # the whole plane is marked in one vectorized assignment.
        H_img, W_img = color_image.shape[:2]
        for i, (_, inlier_indices, _) in enumerate(found_vertical_planes):
            color = plane_colors[i % len(plane_colors)]
            inlier_uv = uv[np.asarray(inlier_indices, dtype=np.intp)]
            u = inlier_uv[:, 0].astype(np.intp)
            v = inlier_uv[:, 1].astype(np.intp)
            in_bounds = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
            color_image[v[in_bounds], u[in_bounds]] = color



    def detect(self, color_image, depth_image_in_meters,
               fx, fy, cx, cy):
        """
        Find and temporally confirm the strongest door-sized plane.

        Candidates must face the camera, satisfy physical size and distance
        gates, and remain geometrically consistent across the tracker's recent
        history. The color image receives candidate and confirmed overlays.

        Args:
            color_image:
                BGR image modified with RANSAC inliers and confirmed outline.

            depth_image_in_meters:
                Aligned ``H x W`` metric depth image.

            fx:
                Horizontal focal length in pixels.

            fy:
                Vertical focal length in pixels.

            cx:
                Horizontal principal point in pixels.

            cy:
                Vertical principal point in pixels.

        Returns:
            A dictionary containing the confirmed plane model, derived metrics,
            inlier points, and flags distinguishing absent from unconfirmed
            candidates.

        Notes:
            ``had_candidates`` is true when geometry was found but rejected by
            distance or not yet confirmed. A confirmed result is selected by
            maximum inlier count among candidates inside the distance gate.
        """


        # Keep pixel coordinates beside the cloud throughout the pipeline so a
        # confirmed 3D plane can be painted back onto the aligned color image.
        all_points, uv, valid_mask = backproject_depth_to_points(
            depth_image_in_meters, fx, fy, cx, cy,
            self.max_depth_backprojection, self.min_depth_backprojection,
            self.subsample
        )


        if all_points.shape[0] == 0:
            return {"plane_model": None, "plane_metrics": None, "inlier_points": None,"had_candidates": False, "confirmed": False}


        points_after_voxel_downsample, uv_after_voxel_downsample = self.voxel_downsample(all_points, uv)
        #points_after_voxel_downsample, uv_after_voxel_downsample = all_points, uv
        points_after_normal_filtering, uv_after_normal_filtering = self.normal_filter(points_after_voxel_downsample, uv_after_voxel_downsample)


        # RANSAC vertical planes
        found_vertical_planes = self.find_vertical_planes(points_after_normal_filtering)
        if not found_vertical_planes:
            return {"plane_model": None, "plane_metrics": None,"inlier_points": None, "had_candidates": False, "confirmed": False}

        self.highlight_planes_on_image(
            color_image,uv_after_normal_filtering,
            found_vertical_planes)

        # Build candidates with size gating; defer drawing until confirmed
        candidates = []
        for plane_model, inlier_indices, inlier_points in found_vertical_planes:
            if inlier_indices.shape[0] == 0:
                continue

            plane_corners_3d = self.check_plane_size(inlier_points, plane_model)
            # if the plane is too small, skip drawing and continue
            if plane_corners_3d is None:
                continue
            a, b, c, d = plane_model
            plane_norm = np.sqrt(a*a + b*b + c*c)
            if plane_norm <= 1e-12:
                continue
            n = np.array([a, b, c], dtype=np.float32) / plane_norm
            distance_m = abs(d) / plane_norm
            candidates.append({
                "plane_model": plane_model,
                "inliers": inlier_indices,
                "inlier_points": inlier_points,
                "corners_3d": plane_corners_3d,
                "normal": n,
                "distance_m": float(distance_m),
            })

        if not candidates:
            return {"plane_model": None, "plane_metrics": None, "inlier_points": None,"had_candidates": False, "confirmed": False}

        # Keep only planes within the reference distance threshold
        # The map-provided stopping distance is a useful prior, but the range
        # allows for odometry and stopping error.
        filtered = [c for c in candidates if abs(c["distance_m"] - self.reference_door_distance_m) <= self.distance_range_m]
        if not filtered:
            # We had candidates, but none within desired distance
            return {"plane_model": None, "plane_metrics": None, "inlier_points": None, "had_candidates": True, "confirmed": False}
        # Pick the strongest by inlier count among filtered
        # this is important to avoid jumping between multiple planes. because sometimes, points that belongs to one plane is segmented into two planes by RANSAC.
        # observed in testing glass door with walls on both sides.
        # Largest support is more stable than first-found when RANSAC splits one
        # physical surface into two similar candidates.
        chosen = max(filtered, key=lambda c: len(c["inliers"]))

        # Update temporal tracker; only proceed once confirmed
        # Do not expose a one-frame fit to the line detectors or navigation.
        confirmed, n_s, d_s = self.tracker.update(chosen["normal"], chosen["distance_m"])
        if not confirmed:
            return {"plane_model": None, "plane_metrics": None, "inlier_points": None, "had_candidates": True, "confirmed": False}

        # Draw outline and compute metrics for confirmed plane. only after temporal smoothing.
        plane_corners_2d = self.draw_plane_outline_on_image(
            color_image, chosen["corners_3d"], fx, fy, cx, cy)

        mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
        plane_metrics = self.find_plane_metrics(
            chosen["plane_model"], depth_image_in_meters, chosen["inliers"], plane_corners_2d, mask)

        result = {
            "plane_model": chosen["plane_model"],
            "plane_metrics": plane_metrics,
            "inlier_points" : chosen["inlier_points"],
            "had_candidates": True,
            "confirmed": True
        }

        return result
