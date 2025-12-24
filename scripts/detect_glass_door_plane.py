#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import open3d as o3d

from processing_classes import backproject_depth_to_points





class TemporalPlaneTracker:
    """
    Tracks and stabilizes a single plane hypothesis over time using
    angle/distance gating and exponential smoothing. Only reports
    confirmed planes after sufficient consistent frames.
    """
    def __init__(self):
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
        norm = np.linalg.norm(n)
        return n / (norm + 1e-12)

    def _angle_deg(self, n1, n2):
        if n1 is None or n2 is None:
            return 180.0

        c = np.clip(float(np.dot(n1, n2)), -1.0, 1.0)
        return np.degrees(np.arccos(c))

    def _push(self, val: bool):
        self.history.append(bool(val))
        if len(self.history) > self.window:
            self.history.pop(0)

    def _update_confirmed(self):
        #counts the number of True in history. If >= confirm_k, set confirmed to True. If <= drop_k, set confirmed to False.
        #self.history is a list of booleans. np.sum of False is 0, True is 1.
        count_true = int(np.sum(self.history))
        #hysteresis
        if not self.confirmed and count_true >= self.confirm_k:
            self.confirmed = True
        elif self.confirmed and count_true <= self.drop_k:
            self.confirmed = False

    def update(self, n_t, d_t):
        """Update with current plane unit normal and distance (meters)."""
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

        angle = self._angle_deg(self.n_s, n_t)
        jump = abs(d_t - self.d_s)
        associated = (angle <= self.max_jump_deg) and (jump <= self.max_jump_m)
        self._push(associated)
        if associated:
            # EMA smoothing
            n_blend = (1.0 - self.alpha) * self.n_s + self.alpha * n_t
            self.n_s = self._normalize(n_blend)
            self.d_s = (1.0 - self.alpha) * self.d_s + self.alpha * d_t
        self._update_confirmed()
        return self.confirmed, self.n_s, self.d_s



class PlaneDetector:
    """
    Encapsulates glass-door plane detection configuration and logic.
    Parameters are read once from ROS params in __init__.
    """
    def __init__(self):
        ns = "~plane_detector"
        self.reference_door_distance_m = rospy.get_param(f"{ns}/reference_door_distance_m", 2.0)
        self.distance_range_m = rospy.get_param(f"{ns}/distance_range_m", 0.3)
        #backprojection
        self.max_depth_backprojection = rospy.get_param(f"{ns}/backproject/max_depth", 3.0)
        self.min_depth_backprojection = rospy.get_param(f"{ns}/backproject/min_depth", 1.0)
        self.subsample = rospy.get_param(f"{ns}/backproject/subsample", 1)
        
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
        # Temporal smoothing tracker
        self.tracker = TemporalPlaneTracker()


    def find_vertical_planes(self,points):
        """
        Iteratively find up to max_planes vertical planes in the point cloud.
        Returns a list of (plane_model, inlier_indices) for vertical planes.
        """

        remaining_points = points.copy()
        remaining_indices = np.arange(points.shape[0])  # Track original indices
        found_vertical_planes = []



        for _ in range(self.max_planes):
            if len(remaining_points) < self.ransac_n:
                break

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(remaining_points)
            plane_model, inliers = pc.segment_plane(distance_threshold=self.ransac_distance_threshold,
                                                    ransac_n=self.ransac_n,
                                                    num_iterations=self.ransac_num_iterations)
            
            inliers = np.array(inliers, dtype=np.int64)
            if len(inliers) < self.min_inliers:
                break

            orig_inlier_indices = remaining_indices[inliers]
            orig_inlier_points = remaining_points[inliers]

            
            # Check if the plane is vertical
            normal = np.array(plane_model[:3])
            normal = normal / np.linalg.norm(normal)
            #print("Detected plane normal:", normal)
            if abs(abs(normal[2]) - 1.0) < self.vertical_tol:
                # Save vertical plane
                found_vertical_planes.append((plane_model, orig_inlier_indices, orig_inlier_points))
                print(f"Found vertical plane with {len(inliers)} inliers.")
            else:
                print("Detected plane is not vertical; skipping.")
            
            
            mask = np.ones(len(remaining_points), dtype=bool)
            mask[inliers] = False
            
            remaining_points = remaining_points[mask]
            remaining_indices = remaining_indices[mask]
        return found_vertical_planes
    

    def draw_plane_outline_on_image(self, color_image, corners_3d, fx, fy, cx, cy):
        """
        Draws a robust outline of the detected plane as a polygon on the color image.
        Supports any number of valid projected corners (>= 3).
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
        # Voxel downsampling
        # Reduces redundant neighboring points while preserving
        # metal frame edges and door boundaries. This balances
        # point density for RANSAC without losing structure.
        #add size heuristics and inliner density based on high depth change(glass)

        vox_coords = np.floor(points / self.voxel_size).astype(np.int64)
        _, unique_idx = np.unique(vox_coords, axis=0, return_index=True)
        unique_idx = np.sort(np.array(unique_idx, dtype=np.int64))
        points = points[unique_idx]
        uv = uv[unique_idx]
        return points, uv

    def normal_filter(self, points, uv):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)

        # Normal estimation
        pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=self.normal_max_nn)
        )
        normals = np.asarray(pc.normals)

        # Normal filtering
        z_axis_mask = (np.abs(normals[:, 0]) < self.nx_thr) & (np.abs(normals[:, 1]) < self.ny_thr) & (np.abs(normals[:, 2]) >= self.nz_min)
        keep_idx = np.where(z_axis_mask)[0]

        pc = pc.select_by_index(keep_idx)
        points = np.asarray(pc.points)
        uv = uv[keep_idx]
        return points, uv
    

    def check_plane_size(self, inlier_points, plane_model):
        """
        Check if the detected plane has reasonable size to be a door.
        Returns the corners if valid, else None.
        """
        if inlier_points.shape[0] == 0:
            return None
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
        pts = np.array(plane_corners_2d, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

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
            "distance_m": distance_m
        }


    def highlight_planes_on_image(self, color_image, uv, found_vertical_planes):
        """
        Overlays each detected plane's inlier pixels on the color_image in a unique color.
        - color_image: (H, W, 3) numpy array (will be modified in-place)
        - uv: (N, 2) array of pixel coordinates corresponding to the original points
        - found_vertical_planes: list of (plane_model, inlier_indices, inlier_points)
        """
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
        for i, (_, inlier_indices, _) in enumerate(found_vertical_planes):
            color = plane_colors[i % len(plane_colors)]
            for idx in inlier_indices:
                u, v = uv[idx]
                u, v = int(u), int(v)
                
                if 0 <= v < color_image.shape[0] and 0 <= u < color_image.shape[1]:
                    cv2.circle(color_image, (u, v), 1, color, -1)  # Draw a small dot
                    


    def detect(self, color_image, depth_image_in_meters,
               fx, fy, cx, cy):
        

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
        filtered = [c for c in candidates if abs(c["distance_m"] - self.reference_door_distance_m) <= self.distance_range_m]
        if not filtered:
            # We had candidates, but none within desired distance
            return {"plane_model": None, "plane_metrics": None, "inlier_points": None, "had_candidates": True, "confirmed": False}
        # Pick the strongest by inlier count among filtered
        # this is important to avoid jumping between multiple planes. because sometimes, points that belongs to one plane is segmented into two planes by RANSAC.
        # observed in testing glass door with walls on both sides.
        chosen = max(filtered, key=lambda c: len(c["inliers"]))

        # Update temporal tracker; only proceed once confirmed
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
    