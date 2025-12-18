#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import open3d as o3d

from processing_classes import backproject_depth_to_points





class PlaneDetector:
    """
    Encapsulates glass-door plane detection configuration and logic.
    Parameters are read once from ROS params in __init__.
    """
    def __init__(self):
        ns = "~plane_detector"
        # Downsampling
        self.voxel_size = rospy.get_param(f"{ns}/voxel_size", 0.008)
        # Normal estimation
        self.normal_radius = rospy.get_param(f"{ns}/normal_radius", 0.03)
        self.normal_max_nn = rospy.get_param(f"{ns}/normal_max_nn", 30)
        # Normal filtering thresholds
        self.nx_thr = rospy.get_param(f"{ns}/nx_thr", 0.30)
        self.ny_thr = rospy.get_param(f"{ns}/ny_thr", 0.30)
        self.nz_min = rospy.get_param(f"{ns}/nz_min", 0.85)
        # RANSAC
        self.ransac_distance_threshold = rospy.get_param(f"{ns}/ransac/distance_threshold", 0.03)
        self.ransac_n = rospy.get_param(f"{ns}/ransac/n", 3)
        self.ransac_num_iterations = rospy.get_param(f"{ns}/ransac/num_iterations", 100)
        self.vertical_tol = rospy.get_param(f"{ns}/vertical_tol", 0.1)
        self.min_inliers = rospy.get_param(f"{ns}/min_inliers", 2000)
        self.max_planes = rospy.get_param(f"{ns}/max_planes", 1)
        # Outline drawing
        self.lower_pct = rospy.get_param(f"{ns}/outline/lower_pct", 5)
        self.upper_pct = rospy.get_param(f"{ns}/outline/upper_pct", 95)
        self.min_width_m = rospy.get_param(f"{ns}/outline/min_width_m", 0.8)
        self.min_height_m = rospy.get_param(f"{ns}/outline/min_height_m", 0.5)


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
    

    def draw_plane_outline_on_image(self, color_image, plane_model, inlier_points, fx, fy, cx, cy):
        """
        Draws a robust outline of the detected plane as a quadrilateral on the color image.
        Uses percentiles to avoid outlier influence.
        """
        # Use percentiles to avoid outliers
        x_min = np.percentile(inlier_points[:, 0], self.lower_pct)
        x_max = np.percentile(inlier_points[:, 0], self.upper_pct)
        y_min = np.percentile(inlier_points[:, 1], self.lower_pct)
        y_max = np.percentile(inlier_points[:, 1], self.upper_pct)

        # to avoid detecting small planes including the human body
        width = x_max - x_min
        height = y_max - y_min
        #print(f"width={width:.2f}, height={height:.2f}")
        if width < self.min_width_m or height < self.min_height_m:
            return []  # Plane too small to consider
        
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

        corners_uv = []
        for x, y, z in corners_3d:
            if z <= 0:
                continue
            u = int(round(x * fx / z + cx))
            v = int(round(y * fy / z + cy))
            corners_uv.append((u, v))
        
        if len(corners_uv) == 4:
            pts = np.array(corners_uv, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(color_image, [pts], isClosed=True, color=(0,0,255), thickness=2)
            return corners_uv  # Return the list of (u, v) tuples

        return []  # Return empty if not enough valid corners



    def detect(self, color_image, depth_image_in_meters,
               fx, fy, cx, cy,
               max_inlier_density=1.0,
               max_depth_consider=4.0,
               min_depth_consider=1.0):
        found_vertical_planes = []
        fraction_of_holes_in_plane = 0.0
        inlier_density = 0.0
        H, W = depth_image_in_meters.shape

        points, uv, valid_mask = backproject_depth_to_points(
            depth_image_in_meters, fx, fy, cx, cy,
            max_depth=max_depth_consider, min_depth=min_depth_consider,
            subsample=1
        )
        if points.shape[0] == 0:
            return {"plane_model": None, "distance_m": None, "is_door_candidate": False}, None, found_vertical_planes

        # Voxel downsampling
            # Setting: voxel_size = 0.01 to 0.015 meters (1–1.5 cm)
        # Why: Reduces redundant neighboring points while preserving
        #      metal frame edges and door boundaries. This balances
        #      point density for RANSAC without losing structure.
                #add size heuristics and inliner density based on high depth change(glass)


        vox_coords = np.floor(points / self.voxel_size).astype(np.int64)
        _, unique_idx = np.unique(vox_coords, axis=0, return_index=True)
        unique_idx = np.sort(np.array(unique_idx, dtype=np.int64))
        points = points[unique_idx]
        uv = uv[unique_idx]

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
        if keep_idx.size == 0:
            return {"plane_model": None, "distance_m": None, "is_door_candidate": False}, None, found_vertical_planes
        pc = pc.select_by_index(keep_idx)
        points = np.asarray(pc.points)
        uv = uv[keep_idx]

        # RANSAC vertical planes
        found_vertical_planes = self.find_vertical_planes(points)
        if not found_vertical_planes or len(found_vertical_planes[0]) < 2:
            return {"plane_model": None, "distance_m": None, "is_door_candidate": False}, None, found_vertical_planes

        detected_plane = None
        for i, (plane_model, inlier_indices, inlier_points) in enumerate(found_vertical_planes):
            detected_plane = self.draw_plane_outline_on_image(
                color_image, plane_model, inlier_points, fx, fy, cx, cy)
            
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

        plane_model, inlier_indices = found_vertical_planes[0][:2]
        if len(inlier_indices) == 0:
            return {"plane_model": None, "distance_m": None, "is_door_candidate": False}, None, found_vertical_planes

        a, b, c, d = plane_model
        plane_norm = np.sqrt(a*a + b*b + c*c)
        distance_m = abs(d) / (plane_norm + 1e-12)

        is_door = False
        if inlier_density <= max_inlier_density:
            is_door = True

        result = {
            "plane_model": plane_model,
            "distance_m": distance_m,
            "is_door_candidate": is_door,
        }

        return result, detected_plane, found_vertical_planes


# Backward-compatible function wrapper
_PLANE_DETECTOR = None

def detect_glass_door_plane(color_image, depth_image_in_meters,
                            fx, fy, cx, cy,
                            max_inlier_density=1.0,
                            max_depth_consider=4.0,
                            min_depth_consider=1.0):
    global _PLANE_DETECTOR
    if _PLANE_DETECTOR is None:
        _PLANE_DETECTOR = PlaneDetector()
    return _PLANE_DETECTOR.detect(color_image, depth_image_in_meters, fx, fy, cx, cy,
                                  max_inlier_density=max_inlier_density,
                                  max_depth_consider=max_depth_consider,
                                  min_depth_consider=min_depth_consider)
