


#!/usr/bin/env python3
import rospy
from door_status import detect_door_state


class Door_Status_Detector:
    def __init__(self, keyword: str = "color_based"):
        ns = "~door_status_detector"
        self.keyword = keyword
        # Core ROI params
        self.roi_width = rospy.get_param(f"{ns}/roi_width",240)
        self.margin = rospy.get_param(f"{ns}/margin", 10)
        self.threshold = rospy.get_param(f"{ns}/threshold", 0.05)

        # Grouped detector params (similar to DepthDoorDetector style)
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

    def detect(self, ctx):

        if getattr(ctx, f"roi_left_{self.keyword}") is None or getattr(ctx, f"roi_right_{self.keyword}") is None:
            return None

        # Call detector and then assign views to context separately
        door_state, passability_view, bird_eye_view = detect_door_state(
            ctx.depth_image_in_meters,
            ctx.fx,
            ctx.fy,
            ctx.cx,
            ctx.cy,
            getattr(ctx, f"color_image_{self.keyword}"),
            getattr(ctx, f"roi_left_{self.keyword}"),
            getattr(ctx, f"roi_right_{self.keyword}"),
            roi_width=self.roi_width,
            margin=self.margin,
            threshold=self.threshold,
            z_door_depth=getattr(ctx, f"door_depth_m_{self.keyword}"),
            keyword=self.keyword,
            back_proj_params=self.back_proj_params,
            filter_points_params=self.filter_points_params,
            bev_params=self.bev_params
            
        )

        # Persist visualizations in context
        setattr(ctx, f"passability_view_{self.keyword}", passability_view)
        setattr(ctx, f"bird_eye_view_{self.keyword}", bird_eye_view)

        if door_state is not None:
            setattr(ctx, f"door_state_{self.keyword}", door_state)
            return door_state

        return None  # Stay in the current state if door state cannot be determined