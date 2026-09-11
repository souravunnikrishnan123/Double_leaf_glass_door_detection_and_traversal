#!/usr/bin/env python3
"""State that runs the color and depth door-frame branches in sequence."""

from Frame_data import BaseState, FrameContext
from typing import Optional
from color_door_detector import ColorDoorDetector
from door_status_detector import Door_Status_Detector
from depth_door_detector import DepthDoorDetector
from find_glass_frame_lines import GlassFrameLineProcessor
from visualization_utils import show_stacked_visualization, build_stacked_visualization
import rospy
import rospkg
import os

class dual_branch_frame_detection_state(BaseState):
    """
    Run both detector branches and store their independent status results.

    Despite the historical state name, both branches currently execute
    sequentially in one callback. Separate door-status instances carry the
    keyword used to select color- or depth-specific context fields.

    Attributes:
        ransac_error:
            Fractional depth tolerance around the latest plane distance.

        color_frame_detection_state:
            Color Canny/Hough line detector.

        depth_frame_detection_state:
            Depth Sobel/Hough line detector.

        glass_frame_detector_based_on_color:
            Pairing and ROI processor for color-derived lines.

        glass_frame_detector_based_on_depth:
            Pairing and ROI processor for depth-derived lines.

        detecting_door_status_based_on_color:
            Branch-specific side-depth classifier.

        detecting_door_status_based_on_depth:
            Branch-specific side-depth classifier.

        door_geometry:
            Latest physical geometry mapping, refreshed per frame.
    """

    def __init__(self):
        """
        Construct reusable processing objects for both detector branches.

        Notes:
            Instances persist across frames to avoid rebuilding configuration
            and to keep branch-specific status keywords.
        """
        super().__init__("dual_branch_frame_detection_state")
        self.ransac_error = rospy.get_param(f"~plane_detector/output/ransac_error", 0.02)
        self.door_state_log_path = rospy.get_param("~result_log_path")+"/door_states.txt"  # Path to log file for door states
        # child current states
        # Reuse the helpers across frames so ROS parameters and timing state are
        # not rebuilt in the camera callback.
        self.color_frame_detection_state = ColorDoorDetector()#only need to create state object once.
        self.depth_frame_detection_state = DepthDoorDetector()#only need to create state object once.
        self.glass_frame_detector_based_on_color = GlassFrameLineProcessor("color_based")
        self.glass_frame_detector_based_on_depth = GlassFrameLineProcessor("depth_based")
        self.detecting_door_status_based_on_color = Door_Status_Detector("color_based")
        self.detecting_door_status_based_on_depth = Door_Status_Detector("depth_based")
        self.enable_windows = rospy.get_param("~enable_visualization", False)
        self.enable_result_log = rospy.get_param("~enable_result_log", False)  # Whether to log results to file

    def do_action(self, ctx: FrameContext) -> Optional[str]:
        """
        Process both branches, create debug views, and request fusion.

        Args:
            ctx:
                Shared context with aligned color/depth input, intrinsics, and
                branch-specific visualization images.

        Returns:
            Always returns ``"combine_door_state"`` after both branch results
            have been written to ``ctx``.

        Notes:
            Door geometry and plane distance are re-read each frame because the
            plane-search state can update them at runtime. A branch without
            valid ROIs receives the label ``"no_frame_detected"``.
        """
        # get the latest door geometry and distance from ROS params
        # these values are changed during runtime. hence need to load the door geometry and plane distance in this state as well to make sure the latest value is used for detection

        # Plane search can update these measurements after examining the current
        # doorway, so cached constructor values would quickly become stale.
        self.door_geometry = {
            "glass_width_cm": rospy.get_param("~door_geometry/glass_width_cm", 40),
            "center_frame_width_cm": rospy.get_param("~door_geometry/center_frame_width_cm", 30),
            "roi_width": rospy.get_param("~door_geometry/roi_width", 240),
            "correction_factor": rospy.get_param("~door_geometry/correction_factor", 1.1),
        }
        ransac_plane_distance = rospy.get_param("~plane_detector/output/ransac_plane_distance", 2.0)
        DEPTH_RANGE = [ransac_plane_distance * (1 - self.ransac_error), ransac_plane_distance * (1 + self.ransac_error)]

        # Keep the two answers independent until fusion; a failure in one branch
        # should remain visible rather than silently borrowing the other's ROI.
        # Advance color branch
        potential_frame_lines_from_c , depth_of_potential_frame_lines_c, edges_c = self.color_frame_detection_state.process_frame(ctx)
        ctx.edges = edges_c
        roi_left_c, roi_right_c, mean_z_depth_along_frame_lines_c = self.glass_frame_detector_based_on_color.find_left_right_roi_and_door_depth(
                ctx.depth_image_in_meters,
                ctx.color_image_color_based,
                ctx.fx,
                potential_frame_lines_from_c,
                depth_of_potential_frame_lines_c,
                self.door_geometry,
                DEPTH_RANGE
            )


        if roi_left_c is not None and roi_right_c is not None:
            #only update context if valid ROIs are found for better debugging
            ctx.roi_left_color_based = roi_left_c
            ctx.roi_right_color_based = roi_right_c
            ctx.door_depth_m_color_based = mean_z_depth_along_frame_lines_c
            door_state_c, roi_open_side_c =self.detecting_door_status_based_on_color.detect(ctx)
        else:
            # "no_mid_door_frame_detected_state"
            #check bev based detection with full image
            door_state_c = "no_frame_detected"
            roi_open_side_c = None
            pass  # could log or handle no detection case here



        # Advance depth branch
        potential_frame_lines_from_d , depth_of_potential_frame_lines_d, edges_d = self.depth_frame_detection_state.process_frame(ctx)
        ctx.sobel_vis_color = edges_d
        roi_left_d, roi_right_d, mean_z_depth_along_frame_lines_d = self.glass_frame_detector_based_on_depth.find_left_right_roi_and_door_depth(
                ctx.depth_image_in_meters,
                ctx.color_image_depth_based,
                ctx.fx,
                potential_frame_lines_from_d,
                depth_of_potential_frame_lines_d,
                self.door_geometry,
                DEPTH_RANGE
            )

        if roi_left_d is not None and roi_right_d is not None:
            #only update context if valid ROIs are found for better debugging
            ctx.roi_left_depth_based = roi_left_d
            ctx.roi_right_depth_based = roi_right_d
            ctx.door_depth_m_depth_based =  mean_z_depth_along_frame_lines_d
            door_state_d, roi_open_side_d = self.detecting_door_status_based_on_depth.detect(ctx)
        else:
            # "no_mid_door_frame_detected_state"
            #check bev based detection with full image
            door_state_d = "no_frame_detected"
            roi_open_side_d = None
            pass  # could log or handle no detection case here


        setattr(ctx, "door_state_color_based", door_state_c)
        setattr(ctx, "roi_open_side_color_based", roi_open_side_c)


        setattr(ctx, "door_state_depth_based", door_state_d)
        setattr(ctx, "roi_open_side_depth_based", roi_open_side_d)

        # Optional window display controlled by param
        if self.enable_windows:
            # Build images even when local windows are disabled; remote ROS tools use them.
            # Build stacked visualizations for publishing
            ctx.viz_color_stack = build_stacked_visualization(
                ctx.color_image_color_based, MIN_DEPTH=0.3, MAX_DEPTH=6.0, edges=ctx.edges, depth_frame=ctx.depth_frame
            )

            ctx.viz_depth_stack = build_stacked_visualization(
                ctx.color_image_depth_based, MIN_DEPTH=1.9, MAX_DEPTH=2.1, edges=ctx.sobel_vis_color, depth_frame=ctx.depth_frame
            )

            show_stacked_visualization(
                ctx.color_image_color_based, MIN_DEPTH=0.3, MAX_DEPTH=6.0, edges=ctx.edges, depth_frame=ctx.depth_frame, window_name="Color | Depth | Edges+ Lines"
            )
            show_stacked_visualization(
                ctx.color_image_depth_based, MIN_DEPTH=1.8, MAX_DEPTH=2.5, edges=ctx.sobel_vis_color, depth_frame=ctx.depth_frame, window_name="depth lines in color image | Depth for depth lines| Sobel + Depth Edges"
            )

        # Persist latest states to a text log for debugging/analysis
        if self.enable_result_log:
            with open(self.door_state_log_path, "a") as f:
                f.write(f"Color-based door state: {door_state_c}, Depth-based door state: {door_state_d}\n")

        return "combine_door_state"



