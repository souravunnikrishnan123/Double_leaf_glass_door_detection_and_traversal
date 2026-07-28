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
    """Run both detector branches and store their independent status results.

    Despite the historical state name, both branches currently execute
    sequentially in one callback. Separate door-status instances carry the
    keyword used to select color- or depth-specific context fields.
    """

    def __init__(self):
        super().__init__("dual_branch_frame_detection_state")
        self.ransac_error = rospy.get_param(f"~plane_detector/output/ransac_error", 0.02)

        # child current states
        self.color_frame_detection_state = ColorDoorDetector()#only need to create state object once.
        self.depth_frame_detection_state = DepthDoorDetector()#only need to create state object once.
        self.glass_frame_detector_based_on_color = GlassFrameLineProcessor("color_based")
        self.glass_frame_detector_based_on_depth = GlassFrameLineProcessor("depth_based")
        self.detecting_door_status_based_on_color = Door_Status_Detector("color_based")
        self.detecting_door_status_based_on_depth = Door_Status_Detector("depth_based")





    def do_action(self, ctx: FrameContext) -> Optional[str]:
        """Process both branches, create debug views, and request fusion."""
        # get the latest door geometry and distance from ROS params
        # these values are changed during runtime. hence need to load the door geometry and plane distance in this state as well to make sure the latest value is used for detection

        self.door_geometry = {
            "glass_width_cm": rospy.get_param("~door_geometry/glass_width_cm", 40),
            "center_frame_width_cm": rospy.get_param("~door_geometry/center_frame_width_cm", 30),
            "roi_width": rospy.get_param("~door_geometry/roi_width", 240),
            "correction_factor": rospy.get_param("~door_geometry/correction_factor", 1.1),
        }
        ransac_plane_distance = rospy.get_param("~plane_detector/output/ransac_plane_distance", 2.0)
        DEPTH_RANGE = [ransac_plane_distance * (1 - self.ransac_error), ransac_plane_distance * (1 + self.ransac_error)]

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


        # Build stacked visualizations for publishing
        ctx.viz_color_stack = build_stacked_visualization(
            ctx.color_image_color_based, MIN_DEPTH=0.3, MAX_DEPTH=6.0, edges=ctx.edges, depth_frame=ctx.depth_frame
        )

        ctx.viz_depth_stack = build_stacked_visualization(
            ctx.color_image_depth_based, MIN_DEPTH=1.9, MAX_DEPTH=2.1, edges=ctx.sobel_vis_color, depth_frame=ctx.depth_frame
        )


        # Optional window display controlled by param
        if rospy.get_param("~enable_windows", False):
            show_stacked_visualization(
                ctx.color_image_color_based, MIN_DEPTH=0.3, MAX_DEPTH=6.0, edges=ctx.edges, depth_frame=ctx.depth_frame, window_name="Color | Depth | Edges+ Lines"
            )
            show_stacked_visualization(
                ctx.color_image_depth_based, MIN_DEPTH=1.8, MAX_DEPTH=2.5, edges=ctx.sobel_vis_color, depth_frame=ctx.depth_frame, window_name="depth lines in color image | Depth for depth lines| Sobel + Depth Edges"
            )
        # Persist latest states to a text log for debugging/analysis

        try:
            # Resolve package path dynamically to avoid hard-coded container paths
            pkg_path = rospkg.RosPack().get_path('robodog_glass_door_detection')
            log_dir = os.path.join(pkg_path, 'scripts')
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'door_states.txt')


            with open(log_path, "a") as f:
                f.write(f"Color-based door state: {door_state_c}, Depth-based door state: {door_state_d}\n")

        except Exception as e:
            rospy.logwarn(f"Failed to write door states: {e}")
        return "combine_door_state"





