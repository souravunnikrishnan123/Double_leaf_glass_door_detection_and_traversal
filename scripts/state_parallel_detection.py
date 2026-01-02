#!/usr/bin/env python3
from Frame_data import BaseState, FrameContext
from typing import Optional
from color_door_detector import ColorDoorDetector
from door_status_detector import Door_Status_Detector
from depth_door_detector import DepthDoorDetector

from visualization_utils import show_stacked_visualization, build_stacked_visualization
import rospy
import rospkg
import os

class parallel_detection_state(BaseState):
    def __init__(self):
        super().__init__("parallel_detection_state")
        # child current states
        self.color_frame_detection_state = ColorDoorDetector()#only need to create state object once.
        self.depth_frame_detection_state = DepthDoorDetector()#only need to create state object once.
        self.detecting_door_status_based_on_color = Door_Status_Detector("color_based")
        self.detecting_door_status_based_on_depth = Door_Status_Detector("depth_based")


    def do_action(self, ctx: FrameContext) -> Optional[str]:
        # Advance color branch
   
        result_c = self.color_frame_detection_state.process_frame(ctx)
        if result_c.roi_left is not None and result_c.roi_right is not None:
            #substate "color_image_based_detecting_door_state"
            door_state_c, roi_open_side_c =self.detecting_door_status_based_on_color.detect(ctx)
        else:
            # "no_mid_door_frame_detected_state"
            #check bev based detection with full image
            door_state_c = "no_frame_detected"
            roi_open_side_c = None
            pass  # could log or handle no detection case here

        result_d = self.depth_frame_detection_state.process_frame(ctx)
        if result_d.roi_left is not None and result_d.roi_right is not None:
            #substate "depth_image_based_detecting_door_state"
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


        


        