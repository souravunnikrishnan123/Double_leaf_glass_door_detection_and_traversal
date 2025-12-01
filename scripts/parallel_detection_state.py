#!/usr/bin/env python3
from Frame_data import BaseState, FrameContext
from typing import Optional
from color_image_based_door_frame_detection_state import color_image_based_door_frame_detection_state
from color_image_based_detecting_door_state import color_image_based_detecting_door_state
from depth_image_based_door_frame_detection_state import depth_image_based_door_frame_detection_state
from depth_image_based_detecting_door_state import depth_image_based_detecting_door_state
from visualization_utils import show_stacked_visualization, build_stacked_visualization
import rospy


class parallel_detection_state(BaseState):
    def __init__(self):
        super().__init__("parallel_detection_state")
        # child current states
        self.color_frame_detection_state = color_image_based_door_frame_detection_state()#only need to create state object once.
        self.depth_frame_detection_state = depth_image_based_door_frame_detection_state()#only need to create state object once.
        self.color_detecting_door_state = color_image_based_detecting_door_state()
        self.depth_detecting_door_state = depth_image_based_detecting_door_state()

    def do_action(self, ctx: FrameContext) -> Optional[str]:
        # Advance color branch
   
        nxt_c = self.color_frame_detection_state.do_action(ctx)
        if nxt_c == "color_image_based_detecting_door_state":
            self.color_detecting_door_state.do_action(ctx)

        # Advance depth branch
        nxt_d = self.depth_frame_detection_state.do_action(ctx)
        if nxt_d == "depth_image_based_detecting_door_state":
            self.depth_detecting_door_state.do_action(ctx)


        # Build stacked visualizations for publishing
        ctx.viz_color_stack = build_stacked_visualization(
            ctx.color_image_color_based, MIN_DEPTH=0.3, MAX_DEPTH=6.0, edges=ctx.edges, depth_frame=ctx.depth_frame
        )
        ctx.viz_depth_stack = build_stacked_visualization(
            ctx.color_image_depth_based, MIN_DEPTH=1.8, MAX_DEPTH=2.5, edges=ctx.sobel_vis_color, depth_frame=ctx.depth_frame
        )

        # Optional window display controlled by param
        if rospy.get_param("~enable_windows", False):
            show_stacked_visualization(
                ctx.color_image_color_based, MIN_DEPTH=0.3, MAX_DEPTH=6.0, edges=ctx.edges, depth_frame=ctx.depth_frame, window_name="Color | Depth | Edges+ Lines"
            )
            show_stacked_visualization(
                ctx.color_image_depth_based, MIN_DEPTH=1.8, MAX_DEPTH=2.5, edges=ctx.sobel_vis_color, depth_frame=ctx.depth_frame, window_name="depth lines in color image | Depth for depth lines| Sobel + Depth Edges"
            )
            #cv2.waitKey(1)


        # If both branches produced a door_state, proceed to combine
        if ctx.door_state_color_based is not None or ctx.door_state_depth_based is not None:
            return "combine_door_state"
        return None

        


        