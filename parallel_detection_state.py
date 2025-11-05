from Frame_data import BaseState, FrameContext
from typing import Optional
from color_image_based_door_frame_detection_state import color_image_based_door_frame_detection_state
from color_image_based_detecting_door_state import color_image_based_detecting_door_state
from depth_image_based_door_frame_detection_state import depth_image_based_door_frame_detection_state
from depth_image_based_detecting_door_state import depth_image_based_detecting_door_state
from visualization_utils import show_stacked_visualization


class parallel_detection_state(BaseState):
    def __init__(self):
        super().__init__("parallel_detection_state")
        # child current states
        self.color_state = color_image_based_door_frame_detection_state()
        self.depth_state = depth_image_based_door_frame_detection_state()

    def entry_action(self, ctx: FrameContext) -> None:
        # reset branch outputs on entry
        pass

    def do_action(self, ctx: FrameContext) -> Optional[str]:
        # Advance color branch
        nxt_c = self.color_state.do_action(ctx)
        if nxt_c == "color_image_based_detecting_door_state":
            self.color_state = color_image_based_detecting_door_state()

        # Advance depth branch
        nxt_d = self.depth_state.do_action(ctx)
        if nxt_d == "depth_image_based_detecting_door_state":
            self.depth_state = depth_image_based_detecting_door_state()

        esc_pressed = show_stacked_visualization(
            ctx.color_image_color_based, MIN_DEPTH = 0.3, MAX_DEPTH = 6.0, edges = ctx.edges, depth_frame = ctx.depth_frame, window_name= "Color | Depth | Edges+ Lines"
        )

        esc_pressed = show_stacked_visualization(
            ctx.color_image_depth_based, MIN_DEPTH = 1.8, MAX_DEPTH = 2.5, edges = ctx.sobel_vis_color, depth_frame=ctx.depth_frame, window_name="depth lines in color image | Depth for depth lines| Sobel + Depth Edges"
            )


        # If both branches produced a door_state, proceed to combine
        if ctx.door_state_color_based is not None and ctx.door_state_depth_based is not None:
            return "combine_door_state"
        return None