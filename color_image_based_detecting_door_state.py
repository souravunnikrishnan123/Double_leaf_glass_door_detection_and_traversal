import numpy as np
from Frame_data import BaseState, FrameContext
from door_status import detect_door_state

class color_image_based_detecting_door_state(BaseState):
    def __init__(self):
        super().__init__("color_image_based_detecting_door_state")

    def do_action(self, ctx: FrameContext):
        # Analyze the door state using the ROI and depth information
        # Update ctx.door_state_label and ctx.passable_ratio accordingly

        if ctx.roi_left_color_based is None or ctx.roi_right_color_based is None:
            return None

        door_state = detect_door_state(ctx.depth_image_in_meters,ctx.fx, ctx.fy, ctx.cx, ctx.cy, ctx.color_image_color_based, ctx.roi_left_color_based, ctx.roi_right_color_based, ctx.found_vertical_planes,
            roi_width=240, margin=10, threshold=0.05, z_door_depth=ctx.door_depth_m_color_based, plotname = "door_state_based_on_color_image")

        if door_state is not None:
            ctx.door_state_color_based = door_state
            return door_state

        return None  # Stay in the current state if door state cannot be determined