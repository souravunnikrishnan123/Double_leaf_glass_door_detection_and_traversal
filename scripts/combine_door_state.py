#!/usr/bin/env python3
import rospy
from typing import Optional
from Frame_data import BaseState, FrameContext

class combine_door_state(BaseState):
    def __init__(self):
        super().__init__("combine_door_state")

    def do_action(self, ctx: FrameContext) -> Optional[str]:
        c = ctx.door_state_color_based
        d = ctx.door_state_depth_based

        # Simple fusion: agree -> that label; else prefer depth; fallback to color; else unknown
        if c is not None and d is not None and c == d:
            final = c
        elif d is not None:
            final = d
        elif c is not None:
            final = c
        else:
            final = "unknown"

        ctx.door_state_label = final

        # Optionally: clear branch state for next cycle
        #ctx.roi_left_color_based = ctx.roi_right_color_based = None
        #ctx.roi_left_depth_based = ctx.roi_right_depth_based = None
        #ctx.door_state_color_based = ctx.door_state_depth_based = None
        #ctx.door_depth_m_color_based = ctx.door_depth_m_depth_based = None

        return "parallel_detection_state"