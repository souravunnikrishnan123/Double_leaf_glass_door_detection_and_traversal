#!/usr/bin/env python3
import rospy
from Frame_data import BaseState , FrameContext
from check_if_passable import Passability_checker
import numpy as np

class check_if_passable_state(BaseState):
    def __init__(self):
        super().__init__("check_if_passable_state")

        self.check_passability = Passability_checker()

    def do_action(self, ctx: FrameContext):


        # "color_based" or "depth_based" or "full_image_view"

        Passability_status, front_clearance, passability_view = self.check_passability.run(
        ctx.depth_image_in_meters, ctx.fx, ctx.fy, ctx.cx, ctx.cy, getattr(ctx, f"color_image_{ctx.type_of_check_passability}"), mid_frame_x_px, ctx.door_state_label, getattr(ctx, f"door_depth_m_{ctx.type_of_check_passability}"), ctx.type_of_check_passability)

        # Append passability status to door state
        if Passability_status: # True or False
            ctx.passability_status = "passable"
        else:
            ctx.passability_status = "not_passable"

        # Persist visualizations in context
        setattr(ctx, f"passability_view_{ctx.type_of_check_passability}", passability_view)
        return None  # Stay in the current state