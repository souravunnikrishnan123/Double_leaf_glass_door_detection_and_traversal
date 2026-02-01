#!/usr/bin/env python3
import rospkg
import rospy
from Frame_data import BaseState, FrameContext

class final_state(BaseState):
    def __init__(self):
        super().__init__("final_state")
    def do_action(self, ctx: FrameContext):
        if ctx.go_to_idle_from_finish_state:
            return "idle_state"

        return  None #stay in final state