#!/usr/bin/env python3
"""Terminal state that holds a completed detection result."""

import rospkg
import rospy
from Frame_data import BaseState, FrameContext

class final_state(BaseState):
    """Keep the published result stable until a reset request arrives."""

    def __init__(self):
        super().__init__("final_state")
    def do_action(self, ctx: FrameContext):
        """Return to idle after the downstream traversal cycle finishes."""
        if ctx.go_to_idle_from_finish_state:
            return "idle_state"

        return  None #stay in final state
