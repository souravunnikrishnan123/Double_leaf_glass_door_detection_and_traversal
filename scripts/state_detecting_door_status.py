#!/usr/bin/env python3
import rospy
from Frame_data import BaseState, FrameContext
from door_status_detector import Door_Status_Detector

class detecting_door_status(BaseState):
    def __init__(self):
        super().__init__("detecting_door_status")
        self.door_status_detector = Door_Status_Detector()

    def do_action(self, ctx: FrameContext, keyword: str = "color_based"):
        # Analyze the door state using the ROI and depth information
        # Update ctx.door_state_label and ctx.passable_ratio accordingly

        door_state = self.door_status_detector.detect(ctx, keyword)
        return door_state