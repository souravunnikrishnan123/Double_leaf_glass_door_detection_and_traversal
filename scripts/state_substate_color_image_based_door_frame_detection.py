#!/usr/bin/env python3
import rospy
from Frame_data import BaseState
from Frame_data import FrameContext
from color_door_detector import ColorDoorDetector

class color_image_based_door_frame_detection_state(BaseState):
    def __init__(self):
        super().__init__("color_image_based_door_frame_detection_state")
        # Instantiate OO facade once to keep config and any future temporal state
        self.detector = ColorDoorDetector()

    def do_action(self, ctx: FrameContext):

        if ctx.detected_plane is None:
            return None

        result = self.detector.process_frame(ctx)

        if result.roi_left is not None and result.roi_right is not None:
            return "color_image_based_detecting_door_state"

        return None  # Stay in the current state if frame detection fails
        