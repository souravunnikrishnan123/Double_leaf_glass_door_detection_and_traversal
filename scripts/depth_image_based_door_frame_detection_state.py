#!/usr/bin/env python3
import rospy
from Frame_data import BaseState
from Frame_data import FrameContext
from depth_door_detector import DepthDoorDetector, DepthDetectorConfig
import numpy as np

class depth_image_based_door_frame_detection_state(BaseState):
    def __init__(self):
        super().__init__("depth_image_based_door_frame_detection_state")
        # Instantiate OO facade once; keeps config and potential temporal state
        self.detector = DepthDoorDetector(DepthDetectorConfig())

    def do_action(self, ctx: FrameContext):
        result = self.detector.process_frame(ctx)

        if result.roi_left is not None and result.roi_right is not None:
            return "depth_image_based_detecting_door_state"
        return None  # Stay in the current state if frame detection fails
        