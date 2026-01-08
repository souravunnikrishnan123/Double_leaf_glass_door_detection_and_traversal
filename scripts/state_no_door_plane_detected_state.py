

from Frame_data import FrameContext,BaseState
import numpy as np

import rospy


class full_image_passability_check_state(BaseState):
    def __init__(self):
        super().__init__("full_image_passability_check_state")
        self.reference_door_distance_m = rospy.get_param("~plane_detector/reference_door_distance_m", 2.0)  # meters

    def do_action(self, ctx: FrameContext):
        ctx.door_state_label = "No_door_plane_detected"  # No door plane detected
        ctx.mid_frame_x_px_for_passability_check = None  # Not applicable
        ctx.door_depth = self.reference_door_distance_m  # Default depth when no door plane is
        return "idle_state"  # Transition back to idle state