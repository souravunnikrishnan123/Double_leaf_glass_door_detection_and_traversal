

"""Fallback state for passability checks without a reliable frame pair."""

from Frame_data import FrameContext,BaseState
import numpy as np

import rospy


class full_image_passability_check_state(BaseState):
    """
    Prepare full-image passability inputs when frame detection failed.

    This fallback cannot provide a central frame pixel. It supplies a depth
    reference instead, allowing the downstream node to search the full view
    for a virtual corridor.

    Attributes:
        reference_door_distance_m:
            Configured nominal door distance used when no plane was found.

        ransac_plane_distance:
            Latest confirmed plane distance used when the plane exists but no
            center-frame pair was detected.
    """

    def __init__(self):
        """
        Initialize fallback depth references from ROS parameters.

        Notes:
            ``~plane_detector/output/ransac_plane_distance`` is required and
            has no local default in this constructor.
        """
        super().__init__("full_image_passability_check_state")
        self.reference_door_distance_m = rospy.get_param("~reference_door_distance_m", 2.0)
        self.ransac_plane_distance = rospy.get_param(f"~plane_detector/output/ransac_plane_distance")
    def do_action(self, ctx: FrameContext):
        """
        Choose a fallback depth reference and finish door detection.

        Args:
            ctx:
                Shared context containing the fused failure label.

        Returns:
            Always returns ``"final_state"``.

        Notes:
            ``No_door_plane_detected`` uses the configured nominal distance,
            whereas ``no_frame_detected`` uses the most recent RANSAC distance.
        """
        ctx.mid_frame_x_px_for_passability_check = None  # Not applicable
        # door state label should have been set in the previous state when determine to transition to this state
        if ctx.door_state_label == "No_door_plane_detected":
            ctx.door_depth = self.reference_door_distance_m  # set a default door depth for passability check when no door plane is detected
            rospy.logwarn("No door plane detected in the previous state, so we have to directly check the passability with full image and depth without relying on mid frame door detection. This is the least desirable case because it means the door detection algorithm fails to detect any reliable door signal in the middle frame. The passability check result in this case will be less reliable and more noisy, so please be cautious when using the passability check result in this case.")
        
        elif ctx.door_state_label == "no_frame_detected":
            rospy.logwarn("No door frame detected in the middle frame, and the smoothed door state is also no_frame_detected, so we have to directly check the passability with full image and depth without relying on mid frame door detection. This is the least desirable case because it means the door detection algorithm fails to detect any reliable door signal in the middle frame. The passability check result in this case will be less reliable and more noisy, so please be cautious when using the passability check result in this case.")
            ctx.door_depth = self.ransac_plane_distance
        
        return "final_state"  # Transition back to final state and wait until  get request to go to idle state again
