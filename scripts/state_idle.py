#!/usr/bin/env python3
import rospkg
import rospy
from Frame_data import BaseState, FrameContext
import os

class idle_state(BaseState):
    def __init__(self):
        super().__init__("idle_state")
    def do_action(self, ctx: FrameContext):
        pkg_path = rospkg.RosPack().get_path('robodog_glass_door_detection')
        log_dir = os.path.join(pkg_path, 'scripts')
        os.makedirs(log_dir, exist_ok=True)
        door_state_file_path = os.path.join(log_dir, 'door_states.txt')
        final_door_status_file_path = os.path.join(log_dir, 'final_door_status_after_temporal_smoothing.txt')
        with open(door_state_file_path, "w") as f:
            f.write(f"Logging door state by detection algorithm\n")
        with open(final_door_status_file_path, "w") as f:
            f.write(f"Logging smoothed_door_state after temporal smoothing\n")
        
        if ctx.start_door_frame_detection:
            return "searching_door_plane_state"
        return None # stay in idle state until get request to start door frame detection