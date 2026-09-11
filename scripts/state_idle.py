#!/usr/bin/env python3
"""Idle state for starting a fresh door-detection cycle."""

import rospkg
import rospy
from Frame_data import BaseState, FrameContext
import os

class idle_state(BaseState):
    """
    Wait for a start request and reset per-run diagnostic logs.

    The state truncates the raw and smoothed door-state log files while it is
    idle, then transitions to plane search after the start flag is received.

    Attributes:
        name:
            Fixed state-machine key ``"idle_state"`` inherited from
            :class:`BaseState`.
    """

    def __init__(self):
        """
        Initialize the idle state with its registered transition key.

        Notes:
            Per-cycle data is reset by the surrounding state-machine workflow;
            construction only assigns the state name.
        """
        self.enable_result_log = rospy.get_param("~enable_result_log", False)  # Whether to log results to file
        super().__init__("idle_state")
        self.door_state_log_path = rospy.get_param("~result_log_path")+"/door_states.txt"  # Path to log file for door states
        self.smoothed_door_state_log_path = rospy.get_param("~result_log_path")+"/final_door_status_after_temporal_smoothing.txt"  # Path to log file for smoothed door states

    def do_action(self, ctx: FrameContext):
        """
        Clear old log output and enter plane search when requested.

        Args:
            ctx:
                Shared context containing ``start_door_frame_detection``.

        Returns:
            ``"searching_door_plane_state"`` when detection is requested;
            otherwise ``None``.

        Raises:
            OSError:
                If the package log directory or either log file cannot be
                created or written.

        Notes:
            This method currently rewrites the two log headers on every idle
            frame, not only when the state is first entered.
        """
        if self.enable_result_log:
            # Clear old log output for this run
            with open(self.door_state_log_path, "w") as f:
                f.write(f"Logging door state by detection algorithm\n")
            with open(self.smoothed_door_state_log_path, "w") as f:
                f.write(f"Logging smoothed_door_state after temporal smoothing\n")
        
        # Consume-style triggering is handled by the surrounding controller;
        # this state only decides when a new detection run may begin.
        if ctx.start_door_frame_detection:
            return "searching_door_plane_state"
        return None # stay in idle state until get request to start door frame detection
