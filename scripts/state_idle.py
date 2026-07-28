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
        super().__init__("idle_state")

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
