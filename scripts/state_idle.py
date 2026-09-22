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

    def entry_action(self, ctx: FrameContext):
        """
        Truncate the per-run diagnostic logs once, on entering idle.

        Args:
            ctx:
                Shared frame context; unused here.

        Raises:
            OSError:
                If either log file cannot be created or written.

        Notes:
            Clearing belongs to entering the state. Doing it in ``do_action``
            rewrote both files on every idle frame, which is two file opens and
            writes per frame for the whole time the node sits idle.
        """
        if self.enable_result_log:
            # Clear old log output for this run
            with open(self.door_state_log_path, "w") as f:
                f.write(f"Logging door state by detection algorithm\n")
            with open(self.smoothed_door_state_log_path, "w") as f:
                f.write(f"Logging smoothed_door_state after temporal smoothing\n")

    def do_action(self, ctx: FrameContext):
        """
        Enter plane search when requested.

        Args:
            ctx:
                Shared context containing ``start_door_frame_detection``.

        Returns:
            ``"searching_door_plane_state"`` when detection is requested;
            otherwise ``None``.

        Notes:
            The log headers are written by :meth:`entry_action`, so they are
            reset once per idle entry rather than once per frame.
        """
        # Consume-style triggering is handled by the surrounding controller;
        # this state only decides when a new detection run may begin.
        if ctx.start_door_frame_detection:
            return "searching_door_plane_state"
        return None # stay in idle state until get request to start door frame detection
