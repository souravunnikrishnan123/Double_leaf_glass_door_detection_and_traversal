#!/usr/bin/env python3
"""Shared data contracts for the glass-door detection state machine.

The state machine keeps one :class:`FrameContext` alive and refreshes its
sensor fields for every synchronized frame. Detection states add their
intermediate and final results to the same object.
"""

import rospy
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np

@dataclass
class FrameContext:
    """Per-frame inputs and results exchanged between detection states.

    The required fields describe the current aligned color/depth frame and its
    camera intrinsics. Optional fields are populated progressively by the
    plane, color, depth, fusion, and visualization stages.
    """

    depth_image_in_meters: np.ndarray
    color_image: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    depth_frame: Optional[Any] = None

    go_to_idle_from_finish_state: bool = False
    start_door_frame_detection : bool = False
    # shared artifacts across states
    plane_result: Optional[dict] = None
    color_image_for_plane_detection: Optional[np.ndarray] = None
    
    # color-based branch
    roi_left_color_based: Optional[np.ndarray] = None
    roi_right_color_based: Optional[np.ndarray] = None
    door_depth_m_color_based: Optional[float] = None
    roi_open_side_color_based: Optional[np.ndarray] = None
    door_state_color_based: Optional[str] = None
    color_image_color_based: Optional[np.ndarray] = None
    edges : Optional[np.ndarray] = None

    # depth-based branch
    roi_left_depth_based: Optional[np.ndarray] = None
    roi_right_depth_based: Optional[np.ndarray] = None
    door_depth_m_depth_based: Optional[float] = None
    roi_open_side_depth_based: Optional[np.ndarray] = None
    door_state_depth_based: Optional[str] = None
    color_image_depth_based: Optional[np.ndarray] = None
    sobel_vis_color : Optional[np.ndarray] = None

    # final
    door_state_label: Optional[str] = None
    mid_frame_x_px_for_passability_check: Optional[float] = None
    door_depth : Optional[float] = None

    # visualization buffers (published as ROS images)
    viz_color_stack: Optional[np.ndarray] = None
    viz_depth_stack: Optional[np.ndarray] = None
    viz_plane_overlay: Optional[np.ndarray] = None


class BaseState:
    """Base class for states driven once per synchronized camera frame."""

    def __init__(self, name):
        self.name = name

    def entry_action(self, ctx : FrameContext) -> None:
        """Called when entering this state."""
        pass

    def do_action(self, ctx : FrameContext)-> Optional[str]:
        """
        Called every frame. Return next state name to transition, or None to stay.
        """
        return None

    def exit_action(self, ctx: FrameContext) -> None:
        """Called before leaving this state."""
        pass
