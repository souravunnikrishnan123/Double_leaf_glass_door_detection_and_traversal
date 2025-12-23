#!/usr/bin/env python3
import rospy
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np

@dataclass
class FrameContext:
    depth_image_in_meters: np.ndarray
    color_image: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    depth_frame: Optional[Any] = None

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
    passable_ratio: Optional[float] = None

    # visualization buffers (published as ROS images)
    viz_color_stack: Optional[np.ndarray] = None
    viz_depth_stack: Optional[np.ndarray] = None
    viz_plane_overlay: Optional[np.ndarray] = None
    bird_eye_view_color_based: Optional[np.ndarray] = None
    bird_eye_view_depth_based: Optional[np.ndarray] = None
    bird_eye_view_full_image_view : Optional[np.ndarray] = None
    passability_view_color_based: Optional[np.ndarray] = None
    passability_view_depth_based: Optional[np.ndarray] = None
    passability_view_full_image_view : Optional[np.ndarray] = None




class BaseState:
    """Base class for all robot states."""
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

