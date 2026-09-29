#!/usr/bin/env python3
"""Shared data contracts for the glass-door detection state machine.

The state machine keeps one :class:`FrameContext` alive and refreshes its
sensor fields for every synchronized frame. Detection states add their
intermediate and final results to the same object.
"""

import rospy
from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np
from visualization_utils import DiagnosticBanner

@dataclass
class FrameContext:
    """
    Hold per-frame inputs and results exchanged between detection states.

    The required fields describe the current aligned color/depth frame and its
    camera intrinsics. Optional fields are populated progressively by the
    plane, color, depth, fusion, and visualization stages. The same instance is
    reused across callbacks, so states should overwrite fields that belong to
    the current frame instead of retaining references to older images.

    Attributes:
        depth_image_in_meters:
            Aligned ``H x W`` depth image expressed in meters.

        color_image:
            Current BGR color image aligned with the depth image.

        fx:
            Horizontal focal length in pixels.

        fy:
            Vertical focal length in pixels.

        cx:
            Horizontal principal point in pixels.

        cy:
            Vertical principal point in pixels.

        depth_frame:
            RealSense-compatible depth adapter used by the visualization
            helpers, or ``None`` when visualization is disabled.

        go_to_idle_from_finish_state:
            Reset signal consumed by the final state.

        start_door_frame_detection:
            Start signal consumed by the idle state.

        plane_result:
            Confirmed plane distance and normal published to downstream nodes.
            It is set by plane search and kept until the next confirmation.

        color_image_for_plane_detection:
            Color image receiving the plane overlays; a private copy only when
            visualization is enabled.

        banner_for_plane_detection:
            Text shown in the banner strip of the plane overlay image.

        door_geometry:
            Physical geometry of the door and its frame. The node seeds it from
            ``~door_geometry`` and plane search refines the glass and
            center-frame widths.

        ransac_plane_distance:
            Distance of the confirmed plane, used to gate line detection and
            as the fallback door depth. The node seeds it from
            ``~plane_detector/output/ransac_plane_distance`` and plane search
            updates it.

        roi_left_color_based:
            Left-side polygon produced by the color line branch.

        roi_right_color_based:
            Right-side polygon produced by the color line branch.

        door_depth_m_color_based:
            Door-frame depth estimated from color-derived lines.

        roi_open_side_color_based:
            Color-branch ROI corresponding to the inferred opening.

        door_state_color_based:
            Color-branch label such as ``"closed"`` or ``"open_left"``.

        color_image_color_based:
            Color image receiving color-branch overlays; a private copy only
            when visualization is enabled.

        banner_color_based:
            Text shown in the banner strip of the color-branch images.

        edges:
            Full-resolution Canny edge visualization, or ``None`` when
            visualization is disabled.

        roi_left_depth_based:
            Left-side polygon produced by the depth-gradient branch.

        roi_right_depth_based:
            Right-side polygon produced by the depth-gradient branch.

        door_depth_m_depth_based:
            Door-frame depth estimated from depth-derived lines.

        roi_open_side_depth_based:
            Depth-branch ROI corresponding to the inferred opening.

        door_state_depth_based:
            Depth-branch door-state label.

        color_image_depth_based:
            Color image receiving depth-branch overlays; a private copy only
            when visualization is enabled.

        banner_depth_based:
            Text shown in the banner strip of the depth-branch image.

        sobel_vis_color:
            Color visualization of the depth Sobel response and thresholded
            edges, or ``None`` when visualization is disabled.

        door_state_label:
            Navigation-facing door label published on every frame. Fusion sets
            it from the smoothed label: an open label only together with
            current-frame geometry, ``"no_frame_detected"`` only after the
            configured number of consecutive frames, and ``"unknown"`` in
            between. Plane search sets ``"No_door_plane_detected"``. Entering
            idle resets it to ``None``, which is published as ``"unknown"``.

        mid_frame_x_px_for_passability_check:
            Image x-coordinate of the central frame edge bordering the opening.
            Fusion clears it every frame and sets it only when an open result
            ends the cycle; it is ``None`` on the fallback path and is reset
            when the machine enters idle.

        door_depth:
            Door reference depth for passability checking. Fusion clears it
            every frame and sets it when an open result ends the cycle; the
            full-image fallback sets the nominal or RANSAC plane distance.
            Entering idle resets it to ``None``.

        viz_color_stack:
            Published color-branch stacked debug image.

        viz_depth_stack:
            Published depth-branch stacked debug image.

        viz_plane_overlay:
            Reserved for a cached plane-detection visualization; currently
            unused.
    """

    # Images are kept in one shared object so state transitions do not copy
    # several megabytes of frame data on every callback.
    depth_image_in_meters: np.ndarray
    color_image: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    depth_frame: Optional[Any] = None

    go_to_idle_from_finish_state: bool = False
    start_door_frame_detection : bool = False
    # Plane search fills these before either line detector is allowed to run.
    plane_result: Optional[dict] = None
    color_image_for_plane_detection: Optional[np.ndarray] = None
    banner_for_plane_detection: DiagnosticBanner = field(default_factory=DiagnosticBanner)

    # Door geometry and confirmed plane distance. Plane search refines these at
    # runtime and every branch re-reads them each frame. They are carried here
    # rather than fetched from the parameter server per frame: producer and
    # consumers are all in this process, and each parameter read is a blocking
    # XML-RPC round trip to the master.
    door_geometry: Optional[dict] = None
    ransac_plane_distance: Optional[float] = None
    
    # color-based branch
    roi_left_color_based: Optional[np.ndarray] = None
    roi_right_color_based: Optional[np.ndarray] = None
    door_depth_m_color_based: Optional[float] = None
    roi_open_side_color_based: Optional[np.ndarray] = None
    door_state_color_based: Optional[str] = None
    color_image_color_based: Optional[np.ndarray] = None
    banner_color_based: DiagnosticBanner = field(default_factory=DiagnosticBanner)
    edges : Optional[np.ndarray] = None

    # depth-based branch
    roi_left_depth_based: Optional[np.ndarray] = None
    roi_right_depth_based: Optional[np.ndarray] = None
    door_depth_m_depth_based: Optional[float] = None
    roi_open_side_depth_based: Optional[np.ndarray] = None
    door_state_depth_based: Optional[str] = None
    color_image_depth_based: Optional[np.ndarray] = None
    banner_depth_based: DiagnosticBanner = field(default_factory=DiagnosticBanner)
    sobel_vis_color : Optional[np.ndarray] = None

    # Fusion owns these values; the individual branches should not set them.
    door_state_label: Optional[str] = None
    mid_frame_x_px_for_passability_check: Optional[float] = None
    door_depth : Optional[float] = None

    # visualization buffers (published as ROS images)
    viz_color_stack: Optional[np.ndarray] = None
    viz_depth_stack: Optional[np.ndarray] = None
    viz_plane_overlay: Optional[np.ndarray] = None


class BaseState:
    """
    Define the interface implemented by frame-driven detector states.

    Subclasses can override the entry and exit hooks and must override
    :meth:`do_action` when they need per-frame behavior.

    Attributes:
        name:
            Unique key used to register and transition to the state.
    """

    def __init__(self, name):
        """
        Initialize a named state.

        Args:
            name:
                Unique transition key used by :class:`StateMachine` to
                register and select this state.

        Notes:
            The base class stores no frame-specific data. Mutable detection
            state is shared through :class:`FrameContext`.
        """
        self.name = name

    def entry_action(self, ctx : FrameContext) -> None:
        """
        Run the state's optional entry hook.

        Args:
            ctx:
                Shared frame context available at transition time.

        Notes:
            The base implementation intentionally does nothing.
        """
        # Most states need no setup, but the hook keeps transitions symmetrical.
        pass

    def do_action(self, ctx : FrameContext)-> Optional[str]:
        """
        Run one frame of state logic.

        Args:
            ctx:
                Shared context containing the latest synchronized frame and
                accumulated detector results.

        Returns:
            Name of the next state, or ``None`` to remain in this state.

        Notes:
            The base implementation remains in the current state.
        """
        # None is the state machine's explicit "stay here for another frame" signal.
        return None

    def exit_action(self, ctx: FrameContext) -> None:
        """
        Run the state's optional exit hook.

        Args:
            ctx:
                Shared frame context at the time of transition.

        Notes:
            The base implementation intentionally does nothing.
        """
        pass
