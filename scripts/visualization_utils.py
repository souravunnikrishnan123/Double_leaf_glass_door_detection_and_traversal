#!/usr/bin/env python3
"""OpenCV helpers for detector overlays and stacked debug views."""

import rospy
import cv2
import numpy as np


def _noop(*args, **kwargs):
    """
    Accept and ignore an arbitrary OpenCV drawing call.

    Args:
        args:
            Positional arguments supplied to the patched OpenCV function.

        kwargs:
            Keyword arguments supplied to the patched OpenCV function.

    Returns:
        Always ``None``.
    """
    # No operation for disabled visualization; return minimal defaults
    return None

def setup_visualization_mode(enable: bool):
    """
    Enable normal drawing or globally replace selected OpenCV calls with no-ops.

    Args:
        enable:
            When false, patch common drawing and window functions in the
            imported ``cv2`` module.

    Notes:
        The patch is process-wide and is not reversed by calling this function
        later with ``True``. Image-producing operations such as color mapping
        remain available.
    """
    if enable:
        return
    # Patch at the OpenCV boundary so the detection code does not need a flag
    # around every diagnostic drawing call.
    # Patch common drawing APIs to no-op
    cv2.line = _noop
    cv2.circle = _noop
    cv2.rectangle = _noop
    cv2.putText = _noop
    cv2.polylines = _noop
    cv2.drawContours = _noop
    cv2.imshow = _noop
    cv2.waitKey = _noop
    # Functions that create images (like applyColorMap) remain unchanged


_VISUALIZATION_ENABLED = None


def _visualization_enabled():
    """
    Report whether diagnostic display is enabled, caching the answer.

    Returns:
        ``True`` when ``~enable_visualization`` is set or unreadable, matching
        the previous permissive default.

    Notes:
        The flag is fixed for the lifetime of a node, so it is fetched once
        instead of on every draw call.
    """
    global _VISUALIZATION_ENABLED
    if _VISUALIZATION_ENABLED is None:
        try:
            _VISUALIZATION_ENABLED = bool(rospy.get_param("~enable_visualization", True))
        except Exception:
            _VISUALIZATION_ENABLED = True
    return _VISUALIZATION_ENABLED


def get_z_depth(depth_frame, x, y):
    """
    Read a valid metric depth at a pixel from a RealSense-like frame.

    Args:
        depth_frame:
            Native RealSense frame or :class:`ros_frame_adapter.DepthFrameAdapter`.

        x:
            Horizontal pixel coordinate.

        y:
            Vertical pixel coordinate.

    Returns:
        Positive finite depth in meters, or ``None`` when the pixel is outside
        the frame or has invalid depth.

    Notes:
        Older bindings that lack direct width/height methods are supported
        through ``frame.profile.as_video_stream_profile()``.
    """
    # Return Z in meters directly; no deprojection needed for Z
    xi, yi = int(x), int(y)
    try:
        w = depth_frame.get_width()
        h = depth_frame.get_height()
    except Exception:
        # Fallback for older bindings
        prof = depth_frame.profile.as_video_stream_profile()
        w, h = prof.width(), prof.height()

    if not (0 <= xi < w and 0 <= yi < h):
        return None
    z = float(depth_frame.get_distance(xi, yi))
    return z if np.isfinite(z) and z > 0 else None

    

def build_stacked_visualization(color_image, MIN_DEPTH, MAX_DEPTH, edges, depth_frame):
    """
    Build a side-by-side color, depth, and edge visualization.

    Args:
        color_image:
            BGR image that defines the output panel dimensions.

        MIN_DEPTH:
            Minimum displayed depth in meters.

        MAX_DEPTH:
            Maximum displayed depth in meters.

        edges:
            Optional grayscale or BGR edge visualization.

        depth_frame:
            RealSense-like frame whose data is stored in millimeters.

    Returns:
        BGR image containing the three horizontally stacked panels.
    """
    # Use the same three-panel order everywhere; this makes recorded diagnostics
    # easy to compare frame by frame.
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = depth_to_colormap(depth_image, MIN_DEPTH, MAX_DEPTH)
    target_height, target_width = color_image.shape[:2]
    if edges is None:
        edges_vis = np.zeros_like(color_image)
    else:
        if edges.ndim == 2 or (edges.ndim == 3 and edges.shape[2] == 1):
            edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        else:
            edges_vis = edges
    edges_resized = cv2.resize(edges_vis, (target_width, target_height))
    stacked = np.hstack((color_image, depth_colormap, edges_resized))
    return stacked

def show_stacked_visualization(color_image, MIN_DEPTH, MAX_DEPTH, edges, depth_frame, window_name="Color | Depth | Edges+ Lines", screen_width=1920, screen_height=1080):
    """
    Display a scaled color, depth, and edge stack in an OpenCV window.

    Args:
        color_image:
            BGR image used for the first panel.

        MIN_DEPTH:
            Minimum displayed depth in meters.

        MAX_DEPTH:
            Maximum displayed depth in meters.

        edges:
            Optional grayscale or BGR edge visualization.

        depth_frame:
            RealSense-like depth frame.

        window_name:
            OpenCV window title.

        screen_width:
            Maximum display width in pixels.

        screen_height:
            Maximum display height in pixels.

    Notes:
        Display is skipped when the ROS parameter ``~enable_visualization`` is
        false. The stack is resized without changing its aspect ratio.
    """
    # Respect global visualization toggle; no functionality changes otherwise.
    # Read once per process: this is called twice per frame and each parameter
    # lookup is a blocking XML-RPC round trip to the master. Callers already gate
    # on the same flag, so this is a backstop rather than the primary check.
    if not _visualization_enabled():
        return
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = depth_to_colormap(depth_image, MIN_DEPTH, MAX_DEPTH)

    target_height, target_width = color_image.shape[:2]
    # Fallback if edges is None
    if edges is None:
        edges_vis = np.zeros_like(color_image)
    else:
        # Normalize to 3-channel BGR for stacking
        if edges.ndim == 2 or (edges.ndim == 3 and edges.shape[2] == 1):
            edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        else:
            edges_vis = edges
    edges_resized = cv2.resize(edges_vis, (target_width, target_height))
    
    stacked = np.hstack((color_image, depth_colormap, edges_resized))

    scale_w = screen_width / stacked.shape[1]
    scale_h = screen_height / stacked.shape[0]
    # Fit inside both screen dimensions without changing image aspect ratio.
    scale_factor = min(scale_w, scale_h)

    stacked_resized = cv2.resize(stacked, None, fx=scale_factor, fy=scale_factor)
    cv2.imshow(window_name, stacked_resized)
    # cv2.setMouseCallback(window_name, click_event, param=(depth_frame, scale_factor))
    cv2.waitKey(1)




# -------------------------------
# Convert raw depth image to color for visualization
# -------------------------------
def depth_to_colormap(depth_image, MIN_DEPTH, MAX_DEPTH):
    """
    Convert a millimeter depth image into a clipped JET color map.

    Args:
        depth_image:
            ``H x W`` depth array in millimeters.

        MIN_DEPTH:
            Lower visualization limit in meters.

        MAX_DEPTH:
            Upper visualization limit in meters.

    Returns:
        ``H x W x 3`` uint8 BGR color-map image.

    Notes:
        Values outside the requested interval are clipped before conversion.
    """
    # Clip depth image to desired range
    # Raw adapter data is millimetres even though display limits are configured
    # in metres.
    depth_scaled = np.clip(depth_image, MIN_DEPTH*1000, MAX_DEPTH*1000)
    # Convert depth to 8-bit for color mapping
    depth_scaled = cv2.convertScaleAbs(depth_scaled, alpha=0.03)
    # Apply colormap (Jet: Blue → Red gradient)
    return cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)

# -------------------------------
# Mouse click event for checking depth at pixel
# -------------------------------
def click_event(event, x, y, flags, param):
    """
    Print the original-frame depth corresponding to a display click.

    Args:
        event:
            OpenCV mouse-event code.

        x:
            Horizontal coordinate in the resized display.

        y:
            Vertical coordinate in the resized display.

        flags:
            OpenCV event flags. They are accepted but not used.

        param:
            Tuple ``(depth_frame, scale_factor)`` registered with the callback.

    Notes:
        Only left-button presses are handled. Display coordinates are divided
        by ``scale_factor`` before reading depth.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_frame, scale_factor = param  # unpack parameters
        # Map coordinates back to original resolution
        orig_x = int(x / scale_factor)
        orig_y = int(y / scale_factor)

        depth = get_z_depth(depth_frame, orig_x, orig_y)
        print(f"Clicked at ({orig_x}, {orig_y}) → Depth: {depth:.2f} m")
