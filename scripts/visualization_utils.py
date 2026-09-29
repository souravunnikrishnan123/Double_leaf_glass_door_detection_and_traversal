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


# -------------------------------
# Diagnostic text banner
# -------------------------------
# Published diagnostic images carry their text in a black strip padded onto the
# frame rather than on the scene, so labels never hide the lines and ROIs they
# describe. The strip has a fixed height, so every frame on a topic has the same
# size, which recorded image sequences and videos rely on.
BANNER_POSITION = "top"  # "top" or "bottom"
BANNER_ROWS = 4
BANNER_FONT = cv2.FONT_HERSHEY_SIMPLEX
BANNER_FONT_SCALE = 0.7
BANNER_THICKNESS = 2
BANNER_MARGIN_PX = 10
BANNER_PART_GAP_PX = 30
BANNER_TEXT_COLOR = (255, 255, 255)
BANNER_NOTE_COLOR = (160, 160, 160)

(_, _BANNER_TEXT_H), _BANNER_BASELINE = cv2.getTextSize(
    "Ag", BANNER_FONT, BANNER_FONT_SCALE, BANNER_THICKNESS
)
BANNER_ROW_PX = _BANNER_TEXT_H + _BANNER_BASELINE + 8


class DiagnosticBanner:
    """
    Collect the diagnostic text belonging to one published image.

    Detection code records its readings here instead of drawing them onto the
    frame; :func:`add_banner` renders them when the image is published.

    Attributes:
        rows:
            Recorded rows, each a list of ``(text, bgr)`` parts drawn left to
            right.
    """

    def __init__(self):
        """
        Start with no rows.
        """
        self.rows = []

    def add(self, *parts, first=False):
        """
        Record one row of text.

        Args:
            parts:
                Plain strings, drawn white, or ``(text, bgr)`` tuples.

            first:
                Place the row above those already recorded. Used for headline
                results so they stay visible when the strip overflows.
        """
        row = [(p, BANNER_TEXT_COLOR) if isinstance(p, str) else p for p in parts]
        if first:
            self.rows.insert(0, row)
        else:
            self.rows.append(row)


def banner_height(rows=BANNER_ROWS):
    """
    Height in pixels of a banner strip reserving ``rows`` text lines.
    """
    return 2 * BANNER_MARGIN_PX + rows * BANNER_ROW_PX


def _wrap_banner_rows(rows, width):
    """
    Lay out banner rows, moving parts that do not fit onto extra lines.

    Args:
        rows:
            ``DiagnosticBanner.rows``.

        width:
            Strip width in pixels.

    Returns:
        List of lines, each a list of ``(text, bgr, x_offset)`` tuples.
    """
    usable = width - 2 * BANNER_MARGIN_PX
    lines = []
    for row in rows:
        line, x = [], 0
        for text, color in row:
            (w, _), _ = cv2.getTextSize(text, BANNER_FONT, BANNER_FONT_SCALE, BANNER_THICKNESS)
            if line and x + w > usable:
                lines.append(line)
                line, x = [], 0
            line.append((text, color, x))
            x += w + BANNER_PART_GAP_PX
        if line:
            lines.append(line)
    return lines


def add_banner(image, banner, rows=BANNER_ROWS, position=BANNER_POSITION):
    """
    Pad an image with a fixed-height black strip holding its diagnostic text.

    Args:
        image:
            BGR image. It is not modified.

        banner:
            :class:`DiagnosticBanner` to render, or ``None`` for an empty strip.

        rows:
            Number of text lines the strip reserves.

        position:
            ``"top"`` or ``"bottom"``.

    Returns:
        New image ``banner_height(rows)`` pixels taller than ``image``.

    Notes:
        The strip is added even when there is no text, so image size never
        changes between frames. Lines beyond ``rows`` are dropped and the last
        line reports how many were hidden.
    """
    width = image.shape[1]
    lines = _wrap_banner_rows(banner.rows if banner is not None else [], width)
    if len(lines) > rows:
        hidden = len(lines) - (rows - 1)
        lines = lines[:rows - 1] + [[(f"... {hidden} more line(s) not shown", BANNER_NOTE_COLOR, 0)]]

    strip = np.zeros((banner_height(rows), width) + image.shape[2:], dtype=image.dtype)
    for i, line in enumerate(lines):
        y = BANNER_MARGIN_PX + i * BANNER_ROW_PX + _BANNER_TEXT_H
        for text, color, x in line:
            cv2.putText(strip, text, (BANNER_MARGIN_PX + x, y), BANNER_FONT,
                        BANNER_FONT_SCALE, color, BANNER_THICKNESS, cv2.LINE_AA)

    if position == "bottom":
        return np.vstack((image, strip))
    return np.vstack((strip, image))


def build_stacked_visualization(color_image, MIN_DEPTH=None, MAX_DEPTH=None, edges=None, depth_frame=None, banner=None):
    """
    Build a side-by-side color, depth, and edge visualization.

    Args:
        color_image:
            BGR image that defines the output panel dimensions.

        MIN_DEPTH:
            Minimum displayed depth in meters, or ``None`` to fit the frame.

        MAX_DEPTH:
            Maximum displayed depth in meters, or ``None`` to fit the frame.

        edges:
            Optional grayscale or BGR edge visualization.

        depth_frame:
            RealSense-like frame whose data is stored in millimeters. Despite
            the ``None`` default it is required; the detection node only
            creates it when visualization is enabled.

        banner:
            Optional :class:`DiagnosticBanner` rendered in a strip spanning all
            three panels.

    Returns:
        BGR image containing the three horizontally stacked panels, plus the
        banner strip when ``banner`` is given.

    Notes:
        With no limits given the depth panel fits the range present in the
        frame. That range is printed on the panel, because an automatic scale
        makes a color meaningless without the numbers beside it.
    """
    # Use the same three-panel order everywhere; this makes recorded diagnostics
    # easy to compare frame by frame.
    depth_image = np.asanyarray(depth_frame.get_data())
    auto_scaled = MIN_DEPTH is None or MAX_DEPTH is None
    if auto_scaled:
        MIN_DEPTH, MAX_DEPTH = auto_depth_range_m(depth_image)
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
    if banner is not None:
        stacked = add_banner(stacked, banner)
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
def auto_depth_range_m(depth_image, lower_pct=1.0, upper_pct=99.0):
    """
    Derive display depth limits from the data actually present in a frame.

    Args:
        depth_image:
            ``H x W`` depth array in millimeters. Zero marks an invalid pixel.

        lower_pct:
            Percentile taken as the near limit.

        upper_pct:
            Percentile taken as the far limit.

    Returns:
        Tuple ``(min_m, max_m)`` in meters.

    Notes:
        Percentiles rather than the raw extremes, because a handful of
        speckle returns at the sensor's limits would otherwise stretch the
        range and flatten everything else. Frames with no valid depth fall
        back to a nominal one-metre span.
    """
    depth = np.asarray(depth_image)
    valid = depth > 0
    if not np.any(valid):
        return 0.0, 1.0
    lo_mm, hi_mm = np.percentile(depth[valid], (lower_pct, upper_pct))
    if hi_mm - lo_mm < 1.0:
        hi_mm = lo_mm + 1.0
    return float(lo_mm) / 1000.0, float(hi_mm) / 1000.0


def depth_to_colormap(depth_image, MIN_DEPTH=None, MAX_DEPTH=None):
    """
    Convert a millimeter depth image into a JET color map.

    Args:
        depth_image:
            ``H x W`` depth array in millimeters.

        MIN_DEPTH:
            Lower visualization limit in meters, or ``None`` to derive it from
            the frame.

        MAX_DEPTH:
            Upper visualization limit in meters, or ``None`` to derive it from
            the frame.

    Returns:
        ``H x W x 3`` uint8 BGR color-map image.

    Notes:
        The requested interval is mapped across the full 0-255 range, so a
        narrow window spends the whole color map on it. Pixels with no depth
        reading are painted black rather than left at the map's near end, where
        they would be indistinguishable from a surface against the lens.
    """
    depth = np.asarray(depth_image)
    valid = depth > 0
    if MIN_DEPTH is None or MAX_DEPTH is None:
        MIN_DEPTH, MAX_DEPTH = auto_depth_range_m(depth)
    lo_mm = float(MIN_DEPTH) * 1000.0
    hi_mm = float(MAX_DEPTH) * 1000.0
    if hi_mm - lo_mm < 1.0:
        hi_mm = lo_mm + 1.0
    # Linear rescale onto the full 8-bit range. The previous fixed alpha of 0.03
    # ignored the requested limits entirely: it saturated everything past 8.5 m
    # and squeezed a 1.9-2.1 m window into 6 of the 255 available levels.
    normalized = (depth.astype(np.float32) - lo_mm) * (255.0 / (hi_mm - lo_mm))
    normalized = np.clip(normalized, 0.0, 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    colored[~valid] = 0
    return colored

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
