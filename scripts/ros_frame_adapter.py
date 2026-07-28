#!/usr/bin/env python3
"""Compatibility wrapper exposing a NumPy depth image as a RealSense frame."""

import numpy as np

class _Intrinsics:
    """
    Store the subset of RealSense pinhole intrinsics used by this package.

    Attributes:
        fx:
            Horizontal focal length in pixels.

        fy:
            Vertical focal length in pixels.

        ppx:
            Horizontal principal point in pixels.

        ppy:
            Vertical principal point in pixels.
    """

    def __init__(self, fx, fy, ppx, ppy):
        """
        Initialize a compact intrinsics object.

        Args:
            fx:
                Horizontal focal length.

            fy:
                Vertical focal length.

            ppx:
                Horizontal principal point.

            ppy:
                Vertical principal point.
        """
        self.fx = float(fx)
        self.fy = float(fy)
        self.ppx = float(ppx)
        self.ppy = float(ppy)

class _VideoStreamProfile:
    """
    Mimic the small RealSense video-profile API used by this package.

    The wrapper supplies callable width and height accessors together with a
    public intrinsics object. It is intentionally limited to the members
    consumed by the existing visualization and geometry helpers.

    Attributes:
        intrinsics:
            Compact pinhole camera intrinsics.

        _width:
            Stored stream width in pixels.

        _height:
            Stored stream height in pixels.
    """

    def __init__(self, width, height, intrinsics: _Intrinsics):
        """
        Initialize stream dimensions and intrinsics.

        Args:
            width:
                Stream width in pixels.

            height:
                Stream height in pixels.

            intrinsics:
                Pinhole intrinsics associated with the stream.
        """
        self._width = int(width)
        self._height = int(height)
        self.intrinsics = intrinsics
    def width(self):
        """
        Return the adapted stream width.

        Returns:
            Positive stream width in pixels, stored during construction.

        Notes:
            A method is used instead of a property to match
            ``pyrealsense2.video_stream_profile``.
        """
        return self._width

    def height(self):
        """
        Return the adapted stream height.

        Returns:
            Positive stream height in pixels, stored during construction.

        Notes:
            A method is used instead of a property to match
            ``pyrealsense2.video_stream_profile``.
        """
        return self._height

class _Profile:
    """
    Provide the ``depth_frame.profile`` access pattern used by RealSense.

    Native RealSense frames expose a generic profile which callers convert to
    a video stream profile. This wrapper preserves that two-step access pattern
    for code receiving a NumPy-backed :class:`DepthFrameAdapter`.

    Attributes:
        _vsp:
            Wrapped video stream profile returned by
            :meth:`as_video_stream_profile`.
    """

    def __init__(self, width, height, intrinsics: _Intrinsics):
        """
        Initialize a profile wrapper.

        Args:
            width:
                Stream width in pixels.

            height:
                Stream height in pixels.

            intrinsics:
                Pinhole intrinsics for the stream.
        """
        self._vsp = _VideoStreamProfile(width, height, intrinsics)

    def as_video_stream_profile(self):
        """
        Return the wrapped video stream profile.

        Returns:
            :class:`_VideoStreamProfile` compatible with the RealSense API.

        Notes:
            The same immutable wrapper is returned on every call.
        """
        return self._vsp

class DepthFrameAdapter:
    """
    Expose a NumPy depth image through the RealSense depth-frame interface.

    The adapter lets visualization code operate identically on ROS depth
    messages and native ``pyrealsense2`` frames.

    Attributes:
        profile:
            RealSense-like profile containing stream dimensions and intrinsics.

        _depth_mm:
            Original uint16 depth array in millimeters.

        _h:
            Cached image height in pixels.

        _w:
            Cached image width in pixels.

        _intr:
            Compact intrinsics object shared by the profile wrapper.
    """

    def __init__(self, depth_mm: np.ndarray, fx: float, fy: float, cx: float, cy: float):
        """
        Initialize an adapted depth frame.

        Args:
            depth_mm:
                ``H x W`` uint16 depth image in millimeters.

            fx:
                Horizontal focal length in pixels.

            fy:
                Vertical focal length in pixels.

            cx:
                Horizontal principal point in pixels.

            cy:
                Vertical principal point in pixels.

        Raises:
            ValueError:
                If ``depth_mm`` is not a uint16 array.
        """
        if depth_mm.dtype != np.uint16:
            raise ValueError("depth_mm must be uint16 (millimeters)")
        self._depth_mm = depth_mm
        self._h, self._w = depth_mm.shape
        self._intr = _Intrinsics(fx, fy, cx, cy)
        self.profile = _Profile(self._w, self._h, self._intr)

    def get_data(self):
        """
        Return the original depth image.

        Returns:
            ``H x W`` uint16 array in millimeters. No copy is made.

        Notes:
            Mutating the returned array also changes subsequent reads through
            this adapter.
        """
        return self._depth_mm

    def get_distance(self, x, y):
        """
        Read metric depth at one pixel.

        Args:
            x:
                Horizontal pixel coordinate.

            y:
                Vertical pixel coordinate.

        Returns:
            Positive depth in meters, or ``0.0`` for an out-of-bounds or
            zero-valued pixel.
        """
        xi = int(x)
        yi = int(y)
        if 0 <= xi < self._w and 0 <= yi < self._h:
            mm = int(self._depth_mm[yi, xi])
            return float(mm) / 1000.0 if mm > 0 else 0.0
        return 0.0

    def get_width(self):
        """
        Return the depth image width.

        Returns:
            Width of the adapted NumPy array in pixels.

        Notes:
            This mirrors the corresponding RealSense depth-frame convenience
            method.
        """
        return self._w

    def get_height(self):
        """
        Return the depth image height.

        Returns:
            Height of the adapted NumPy array in pixels.

        Notes:
            This mirrors the corresponding RealSense depth-frame convenience
            method.
        """
        return self._h
