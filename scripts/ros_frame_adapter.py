#!/usr/bin/env python3
import numpy as np

class _Intrinsics:
    def __init__(self, fx, fy, ppx, ppy):
        self.fx = float(fx)
        self.fy = float(fy)
        self.ppx = float(ppx)
        self.ppy = float(ppy)

class _VideoStreamProfile:
    def __init__(self, width, height, intrinsics: _Intrinsics):
        self._width = int(width)
        self._height = int(height)
        self.intrinsics = intrinsics
    def width(self):
        return self._width
    def height(self):
        return self._height

class _Profile:
    def __init__(self, width, height, intrinsics: _Intrinsics):
        self._vsp = _VideoStreamProfile(width, height, intrinsics)
    def as_video_stream_profile(self):
        return self._vsp

class DepthFrameAdapter:
    """
    Adapter to mimic pyrealsense2 depth frame API from numpy depth image and intrinsics.
    - depth_mm: HxW uint16 depth in millimeters
    - fx, fy, cx, cy: camera intrinsics (for aligned depth in color frame)
    """
    def __init__(self, depth_mm: np.ndarray, fx: float, fy: float, cx: float, cy: float):
        if depth_mm.dtype != np.uint16:
            raise ValueError("depth_mm must be uint16 (millimeters)")
        self._depth_mm = depth_mm
        self._h, self._w = depth_mm.shape
        self._intr = _Intrinsics(fx, fy, cx, cy)
        self.profile = _Profile(self._w, self._h, self._intr)

    def get_data(self):
        return self._depth_mm

    def get_distance(self, x, y):
        xi = int(x)
        yi = int(y)
        if 0 <= xi < self._w and 0 <= yi < self._h:
            mm = int(self._depth_mm[yi, xi])
            return float(mm) / 1000.0 if mm > 0 else 0.0
        return 0.0

    def get_width(self):
        return self._w

    def get_height(self):
        return self._h
