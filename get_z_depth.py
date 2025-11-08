import profile
import pyrealsense2 as rs
import numpy as np

def get_z_depth(depth_frame, x, y):
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