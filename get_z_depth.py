import profile
import pyrealsense2 as rs
import numpy as np

def get_z_depth(depth_frame, x, y):
    # Your get_z_depth function remains the same

    if not (0 <= x < depth_frame.width and 0 <= y < depth_frame.height):
        return None
    try:
        depth = depth_frame.get_distance(int(x), int(y))
        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        _, _, z = rs.rs2_deproject_pixel_to_point(intr, [int(x), int(y)], depth)
        if np.isfinite(z):
            return z
        else:
            return None
    except Exception as e:
        return None