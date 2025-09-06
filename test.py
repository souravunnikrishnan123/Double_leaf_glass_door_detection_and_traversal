import pyrealsense2 as rs 
import numpy as np

from get_z_depth import get_z_depth

pipeline = rs.pipeline()
config = rs.config()

# Enable depth stream (z16 = 16-bit grayscale)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
# Enable color stream (bgr8 = standard OpenCV format)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

# Start streaming
pipeline.start(config)

# Align depth to color stream so depth and color pixels correspond
align = rs.align(rs.stream.color)

while True:  
  # Wait for frames
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    # --- Debug check for depth meaning ---
    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    x, y = 320, 240  # center pixel, or pick a point of interest

    z_value = depth_frame.get_distance(x, y)
    z = get_z_depth(depth_frame, x, y)
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [x, y], z_value)

    print("Z from depth frame:", z_value)
    print("Z from get_z_depth:", z)
    print("Euclidean distance:", np.sqrt(X*X + Y*Y + Z*Z))
