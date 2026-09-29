# Outcome of one door-detection cycle, sent as a single latched message.
# The detector publishes it once when a cycle finishes, and once more as a
# reset (door_state "unknown", nothing valid) when it returns to idle.

# Stamp of the camera frame that produced the result.
Header header

# open_left, open_right, no_frame_detected, No_door_plane_detected, or unknown for a reset.
string door_state

# Door reference depth in metres; NaN when not available.
float32 door_depth

# Image x-coordinate of the central frame edge bordering the opening; -1 when not available.
int32 mid_frame_x_px

# Distance from the camera to the door plane, in metres. For No_door_plane_detected
# this is the configured reference door distance; NaN in the reset.
float32 plane_distance
# Unit normal of the door plane in the camera optical frame, pointing away from the
# camera. (0, 0, 1) for No_door_plane_detected; zero in the reset.
geometry_msgs/Vector3 plane_normal
