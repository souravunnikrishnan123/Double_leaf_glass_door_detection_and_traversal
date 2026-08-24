# Robot Sensing and Passability Data

# Nearest filtered obstacle along the requested corridor, in metres.
float64 front_clearance

# Lateral center error in the fixed yaw frame used when this corridor was defined.
float64 x_corridor_center_start_yaw_frame
# Lateral center coordinate in the current optical camera frame, in metres.
float64 x_corridor_center_cam

# 1 checks the stored door corridor; 2 checks short-range space ahead of the robot.
int32 type_of_passability_check

# Signed forward distance from the robot to the corridor midpoint, in metres.
float64 corridor_middle_point_depth
# Extra clear distance required beyond that midpoint before traversal is allowed.
float64 clearance_needed_beyond_mid_corridor

# Signed angle from the robot heading to the corridor axis, in radians.
float64 heading_error
