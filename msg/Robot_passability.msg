# Robot Sensing and Passability Data

# Front distance information
float64 front_clearance

# Corridor center coordinates in different frames
float64 x_corridor_center_start_yaw_frame
float64 x_corridor_center_cam

# Passability check type (1: corridor, 2: local)
int32 type_of_passability_check

# Depth and clearance measurements
float64 corridor_middle_point_depth
float64 clearance_needed_beyond_mid_corridor

#heading
float64 heading_error 