# Import necessary libraries
import pyrealsense2 as rs   # RealSense SDK for Python
import numpy as np          # For array and matrix operations
import cv2                  # OpenCV for image processing
from collections import deque, Counter



from depth_based_detection import depth_based_edge_detection
from detect_glass_door_plane import detect_glass_door_plane
from door_status import check_if_passable, detect_door_state
from setup_realsense_pipeline import setup_realsense_pipeline
from visualization_utils import show_stacked_visualization
from color_image_based_frame_detection import color_image_based_frame_detection

# -------------------------------
# Constants for Depth Display
# -------------------------------
MIN_DEPTH = 0.3  # Minimum depth (in meters)
MAX_DEPTH = 6.0  # Maximum depth (in meters)

MIN_DEPTH_DEPTH_EDGE_DETECTION = 1.0  # Minimum depth for edge detection (in meters)
MAX_DEPTH_DEPTH_EDGE_DETECTION = 4.0  # Maximum depth for edge detection (in meters). Because the algo works best in this range. if the glass is open, or closed if the object is beyonod 3m, then its okay we get the depth as zero. anyway we want to find large depth gradient

DEPTH_RANGE = (1.7, 2.3)  # in meters

ROI_WIDTH = 60            # width of ROI in pixels

MAX_CLIP_DEPTH = 0.1
DOOR_MIN_DEPTH = 1.7  # e.g., 1 meter away
DOOR_MAX_DEPTH = 2.3


# Parameters
HISTORY_LENGTH = 30  # Frames to track
MAX_LINES_TO_TRACK = 5  # Keep top N stable lines
DISTANCE_THRESHOLD = 15  # Pixels for grouping similar lines
PHYSICAL_GRADIENT_THRESHOLD = 0.25  # in meters 
# History: store list of detected lines (each as a tuple: (avg_x, points))
line_history = deque(maxlen=HISTORY_LENGTH)

pipeline,config,align = setup_realsense_pipeline(bag_file="/app/realsense_camera_feed/grey_door/grey_door_always_open_night_with_flat_wall_on_both_sides.bag")



# -------------------------------
# Main loop for live streaming
# -------------------------------
try:
    while True:
        # Wait for next set of frames
        frames = pipeline.wait_for_frames()

        # Align depth to color
        aligned = align.process(frames)

        # Extract depth and color frames
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        # If either frame is not available, skip this loop
        if not depth_frame or not color_frame:
            continue

        # --- Apply RealSense filters ---
        #depth_frame = spatial.process(depth_frame)
        #depth_frame = temporal.process(depth_frame)
        #depth_frame = hole_filling.process(depth_frame)

        # Convert depth to numpy arrays for OpenCV
        # Get depth sensor from the pipeline
        depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()

        # Depth: must convert to float meters
        depth_scale = depth_sensor.get_depth_scale()
        depth_image_raw = np.asanyarray(depth_frame.get_data())
        # Convert to meters
        depth_image_in_meters = depth_image_raw.astype(float) * depth_scale
        
        # Convert colour image to numpy arrays for OpenCV
        #raw color (8-bit RGB values, already fine for OpenCV, no need of any conversion)
        color_image = np.asanyarray(color_frame.get_data())
        color_image_for_depth_line = color_image.copy()
        color_image_for_ransac = color_image.copy()
        color_image_for_bev = color_image.copy()
        

        # Get intrinsics
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy

        #check if there is a glass door plane in front of the camera
        # if yes, then proceed with line detection and frame detection
        result , detected_plane, found_vertical_planes = detect_glass_door_plane(color_image_for_ransac, depth_image_in_meters, fx, fy, cx, cy)
        #cv2.imshow("ransac", color_image_for_ransac)
        #print(result)

        edges = None
        sobel_vis_color = None

        

        if result["is_door_candidate"]:
            distance = result["distance_m"]
            #print(distance)

            # Only proceed if around 2 m (add ± tolerance)
            if abs(distance - 2.0) < 0.3:

                roi_polygon_left_based_on_depth_image, roi_polygon_right_based_on_depth_image, mean_z_depth_to_frame_based_on_depth_image, sobel_vis_color = depth_based_edge_detection(depth_frame, depth_image_in_meters, color_image_for_depth_line, MIN_DEPTH_DEPTH_EDGE_DETECTION, MAX_DEPTH_DEPTH_EDGE_DETECTION, DEPTH_RANGE, PHYSICAL_GRADIENT_THRESHOLD)

                roi_polygon_left_based_on_color_image, roi_polygon_right_based_on_color_image, mean_z_depth_to_frame_based_on_color_image, edges = color_image_based_frame_detection(depth_image_in_meters, color_image, depth_frame, DEPTH_RANGE)

                door_state_based_on_color_image = detect_door_state(depth_image_in_meters,fx, fy, cx, cy, color_image, roi_polygon_left_based_on_color_image, roi_polygon_right_based_on_color_image,
                roi_width=240, margin=10, threshold=0.05, z_door_depth = mean_z_depth_to_frame_based_on_color_image, plotname = "door_state_based_on_color_image")

                door_state_based_on_depth_image = detect_door_state(depth_image_in_meters,fx, fy, cx, cy, color_image_for_depth_line, roi_polygon_left_based_on_depth_image, roi_polygon_right_based_on_depth_image,
                roi_width=240, margin=10, threshold=0.05, z_door_depth = mean_z_depth_to_frame_based_on_depth_image, plotname = "door_state_based_on_depth_image")


                #create BEV of entire view
                full_image_roi = np.array([[0,0],[color_image.shape[1]-1,0],[color_image.shape[1]-1,color_image.shape[0]-1],[0,color_image.shape[0]-1]])
                ratio = check_if_passable(depth_image_in_meters, fx, fy, cx, cy, color_image_for_bev, full_image_roi, door_depth = mean_z_depth_to_frame_based_on_color_image, plotname = "full_view_bev")


        else:
            cv2.putText(color_image, "No door-like plane detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
            
        # Stack visualizations horizontally:
        # Show the result in one window using the new utility function
        show_stacked_visualization(
            color_image, MIN_DEPTH, MAX_DEPTH, edges, depth_frame, "Color | Depth | Edges+ Lines"
        )

        show_stacked_visualization(
                color_image_for_depth_line, MIN_DEPTH_DEPTH_EDGE_DETECTION, MAX_DEPTH_DEPTH_EDGE_DETECTION, sobel_vis_color, depth_frame, "depth lines in color image | Depth for depth lines| Sobel + Depth Edges"
            )
       

# ------------------------------------
# Stop pipeline and clean up on exit
# ------------------------------------
finally:
    pipeline.stop()
    cv2.destroyAllWindows()