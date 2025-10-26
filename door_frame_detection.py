# Import necessary libraries
import pyrealsense2 as rs   # RealSense SDK for Python
import numpy as np          # For array and matrix operations
import cv2                  # OpenCV for image processing
import math
from collections import deque, Counter



from depth_based_detection import depth_based_edge_detection, depth_based_edge_detection_within_rgb_based_frame_lines_roi
from detect_glass_door_plane import detect_glass_door_plane
from roi import process_filtered_lines
from door_status import detect_door_state
from get_z_depth import get_z_depth
from get_roi_bounding_box_for_depth_line_detection import get_roi_bounding_box_from_frame_lines_for_depth_lines_detection
from post_processing_of_detected_vertical_lines import extrapolate_along_line_segment,extract_smooth_line_segment_with_moving_avg, get_median_depth_along_detected_line
from setup_realsense_pipeline import setup_realsense_pipeline
from setup_segmentation_model import setup_segmentation_model
from temporal_smoothing_of_frame_line_candidates import update_line_history
from find_frame_line_pair import filter_vertical_lines_glass_contact
from rgb_based_line_detected import get_rgb_based_lines_using_canny_and_hough_lines
from visualization_utils import show_stacked_visualization


# -------------------------------
# Constants for Depth Display
# -------------------------------
MIN_DEPTH = 0.3  # Minimum depth (in meters)
MAX_DEPTH = 6.0  # Maximum depth (in meters)

MIN_DEPTH_DEPTH_EDGE_DETECTION = 1.8  # Minimum depth for edge detection (in meters)
MAX_DEPTH_DEPTH_EDGE_DETECTION = 2.5  # Maximum depth for edge detection (in meters). Because the algo works best in this range. if the glass is open, or closed if the object is beyonod 3m, then its okay we get the depth as zero. anyway we want to find large depth gradient

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

pipeline,config,align = setup_realsense_pipeline(bag_file="/app/realsense_camera_feed/grey_door_always_open_night_from_IAS_lab_side.bag")

mp_drawing , segmentation = setup_segmentation_model()




def sort_line_by_y(line_points):
    # Sort points by y-coordinate (ascending)
    return sorted(line_points, key=lambda pt: pt[1])

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
        

        # Get intrinsics
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy

        #check if there is a glass door plane in front of the camera
        # if yes, then proceed with line detection and frame detection
        result , detected_plane, found_vertical_planes = detect_glass_door_plane(color_image_for_ransac, depth_image_in_meters, fx, fy, cx, cy,segmentation)
        cv2.imshow("ransac", color_image_for_ransac)
        #print(result)
        edges = np.zeros_like(color_image[:,:,0])  # Empty edges

        """
        # Highlight detected plane in color image (always, if valid)
        stats = result.get("stats")
        if stats and stats.get("valid"):
            uv_inliers = stats.get("uv_inliers")
            if uv_inliers is not None:
                for u, v in uv_inliers:
                    u, v = int(u), int(v)
                    if 0 <= v < color_image.shape[0] and 0 <= u < color_image.shape[1]:
                        color_image[v, u] = [255, 0, 255]  # Magenta
        """

        
        

        if result["is_door_candidate"]:
            distance = result["distance_m"]
            #print(distance)

            # Only proceed if around 2 m (add ± tolerance)
            if abs(distance - 2.0) < 0.3:

                # Detect vertical lines in color image using Canny + Hough
                lines, edges = get_rgb_based_lines_using_canny_and_hough_lines(color_image)
                
                
                vertical_lines = []
                depth_of_each_lines = []
                depth_based_edge_detection(depth_frame, depth_image_in_meters, color_image_for_depth_line, MIN_DEPTH_DEPTH_EDGE_DETECTION, MAX_DEPTH_DEPTH_EDGE_DETECTION, DEPTH_RANGE,detected_plane, found_vertical_planes, PHYSICAL_GRADIENT_THRESHOLD, None, None, depth_image_raw)
                if lines is not None:
                    for line in lines:
                        filtered_segment = []
                        x1, y1, x2, y2 = line[0]
                        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                        #cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # All lines: blue
                        if 80 < abs(angle) < 100:  # near-vertical
                            cv2.line(color_image, (x1, y1), (x2, y2), (0, 165, 255), 2)  # All vertical lines: orange
                            image_height = depth_image_in_meters.shape[0]
                            y_top = 0
                            y_bottom = image_height - 1

                            #center_depth = get_median_depth_window(depth_frame, x_center, y_center, window=10)
                            #if center_depth is None:
                                #center_depth = get_median_depth_along_line(depth_frame, x_center, y1, y2)

                        # Extract smooth portion along detected Hough line
                            pixel_length = math.hypot(x2 - x1, y2 - y1)
                            num_samples = int(pixel_length)
                            """
                            filtered_segment,center_depth = extract_smooth_line_segment_with_moving_avg(
                                depth_frame, x1, y1, x2, y2,
                                gradient_threshold=0.1, window=10, num_samples=num_samples
                            ) 
                            """

                            center_depth = get_median_depth_along_detected_line(depth_frame, x1, y1, x2, y2, num_samples=num_samples, min_num_of_valid_depths=10)
                            # Compute median depth and center
                            
                            if center_depth is None:
                                continue  # Skip line if no valid depth
                            if not (DEPTH_RANGE[0] <= center_depth <= DEPTH_RANGE[1]):
                                continue  # Still skip if out of expected depth range
                            
                            filtered_segment.append((x1, y1))
                            filtered_segment.append((x2, y2))
                           
                            # Draw vertical_lines in cyan
                            
                            cv2.line(color_image, (x1, y1), (x2, y2), (255, 255, 0), 2)  # cyan

                            
                            # Use first and last points of filtered segment
                            start_fwd = filtered_segment[-1] # take only x and y coordinate. donot take depth
                            start_back = filtered_segment[0]

                            # Compute direction vector of the line (normalized)
                            dx = x2 - x1
                            dy = y2 - y1
                            norm = np.hypot(dx, dy)
                            dx /= norm
                            dy /= norm
                            
                            # Extrapolate forward
                            
                            extrapolated_forward = extrapolate_along_line_segment(
                                depth_frame, start_fwd, (dx, dy), center_depth, gradient_threshold=0.1, window=5
                            )

                            # Extrapolate backward
                            extrapolated_backward = extrapolate_along_line_segment(
                                depth_frame, start_back, (-dx, -dy), center_depth, gradient_threshold=0.1, window=5
                            )

                            # Combine all
                            full_line_segment = extrapolated_backward[::-1] + filtered_segment + extrapolated_forward
                            
                            

                            if len(full_line_segment) >= 2:
                                
                                cv2.line(color_image, full_line_segment[0], full_line_segment[-1], (0, 255, 0), 2)  # Green
                                vertical_lines.append(full_line_segment)
                                depth_of_each_lines.append(center_depth)
                            
                            """
                            if len(filtered_segment) >= 2:
                                pt1 = tuple(map(int, filtered_segment[0][:2]))
                                pt2 = tuple(map(int, filtered_segment[-1][:2]))
                                cv2.line(color_image, pt1, pt2, (255, 255, 0), 2)  # Cyan
                            
                            if len(full_line_segment) >= 2:
                                pt1 = tuple(map(int, full_line_segment[0][:2]))
                                pt2 = tuple(map(int, full_line_segment[-1][:2]))
                                cv2.line(color_image, pt1, pt2, (255, 0, 255), 2)  # Magenta
                            

                            # Append clipped vertical line
                            MIN_LINE_LENGTH = 50  # Minimum number of points required. because otherwise a small line segment
                            #on the frame ( which is clipped by the previous logic) will be still considered as valid line and cause issue with detection

                            if len(full_line_segment) >= MIN_LINE_LENGTH:
                                vertical_lines.append(full_line_segment)
                                depth_of_each_lines.append(center_depth)
                                # Draw vertical_lines in green
                                pt1 = tuple(map(int, full_line_segment[0][:2]))
                                pt2 = tuple(map(int, full_line_segment[-1][:2]))
                                cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)  # Green
                                #print(f"filtered_segment depth {center_depth:.2f}m")
                            """
                    
                    
                    # there were some problem with stable_lines calculation. it was not working properly. so commenting it out for now
                    # Instead, we will just use vertical_lines directly for pairing
                    #update_line_history(vertical_lines, line_history, DISTANCE_THRESHOLD, MAX_LINES_TO_TRACK, color_image)
        

                    # Pass only line points to filter function
                    #stable_line_points = [line_pts for avg_x, line_pts, confidence in stable_lines]
                    
                    # Visualize stable_line_points (lines passed to filter_vertical_lines_glass_contact)
                    #for line_pts in stable_line_points:
                        #if len(line_pts) >= 2:
                            #pt1 = tuple(map(int, line_pts[0][:2]))
                            #pt2 = tuple(map(int, line_pts[-1][:2]))
                            #cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)  # Green for stable lines

                    #print(len(vertical_lines))
                    paired_lines = filter_vertical_lines_glass_contact(
                        vertical_lines, depth_of_each_lines, depth_frame, glass_width_cm=40, center_frame_width_cm=30
                    )
                    
                    # Adjust all pairs so each line's points are sorted by y. so that gradient ccan be calculated correctly
                    paired_lines_sorted = [
                        (sort_line_by_y(left_line), sort_line_by_y(right_line), left_depth, right_depth)
                        for left_line, right_line, left_depth, right_depth in paired_lines]
                    

                    # Visualize paired lines (glass frame candidates)
                    for left_line, right_line, left_depth, right_depth in paired_lines_sorted:
                        # Draw left line in red
                        if len(left_line) >= 2:
                            pt1 = tuple(map(int, left_line[0][:2]))
                            pt2 = tuple(map(int, left_line[-1][:2]))
                            cv2.line(color_image, pt1, pt2, (0, 0, 255), 2)
                        # Draw right line in cyan
                        if len(right_line) >= 2:
                            pt1 = tuple(map(int, right_line[0][:2]))
                            pt2 = tuple(map(int, right_line[-1][:2]))
                            cv2.line(color_image, pt1, pt2, (0, 255, 255), 2)



                    # Example: Extract full coordinates of each stable line
                    #for avg_x, line_pts, confidence in stable_lines:
                        # Print the number of points instead of shape
                        #print(f"Stable Line X={avg_x}, Confidence={confidence:.2f}, NumPoints={len(line_pts)}")


                    roi_polygon_left_list, roi_polygon_right_list, filtered_pairs , mean_z_depth_along_frame_lines_list = process_filtered_lines(paired_lines_sorted, depth_frame, color_image, detected_plane)

                    #Filter for the leftmost pair (lowest average x of left line). this is temporary logic to avoid getting the lines near to the tv in the PC lab being detected as door frame lines. need to improve it
                    
                    if filtered_pairs: 
                        if len(filtered_pairs) > 1:# more than one pair detected as glass frame. 
                            #in this case we need to filter out the correct frame line at the center. if we are getting more than one glass -frame candidate means, mostly it is due to the glass area in the  inward opening door 
                            #so in this case chances are high that ransac detected the whole door plane. hence we can use the width of detected ransac plane to filter out the correct frame line pair
                            #correct frame line pair will be close to the center of detected ransac door plane
                            # 1. Compute the center x of the detected plane (average of its 4 corners)
                            plane_center_x = np.mean([pt[0] for pt in detected_plane]) if detected_plane is not None else color_image.shape[1] // 2
                        
                            # 2. Find the pair whose center is closest to the plane center
                            min_dist = float('inf')
                            best_pair = None
                            for left_line, right_line, left_depth, right_depth in filtered_pairs:
                                # Compute mean x of left and right line
                                left_x = np.mean([pt[0] for pt in left_line])
                                right_x = np.mean([pt[0] for pt in right_line])
                                pair_center_x = (left_x + right_x) / 2
                                dist = abs(pair_center_x - plane_center_x)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_pair = (left_line, right_line, left_depth, right_depth)
                            
                            idx = filtered_pairs.index(best_pair)

                        else:# filtered pairs has only one pair.
                            best_pair = filtered_pairs[0]
                            idx = 0


                        final_left_frame_line = best_pair[0]
                        final_right_frame_line = best_pair[1]
                        roi_polygon_left = roi_polygon_left_list[idx]
                        roi_polygon_right = roi_polygon_right_list[idx]
                        mean_z_depth_along_frame_lines = mean_z_depth_along_frame_lines_list[idx]

                        door_state = detect_door_state(depth_image_in_meters,fx, fy, cx, cy, color_image, roi_polygon_left, roi_polygon_right,found_vertical_planes,
                            roi_width=240, margin=10, threshold=0.1, z_door_depth = mean_z_depth_along_frame_lines)

                        
                        


                        #print(f"Door is {door_state}")


                        # Compute ROI bounding box for depth based line detection
                        min_x, max_x, min_y, max_y = get_roi_bounding_box_from_frame_lines_for_depth_lines_detection(final_left_frame_line, final_right_frame_line, margin=40, img_shape=color_image.shape)
                        # Crop color and depth images
                        roi_color = color_image[min_y:max_y, min_x:max_x]
                        roi_depth_np = np.asanyarray(depth_frame.get_data())[min_y:max_y, min_x:max_x]

            
                        # need to adapt depth_based_edge_detection to accept numpy arrays for depth
                        #depth_based_edge_detection_within_rgb_based_frame_lines_roi(roi_depth_np, roi_color, MIN_DEPTH_DEPTH_EDGE_DETECTION, MAX_DEPTH_DEPTH_EDGE_DETECTION, DEPTH_RANGE)
                        #depth_based_edge_detection(depth_frame, color_image_for_depth_line, MIN_DEPTH_DEPTH_EDGE_DETECTION, MAX_DEPTH_DEPTH_EDGE_DETECTION, DEPTH_RANGE, PHYSICAL_GRADIENT_THRESHOLD, final_left_frame_line, final_right_frame_line, depth_image_raw)
        else:
            cv2.putText(color_image, "No door-like plane detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    
            
        # Stack visualizations horizontally:
        # Show the result in one window using the new utility function
        esc_pressed = show_stacked_visualization(
            color_image, depth_image_raw, MIN_DEPTH, MAX_DEPTH, edges, depth_frame, "Color | Depth | Edges+ Lines"
        )
        if esc_pressed:
            break

# ------------------------------------
# Stop pipeline and clean up on exit
# ------------------------------------
finally:
    segmentation.close()
    pipeline.stop()
    cv2.destroyAllWindows()