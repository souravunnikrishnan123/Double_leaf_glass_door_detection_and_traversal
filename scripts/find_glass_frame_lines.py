import cv2
import numpy as np
from find_frame_line_pair import filter_vertical_lines_glass_contact
from roi import process_filtered_lines
from duration import get_duration_seconds


def sort_lines_by_y(line_points):
    # Sort points by y-coordinate (ascending)
    return sorted(line_points, key=lambda pt: pt[1])

def left_right_roi_and_door_depth(
    depth_image_in_meters,
    color_image,
    depth_frame,
    lines,
    depth_of_each_lines,
    door_geometry,
    keyword
):
    timer = get_duration_seconds()
    timer.start(f"{keyword}_image_based_frame_detection pairing and roi processing")

    paired_lines = filter_vertical_lines_glass_contact(
        lines, depth_of_each_lines, depth_frame, door_geometry["glass_width_cm"], door_geometry["center_frame_width_cm"]
    )
    
    # Adjust all pairs so each line's points are sorted by y. so that gradient ccan be calculated correctly
    paired_lines_sorted = [
        (sort_lines_by_y(left_line), sort_lines_by_y(right_line), left_depth, right_depth)
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


    roi_polygon_left, roi_polygon_right , mean_z_depth_along_frame_lines = process_filtered_lines(paired_lines_sorted, depth_image_in_meters, color_image, door_geometry["roi_width"], door_geometry["min_depth"], door_geometry["correction_factor"])

    
    timer.stop(f"{keyword}_image_based_frame_detection pairing and roi processing")
    return roi_polygon_left, roi_polygon_right , mean_z_depth_along_frame_lines
