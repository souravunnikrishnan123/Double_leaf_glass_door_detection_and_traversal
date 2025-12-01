import cv2
import numpy as np
from find_frame_line_pair import filter_vertical_lines_glass_contact
from roi import process_filtered_lines
from sort_lines_by_y import sort_lines_by_y
from duration import get_duration_seconds


def left_right_roi_and_door_depth(
    depth_image_in_meters,
    color_image,
    depth_frame,
    lines,
    depth_of_each_lines,
    keyword
):
    timer = get_duration_seconds()
    timer.start(f"{keyword}_image_based_frame_detection pairing and roi processing")

    paired_lines = filter_vertical_lines_glass_contact(
        lines, depth_of_each_lines, depth_frame, glass_width_cm=40, center_frame_width_cm=30
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


    roi_polygon_left_list, roi_polygon_right_list, filtered_pairs , mean_z_depth_along_frame_lines_list = process_filtered_lines(paired_lines_sorted, depth_image_in_meters, color_image)

    #Filter for the leftmost pair (lowest average x of left line). this is temporary logic to avoid getting the lines near to the tv in the PC lab being detected as door frame lines. need to improve it
    
    if filtered_pairs:
        if len(filtered_pairs) > 1:# more than one pair detected as glass frame. 
            #in this case we need to filter out the correct frame line at the center. if we are getting more than one glass -frame candidate means, mostly it is due to the glass area in the  inward opening door 
            #so in this case chances are high that ransac detected the whole door plane. hence we can use the width of detected ransac plane to filter out the correct frame line pair
            #correct frame line pair will be close to the center of detected ransac door plane
            # 1. Compute the center x of the detected plane (average of its 4 corners)
            #plane_center_x = np.mean([pt[0] for pt in detected_plane]) if detected_plane is not None else color_image.shape[1] // 2
            plane_center_x = color_image.shape[1] // 2 #since wer are stopping about 2m far from glass door plane. the center of entire view would be almost same as the center of glass door plane
            #this is to avoid dependancy to glass detection algorithm
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
        
        timer.stop(f"{keyword}_image_based_frame_detection pairing and roi processing")
        return roi_polygon_left, roi_polygon_right, mean_z_depth_along_frame_lines
    else:
        return None, None, 0