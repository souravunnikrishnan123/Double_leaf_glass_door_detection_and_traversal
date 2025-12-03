#!/usr/bin/env python3
import rospy
import numpy as np



def filter_vertical_lines_glass_contact(lines, depth_of_each_lines, depth_frame, glass_width_cm, center_frame_width_cm):
    """
    Filters vertical lines that are likely in contact with a glass pane.

    Keeps lines that:
    - Have no neighbor on one side (left or right), OR
    - The neighbor is farther than `glass_width_cm` (in centimeters)

    Args:
        lines: list of tuples (x_center, y_top, y_bottom, center_depth)
        depth_frame: RealSense depth frame for accurate distance lookup
        fx: focal length in pixels (from intrinsics)
        glass_width_cm: minimum distance (in cm) we expect between the frame and its neighbor
        
    Returns:
        filtered: list of lines that likely represent frame-glass boundary
    """
    if not lines:
        return []
    
    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    fx = intr.fx  # in pixels

    
    
    
    # Sort lines left to right based on x coordinate avg. as there can be a a lot of same coordinate for a line, it can cause bias. hence it is better to take mean to sort the line
    # Sort lines and keep track of the original indices
    sorted_indices = sorted(range(len(lines)), key=lambda i: sum(point[0] for point in lines[i]) / len(lines[i]))
    lines = [lines[i] for i in sorted_indices]
    depth_of_each_lines = [depth_of_each_lines[i] for i in sorted_indices]

    # After sorting lines and depths
    line_to_depth = {id(line): depth for line, depth in zip(lines, depth_of_each_lines)}

    filtered = []
    # find the mean x-coordinate for each sorted line
    # and store it in a new list.
    mean_x_coords = []
    for line in lines:
        if len(line) > 0:

            # Calculate the mean x-coordinate 
            mean_x = sum(point[0] for point in line) / len(line) #as there can be a a lot of same coordinate for a line, it can cause bias. hence it is better to take mean to sort the line
            # Append it to the new list
            mean_x_coords.append(mean_x)
        else:
            # Handle empty lines
            mean_x_coords.append(None) # Or some other placeholder

    paired_lines = []

    for i, line in enumerate(lines):

        # Convert depth to centimeters and compute pixel gaps using focal length
        depth_cm = depth_of_each_lines[i] * 100.0
        if depth_cm <= 0:
            # Skip invalid depths gracefully
            continue
        required_pixel_gap = (glass_width_cm / depth_cm) * fx
        frame_pixel_gap = (center_frame_width_cm / depth_cm) * fx
        #print(fx)
        #print(frame_pixel_gap)

        
        left_exists = i > 0
        right_exists = i < len(lines) - 1

        x0 = mean_x_coords[i]
        
        # --- Case 1: Both left and right neighbors exist ---
        if left_exists and right_exists:
            x_left = mean_x_coords[i - 1]
            x_right = mean_x_coords[i + 1]
            dist_left = abs(x0 - x_left)
            dist_right = abs(x0 - x_right)

            # One must be wide, one must be narrow
            if ((dist_left >= required_pixel_gap and dist_right <= frame_pixel_gap) or
                (dist_right >= required_pixel_gap and dist_left <= frame_pixel_gap)):
                filtered.append(line)
                


         # --- Case 2: Only one neighbor exists ---
        elif left_exists:
            dist_left = abs(x0 - mean_x_coords[i - 1])
            if dist_left <= frame_pixel_gap:
                filtered.append(line)
                


        elif right_exists:
            dist_right = abs(x0 - mean_x_coords[i + 1])
            if dist_right <= frame_pixel_gap:
                filtered.append(line)
                

        
    paired_lines = get_paired_lines(filtered, frame_pixel_gap, line_to_depth)

    #print("Filtered lines:", filtered)
    #print("Paired lines:", paired_lines)

    return paired_lines


def get_paired_lines(filtered, frame_pixel_gap, line_to_depth):
    """
    Given a sorted list of filtered lines, return a list of tuples (line, neighbor_line)
    where each pair is within frame_pixel_gap.
    """
    paired_lines = []

    filtered = sorted(filtered, key=lambda line: sum(point[0] for point in line) / len(line))

    mean_x_coords = [np.mean([pt[0] for pt in line]) for line in filtered]

    for i, line in enumerate(filtered):
        x0 = mean_x_coords[i]
        left_exists = i > 0
        right_exists = i < len(filtered) - 1

        # Both neighbors exist
        if left_exists and right_exists:
            x_left = mean_x_coords[i - 1]
            x_right = mean_x_coords[i + 1]
            dist_left = abs(x0 - x_left)
            dist_right = abs(x0 - x_right)
            # Pair with the neighbor within frame_pixel_gap
            if dist_left <= frame_pixel_gap and dist_right > frame_pixel_gap:
                paired_lines.append(( filtered[i - 1], line, line_to_depth[id(filtered[i - 1])], line_to_depth[id(line)]))
            elif dist_right <= frame_pixel_gap and dist_left > frame_pixel_gap:
                paired_lines.append((line, filtered[i + 1], line_to_depth[id(line)], line_to_depth[id(filtered[i + 1])]))
            # If both are within gap, you can choose one or both (here, choose left)
            elif dist_left <= frame_pixel_gap and dist_right <= frame_pixel_gap:
                paired_lines.append(( filtered[i - 1], line, line_to_depth[id(filtered[i - 1])], line_to_depth[id(line)]))
        # Only left neighbor
        elif left_exists:
            x_left = mean_x_coords[i - 1]
            dist_left = abs(x0 - x_left)
            if dist_left <= frame_pixel_gap:
                paired_lines.append((filtered[i - 1], line, line_to_depth[id(filtered[i - 1])], line_to_depth[id(line)]))
        # Only right neighbor
        elif right_exists:
            x_right = mean_x_coords[i + 1]
            dist_right = abs(x0 - x_right)
            if dist_right <= frame_pixel_gap:
                paired_lines.append((line, filtered[i + 1], line_to_depth[id(line)], line_to_depth[id(filtered[i + 1])]))


    # Remove duplicate pairs (order-insensitive)
    unique_pairs = []
    seen = set()
    for l1, l2, d1, d2 in paired_lines:
        # Use tuple of sorted ids to avoid (A,B) and (B,A) duplicates
        key = tuple(sorted([id(l1), id(l2)]))
        if key not in seen:
            unique_pairs.append((l1, l2, d1, d2))
            seen.add(key)
    #print("Unique pairs found:", unique_pairs)
    return unique_pairs