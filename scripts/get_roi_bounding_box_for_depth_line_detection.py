#!/usr/bin/env python3
import rospy
import numpy as np

def get_roi_bounding_box_from_frame_lines_for_depth_lines_detection(line1, line2, margin=20, img_shape=None):
    # line1 and line2 are lists of (x, y) tuples
    all_points = np.array(line1 + line2)
    min_x = max(int(np.min(all_points[:, 0]) - margin), 0)
    max_x = min(int(np.max(all_points[:, 0]) + margin), img_shape[1])
    min_y = max(int(np.min(all_points[:, 1]) - margin), 0)
    max_y = min(int(np.max(all_points[:, 1]) + margin), img_shape[0])
    return min_x, max_x, min_y, max_y