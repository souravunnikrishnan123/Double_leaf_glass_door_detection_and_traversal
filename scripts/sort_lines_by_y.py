#!/usr/bin/env python3
import rospy
def sort_lines_by_y(line_points):
    # Sort points by y-coordinate (ascending)
    return sorted(line_points, key=lambda pt: pt[1])