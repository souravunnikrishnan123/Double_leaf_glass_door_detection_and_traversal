#!/usr/bin/env python3
import rospy
from Frame_data import BaseState, FrameContext
from detect_glass_door_plane import detect_glass_door_plane

class searching_door_plane_state(BaseState):
    def __init__(self):
        super().__init__("searching_door_plane_state")

    def do_action(self, ctx: FrameContext):
        #check if there is a glass door plane in front of the camera
        # if yes, then proceed with line detection and frame detection
        result , detected_plane, found_vertical_planes = detect_glass_door_plane(ctx.color_image, ctx.depth_image_in_meters, ctx.fx, ctx.fy, ctx.cx, ctx.cy)
        # plane overlays are drawn into ctx.color_image; main publishes as ~viz/plane_overlay
    
        ctx.plane_result = result
        ctx.detected_plane = detected_plane
        ctx.found_vertical_planes = found_vertical_planes

        if result["is_door_candidate"]:
            distance = result["distance_m"]
            if abs(distance - 2.0) < 0.3:
                return "parallel_detection_state"
            else:
                return "start_movement_to_reach_within_distance_range_to_door_plane"
        else:
            return "no_door_plane_detected_state"
            
