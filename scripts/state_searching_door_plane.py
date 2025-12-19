#!/usr/bin/env python3
import rospy
from Frame_data import BaseState, FrameContext
from detect_glass_door_plane import PlaneDetector

class searching_door_plane_state(BaseState):
    def __init__(self):
        super().__init__("searching_door_plane_state")
        self.find_door_plane = PlaneDetector()

    def do_action(self, ctx: FrameContext):
        #check if there is a glass door plane in front of the camera
        # if yes, then proceed with line detection and frame detection
        result = self.find_door_plane.detect(ctx.color_image, ctx.depth_image_in_meters, ctx.fx, ctx.fy, ctx.cx, ctx.cy)
        # plane overlays are drawn into ctx.color_image; main publishes as ~viz/plane_overlay
    
        ctx.plane_result = result

        if result["plane_model"] is not None:
            distance = result["plane_metrics"]["distance_m"]
            if abs(distance - 2.0) < 0.3:
                #return "checking_door_type_state"
                return "parallel_detection_state"
            else:
                return "movement_state"
        else:
            #stay in this state and keep searching for door plane
            return None
            
