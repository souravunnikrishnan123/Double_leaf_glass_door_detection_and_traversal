#!/usr/bin/env python3
from door_type_detector import DoorTypeDetector
import rospy
from Frame_data import BaseState, FrameContext
from detect_glass_door_plane import PlaneDetector

class searching_door_plane_state(BaseState):
    def __init__(self):
        super().__init__("searching_door_plane_state")
        self.find_door_plane = PlaneDetector()
        self.door_type_detector = DoorTypeDetector()
        ns = "~plane_detector"
        # Thresholds to decide "no plane present" path
        self.max_no_candidate_frames = rospy.get_param(f"{ns}/max_no_candidate_frames", 5)
        self.reference_door_distance_m = rospy.get_param(f"{ns}/reference_door_distance_m", 2.0)
        self.distance_range_m = rospy.get_param(f"{ns}/distance_range_m", 0.3)
        self._no_candidate_count = 0

    def do_action(self, ctx: FrameContext):
        #check if there is a glass door plane in front of the camera
        # if yes, then proceed with line detection and frame detection
        result = self.find_door_plane.detect(ctx.color_image_for_plane_detection, ctx.depth_image_in_meters, ctx.fx, ctx.fy, ctx.cx, ctx.cy)
        # plane overlays are drawn into ctx.color_image_for_plane_detection; main publishes as ~viz/plane_overlay
    

        if result["plane_model"] is not None:

            glass_and_door_width = self.door_type_detector.estimate_glass_and_frame_widths(
                result["plane_model"],
                result["inlier_points"]
            )
            
            vis_img = self.door_type_detector.visualize_door_bins_and_widths(
            result["plane_model"],
            inlier_points=result["inlier_points"],
            bin_edges=glass_and_door_width["bins"],
            bin_labels=glass_and_door_width["bin_labels"],
            segments=glass_and_door_width["segments"],
            fx=ctx.fx,
            cx=ctx.cx,
            color_image=ctx.color_image_for_plane_detection,
            )

            ctx.color_image_for_plane_detection = vis_img
            print("Glass widths (m):", glass_and_door_width["glass_widths_m"])
            print("Frame widths (m):", glass_and_door_width["frame_widths_m"])


            ctx.plane_result = result
            self._no_candidate_count = 0 # reset counter on successful detection

            distance = result["plane_metrics"]["distance_m"]
            if abs(distance - self.reference_door_distance_m) < self.distance_range_m:
                #return "parallel_detection_state"
                return None
            else:
                return "movement_state"
        else:
            # Distinguish between "unconfirmed yet" vs "no candidates at all"
            had_candidates = result["had_candidates"]

            if had_candidates is True:
                # There is a potential plane, keep trying and reset miss counter
                self._no_candidate_count = 0
                return None
            else: # had_candidates is False
                # No candidates detected this frame
                self._no_candidate_count += 1
                if self._no_candidate_count >= self.max_no_candidate_frames:
                    # Transition to full-view bird-eye processing (no door plane detected path)
                    # Reset counters on transition
                    self._no_candidate_count = 0
                    # to directly check if the door is passable without plane detection as the plane is not detectable. this is useful when the both doors halves are wide open already and ransac couldnt detect frame around it as well
                    return "create_full_view_bird_eye_view_state"
                #stay in this state
                return None


            
