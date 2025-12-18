#!/usr/bin/env python3
import os
import rospkg
import rospy
from typing import Optional
from Frame_data import BaseState, FrameContext
from check_if_passable import BirdsEyePassabilityPipeline
from Frame_data import FrameContext
import numpy as np
import cv2
from collections import deque, defaultdict

class TemporalSmoother:
    def __init__(self, window_size=10, min_consistent=3, hysteresis=True, stable_hold=2):
        self.window_size = window_size
        self.min_consistent = min_consistent
        self.hysteresis = hysteresis
        self.stable_hold = stable_hold
        self.buffer = deque(maxlen=window_size)
        self.current_stable = None
        self.stable_count = 0

    def update(self, label: str) -> str:
        # Push new value
        self.buffer.append(label)

        # Count frequencies over window
        freq = defaultdict(int)
        for v in self.buffer:
            freq[v] += 1

        # Majority vote
        majority_label = max(freq.items(), key=lambda x: x[1])[0]
        majority_count = freq[majority_label]

        # Require at least min_consistent within window
        candidate = majority_label if majority_count >= self.min_consistent else (self.current_stable or label)

        if not self.hysteresis:
            self.current_stable = candidate
            self.stable_count = majority_count
            return self.current_stable

        # Hysteresis: only switch after "stable_hold" confirmations for new candidate
        if self.current_stable is None:
            self.current_stable = candidate
            self.stable_count = 1
            return self.current_stable

        if candidate == self.current_stable:
            # reinforce stability
            self.stable_count = min(self.window_size, self.stable_count + 1)
            return self.current_stable
        else:
            # moving toward a new state: require repeated confirmations
            if majority_count >= self.stable_hold:
                self.current_stable = candidate
                self.stable_count = majority_count
            else:
                # keep previous until enough evidence
                self.stable_count = max(0, self.stable_count - 1)
            return self.current_stable

class combine_door_state(BaseState):
    def __init__(self):
        super().__init__("combine_door_state")
        self.check_passability = BirdsEyePassabilityPipeline()
        self.height = 0
        self.width = 0
        # Temporal smoothing parameters can be tuned here
        self.smoother = TemporalSmoother(window_size=8, min_consistent=3, hysteresis=True, stable_hold=2)

    def rois_match(self, roi1, roi2, iou_threshold):
        mask1 = np.zeros((self.height, self.width), dtype=np.uint8)
        mask2 = np.zeros((self.height, self.width), dtype=np.uint8)

        if roi1 is None or roi2 is None:
            return False
        
        cv2.fillPoly(mask1, [roi1.astype(np.int32)], 255)
        cv2.fillPoly(mask2, [roi2.astype(np.int32)], 255)

        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return False

        iou= intersection / union
        return iou >= iou_threshold
    


    def resolve_door_status(self, color_pipline_result, depth_pipline_result, roi_open_side_color_based, roi_open_side_depth_based, iou_threshold):

        # Default result container
        result = dict(final_door_status="unknown", bev=None, ask_human=False, reason="")

        # -------------------------------------------------------------
        # DEPTH = NO FRAME
        # -------------------------------------------------------------
        if depth_pipline_result == "no_frame_detected":
            if color_pipline_result == "open_left" or color_pipline_result == "open_right":
                result["final_door_status"] = color_pipline_result # "open_left" or "open_right"
                result["bev"] = "color"
                result["reason"] = f"Color {color_pipline_result} + depth no-frame → use color BEV"
            elif color_pipline_result == "closed":
                result["final_door_status"] = "closed"
                result["reason"] = "Color closed + depth no-frame → closed"
            elif color_pipline_result == "no_frame_detected":
                result["final_door_status"] = "no_frame_detected"
                result["bev"] = "full"
                result["ask_human"] = True
                result["reason"] = "No frame in both → no_frame_detected, ask human/full BEV"
            elif color_pipline_result == "unknown":
                result["final_door_status"] = "unknown"
                result["bev"] = "full"
                result["ask_human"] = True
                result["reason"] = "Depth no-frame + color unknown → unknown"

        # -------------------------------------------------------------
        # DEPTH = UNKNOWN
        # -------------------------------------------------------------
        elif depth_pipline_result == "unknown":
            if color_pipline_result == "open_left" or color_pipline_result == "open_right":
                result["final_door_status"] = color_pipline_result # "open_left" or "open_right"
                result["bev"] = "color"
                result["reason"] = f"Color {color_pipline_result} + depth unknown → use color BEV"
            elif color_pipline_result == "closed":
                result["final_door_status"] = "closed"
                result["reason"] = "Color closed + depth unknown → closed"
            elif color_pipline_result == "no_frame_detected":
                result["final_door_status"] = "unknown"
                result["bev"] = "full"
                result["ask_human"] = True
                result["reason"] = "Color no-frame + depth unknown → ambiguous"
            elif color_pipline_result == "unknown":
                result["final_door_status"] = "unknown"
                result["bev"] = "full"
                result["ask_human"] = True
                result["reason"] = "Both unknown → unknown"

        # -------------------------------------------------------------
        # DEPTH = CLOSED
        # -------------------------------------------------------------
        elif depth_pipline_result == "closed":
            if color_pipline_result == "open_left" or color_pipline_result == "open_right":
                result["final_door_status"] = "unknown"
                result["bev"] = "full"
                result["ask_human"] = True
                result["reason"] = "Conflict: color open but depth closed"
            elif color_pipline_result == "closed":
                result["final_door_status"] = "closed"
                result["reason"] = "Both closed → closed"
            elif color_pipline_result == "no_frame_detected" or color_pipline_result == "unknown":
                result["final_door_status"] = "closed"
                result["reason"] = "Depth closed overrides color no-frame/unknown"

        # -------------------------------------------------------------
        # DEPTH = OPEN LEFT
        # -------------------------------------------------------------
        elif depth_pipline_result == "open_left":
            if color_pipline_result == "open_left":
                if self.rois_match(roi_open_side_color_based, roi_open_side_depth_based, iou_threshold):
                    result["final_door_status"] = "open_left"
                    result["bev"] = "color"
                    result["reason"] = "Both open_left + ROI match → use color BEV"
                else:
                    result["final_door_status"] = "unknown"
                    result["bev"] = "full"
                    result["ask_human"] = True
                    result["reason"] = "Both open_left but ROI mismatch → full BEV"
            elif color_pipline_result == "open_right" or color_pipline_result == "closed":
                result["final_door_status"] = "unknown"
                result["bev"] = "full"
                result["ask_human"] = True
                result["reason"] = "Conflict with depth open_left"
            elif color_pipline_result == "no_frame_detected" or color_pipline_result == "unknown":
                result["final_door_status"] = "open_left"
                result["bev"] = "depth"
                result["reason"] = "Depth open_left + weak color → use depth BEV"
        # -------------------------------------------------------------
        # DEPTH = OPEN RIGHT
        # -------------------------------------------------------------
        elif depth_pipline_result == "open_right":
            if color_pipline_result == "open_right":
                if self.rois_match(roi_open_side_color_based, roi_open_side_depth_based, iou_threshold):
                    result["final_door_status"] = "open_right"
                    result["bev"] = "color"
                    result["reason"] = "Both open_right + ROI match → use color BEV"
                else:
                    result["final_door_status"] = "unknown"
                    result["bev"] = "full"
                    result["ask_human"] = True
                    result["reason"] = "Both open_right but ROI mismatch"
            elif color_pipline_result == "open_left" or color_pipline_result == "closed":
                result["final_door_status"] = "unknown"
                result["bev"] = "full"
                result["ask_human"] = True
                result["reason"] = "Conflict with depth open_right"
            elif color_pipline_result == "no_frame_detected" or color_pipline_result == "unknown":
                result["final_door_status"] = "open_right"
                result["bev"] = "depth"
                result["reason"] = "Depth open_right + weak color → use depth BEV"
        # -------------------------------------------------------------
        # FALLBACK
        # -------------------------------------------------------------
        else:
            result["final_door_status"] = "unknown"
            result["bev"] = "full"
            result["ask_human"] = True
            result["reason"] = f"Unhandled combination ({depth_pipline_result}, {color_pipline_result})"

        return result


    def do_action(self, ctx: FrameContext) -> Optional[str]:
        self.height, self.width = ctx.color_image_color_based.shape[:2]

        result = self.resolve_door_status(ctx.door_state_color_based, ctx.door_state_depth_based, ctx.roi_open_side_color_based, ctx.roi_open_side_depth_based, iou_threshold=0.5)


        # Optionally: clear branch state for next cycle
        #ctx.roi_left_color_based = ctx.roi_right_color_based = None
        #ctx.roi_left_depth_based = ctx.roi_right_depth_based = None
        #ctx.door_state_color_based = ctx.door_state_depth_based = None
        #ctx.door_depth_m_color_based = ctx.door_depth_m_depth_based = None




        if result["final_door_status"] == "open_left" or result["final_door_status"] == "open_right":
            #create bird eye view and passability view for the open side
            roi_open_side = ctx.roi_open_side_color_based if result["bev"] == "color" else ctx.roi_open_side_depth_based
            keyword = "color_based" if result["bev"] == "color" else "depth_based"
            Passability_status, passability_view, bird_eye_view = self.check_passability.run(
            ctx.depth_image_in_meters, ctx.fx, ctx.fy, ctx.cx, ctx.cy, getattr(ctx, f"color_image_{keyword}"), roi_open_side, getattr(ctx, f"door_depth_m_{keyword}"), keyword)

            # Append passability status to door state
            if Passability_status: # True or False
                result["final_door_status"] += "_passable"
            else:
                result["final_door_status"] += "_not_passable"

            # Persist visualizations in context
            setattr(ctx, f"passability_view_{keyword}", passability_view)
            setattr(ctx, f"bird_eye_view_{keyword}", bird_eye_view)
        

        # Apply temporal smoothing on final label
        smoothed_door_state = self.smoother.update(result["final_door_status"])
        ctx.door_state_label = smoothed_door_state
        
        pkg_path = rospkg.RosPack().get_path('robodog_glass_door_detection')
        log_dir = os.path.join(pkg_path, 'scripts')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'final_door_status_after_temporal_smoothing.txt')
        with open(log_path, "a") as f:
                f.write(f"smoothed_door_state: {smoothed_door_state}\n")


        return "parallel_detection_state"