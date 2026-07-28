#!/usr/bin/env python3
"""Fuse color/depth door states and smooth the selected result over time."""

import os
import rospkg
import rospy
from typing import Optional
from Frame_data import BaseState, FrameContext
from Frame_data import FrameContext
import numpy as np
import cv2
from collections import deque, defaultdict

class TemporalSmoother:
    """
    Smooth categorical door states with majority voting and hysteresis.

    A bounded window supplies the majority candidate. Initial output is held
    until enough consistent samples exist, and an optional hysteresis rule
    prevents weak candidates from replacing an established result.

    Attributes:
        window_size:
            Maximum number of recent labels retained.

        min_consistent:
            Minimum majority count required for a candidate.

        hysteresis:
            Whether switching away from the stable label is guarded.

        stable_hold:
            Minimum majority count used by the switch guard.

        buffer:
            Bounded deque of recent labels.

        current_stable:
            Label currently exposed to downstream states, or ``None`` during
            warm-up.

        stable_count:
            Confidence-like count maintained for the current stable label.
    """

    def __init__(self, window_size=10, min_consistent=3, hysteresis=True, stable_hold=2):
        """
        Initialize an empty categorical smoothing window.

        Args:
            window_size:
                Maximum retained labels.

            min_consistent:
                Minimum repeated-label count required during warm-up and
                candidate selection.

            hysteresis:
                Enable guarded switching between stable labels.

            stable_hold:
                Minimum candidate count required before switching.
        """
        self.window_size = window_size
        self.min_consistent = min_consistent
        self.hysteresis = hysteresis
        self.stable_hold = stable_hold
        self.buffer = deque(maxlen=window_size)
        self.current_stable = None
        self.stable_count = 0

    def update(self, label: str) -> Optional[str]:
        """
        Add one observation and return the current stable label.

        Args:
            label:
                Current fused door-state observation.

        Returns:
            Stable label, or ``None`` while the initial window lacks
            ``min_consistent`` evidence.

        Notes:
            Majority ties follow insertion order in the frequency mapping.
            With hysteresis disabled, every sufficiently supported majority
            replaces the current stable label immediately.
        """
        # Push new value
        self.buffer.append(label)

        # Count frequencies over window
        freq = defaultdict(int)
        for v in self.buffer:
            freq[v] += 1

        # Majority vote
        majority_label = max(freq.items(), key=lambda x: x[1])[0]
        majority_count = freq[majority_label]

        # Initial warm-up: until enough evidence, return None
        if self.current_stable is None:
            if majority_count >= self.min_consistent:
                self.current_stable = majority_label
                self.stable_count = majority_count
                return self.current_stable
            else:
                return None

        # Require at least min_consistent within window
        candidate = majority_label if majority_count >= self.min_consistent else self.current_stable

        if not self.hysteresis:
            self.current_stable = candidate
            self.stable_count = majority_count
            return self.current_stable

        # Hysteresis: only switch after "stable_hold" confirmations for new candidate
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
    """
    Resolve detector-branch disagreements and route the state machine.

    A fixed decision table chooses a label, source pipeline, and door depth.
    Matching open-side polygons are compared with intersection-over-union.
    The selected label is then temporally smoothed before navigation outputs
    are written to the frame context.

    Attributes:
        height:
            Current branch image height used to rasterize ROI masks.

        width:
            Current branch image width used to rasterize ROI masks.

        smoother:
            :class:`TemporalSmoother` applied to fused labels.
    """

    def __init__(self):
        """
        Initialize ROI dimensions and the final-label temporal smoother.

        The image dimensions remain zero until the first pair of detector
        results is fused. The smoother requires three consistent labels within
        an eight-frame window and briefly holds the last stable result while
        evidence changes.
        """
        super().__init__("combine_door_state")
        self.height = 0
        self.width = 0
        # Temporal smoothing parameters can be tuned here
        self.smoother = TemporalSmoother(window_size=8, min_consistent=3, hysteresis=True, stable_hold=2)

    def rois_match(self, roi1, roi2, iou_threshold):
        """
        Check polygon agreement using image-space intersection over union.

        Args:
            roi1:
                First ``N x 2`` polygon, or ``None``.

            roi2:
                Second ``N x 2`` polygon, or ``None``.

            iou_threshold:
                Minimum intersection-over-union required for a match.

        Returns:
            ``True`` when both polygons have nonzero union and meet the
            threshold; otherwise ``False``.

        Notes:
            :attr:`height` and :attr:`width` must be set for the current image
            before this method is called.
        """
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



    def resolve_door_status(self, color_pipline_result, depth_pipline_result, roi_open_side_color_based, roi_open_side_depth_based, door_depth_m_color_based, door_depth_m_depth_based, iou_threshold):
        """
        Apply the color/depth branch-fusion decision table.

        Depth detections are trusted when color is weak, while valid color
        detections fill gaps when depth finds no frame. Direct conflicts fall
        back to an unknown/full-image result. Matching open labels must also
        agree spatially.

        Args:
            color_pipline_result:
                Color-branch label.

            depth_pipline_result:
                Depth-branch label.

            roi_open_side_color_based:
                Color-branch opening polygon, if any.

            roi_open_side_depth_based:
                Depth-branch opening polygon, if any.

            door_depth_m_color_based:
                Color-branch door-depth estimate.

            door_depth_m_depth_based:
                Depth-branch door-depth estimate.

            iou_threshold:
                ROI agreement threshold for matching open labels.

        Returns:
            A dictionary with the final label, trusted pipeline, selected door
            depth, escalation flag, and a human-readable reason.

        Notes:
            The dictionary always contains ``final_door_status``, ``pipeline``,
            ``ask_human``, ``door_depth``, and ``reason``.
        """

        # Default result container
        result = dict(final_door_status="unknown", pipeline=None, ask_human=False, door_depth = None, reason="")

        # -------------------------------------------------------------
        # DEPTH = NO FRAME
        # -------------------------------------------------------------
        if depth_pipline_result == "no_frame_detected":
            if color_pipline_result == "open_left" or color_pipline_result == "open_right":
                result["final_door_status"] = color_pipline_result # "open_left" or "open_right"
                result["pipeline"] = "color_based"
                result["door_depth"] = door_depth_m_color_based # use color-based door depth
                result["reason"] = f"Color {color_pipline_result} + depth no-frame → use color pipeline"
            elif color_pipline_result == "closed":
                result["final_door_status"] = "closed"
                result["pipeline"] = "color_based"
                result["door_depth"] = door_depth_m_color_based # use color-based door depth
                result["reason"] = "Color closed + depth no-frame → closed"
            elif color_pipline_result == "no_frame_detected":
                result["final_door_status"] = "no_frame_detected"
                result["pipeline"] = "full_image_view"
                result["ask_human"] = True
                result["door_depth"] = None
                result["reason"] = "No frame in both → no_frame_detected, ask human/full image"
            elif color_pipline_result == "unknown":
                result["final_door_status"] = "unknown"
                result["pipeline"] = "full_image_view"
                result["ask_human"] = True
                result["door_depth"] = None
                result["reason"] = "Depth no-frame + color unknown → unknown"

        # -------------------------------------------------------------
        # DEPTH = UNKNOWN
        # -------------------------------------------------------------
        elif depth_pipline_result == "unknown":
            if color_pipline_result == "open_left" or color_pipline_result == "open_right":
                result["final_door_status"] = color_pipline_result # "open_left" or "open_right"
                result["pipeline"] = "color_based"
                result["door_depth"] = door_depth_m_color_based # use color-based door depth
                result["reason"] = f"Color {color_pipline_result} + depth unknown → use color pipeline"
            elif color_pipline_result == "closed":
                result["final_door_status"] = "closed"
                result["pipeline"] = "color_based"
                result["door_depth"] = door_depth_m_color_based # use color-based door depth
                result["reason"] = "Color closed + depth unknown → closed"
            elif color_pipline_result == "no_frame_detected":
                result["final_door_status"] = "unknown"
                result["pipeline"] = "full_image_view"
                result["ask_human"] = True
                result["door_depth"] = None
                result["reason"] = "Color no-frame + depth unknown → ambiguous"
            elif color_pipline_result == "unknown":
                result["final_door_status"] = "unknown"
                result["pipeline"] = "full_image_view"
                result["ask_human"] = True
                result["door_depth"] = None
                result["reason"] = "Both unknown → unknown"

        # -------------------------------------------------------------
        # DEPTH = CLOSED
        # -------------------------------------------------------------
        elif depth_pipline_result == "closed":
            if color_pipline_result == "open_left" or color_pipline_result == "open_right":
                result["final_door_status"] = "unknown"
                result["pipeline"] = "full_image_view"
                result["ask_human"] = True
                result["door_depth"] = None
                result["reason"] = "Conflict: color open but depth closed"
            elif color_pipline_result == "closed":
                result["final_door_status"] = "closed"
                result["pipeline"] = "depth_based"
                result["door_depth"] = door_depth_m_depth_based # use depth-based door depth
                result["reason"] = "Both closed → closed"
            elif color_pipline_result == "no_frame_detected" or color_pipline_result == "unknown":
                result["final_door_status"] = "closed"
                result["pipeline"] = "depth_based"
                result["door_depth"] = door_depth_m_depth_based # use depth-based door depth
                result["reason"] = "Depth closed overrides color no-frame/unknown"

        # -------------------------------------------------------------
        # DEPTH = OPEN LEFT
        # -------------------------------------------------------------
        elif depth_pipline_result == "open_left":
            if color_pipline_result == "open_left":
                if self.rois_match(roi_open_side_color_based, roi_open_side_depth_based, iou_threshold):
                    result["final_door_status"] = "open_left"
                    result["pipeline"] = "color_based"
                    result["door_depth"] = door_depth_m_color_based # use color-based door depth
                    result["reason"] = "Both open_left + ROI match → use color pipeline"
                else:
                    result["final_door_status"] = "unknown"
                    result["pipeline"] = "full_image_view"
                    result["ask_human"] = True
                    result["door_depth"] = None
                    result["reason"] = "Both open_left but ROI mismatch → full image"
            elif color_pipline_result == "open_right" or color_pipline_result == "closed":
                result["final_door_status"] = "unknown"
                result["pipeline"] = "full_image_view"
                result["ask_human"] = True
                result["door_depth"] = None
                result["reason"] = "Conflict with depth open_left"
            elif color_pipline_result == "no_frame_detected" or color_pipline_result == "unknown":
                result["final_door_status"] = "open_left"
                result["pipeline"] = "depth_based"
                result["door_depth"] = door_depth_m_depth_based # use depth-based door depth
                result["reason"] = "Depth open_left + weak color → use depth pipeline"
        # -------------------------------------------------------------
        # DEPTH = OPEN RIGHT
        # -------------------------------------------------------------
        elif depth_pipline_result == "open_right":
            if color_pipline_result == "open_right":
                if self.rois_match(roi_open_side_color_based, roi_open_side_depth_based, iou_threshold):
                    result["final_door_status"] = "open_right"
                    result["pipeline"] = "color_based"
                    result["door_depth"] = door_depth_m_color_based # use color-based door depth
                    result["reason"] = "Both open_right + ROI match → use color pipeline"
                else:
                    result["final_door_status"] = "unknown"
                    result["pipeline"] = "full_image_view"
                    result["ask_human"] = True
                    result["door_depth"] = None
                    result["reason"] = "Both open_right but ROI mismatch"
            elif color_pipline_result == "open_left" or color_pipline_result == "closed":
                result["final_door_status"] = "unknown"
                result["pipeline"] = "full_image_view"
                result["ask_human"] = True
                result["door_depth"] = None
                result["reason"] = "Conflict with depth open_right"
            elif color_pipline_result == "no_frame_detected" or color_pipline_result == "unknown":
                result["final_door_status"] = "open_right"
                result["pipeline"] = "depth_based"
                result["door_depth"] = door_depth_m_depth_based # use depth-based door depth
                result["reason"] = "Depth open_right + weak color → use depth pipeline"
        # -------------------------------------------------------------
        # FALLBACK
        # -------------------------------------------------------------
        else:
            result["final_door_status"] = "unknown"
            result["pipeline"] = "full_image_view"
            result["ask_human"] = True
            result["door_depth"] = None
            result["reason"] = f"Unhandled combination ({depth_pipline_result}, {color_pipline_result})"

        return result


    def do_action(self, ctx: FrameContext) -> Optional[str]:
        """
        Fuse one frame, update navigation outputs, and select the next state.

        Args:
            ctx:
                Shared frame context containing both branch results and ROIs.

        Returns:
            Next state name. Open labels go to ``"final_state"``; a stable
            no-frame label goes to ``"full_image_passability_check_state"``;
            other labels return to ``"dual_branch_frame_detection_state"``.

        Raises:
            OSError:
                If the smoothed-state diagnostic file cannot be written.

        Notes:
            For an open-left result the 95th ROI x-percentile marks the central
            frame edge; open-right uses the 5th percentile.
        """
        self.height, self.width = ctx.color_image_color_based.shape[:2]

        result = self.resolve_door_status(ctx.door_state_color_based, ctx.door_state_depth_based, ctx.roi_open_side_color_based, ctx.roi_open_side_depth_based, ctx.door_depth_m_color_based, ctx.door_depth_m_depth_based, iou_threshold=0.5)

        # Apply temporal smoothing on final label
        smoothed_door_state = self.smoother.update(result["final_door_status"])

        pkg_path = rospkg.RosPack().get_path('robodog_glass_door_detection')
        log_dir = os.path.join(pkg_path, 'scripts')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'final_door_status_after_temporal_smoothing.txt')
        with open(log_path, "a") as f:
                f.write(f"smoothed_door_state: {smoothed_door_state}\n")


        ctx.door_state_label = smoothed_door_state
        ctx.door_depth = result["door_depth"]
        pipeline_used = result["pipeline"]

        if smoothed_door_state == "open_left":
            roi_open_side = getattr(ctx, f"roi_open_side_{pipeline_used}") # get the open side roi polygon
            ctx.mid_frame_x_px_for_passability_check = np.percentile(roi_open_side[:,0], 95)  # get the max x position of the open side roi polygon, to find the center of the central frame
            return "final_state"  # go to final state
        elif smoothed_door_state == "open_right":
            roi_open_side = getattr(ctx, f"roi_open_side_{pipeline_used}") # get the open side roi polygon
            ctx.mid_frame_x_px_for_passability_check = np.percentile(roi_open_side[:,0], 5)  # get the min x position of the open side roi polygon, to find the center of the central frame
            return "final_state"  # go to final state
        elif smoothed_door_state == "no_frame_detected":
            ctx.mid_frame_x_px_for_passability_check = None
            return "full_image_passability_check_state"  # go to a state that directly check the passability with full image and depth without relying on mid frame door detection, because no mid frame door detected
        else:
            #door state is either closed, unknown or no_frame_detected
            ctx.mid_frame_x_px_for_passability_check = None
            #loop back to parallel detection, because still need to monitor for door opening
            return "dual_branch_frame_detection_state"

