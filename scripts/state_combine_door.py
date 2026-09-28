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

    def reset(self) -> None:
        """Discard all history so the next observation starts a fresh warm-up."""
        self.buffer.clear()
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
        # Keep raw categorical labels; averaging numeric encodings would invent
        # states that have no physical meaning.
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

        # Hysteresis keeps a single bad depth frame from flipping an already
        # published navigation decision.
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
        self.smoothed_door_state_log_path = rospy.get_param("~result_log_path")+"/final_door_status_after_temporal_smoothing.txt"  # Path to log file for smoothed door states
        self.enable_result_log = rospy.get_param("~enable_result_log", False)  # Whether to log results to file
        self.number_of_times_no_frame_detected_as_smoothed_door_state = 0
        self.maximum_allowed_number_of_times_no_frame_detected_as_smoothed_door_state = rospy.get_param("~maximum_allowed_number_of_times_no_frame_detected_as_smoothed_door_state", 10)  # Maximum allowed consecutive frames with no frame detected before concluding door state as no_frame_detected
        # Held open for the lifetime of the node; see _write_result_log.
        self._result_log_file = None
        if self.enable_result_log:
            try:
                self._result_log_file = open(self.smoothed_door_state_log_path, "a")
                rospy.on_shutdown(self._close_result_log)
            except Exception as exc:
                rospy.logwarn("Could not open result log: %s", exc)


    def _write_result_log(self, line):
        """
        Append one diagnostic line to the result log.

        Notes:
            The handle is kept open across frames. Opening and closing the file on
            every frame cost a pair of syscalls plus a directory lookup inside the
            image callback, which is measurable on flash storage. Each line is
            flushed so the file stays readable while the pipeline runs.
        """
        handle = self._result_log_file
        if handle is None:
            return
        try:
            handle.write(line)
            handle.flush()
        except Exception as exc:
            rospy.logwarn_throttle(30.0, "Could not write result log: %s", exc)

    def _close_result_log(self):
        """Close the diagnostic log handle at shutdown."""
        if self._result_log_file is not None:
            try:
                self._result_log_file.close()
            except Exception:
                pass
            self._result_log_file = None

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

        # Raster masks make IoU work for arbitrary quadrilaterals, including
        # the slanted ROIs produced by perspective.
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

        # Start conservatively. Every trusted combination below must opt into a
        # pipeline and a usable door depth.
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

        # Label agreement is not enough for an open door: both branches must be
        # talking about substantially the same image region.
        result = self.resolve_door_status(ctx.door_state_color_based, ctx.door_state_depth_based, ctx.roi_open_side_color_based, ctx.roi_open_side_depth_based, ctx.door_depth_m_color_based, ctx.door_depth_m_depth_based, iou_threshold=0.5)

        raw_door_state = result["final_door_status"]

        # Apply temporal smoothing on final label
        smoothed_door_state = self.smoother.update(raw_door_state)

        if self.enable_result_log:
            self._write_result_log(f"smoothed_door_state: {smoothed_door_state}\n")

        ctx.door_depth = None
        ctx.mid_frame_x_px_for_passability_check = None

        if smoothed_door_state in ("open_left", "open_right"):
            # Smoothing confirms the label only. ROI x and door depth are tied to the
            # camera pose of the frame they were measured in, so they must come from
            # the current frame, and only if it agrees with the smoothed label (a
            # conflict frame has pipeline "full_image_view" and no ROI).
            if raw_door_state == smoothed_door_state:
                roi_open_side = getattr(ctx, f"roi_open_side_{result['pipeline']}") # get the open side roi polygon
                if roi_open_side is not None and result["door_depth"] is not None:
                    # Use an inner-edge percentile rather than a single polygon vertex; small
                    # ROI rotations then have little effect on the corridor anchor.
                    # open_left: max x of the open side roi, open_right: min x (center of the central frame)
                    ctx.door_state_label = smoothed_door_state  #opem_left or open_right
                    percentile = 95 if smoothed_door_state == "open_left" else 5
                    ctx.mid_frame_x_px_for_passability_check = np.percentile(roi_open_side[:, 0], percentile)
                    ctx.door_depth = result["door_depth"]
                    # This detection cycle ends here; the next one (after idle) must not
                    # inherit this door's votes. ctx already holds the latched outputs.
                    self.smoother.reset()
                    # Reset counters on transition
                    self.number_of_times_no_frame_detected_as_smoothed_door_state = 0
                    return "final_state"  # go to final state
            # No usable geometry in this frame. Do not publish an open label yet (the
            # passability node activates on it); wait for the next agreeing frame.
            ctx.door_state_label = "unknown"
            # Reset counters on transition
            self.number_of_times_no_frame_detected_as_smoothed_door_state = 0
            return "dual_branch_frame_detection_state"
        elif smoothed_door_state == "no_frame_detected": 
            #need higher waiting time before concluding door state as no_frame_detected.because it is possible that human is still opening the door causing 
            #depth branch and color branch to not detect the door in mid frame. So wait for a longer period to ensure the door is truly not detected .
            # Wait for a longer period to ensure the door is truly not detected .
            self.number_of_times_no_frame_detected_as_smoothed_door_state += 1
            if self.number_of_times_no_frame_detected_as_smoothed_door_state >= self.maximum_allowed_number_of_times_no_frame_detected_as_smoothed_door_state:
                # Reset counters on transition
                self.number_of_times_no_frame_detected_as_smoothed_door_state = 0
                ctx.door_state_label = "no_frame_detected"
                self.smoother.reset()  # detection cycle ends here, see final_state branch above
                return "full_image_passability_check_state"  # go to a state that directly check the passability with full image and depth without relying on mid frame door detection, because no mid frame door detected
            else:
                #stay in this state and keep waiting for a few more frames to ensure the door is truly not detected. door could be still opening by the human and causing the mid frame detection to not detect the door in some frames. So wait for a few more frames to ensure the door is truly not detected.
                ctx.door_state_label = "unknown"
                return "dual_branch_frame_detection_state"
        elif smoothed_door_state == "closed":
            #door state is either closed
            ctx.door_state_label = "closed" # closed
            #loop back to parallel detection, because still need to monitor for door opening
            # Reset counters on transition
            self.number_of_times_no_frame_detected_as_smoothed_door_state = 0
            return "dual_branch_frame_detection_state"
        else:
            #door state is unknown, loop back to parallel detection, because still need to monitor for door opening
            ctx.door_state_label = "unknown" # unknown
            # Reset counters on transition
            self.number_of_times_no_frame_detected_as_smoothed_door_state = 0
            return "dual_branch_frame_detection_state"
