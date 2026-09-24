#!/usr/bin/env python3
"""State that runs the color and depth door-frame branches in sequence."""

from Frame_data import BaseState, FrameContext
from typing import Optional
from color_door_detector import ColorDoorDetector
from door_status_detector import Door_Status_Detector
from depth_door_detector import DepthDoorDetector
from find_glass_frame_lines import GlassFrameLineProcessor
from visualization_utils import show_stacked_visualization, build_stacked_visualization
import rospy
import rospkg
import os

class dual_branch_frame_detection_state(BaseState):
    """
    Run both detector branches and store their independent status results.

    Despite the historical state name, both branches currently execute
    sequentially in one callback. Separate door-status instances carry the
    keyword used to select color- or depth-specific context fields.

    Attributes:
        ransac_error:
            Fractional depth tolerance around the latest plane distance.

        color_frame_detection_state:
            Color Canny/Hough line detector.

        depth_frame_detection_state:
            Depth Sobel/Hough line detector.

        glass_frame_detector_based_on_color:
            Pairing and ROI processor for color-derived lines.

        glass_frame_detector_based_on_depth:
            Pairing and ROI processor for depth-derived lines.

        detecting_door_status_based_on_color:
            Branch-specific side-depth classifier.

        detecting_door_status_based_on_depth:
            Branch-specific side-depth classifier.

        door_geometry:
            Latest physical geometry mapping, refreshed per frame.
    """

    def __init__(self):
        """
        Construct reusable processing objects for both detector branches.

        Notes:
            Instances persist across frames to avoid rebuilding configuration
            and to keep branch-specific status keywords.
        """
        super().__init__("dual_branch_frame_detection_state")
        self.ransac_error = rospy.get_param(f"~plane_detector/output/ransac_error", 0.02)
        self.door_state_log_path = rospy.get_param("~result_log_path")+"/door_states.txt"  # Path to log file for door states
        # child current states
        # Reuse the helpers across frames so ROS parameters and timing state are
        # not rebuilt in the camera callback.
        self.color_frame_detection_state = ColorDoorDetector()#only need to create state object once.
        self.depth_frame_detection_state = DepthDoorDetector()#only need to create state object once.
        self.glass_frame_detector_based_on_color = GlassFrameLineProcessor("color_based")
        self.glass_frame_detector_based_on_depth = GlassFrameLineProcessor("depth_based")
        self.detecting_door_status_based_on_color = Door_Status_Detector("color_based")
        self.detecting_door_status_based_on_depth = Door_Status_Detector("depth_based")
        self.enable_windows = rospy.get_param("~enable_visualization", False)
        self.enable_result_log = rospy.get_param("~enable_result_log", False)  # Whether to log results to file
        # Held open for the lifetime of the node; see _write_result_log.
        self._result_log_file = None
        if self.enable_result_log:
            try:
                self._result_log_file = open(self.door_state_log_path, "a")
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

    def do_action(self, ctx: FrameContext) -> Optional[str]:
        """
        Process both branches, create debug views, and request fusion.

        Args:
            ctx:
                Shared context with aligned color/depth input, intrinsics, and
                branch-specific visualization images.

        Returns:
            Always returns ``"combine_door_state"`` after both branch results
            have been written to ``ctx``.

        Notes:
            Door geometry and plane distance are re-read each frame because the
            plane-search state can update them at runtime. A branch without
            valid ROIs receives the label ``"no_frame_detected"``.
        """
        # get the latest door geometry and distance from ROS params
        # these values are changed during runtime. hence need to load the door geometry and plane distance in this state as well to make sure the latest value is used for detection

        # Plane search can update these measurements after examining the current
        # doorway, so cached constructor values would quickly become stale. They
        # are read from the shared context, which the plane-search state keeps
        # current, instead of from the parameter server once per frame.
        self.door_geometry = ctx.door_geometry
        ransac_plane_distance = ctx.ransac_plane_distance
        DEPTH_RANGE = [ransac_plane_distance * (1 - self.ransac_error), ransac_plane_distance * (1 + self.ransac_error)]

        # Keep the two answers independent until fusion; a failure in one branch
        # should remain visible rather than silently borrowing the other's ROI.
        # Advance color branch
        potential_frame_lines_from_c , depth_of_potential_frame_lines_c, edges_c = self.color_frame_detection_state.process_frame(ctx)
        ctx.edges = edges_c
        roi_left_c, roi_right_c, mean_z_depth_along_frame_lines_c = self.glass_frame_detector_based_on_color.find_left_right_roi_and_door_depth(
                ctx.depth_image_in_meters,
                ctx.color_image_color_based,
                ctx.fx,
                potential_frame_lines_from_c,
                depth_of_potential_frame_lines_c,
                self.door_geometry,
                DEPTH_RANGE
            )


        if roi_left_c is not None and roi_right_c is not None:
            #only update context if valid ROIs are found for better debugging
            ctx.roi_left_color_based = roi_left_c
            ctx.roi_right_color_based = roi_right_c
            ctx.door_depth_m_color_based = mean_z_depth_along_frame_lines_c
            door_state_c, roi_open_side_c =self.detecting_door_status_based_on_color.detect(ctx)
        else:
            # "no_mid_door_frame_detected_state"
            #check bev based detection with full image
            door_state_c = "no_frame_detected"
            roi_open_side_c = None
            pass  # could log or handle no detection case here



        # Advance depth branch
        potential_frame_lines_from_d , depth_of_potential_frame_lines_d, edges_d = self.depth_frame_detection_state.process_frame(ctx)
        ctx.sobel_vis_color = edges_d
        roi_left_d, roi_right_d, mean_z_depth_along_frame_lines_d = self.glass_frame_detector_based_on_depth.find_left_right_roi_and_door_depth(
                ctx.depth_image_in_meters,
                ctx.color_image_depth_based,
                ctx.fx,
                potential_frame_lines_from_d,
                depth_of_potential_frame_lines_d,
                self.door_geometry,
                DEPTH_RANGE
            )

        if roi_left_d is not None and roi_right_d is not None:
            #only update context if valid ROIs are found for better debugging
            ctx.roi_left_depth_based = roi_left_d
            ctx.roi_right_depth_based = roi_right_d
            ctx.door_depth_m_depth_based =  mean_z_depth_along_frame_lines_d
            door_state_d, roi_open_side_d = self.detecting_door_status_based_on_depth.detect(ctx)
        else:
            # "no_mid_door_frame_detected_state"
            #check bev based detection with full image
            door_state_d = "no_frame_detected"
            roi_open_side_d = None
            pass  # could log or handle no detection case here


        setattr(ctx, "door_state_color_based", door_state_c)
        setattr(ctx, "roi_open_side_color_based", roi_open_side_c)


        setattr(ctx, "door_state_depth_based", door_state_d)
        setattr(ctx, "roi_open_side_depth_based", roi_open_side_d)

        # Optional window display controlled by param
        if self.enable_windows:
            # Build images even when local windows are disabled; remote ROS tools use them.
            # Build stacked visualizations for publishing
            ctx.viz_color_stack = build_stacked_visualization(
                ctx.color_image_color_based, MIN_DEPTH=0.3, MAX_DEPTH=6.0, edges=ctx.edges, depth_frame=ctx.depth_frame
            )

            ctx.viz_depth_stack = build_stacked_visualization(
                ctx.color_image_depth_based, MIN_DEPTH=1.9, MAX_DEPTH=2.1, edges=ctx.sobel_vis_color, depth_frame=ctx.depth_frame
            )

            show_stacked_visualization(
                ctx.color_image_color_based, MIN_DEPTH=0.3, MAX_DEPTH=6.0, edges=ctx.edges, depth_frame=ctx.depth_frame, window_name="Color | Depth | Edges+ Lines"
            )
            show_stacked_visualization(
                ctx.color_image_depth_based, MIN_DEPTH=1.8, MAX_DEPTH=2.5, edges=ctx.sobel_vis_color, depth_frame=ctx.depth_frame, window_name="depth lines in color image | Depth for depth lines| Sobel + Depth Edges"
            )

        # Persist latest states to a text log for debugging/analysis
        if self.enable_result_log:
            self._write_result_log(
                f"Color-based door state: {door_state_c}, Depth-based door state: {door_state_d}\n")

        return "combine_door_state"



