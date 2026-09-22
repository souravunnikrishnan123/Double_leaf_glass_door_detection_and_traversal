#!/usr/bin/env python3
"""State that waits for a stable door-sized plane before frame detection."""

from door_type_detector import DoorTypeDetector
import rospy
from Frame_data import BaseState, FrameContext
from detect_glass_door_plane import PlaneDetector

class searching_door_plane_state(BaseState):
    """
    Confirm a door plane and update its estimated physical geometry.

    The state runs plane RANSAC until temporal confirmation succeeds. Confirmed
    inliers are then analyzed for pane and center-frame widths, which are written
    into the shared frame context for use by both line branches.

    Attributes:
        find_door_plane:
            Stateful plane detector and temporal tracker.

        door_type_detector:
            Physical pane/frame width estimator.

        max_no_candidate_frames:
            Consecutive empty frames tolerated before full-image fallback.

        reference_door_distance_m:
            Nominal door distance from configuration.

        global_map_distance_accuracy_to_door_plane:
            Fractional uncertainty applied to the nominal map distance.

        distance_range_m:
            Derived accepted distance error around the nominal door position.

        _no_candidate_count:
            Consecutive frames with no plane candidate.

        margin_for_glass_width_inaccuracy:
            Fraction subtracted from estimated pane width as a safety margin.
    """

    def __init__(self):
        """
        Initialize plane search, width estimation, and fallback thresholds.

        Notes:
            The plane detector is intentionally reused so its temporal history
            survives across calls to :meth:`do_action`.
        """
        super().__init__("searching_door_plane_state")
        self.find_door_plane = PlaneDetector()
        self.door_type_detector = DoorTypeDetector()
        ns = "~plane_detector"
        # Thresholds to decide "no plane present" path
        self.max_no_candidate_frames = rospy.get_param(f"{ns}/max_no_candidate_frames", 5)
        self.reference_door_distance_m = rospy.get_param("~reference_door_distance_m", 2.0)
        self.global_map_distance_accuracy_to_door_plane = rospy.get_param(f"{ns}/global_map_distance_accuracy_to_door_plane", 0.15)
        self.distance_range_m = self.reference_door_distance_m * self.global_map_distance_accuracy_to_door_plane
        self._no_candidate_count = 0
        self.margin_for_glass_width_inaccuracy = rospy.get_param(f"{ns}/margin_for_glass_width_inaccuracy", 0.2)  # 20% margin of safety

    def do_action(self, ctx: FrameContext):
        """
        Search the current depth frame and choose the next detection path.

        Args:
            ctx:
                Shared frame context containing metric depth, plane-overlay
                image, and camera intrinsics.

        Returns:
            ``"dual_branch_frame_detection_state"`` after a confirmed plane;
            ``"full_image_passability_check_state"`` after too many frames
            without candidates; otherwise ``None`` to continue searching.

        Notes:
            Frames containing an unconfirmed candidate reset the empty-frame
            counter. Successful width estimates update ``ctx.door_geometry``
            before line detection begins.
        """
        #check if there is a glass door plane in front of the camera
        # if yes, then proceed with line detection and frame detection
        # Plane search is intentionally the first expensive gate; frame-line
        # detection is noisy when it runs without a trusted depth and orientation.
        result = self.find_door_plane.detect(ctx.color_image_for_plane_detection, ctx.depth_image_in_meters, ctx.fx, ctx.fy, ctx.cx, ctx.cy)
        # plane overlays are drawn into ctx.color_image_for_plane_detection; main publishes as ~viz/plane_overlay
    

        if result["plane_model"] is not None:

            # Use the confirmed plane's 3D support to tune the physical spacing
            # expected later by both Hough-line branches.
            glass_and_door_width = self.door_type_detector.estimate_glass_and_frame_widths(
                result["inlier_points"]
            )
            

            vis_img = self.door_type_detector.visualize_door_bins_and_widths(
            inlier_points=result["inlier_points"],
            bin_edges=glass_and_door_width["bins"],
            bin_labels=glass_and_door_width["bin_labels"],
            segments=glass_and_door_width["segments"],
            final_glass_index=glass_and_door_width["final_glass_index"],
            final_frame_index=glass_and_door_width["final_frame_index"],
            fx=ctx.fx,
            cx=ctx.cx,
            color_image=ctx.color_image_for_plane_detection,
            )

            if glass_and_door_width["final_glass_width_m"] is not None and glass_and_door_width["final_frame_width_m"] is not None:
                # Cast numpy scalars to native Python floats before storing them.
                # Underestimate pane width slightly so the pairing test does not
                # discard a real frame because of a borderline measurement.
                glass_width_cm_safe_value = float(glass_and_door_width["final_glass_width_m"]) * 100.0 * (1 - self.margin_for_glass_width_inaccuracy) # margin of safety
                center_frame_width_cm_safe_value = float(glass_and_door_width["final_frame_width_m"]) * 100.0
                #for glass width dont have to check the default value as the glass width detection is usually accurate enough and also, the default value is set to a lower value to handle the case, in which the algorithm couldnt detect the glass width

                default_value_of_center_frame_width = ctx.door_geometry["center_frame_width_cm"]*100.0

                center_frame_width_cm_safe_value = max(center_frame_width_cm_safe_value, default_value_of_center_frame_width)  # enforce minimum frame width.
                #it is needed because sometimes the frame width detection can be way off when there is clutter around the door frame, especially when the door is already open


                # The context is the channel downstream states read. The matching
                # ROS parameters are not written back: nothing reads them after
                # startup, and each write is a blocking call to the master plus a
                # parameter-update broadcast.
                ctx.door_geometry["glass_width_cm"] = glass_width_cm_safe_value
                ctx.door_geometry["center_frame_width_cm"] = center_frame_width_cm_safe_value
                # if final_glass_width_m and final_frame_width_m are None, do not update the params (keep default valid values)

            ctx.color_image_for_plane_detection = vis_img
            
            self._no_candidate_count = 0 # reset counter on successful detection

            distance = float(result["plane_metrics"]["distance_m"])
            rospy.loginfo(f"Detected door plane at distance: {distance:.2f} m")
            ctx.ransac_plane_distance = distance
            # Carried on the context only; see the note on door geometry above.
            ctx.plane_result = {"distance_m": distance, "plane_norm_vector": result["plane_metrics"]["plane_norm_vector"]}
            rospy.loginfo(f"plane detected at a distance of : {distance:.2f} m and its normal vector is {result['plane_metrics']['plane_norm_vector']}")
            """
            if abs(distance - self.reference_door_distance_m) < self.distance_range_m:
                #return "dual_branch_frame_detection_state"
                return None
            else:
                return "movement_state"
            """
            return "dual_branch_frame_detection_state"
        else:
            # Distinguish between "unconfirmed yet" vs "no candidates at all"
            had_candidates = result["had_candidates"]

            # An unstable candidate is different from an empty scene. It should
            # be given time to confirm rather than counting toward fallback.
            if had_candidates is True:
                # There is a potential plane, keep trying and reset miss counter
                self._no_candidate_count = 0
                return None
            else: # had_candidates is False
                # No candidates detected this frame
                self._no_candidate_count += 1
                if self._no_candidate_count >= self.max_no_candidate_frames:
                    # Transition to full-view image passability check (no door plane detected path)
                    # Reset counters on transition
                    self._no_candidate_count = 0
                    # to directly check if the door is passable without plane detection as the plane is not detectable. this is useful when the both doors halves are wide open already and ransac couldnt detect frame around it as well
                    # A wide-open doorway may offer no pane to fit, so route to
                    # direct clearance checking instead of waiting forever.
                    ctx.door_state_label = "No_door_plane_detected"  # No door plane detected
                    return "full_image_passability_check_state"
                #stay in this state
                return None


            
