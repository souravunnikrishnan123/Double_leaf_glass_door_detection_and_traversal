#!/usr/bin/env python3
"""ROS node coordinating synchronized glass-door detection.

Color and aligned depth messages are converted into a shared ``FrameContext``.
The frame-driven state machine then performs plane search, two-branch frame
detection, status fusion, and publication of navigation-facing results.
"""

import rospy
import numpy as np
from std_msgs.msg import String, Float32, Int32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from pyinstrument import Profiler
import message_filters
import os
import sys
import rospkg
import cv2

# Ensure Python can import modules from this package's scripts directory
_pkg_path = rospkg.RosPack().get_path('robodog_glass_door_detection')
_scripts_path = os.path.join(_pkg_path, 'scripts')
if _scripts_path not in sys.path:
    sys.path.insert(0, _scripts_path)

from state_machine_base_classes import StateMachine
from Frame_data import FrameContext
from state_idle import idle_state
from state_searching_door_plane import searching_door_plane_state
from state_combine_door import combine_door_state
from state_parallel_detection import dual_branch_frame_detection_state
from state_final import final_state
from ros_frame_adapter import DepthFrameAdapter
from duration import get_duration_seconds
from resolution_scaling import ResolutionScaler
from state_full_image_passability_check import full_image_passability_check_state
from setup_realsense_pipeline import setup_realsense_pipeline
from visualization_utils import  show_stacked_visualization, setup_visualization_mode
from std_msgs.msg import Bool
import threading


class DoorDetectionNode:
    """
    Own ROS interfaces and drive detection for synchronized camera frames.

    The node caches camera intrinsics, converts ROS image encodings, refreshes a
    shared :class:`FrameContext`, advances the detector state machine, and
    publishes navigation results and optional debug images.

    Attributes:
        latest_info:
            First received color-camera ``CameraInfo`` message.

        fx:
            Cached horizontal focal length in pixels.

        fy:
            Cached vertical focal length in pixels.

        cx:
            Cached horizontal principal point.

        cy:
            Cached vertical principal point.

        go_to_idle_from_finish_state:
            Reset request forwarded into the frame context.

        start_door_frame_detection:
            Start request forwarded into the frame context.

        bridge:
            ``CvBridge`` used for ROS/OpenCV image conversion.

        enable_visualization:
            Whether debug images and OpenCV drawing remain enabled.

        sm:
            Registered frame-driven detection state machine.

        door_state_pub:
            Publisher for the final smoothed door-state label.

        mid_frame_x_px_for_passability_check_pub:
            Publisher for the opening's central-frame x-coordinate.

        door_depth_pub:
            Publisher for the selected door reference depth.

        plane_info_pub:
            Publisher encoding plane distance and normal in a ``Twist``.

        debug_pub:
            Publisher for the selected color-branch debug image.

        viz_color_pub:
            Publisher for the stacked color-processing visualization.

        viz_depth_pub:
            Publisher for the stacked depth-processing visualization.

        viz_plane_pub:
            Publisher for the detected-plane overlay.
    """

    def __init__(self):
        """
        Configure ROS interfaces and construct the detection state machine.

        Topic names, synchronization queue size, synchronization tolerance, and
        visualization behavior are read from private ROS parameters.

        Notes:
            Camera intrinsics are received separately from the synchronized
            color/depth pair and must arrive before frame processing can begin.
        """
        # --- profiling setup ---
        self.profile_enabled = rospy.get_param("~profiling_enabled", False)
        self.profiler_output_path = rospy.get_param("~profiler_output_path")
        self.profile_frame_count = 0
        self.profile_frames = 50          # captures 10 synchronized frames then saves
        self.profile_done = False
        self.profiler = None
        # pyinstrument requires that a session is started and stopped on the same
        # thread, and all real work happens on the subscriber callback thread. The
        # profiler is therefore created here but started lazily on the first
        # callback, so both calls land on the thread being measured.
        self.profile_started = False
        if self.profile_enabled:
            self.profiler = Profiler()
            rospy.on_shutdown(self._save_profile)
        # Timing averages are written on a timer instead of once per frame: the
        # writer does a full read-modify-write of the output file, which is far too
        # expensive to run at frame rate on flash storage.
        if self.profile_enabled:
            rospy.Timer(rospy.Duration(5.0), self._write_durations)
                 

        color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        queue_size = rospy.get_param("~queue_size", 30)
        slop = rospy.get_param("~sync_slop", 0.2)
        self.sync_count = 0
        # Serializes frame processing so a frame arriving mid-computation is dropped
        # rather than queued behind the one in flight.
        self._frame_lock = threading.Lock()
        self.dropped_frame_count = 0
        # Per-frame pair logging is a diagnostic; it publishes to /rosout every frame.
        self.verbose_frame_logging = rospy.get_param("~verbose_frame_logging", False)
        self.latest_info = None
        self.fx = self.fy = self.cx = self.cy = None
        self.go_to_idle_from_finish_state = False
        self.start_door_frame_detection = False

        self.bridge = CvBridge()
        # Global visualization toggle (default true)
        self.enable_visualization = rospy.get_param("~enable_visualization", True)
        # Setup cv2 drawing to no-op when disabled
        setup_visualization_mode(self.enable_visualization)
        # The states form a gate: plane confirmation comes before frame lines,
        # then fusion holds the result until the traversal side acknowledges it.
        self.sm = StateMachine(ctx=None)
        self.sm.add_state(idle_state())
        self.sm.add_state(searching_door_plane_state())
        self.sm.add_state(full_image_passability_check_state())
        self.sm.add_state(dual_branch_frame_detection_state())
        self.sm.add_state(combine_door_state())
        self.sm.add_state(final_state())  # terminal state
        self.sm.set_state("idle_state")

        # state Publishers
        self.door_state_pub = rospy.Publisher("~door_state", String, queue_size=10)
        self.mid_frame_x_px_for_passability_check_pub = rospy.Publisher("~mid_frame_x_px_for_passability_check", Int32, queue_size=10)
        self.door_depth_pub = rospy.Publisher("~door_depth", Float32, queue_size=10)


        self.plane_info_pub = rospy.Publisher("~plane_info", Twist, queue_size=10)
        # visualization publishers
        self.debug_pub = rospy.Publisher("~debug_image", Image, queue_size=1)
        self.viz_color_pub = rospy.Publisher("~viz/color_branch", Image, queue_size=1)
        self.viz_depth_pub = rospy.Publisher("~viz/depth_branch", Image, queue_size=1)
        self.viz_plane_pub = rospy.Publisher("~viz/plane_overlay", Image, queue_size=1)
        

        # Subscribers: sync color + depth; cache camera info separately for robustness
        # queue_size=1 is essential: rospy defaults to an unbounded receive queue, so a
        # consumer slower than the camera would buffer full-resolution images without
        # limit and act on ever more stale frames. buff_size must exceed one image or
        # each frame costs many socket reads.
        color_sub = message_filters.Subscriber(color_topic, Image, queue_size=1, buff_size=2**22)
        depth_sub = message_filters.Subscriber(depth_topic, Image, queue_size=1, buff_size=2**22)
        
        # camera intrinsics will be cached on first receipt
        rospy.Subscriber(info_topic, CameraInfo, self._info_cb, queue_size=10)
        rospy.Subscriber("/check_if_corridor_is_passable/retrigger_door_detection_node", Bool, self._retrigger_door_detection_node_cb, queue_size=1)
        
        # published by top level path planning and navigation algorithm to trigger start of door frame detection and subsequent steps. Published once per door traversal attempt.
        rospy.Subscriber("/trigger_start_door_frame_detection", Bool, self._trigger_start_door_frame_detection, queue_size=1)

        # Exact timestamps differ between several camera drivers, so tolerate a
        # small skew while still processing color and depth as one observation.
        ats = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=queue_size, slop=slop
        )
        ats.registerCallback(self.callback)


    def _write_durations(self, _event=None):
        """
        Flush stage timing averages to disk.

        Notes:
            Driven by a low-rate timer. Failures are logged rather than raised so
            a profiling problem cannot stop detection.
        """
        try:
            get_duration_seconds.write_text_file()
        except Exception as exc:
            rospy.logwarn_throttle(30.0, "Could not write durations file: %s", exc)

    def _stop_profile(self):
        """
        End the sampling session and write the HTML report.

        Notes:
            Must run on the thread that started the session. Called from the
            frame callback once enough frames are captured. Any failure is logged
            rather than raised so profiling cannot break detection.
        """
        if self.profiler is None or self.profile_done:
            return
        self.profile_done = True
        try:
            if self.profile_started:
                self.profiler.stop()
                with open(self.profiler_output_path, "w") as f:
                    f.write(self.profiler.output_html())
                # A plain-text copy is far easier to read over a terminal session
                # on the robot than the HTML report.
                text_path = os.path.splitext(self.profiler_output_path)[0] + ".txt"
                with open(text_path, "w") as f:
                    f.write(self.profiler.output_text(unicode=False, color=False))
                rospy.loginfo("[Profiler] Saved to %s and %s",
                              self.profiler_output_path, text_path)
        except Exception as exc:
            rospy.logwarn("[Profiler] Could not save profile: %s", exc)

    def _save_profile(self):
        """
        Flush profiling output at shutdown.

        Notes:
            The sampling session can only be stopped from the thread that started
            it, so if it is still running at shutdown the report is skipped; the
            frame-count path above is the normal way it is written. Timing
            averages are flushed here regardless.
        """
        if self.profiler is not None and not self.profile_done:
            rospy.loginfo(
                "[Profiler] Shutting down before %d frames were captured; "
                "no call-graph report written.", self.profile_frames,
            )
        # The durations file is written on a timer, so flush a final sample here.
        self._write_durations()


    def _assign_branch_wise_debug_images(self, ctx, color_image: np.ndarray):
        """
        Give each detection branch the image it draws diagnostics onto.

        Args:
            ctx:
                Shared frame context to populate.

            color_image:
                Current BGR frame.

        Notes:
            Each branch needs a private copy only when diagnostics are actually
            drawn. With visualization disabled nothing mutates these buffers, so
            all three share the incoming frame and three full-frame copies per
            callback are avoided.
        """
        if self.enable_visualization:
            ctx.color_image_color_based = color_image.copy()
            ctx.color_image_depth_based = color_image.copy()
            ctx.color_image_for_plane_detection = color_image.copy()
        else:
            ctx.color_image_color_based = color_image
            ctx.color_image_depth_based = color_image
            ctx.color_image_for_plane_detection = color_image

    def _seed_runtime_door_geometry(self, ctx):
        """
        Load the initial door geometry and plane distance into the context.

        Args:
            ctx:
                Shared frame context to seed.

        Notes:
            These values are refined at runtime by the plane-search state and are
            then read once per frame by both branches. Seeding them here means the
            per-frame reads never touch the parameter server.
        """
        ctx.door_geometry = {
            # Widths are physical measurements in centimetres and do not depend
            # on image size; roi_width is a pixel width and does.
            "glass_width_cm": rospy.get_param("~door_geometry/glass_width_cm", 40),
            "center_frame_width_cm": rospy.get_param("~door_geometry/center_frame_width_cm", 30),
            "roi_width": ResolutionScaler().length(
                rospy.get_param("~door_geometry/roi_width", 240)
            ),
            "correction_factor": rospy.get_param("~door_geometry/correction_factor", 1.1),
        }
        ctx.ransac_plane_distance = rospy.get_param(
            "~plane_detector/output/ransac_plane_distance", 2.0
        )

    def _to_cv_color(self, color_msg: Image) -> np.ndarray:
        """
        Convert a ROS color message to an OpenCV array.

        Args:
            color_msg:
                ROS image message.

        Returns:
            BGR image when conversion to ``bgr8`` succeeds; otherwise an array
            using the message's passthrough encoding.

        Raises:
            CvBridgeError:
                If both requested conversions fail.
        """
        try:
            return self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        except Exception:
            return self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="passthrough")

    def _to_depth_mm_and_m(self, depth_msg: Image):
        """
        Convert one ROS depth message into millimeter and meter arrays.

        Args:
            depth_msg:
                Depth image encoded as uint16 millimeters or floating-point
                meters.

        Returns:
            Tuple ``(depth_mm, depth_m)`` containing uint16 millimeters and
            float32 meters.

        Notes:
            Floating-point NaN and infinity values are replaced with zero only
            for the uint16 adapter image. The metric float image retains the
            original nonfinite values for later validity masking.

            The millimetre image feeds only the RealSense-compatible adapter used
            by the diagnostic views, so it is produced only when visualization is
            enabled; otherwise ``None`` is returned in its place.
        """
        img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        # Keep both units: geometry is easier to reason about in metres, while
        # the RealSense-compatible adapter expects the original millimetres.
        if depth_msg.encoding in ("16UC1", "mono16"):
            # astype() always copies, and the array is already uint16 here, so
            # convert straight to metres and scale by a reciprocal.
            depth_m = img.astype(np.float32) * np.float32(0.001)
            depth_mm = img.astype(np.uint16, copy=False) if self.enable_visualization else None
        else:
            depth_m = img.astype(np.float32)
            if self.enable_visualization:
                # Sanitize invalid values before casting (NaN/Inf can trigger warnings)
                scaled_mm = depth_m * 1000.0
                scaled_mm = np.nan_to_num(scaled_mm, nan=0.0, posinf=0.0, neginf=0.0)
                depth_mm = np.clip(np.rint(scaled_mm), 0, 65535).astype(np.uint16)
            else:
                depth_mm = None
        return depth_mm, depth_m

    def _info_cb(self, info_msg: CameraInfo):
        """
        Cache camera intrinsics from the first ``CameraInfo`` message.

        Args:
            info_msg:
                Color-camera calibration message.

        Notes:
            Later calibration messages are ignored because the current camera
            model is assumed constant during a run.
        """
        # Intrinsics describe the stream, not an individual frame, so cache once.
        if self.latest_info is None:
            self.latest_info = info_msg
            K = info_msg.K
            self.fx, self.fy = K[0], K[4]
            self.cx, self.cy = K[2], K[5]
            rospy.loginfo("Camera intrinsics received and stored.")

    def _retrigger_door_detection_node_cb(self, msg: Bool):
        """
        Store the downstream request to return the detector to idle.

        Args:
            msg:
                Boolean reset message from the passability node.
        """
        if msg.data:
            self.go_to_idle_from_finish_state = True
            rospy.loginfo("Door detection retriggered to go to idle state.")
        else:
            self.go_to_idle_from_finish_state = False
    
    def _trigger_start_door_frame_detection(self, msg: Bool):
        """
        Store the top-level request to start or stop door-frame detection.

        Args:
            msg:
                Boolean command from navigation. A true value enables detector
                state transitions; a false value prevents a new detection run
                from starting.

        Notes:
            The callback only updates the shared control flag. Processing
            remains in the synchronized image callback.
        """
        if msg.data:
            self.start_door_frame_detection = True
            rospy.loginfo("door frame detection triggered.")
        else:
            self.start_door_frame_detection = False
            rospy.loginfo("door frame detection stopped.")


    def callback(self, color_msg: Image, depth_msg: Image):
        """
        Process one approximately synchronized color/depth message pair.

        The callback updates or creates the shared frame context, runs one state
        step, writes timing averages, and publishes every currently available
        result.

        Args:
            color_msg:
                Color image selected by ``ApproximateTimeSynchronizer``.

            depth_msg:
                Aligned depth image paired with ``color_msg``.

        Notes:
            Processing is deferred until camera intrinsics are available.
            Separate color copies are created because each branch draws its own
            diagnostics. Missing scalar results are not published, while a
            missing door label is published as ``"unknown"``.
        """
        # Drop frames that arrive while a previous one is still being processed.
        # Without this the node would fall progressively further behind the camera
        # and publish a door state describing an increasingly old observation.
        if not self._frame_lock.acquire(blocking=False):
            self.dropped_frame_count += 1
            rospy.logwarn_throttle(
                5.0,
                "Detection is slower than the camera; dropped %d frame(s) so far.",
                self.dropped_frame_count,
            )
            return
        try:
            self._process_frame(color_msg, depth_msg)
        finally:
            self._frame_lock.release()

    def _process_frame(self, color_msg: Image, depth_msg: Image):
        """
        Run one detection step for an already-admitted frame pair.

        Split out of :meth:`callback` so the drop-if-busy guard wraps the whole
        body. See :meth:`callback` for the argument contract.
        """
        # Start sampling on this thread, which is where the work happens and the
        # only thread the session may later be stopped from.
        if self.profile_enabled and not self.profile_started and not self.profile_done:
            self.profiler.start()
            self.profile_started = True

        self.sync_count += 1
        if self.verbose_frame_logging:
            rospy.loginfo(
                "pair=%d color=%.6f depth=%.6f delta=%.6f thread=%s",
                self.sync_count,
                color_msg.header.stamp.to_sec(),
                depth_msg.header.stamp.to_sec(),
                abs((color_msg.header.stamp - depth_msg.header.stamp).to_sec()),
                threading.current_thread().name,
            )
        color_image = self._to_cv_color(color_msg)
        depth_mm, depth_m = self._to_depth_mm_and_m(depth_msg)
        # Use latest camera info; require not None
        # Running geometry with guessed intrinsics would yield plausible-looking
        # but physically wrong plane sizes, so wait instead.
        if self.fx is None:
            rospy.logwarn_throttle(5.0, "Waiting for CameraInfo (only needed once)...")
            return

        # The adapter exists purely to satisfy the RealSense-style access pattern
        # used by the diagnostic views, so build it only when those run.
        depth_frame_adapter = (
            DepthFrameAdapter(depth_mm, self.fx, self.fy, self.cx, self.cy)
            if self.enable_visualization
            else None
        )

        # Reuse the context to preserve state outputs, but give each detector its
        # own image copy because overlays are drawn in place.
        if not hasattr(self.sm, "ctx") or self.sm.ctx is None: # create object of context class once.
            self.sm.ctx = FrameContext(
                depth_image_in_meters=depth_m,
                color_image=color_image,
                fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
                depth_frame=depth_frame_adapter,
            )
            self._assign_branch_wise_debug_images(self.sm.ctx, color_image)
            self._seed_runtime_door_geometry(self.sm.ctx)
            self.sm.ctx.go_to_idle_from_finish_state = False
            self.sm.ctx.start_door_frame_detection = False
        else: # update existing context object with new subscriped data.
            ctx = self.sm.ctx
            ctx.depth_image_in_meters = depth_m
            ctx.color_image = color_image
            self._assign_branch_wise_debug_images(ctx, color_image)
            ctx.fx, ctx.fy, ctx.cx, ctx.cy = self.fx, self.fy, self.cx, self.cy
            ctx.depth_frame = depth_frame_adapter
            ctx.go_to_idle_from_finish_state = self.go_to_idle_from_finish_state
            ctx.start_door_frame_detection = self.start_door_frame_detection

        
        # One synchronized frame drives exactly one state-machine step.
        self.sm.update(self.sm.ctx)
        # Timing averages are flushed by a timer, not here: the writer rewrites the
        # whole output file and must not run at frame rate.
        # Stop sampling once enough frames are captured so the report stays bounded.
        if self.profile_enabled and not self.profile_done:
            self.profile_frame_count += 1
            if self.profile_frame_count >= self.profile_frames:
                self._stop_profile()
                rospy.loginfo("[Profiler] Captured %d frames.", self.profile_frames)

       

        

        # Publish current door state (guard None)
        door_state = self.sm.ctx.door_state_label
        if door_state is None:
            door_state = "unknown"
        self.door_state_pub.publish(String(data=str(door_state)))

        # Publish mid_frame_x for passability (must be an int)
        mfx = self.sm.ctx.mid_frame_x_px_for_passability_check
        if mfx is not None:
            try:
                self.mid_frame_x_px_for_passability_check_pub.publish(Int32(data=int(mfx)))
            except (ValueError, TypeError) as e:
                rospy.logwarn_throttle(5.0, f"mid_frame_x publish skipped (non-integer): {e}")

        # Publish door depth (must be a float)
        dd = self.sm.ctx.door_depth
        if dd is not None:
            try:
                self.door_depth_pub.publish(Float32(data=float(dd)))
            except (ValueError, TypeError) as e:
                rospy.logwarn_throttle(5.0, f"door_depth publish skipped (non-float): {e}")
        
        # Twist is used as a compact existing carrier: linear.x is distance and
        # angular.xyz stores the unit plane normal. It is not a velocity command.
        pd = self.sm.ctx.plane_result
        if pd is not None:
            plane_distance = float(pd["distance_m"])
            plane_norm = pd["plane_norm_vector"]
            msg = Twist()
            msg.linear.x = plane_distance
            msg.angular.x = plane_norm[0]
            msg.angular.y = plane_norm[1]
            msg.angular.z = plane_norm[2]
        else:
            # Publish a well-formed sentinel rather than leaving subscribers with
            # a stale plane from the previous detection cycle.
            msg = Twist()
            msg.linear.x = 0.0  # send 0 distance if no plane detected, to avoid issues with downstream consumers expecting a distance value. The normal vector will be set to a default value which can be ignored by downstream consumers since the distance is 0, which can be used by downstream consumers to identify that no plane was detected.
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 1.0  # send a default normal vector pointing straight out if no plane detected, to avoid issues with downstream consumers expecting a normal vector. The distance will be None which can be used by downstream consumers to identify that no plane was detected.

        self.plane_info_pub.publish(msg)

        # Visualization failures must never stop the navigation-facing outputs above.
        # Publish visualizations
        try:
            if self.enable_visualization:
                dbg = self.sm.ctx.color_image_color_based
                if dbg is not None:
                    self.debug_pub.publish(self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8"))
                if self.sm.ctx.viz_color_stack is not None:
                    self.viz_color_pub.publish(self.bridge.cv2_to_imgmsg(self.sm.ctx.viz_color_stack, encoding="bgr8"))
                if self.sm.ctx.viz_depth_stack is not None:
                    self.viz_depth_pub.publish(self.bridge.cv2_to_imgmsg(self.sm.ctx.viz_depth_stack, encoding="bgr8"))
                # plane overlay: use base color image (with plane outlines drawn)
                if self.sm.ctx.color_image_for_plane_detection is not None:
                    self.viz_plane_pub.publish(self.bridge.cv2_to_imgmsg(self.sm.ctx.color_image_for_plane_detection, encoding="bgr8"))

        except Exception as e:
            rospy.logdebug(f"Viz publish exception: {e}")

        #cv2.waitKey(1)


def main():
    """
    Initialize the glass-door detection node and enter the ROS event loop.

    Returns:
        ``None`` after ROS shutdown.

    Raises:
        rospy.ROSException:
            If the node or one of its ROS interfaces cannot be initialized.

    Notes:
        All frame processing is driven by subscriber callbacks while
        ``rospy.spin`` keeps the process alive.
    """
    rospy.init_node("glass_door_detection")
    node = DoorDetectionNode()
    rospy.loginfo("glass_door_detection node started.")
    rospy.spin()


if __name__ == "__main__":
    main()
