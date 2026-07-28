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
from state_full_image_passability_check import full_image_passability_check_state
from setup_realsense_pipeline import setup_realsense_pipeline
from visualization_utils import  show_stacked_visualization, setup_visualization_mode
from std_msgs.msg import Bool


class DoorDetectionNode:
    """Own ROS interfaces and drive the detector once per synchronized frame."""

    def __init__(self):

        color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        queue_size = rospy.get_param("~queue_size", 30)
        slop = rospy.get_param("~sync_slop", 0.2)

        self.latest_info = None
        self.fx = self.fy = self.cx = self.cy = None
        self.go_to_idle_from_finish_state = False
        self.start_door_frame_detection = False

        self.bridge = CvBridge()
        # Global visualization toggle (default true)
        self.enable_visualization = rospy.get_param("~enable_visualization", True)
        # Setup cv2 drawing to no-op when disabled
        setup_visualization_mode(self.enable_visualization)
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
        color_sub = message_filters.Subscriber(color_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        
        # camera intrinsics will be cached on first receipt
        rospy.Subscriber(info_topic, CameraInfo, self._info_cb, queue_size=10)
        rospy.Subscriber("/check_if_corridor_is_passable/retrigger_door_detection_node", Bool, self._retrigger_door_detection_node_cb, queue_size=1)
        
        # published by top level path planning and navigation algorithm to trigger start of door frame detection and subsequent steps. Published once per door traversal attempt.
        rospy.Subscriber("/trigger_start_door_frame_detection", Bool, self._trigger_start_door_frame_detection, queue_size=1)

        ats = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=queue_size, slop=slop
        )
        ats.registerCallback(self.callback)




    def _to_cv_color(self, color_msg: Image) -> np.ndarray:
        try:
            return self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        except Exception:
            return self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="passthrough")

    def _to_depth_mm_and_m(self, depth_msg: Image):
        img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth_msg.encoding in ("16UC1", "mono16"):
            depth_mm = img.astype(np.uint16)
            depth_m = depth_mm.astype(np.float32) / 1000.0
        else:
            depth_m = img.astype(np.float32)
            # Sanitize invalid values before casting (NaN/Inf can trigger warnings)
            scaled_mm = depth_m * 1000.0
            scaled_mm = np.nan_to_num(scaled_mm, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = np.clip(np.rint(scaled_mm), 0, 65535).astype(np.uint16)
        return depth_mm, depth_m

    def _info_cb(self, info_msg: CameraInfo):
        if self.latest_info is None:
            self.latest_info = info_msg
            K = info_msg.K
            self.fx, self.fy = K[0], K[4]
            self.cx, self.cy = K[2], K[5]
            rospy.loginfo("Camera intrinsics received and stored.")

    def _retrigger_door_detection_node_cb(self, msg: Bool):
        if msg.data:
            self.go_to_idle_from_finish_state = True
            rospy.loginfo("Door detection retriggered to go to idle state.")
        else:
            self.go_to_idle_from_finish_state = False
    
    def _trigger_start_door_frame_detection(self, msg: Bool):
        if msg.data:
            self.start_door_frame_detection = True
            rospy.loginfo("door frame detection triggered.")
        else:
            self.start_door_frame_detection = False
            rospy.loginfo("door frame detection stopped.")


    def callback(self, color_msg: Image, depth_msg: Image):
        """Process one approximately synchronized color/depth message pair."""
        color_image = self._to_cv_color(color_msg)
        depth_mm, depth_m = self._to_depth_mm_and_m(depth_msg)
        # Use latest camera info; require not None
        if self.fx is None:
            rospy.logwarn_throttle(5.0, "Waiting for CameraInfo (only needed once)...")
            return

        depth_frame_adapter = DepthFrameAdapter(depth_mm, self.fx, self.fy, self.cx, self.cy)

        if not hasattr(self.sm, "ctx") or self.sm.ctx is None: # create object of context class once.
            self.sm.ctx = FrameContext(
                depth_image_in_meters=depth_m,
                color_image=color_image,
                fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
                depth_frame=depth_frame_adapter,
            )
            self.sm.ctx.color_image_color_based = color_image.copy()
            self.sm.ctx.color_image_depth_based = color_image.copy()
            self.sm.ctx.color_image_for_plane_detection = color_image.copy()
            self.sm.ctx.go_to_idle_from_finish_state = False
            self.sm.ctx.start_door_frame_detection = False
        else: # update existing context object with new subscriped data.
            ctx = self.sm.ctx
            ctx.depth_image_in_meters = depth_m
            ctx.color_image = color_image
            ctx.color_image_color_based = color_image.copy()
            ctx.color_image_depth_based = color_image.copy()
            ctx.color_image_for_plane_detection = color_image.copy()
            ctx.fx, ctx.fy, ctx.cx, ctx.cy = self.fx, self.fy, self.cx, self.cy
            ctx.depth_frame = depth_frame_adapter
            ctx.go_to_idle_from_finish_state = self.go_to_idle_from_finish_state
            ctx.start_door_frame_detection = self.start_door_frame_detection

        
        self.sm.update(self.sm.ctx)
        #write durations to file once per callback.
        # file is overwritten each time.filepath is specified by ROS param ~durations_file_path and read by duration.py 
        #when we create get_duration_seconds object in each usage
        get_duration_seconds.write_text_file()

       

        

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
            msg = Twist()
            msg.linear.x = 0.0  # send 0 distance if no plane detected, to avoid issues with downstream consumers expecting a distance value. The normal vector will be set to a default value which can be ignored by downstream consumers since the distance is 0, which can be used by downstream consumers to identify that no plane was detected.
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 1.0  # send a default normal vector pointing straight out if no plane detected, to avoid issues with downstream consumers expecting a normal vector. The distance will be None which can be used by downstream consumers to identify that no plane was detected.

        self.plane_info_pub.publish(msg)

        

                
                


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
    """Initialize the glass-door detection node and enter the ROS event loop."""
    rospy.init_node("glass_door_detection")
    node = DoorDetectionNode()
    rospy.loginfo("glass_door_detection node started.")
    rospy.spin()


if __name__ == "__main__":
    main()
