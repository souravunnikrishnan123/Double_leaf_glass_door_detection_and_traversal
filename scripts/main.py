#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import String
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
from state_parallel_detection import parallel_detection_state
from ros_frame_adapter import DepthFrameAdapter
from duration import get_duration_seconds
from state_no_door_plane_detected_state import create_full_view_bird_eye_view_state
from setup_realsense_pipeline import setup_realsense_pipeline
from visualization_utils import  show_stacked_visualization, setup_visualization_mode


class DoorDetectionNode:
    def __init__(self):
        rospy.init_node("glass_door_detection_node")

        color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        queue_size = rospy.get_param("~queue_size", 30)
        slop = rospy.get_param("~sync_slop", 0.2)

        self.bridge = CvBridge()
        # Global visualization toggle (default true)
        self.enable_visualization = rospy.get_param("~enable_visualization", True)
        # Setup cv2 drawing to no-op when disabled
        setup_visualization_mode(self.enable_visualization)
        self.sm = StateMachine(ctx=None)
        self.sm.add_state(idle_state())
        self.sm.add_state(searching_door_plane_state())
        self.sm.add_state(create_full_view_bird_eye_view_state())
        self.sm.add_state(parallel_detection_state())
        self.sm.add_state(combine_door_state())
        self.sm.set_state("idle")

        # Publishers
        self.state_pub = rospy.Publisher("~door_state", String, queue_size=10)
        self.debug_pub = rospy.Publisher("~debug_image", Image, queue_size=1)
        self.viz_color_pub = rospy.Publisher("~viz/color_branch", Image, queue_size=1)
        self.viz_depth_pub = rospy.Publisher("~viz/depth_branch", Image, queue_size=1)
        self.viz_plane_pub = rospy.Publisher("~viz/plane_overlay", Image, queue_size=1)
        
        self.viz_bird_eye_pub_color = rospy.Publisher("~viz/bird_eye_view_color", Image, queue_size=1)
        self.viz_bird_eye_pub_depth = rospy.Publisher("~viz/bird_eye_view_depth", Image, queue_size=1)
        self.viz_bird_eye_pub_full_image = rospy.Publisher("~viz/bird_eye_view_full_image", Image, queue_size=1)
        
        self.viz_passability_pub_depth = rospy.Publisher("~viz/passability_view_depth", Image, queue_size=1)
        self.viz_passability_pub_color = rospy.Publisher("~viz/passability_view_color", Image, queue_size=1)
        self.viz_passability_pub_full_image = rospy.Publisher("~viz/passability_view_full_image", Image, queue_size=1)

        # Subscribers: sync color + depth; cache camera info separately for robustness
        color_sub = message_filters.Subscriber(color_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        self.latest_info = None
        rospy.Subscriber(info_topic, CameraInfo, self._info_cb, queue_size=10)
        #camera intrinsics will be cached on first receipt
        self.fx = self.fy = self.cx = self.cy = None

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
            depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        return depth_mm, depth_m

    def _info_cb(self, info_msg: CameraInfo):
        if self.latest_info is None:
            self.latest_info = info_msg
            K = info_msg.K
            self.fx, self.fy = K[0], K[4]
            self.cx, self.cy = K[2], K[5]
            rospy.loginfo("Camera intrinsics received and stored.")


    def callback(self, color_msg: Image, depth_msg: Image):
        color_image = self._to_cv_color(color_msg)
        depth_mm, depth_m = self._to_depth_mm_and_m(depth_msg)
        # Use latest camera info; require not None
        if self.fx is None:
            rospy.logwarn_throttle(5.0, "Waiting for CameraInfo (only needed once)...")
            return

        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        depth_frame_adapter = DepthFrameAdapter(depth_mm, fx, fy, cx, cy)

        if not hasattr(self.sm, "ctx") or self.sm.ctx is None:
            self.sm.ctx = FrameContext(
                depth_image_in_meters=depth_m,
                color_image=color_image,
                fx=fx, fy=fy, cx=cx, cy=cy,
                depth_frame=depth_frame_adapter,
            )
            self.sm.ctx.color_image_color_based = color_image.copy()
            self.sm.ctx.color_image_depth_based = color_image.copy()
        else:
            ctx = self.sm.ctx
            ctx.depth_image_in_meters = depth_m
            ctx.color_image = color_image
            ctx.color_image_color_based = color_image.copy()
            ctx.color_image_depth_based = color_image.copy()
            ctx.fx, ctx.fy, ctx.cx, ctx.cy = fx, fy, cx, cy
            ctx.depth_frame = depth_frame_adapter

        
        self.sm.update(self.sm.ctx)

       

        

        label = self.sm.ctx.door_state_label
        if label is not None:
            self.state_pub.publish(String(data=label))

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
                if self.sm.ctx.viz_plane_overlay is not None:
                    self.viz_plane_pub.publish(self.bridge.cv2_to_imgmsg(self.sm.ctx.viz_plane_overlay, encoding="bgr8"))
                if self.sm.ctx.bird_eye_view_color_based is not None:
                    self.viz_bird_eye_pub_color.publish(self.bridge.cv2_to_imgmsg(self.sm.ctx.bird_eye_view_color_based, encoding="bgr8"))
                if self.sm.ctx.bird_eye_view_depth_based is not None:
                    self.viz_bird_eye_pub_depth.publish(self.bridge.cv2_to_imgmsg(self.sm.ctx.bird_eye_view_depth_based, encoding="bgr8"))
                if self.sm.ctx.bird_eye_view_full_image_view is not None:
                    self.viz_bird_eye_pub_full_image.publish(self.bridge.cv2_to_imgmsg(self.sm.ctx.bird_eye_view_full_image_view, encoding="bgr8"))
                if self.sm.ctx.passability_view_color_based is not None:
                    self.viz_passability_pub_color.publish(self.bridge.cv2_to_imgmsg(self.sm.ctx.passability_view_color_based, encoding="bgr8"))
                if self.sm.ctx.passability_view_depth_based is not None:
                    self.viz_passability_pub_depth.publish(self.bridge.cv2_to_imgmsg(self.sm.ctx.passability_view_depth_based, encoding="bgr8"))
                if self.sm.ctx.passability_view_full_image_view is not None:
                    self.viz_passability_pub_full_image.publish(self.bridge.cv2_to_imgmsg(self.sm.ctx.passability_view_full_image_view, encoding="bgr8"))
        except Exception as e:
            rospy.logdebug(f"Viz publish exception: {e}")

        #cv2.waitKey(1)

def main():
    node = DoorDetectionNode()
    rospy.loginfo("glass_door_detection_node started.")
    rospy.spin()


if __name__ == "__main__":
    main()