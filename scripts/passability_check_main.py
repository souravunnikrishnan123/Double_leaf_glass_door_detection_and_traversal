#!/usr/bin/env python3
from check_if_passable import Passability_checker
from duration import get_duration_seconds
import rospy
from std_msgs.msg import String, Float32, Int32
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np  
from geometry_msgs.msg import Twist
from processing_classes import backproject_depth_to_points

class PassabilityCheckerNode:
    def __init__(self):
        self.enable_visualization = rospy.get_param("~enable_visualization", False)
        self.bridge = CvBridge()
        ns = "~passabilility_check"
        # ---- State ----
        self.active = False
        self.open_side = None
        self.mid_frame_x_px = None
        self.door_depth = None

        # Latest sensor data
        self.color_image = None
        self.depth_image = None
        self.latest_info = None 
        self.fx = self.fy = self.cx = self.cy = None

        # ---- Subscribers (conditional) ----
        self.color_sub = None
        self.depth_sub = None
        self.info_sub = None
        self.mid_frame_x_px_sub = None
        self.door_depth_sub = None
        
        # -----------------------------
        # Backprojection parameters
        # -----------------------------
        self.minimum_depth_for_back_projection = rospy.get_param(f"{ns}/back_proj_params/minimum_depth", 0.05)  # meters
        self.maximum_depth_beyond_door_depth_for_back_projection = rospy.get_param(f"{ns}/back_proj_params/maximum_depth_beyond_door_depth", 1.5)  # meters
        self.subsample = rospy.get_param(f"{ns}/back_proj_params/subsample", 2)

        self.robot_width = rospy.get_param(f"{ns}/traversal_params/robot_width", 0.45)  # meters ,robot  width
        self.safety_margin_robot_width = rospy.get_param(f"{ns}/traversal_params/safety_margin_robot_width", 0.05)  # meters
        self.door_frame_margin = rospy.get_param(f"{ns}/traversal_params/door_frame_margin", 0.05)  # meters

        # ---- Always-on subscriber ----
        self.door_state_sub = rospy.Subscriber(
            "/glass_door_detection/door_state",
            String,
            self._door_state_cb,
            queue_size=1
        )

        #publishers.publish always because for rqt_graph stability and silence should never be a state
        #result publishers
        self.Passability_pub = rospy.Publisher(
            "~door_passability",
            Twist,
            queue_size=1
        )

        #visualization publisher
        self.passability_view_pub = rospy.Publisher(
            "~passability_view",
            Image,
            queue_size=1
        )

        # Passability checker instance
        self.passability_checker = Passability_checker()
        self.rate = rospy.Rate(5)  # 5 Hz decision loop

    # -----------------------------
    # Activation logic called upon door state changes
    # only update states, store data and activation/deactivation of camera subscriptions
    # call backs shall be non blocking and lightweight
    # -----------------------------
    def _door_state_cb(self, msg):
        if msg.data in ["open_left", "open_right"]:
            if not self.active:
                rospy.loginfo("PassabilityChecker: activated")
                #to create subscriptions
                self.activate_camera()
                self.get_door_depth_and_midframe()
            self.active = True
            self.open_side = msg.data
        else:
            if self.active:
                rospy.loginfo("PassabilityChecker: deactivated")
                self.deactivate_camera()
                self.deactivate_door_depth_and_midframe()
            self.active = False


    
    # -----------------------------
    # Conditional subscriptions
    # -----------------------------
    def activate_camera(self):
        if self.color_sub is None:
            self.color_sub = rospy.Subscriber(
                "/camera/color/image_raw",
                Image,
                self._color_cb,
                queue_size=1
            )
            self.depth_sub = rospy.Subscriber(
                "/camera/aligned_depth_to_color/image_raw",
                Image,
                self._depth_cb,
                queue_size=1
            )
            self.info_sub = rospy.Subscriber(
                "/camera/color/camera_info",
                CameraInfo,
                self._info_cb,
                queue_size=1
            )
    
    def deactivate_camera(self):
        for sub in [self.color_sub, self.depth_sub, self.info_sub]:
            if sub is not None:
                sub.unregister()

        self.color_sub = None
        self.depth_sub = None
        self.info_sub = None

        self.color_image = None
        self.depth_image = None
    
    def get_door_depth_and_midframe(self):
        self.door_depth_sub = rospy.Subscriber(
            "/glass_door_detection/door_depth",
            Float32,
            self._door_depth_cb,
            queue_size=1
        )
        self.mid_frame_x_px_sub = rospy.Subscriber(
            "/glass_door_detection/mid_frame_x_px_for_passability_check",
            Int32,
            self._mid_frame_x_px_cb,
            queue_size=1
        )
    
    def deactivate_door_depth_and_midframe(self):
        for sub in [self.door_depth_sub, self.mid_frame_x_px_sub]:
            if sub is not None:
                sub.unregister()
        
        self.door_depth_sub = None
        self.mid_frame_x_px_sub = None

        self.door_depth = None
        self.mid_frame_x_px = None

    # -----------------------------
    # Callbacks (store only!)
    # -----------------------------
    def _color_cb(self, color_msg):
        self.color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

    def _depth_cb(self, depth_msg):
        img = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        if depth_msg.encoding in ("16UC1", "mono16"):
            depth_mm = img.astype(np.uint16)
            depth_m = depth_mm.astype(np.float32) / 1000.0
        else:
            depth_m = img.astype(np.float32)

        self.depth_image = depth_m

    def _info_cb(self, info_msg: CameraInfo):
        if self.latest_info is None:
            self.latest_info = info_msg
            K = info_msg.K
            self.fx, self.fy = K[0], K[4]
            self.cx, self.cy = K[2], K[5]
            rospy.loginfo("Camera intrinsics received and stored.")
    
    def _mid_frame_x_px_cb(self, msg):
        self.mid_frame_x_px = msg.data
    
    def _door_depth_cb(self, msg):
        self.door_depth = msg.data
    
    # -----------------------------
    # Check if image data is available
    # -----------------------------

    def data_ready(self):
        return (
            self.color_image is not None and
            self.depth_image is not None and
            self.fx is not None and
            self.mid_frame_x_px is not None and
            self.door_depth is not None
        )
    
    def run(self):
        while not rospy.is_shutdown():
            if not self.active:
                self.rate.sleep()
                continue

            if not self.data_ready():
                self.rate.sleep()
                continue
            

            # ---------------------------------------
            # Convert door mid-frame pixel to X (meters)
            # ---------------------------------------
            x_center_frame = (self.mid_frame_x_px - self.cx) * self.door_depth / self.fx

            # ---------------------------------------
            # Define robot-centric traversal corridor
            # ---------------------------------------
            if self.open_side == "open_left":
                x_left_boundary   = x_center_frame - (self.robot_width + self.safety_margin_robot_width)
                x_right_boundary  = x_center_frame - self.door_frame_margin
            elif self.open_side == "open_right":
                x_left_boundary   = x_center_frame + self.door_frame_margin
                x_right_boundary  = x_center_frame + (self.robot_width + self.safety_margin_robot_width)
            else:
                x_left_boundary = None
                x_right_boundary = None

            if (x_left_boundary is not None) and (x_right_boundary is not None):
                # Center of corridor
                x_corridor_center = (x_left_boundary + x_right_boundary) / 2.0
                # Define Z extents relative to door
                z_min = self.minimum_depth_for_back_projection
                z_max = self.door_depth + self.maximum_depth_beyond_door_depth_for_back_projection
                valid_points, uv, _ = backproject_depth_to_points(
                    self.depth_image, self.fx, self.fy, self.cx, self.cy,
                    max_depth=z_max, min_depth=z_min,
                    subsample=self.subsample, roi_polygon=None
                )
                Passability_status, front_clearance, passability_view = self.passability_checker.run(
                    self.depth_image,
                    self.color_image,
                    x_left_boundary,
                    x_right_boundary,
                    valid_points,
                    uv,
                    z_max
                )
            else:
                Passability_status = False
                front_clearance = 0.0
                passability_view = None
                x_corridor_center = 0.0


            get_duration_seconds.write_text_file()

            msg = Twist()
            msg.linear.x = float(front_clearance) if front_clearance is not None else 0.0
            msg.linear.y = float(x_corridor_center) if x_corridor_center is not None else 0.0
            msg.linear.z = 1.0 if Passability_status else 0.0

            #publish results
            self.Passability_pub.publish(msg)

            # Publish visualizations
            try:
                if self.enable_visualization and passability_view is not None:
                    self.passability_view_pub.publish(self.bridge.cv2_to_imgmsg(passability_view, encoding="bgr8"))
            
            except Exception as e:
                rospy.logdebug(f"Viz publish exception: {e}")

            self.rate.sleep()

    

def main():
    rospy.init_node("check_if_corridor_is_passable")
    node = PassabilityCheckerNode()
    rospy.loginfo("Check if corridor is passable node started.")
    node.run()


if __name__ == "__main__":
    main()