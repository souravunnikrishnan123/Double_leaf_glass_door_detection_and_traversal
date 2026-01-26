#!/usr/bin/env python3
from check_if_passable import Passability_checker
from duration import get_duration_seconds
import rospy
from std_msgs.msg import String, Float32, Int32, Bool
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np  
from geometry_msgs.msg import Twist
from processing_classes import backproject_depth_to_points
import cv2

class PassabilityCheckerNode:
    def __init__(self):
        self.enable_visualization = rospy.get_param("~enable_visualization", False)
        self.color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        self.bridge = CvBridge()
        ns = "~passabilility_check"
        # ---- State ----
        self.active = False
        self.open_side = None
        self.mid_frame_x_px_from_frame_detection_node = None
        self.door_depth_from_frame_detection_node = None
        self.x_center_frame_first_time = None

        # Odometry
        self.start_pose_at_activation = None
        self.start_yaw_at_activation = None
        self.current_pose = None
        self.current_yaw = None

        #dynamic values
        self.dynamic_door_depth = None
        self.x_center_frame_dynamic = None

        # Latest sensor data
        self.color_image = None
        self.depth_image = None
        self.latest_info = None 
        self.fx = self.fy = self.cx = self.cy = None

        # ---- Subscribers (conditional) ----
        self.color_sub = None
        self.depth_sub = None
        self.info_sub = None
        self.mid_frame_x_px_from_frame_detection_node_sub = None
        self.door_depth_from_frame_detection_node_sub = None
        self.odom_sub = None    
        
        # -----------------------------
        # Backprojection parameters
        # -----------------------------
        self.minimum_depth_for_back_projection = rospy.get_param(f"{ns}/back_proj_params/minimum_depth", 0.05)  # meters
        self.maximum_depth_beyond_door_depth_for_back_projection = rospy.get_param(f"{ns}/back_proj_params/maximum_depth_beyond_door_depth", 1.5)  # meters
        self.subsample = rospy.get_param(f"{ns}/back_proj_params/subsample", 2)

        self.robot_width = rospy.get_param(f"{ns}/traversal_params/robot_width", 0.45)  # meters ,robot  width
        self.safety_margin_robot_width = rospy.get_param(f"{ns}/traversal_params/safety_margin_robot_width", 0.05)  # meters
        self.door_frame_margin = rospy.get_param(f"{ns}/traversal_params/door_frame_margin", 0.05)  # meters
        
        self.trigger_traversal_node = False
        # ---- Always-on subscriber ----
        self.door_state_sub = rospy.Subscriber(
            "/glass_door_detection/door_state",
            String,
            self._door_state_cb,
            queue_size=1
        )


        #publishers.publish always because for rqt_graph stability and silence should never be a state
        #result publishers
        self.trigger_traversal_node_pub = rospy.Publisher(
            "~trigger_traversal_node",
            Bool,
            queue_size=1
        )

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
                self.get_odom()
            self.active = True
            self.open_side = msg.data
        else:
            if self.active:
                rospy.loginfo("PassabilityChecker: deactivated")
                self.deactivate_camera()
                self.deactivate_door_depth_and_midframe()
                self.deactivate_odom()
            self.active = False
            self.trigger_traversal_node = False
            self.x_center_frame_first_time = None
            self.door_depth_first_time = None

    
    # -----------------------------
    # Conditional subscriptions
    # -----------------------------
    def activate_camera(self):
        if self.color_sub is None:
            self.color_sub = rospy.Subscriber(
                self.color_topic,
                Image,
                self._color_cb,
                queue_size=1
            )
        if self.depth_sub is None:
            self.depth_sub = rospy.Subscriber(
                self.depth_topic,
                Image,
                self._depth_cb,
                queue_size=1
            )
        if self.info_sub is None:
            self.info_sub = rospy.Subscriber(
                self.camera_info_topic,
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


        # Clear latest data, so that on the next activation we wait for fresh data
        self.color_image = None
        self.depth_image = None
        self.latest_info = None
    
    def get_door_depth_and_midframe(self):
        if self.door_depth_from_frame_detection_node_sub is None:
            self.door_depth_from_frame_detection_node_sub = rospy.Subscriber(
                "/glass_door_detection/door_depth",
                Float32,
                self._door_depth_cb,
                queue_size=1
            )
        if self.mid_frame_x_px_from_frame_detection_node_sub is None:
            self.mid_frame_x_px_from_frame_detection_node_sub = rospy.Subscriber(
                "/glass_door_detection/mid_frame_x_px_for_passability_check",
                Int32,
                self._mid_frame_x_px_cb,
                queue_size=1
            )
    
    def deactivate_door_depth_and_midframe(self):
        for sub in [self.door_depth_from_frame_detection_node_sub, self.mid_frame_x_px_from_frame_detection_node_sub]:
            if sub is not None:
                sub.unregister()
        
        self.door_depth_from_frame_detection_node_sub = None
        self.mid_frame_x_px_from_frame_detection_node_sub = None

        # Clear latest data, so that on the next activation we wait for fresh data
        self.door_depth_from_frame_detection_node = None
        self.mid_frame_x_px_from_frame_detection_node = None

    def get_odom(self):
        if self.odom_sub is None:
            self.odom_sub = rospy.Subscriber(
                "/odom_bridge_output",
                Odometry,
                self._odom_callback,
                queue_size=1
            )

    def deactivate_odom(self):
        if self.odom_sub is not None:
            self.odom_sub.unregister()
            self.odom_sub = None
        # Clear latest data, so that on the next activation we wait for fresh data
        self.start_pose_at_activation = None
        self.start_yaw_at_activation = None
        self.current_pose = None
        self.current_yaw = None

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
        # storing the camera intrinsics only once. it will not change during runtime
        if self.latest_info is None:
            self.latest_info = info_msg
            K = info_msg.K
            self.fx, self.fy = K[0], K[4]
            self.cx, self.cy = K[2], K[5]
            rospy.loginfo("Camera intrinsics received and stored.")
    
    def _mid_frame_x_px_cb(self, msg):
        if msg.data <= 0: # invalid pixel
            return
        # only update if not set. which means take only first valid pixel after activation
        if self.mid_frame_x_px_from_frame_detection_node is None:
            self.mid_frame_x_px_from_frame_detection_node = msg.data
    
    def _door_depth_cb(self, msg):
        if msg.data <= 0.0: # invalid depth
            return
        # only update if not set. which means take only first valid depth after activation
        if self.door_depth_from_frame_detection_node is None:
            self.door_depth_from_frame_detection_node = msg.data

    def _odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation


        self.current_pose = (p.x, p.y)

        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)
        self.current_yaw = yaw
        # store the start pose and yaw at activation.run only once
        if self.start_pose_at_activation is None:
            self.start_pose_at_activation = (p.x, p.y)
            self.start_yaw_at_activation = yaw  
        

    
    # -----------------------------
    # Check if image data is available
    # -----------------------------

    def first_time_data_ready(self):
        return (
            self.color_image is not None and
            self.depth_image is not None and
            self.fx is not None and
            self.mid_frame_x_px_from_frame_detection_node is not None and
            self.door_depth_from_frame_detection_node is not None and
            self.start_pose_at_activation is not None and
            self.start_yaw_at_activation is not None
        )

    def get_forward_displacement(self):
        """
        Computes displacement along initial heading.
        Robust to yaw corrections.
        """

        dx = self.current_pose[0] - self.start_pose_at_activation[0]
        dy = self.current_pose[1] - self.start_pose_at_activation[1]

        return dx * np.cos(self.start_yaw_at_activation) + dy * np.sin(self.start_yaw_at_activation)
    
    def calculate_dynamic_x_center_frame(self):
        """
        Calculates dynamic x_center_frame based on odometry.
        """

        dx = self.current_pose[0] - self.start_pose_at_activation[0]
        dy = self.current_pose[1] - self.start_pose_at_activation[1]
        
        #translation effect
        # Project displacement onto perpendicular to initial heading
        lateral_displacement = -dx * np.sin(self.start_yaw_at_activation) + dy * np.cos(self.start_yaw_at_activation)
        
        #rotation effect
        # rotation effect
        delta_yaw = self.current_yaw - self.start_yaw_at_activation
        delta_yaw = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))

        rotation_shift = (self.door_depth_first_time * np.tan(delta_yaw))
        
        rospy.loginfo(f"start pose is {self.start_pose_at_activation} and current pose is {self.current_pose}   Lateral displacement from start: {lateral_displacement:.3f} m   Rotation shift: {rotation_shift:.3f} m")
        self.x_center_frame_dynamic = (
        self.x_center_frame_first_time
        + lateral_displacement
        + rotation_shift
    )
        
        return self.x_center_frame_dynamic
    
    def run(self):
        while not rospy.is_shutdown():
            if not self.active:
                self.rate.sleep()
                continue
            
            #need to wait for first time data to be ready and calculate x_center_frame_first_time once
            if self.x_center_frame_first_time is None:
                if not self.first_time_data_ready():
                    self.rate.sleep()
                    continue

                # Convert door mid-frame pixel to X (meters)
                self.x_center_frame_first_time = (self.mid_frame_x_px_from_frame_detection_node - self.cx) * self.door_depth_from_frame_detection_node / self.fx
                self.door_depth_first_time = self.door_depth_from_frame_detection_node
            
            # ---------------------------------------
            # Dynamic update of center frame X and door_depth based on odometry
            # ---------------------------------------

            self.dynamic_door_depth = self.door_depth_first_time - self.get_forward_displacement()
            self.x_center_frame_dynamic = self.calculate_dynamic_x_center_frame()

            # ---------------------------------------
            # Define robot-centric traversal corridor
            # ---------------------------------------
            if self.open_side == "open_left":
                x_left_boundary   = self.x_center_frame_dynamic - (self.robot_width + self.safety_margin_robot_width)
                x_right_boundary  = self.x_center_frame_dynamic - self.door_frame_margin
            elif self.open_side == "open_right":
                x_left_boundary   = self.x_center_frame_dynamic + self.door_frame_margin
                x_right_boundary  = self.x_center_frame_dynamic + (self.robot_width + self.safety_margin_robot_width)
            else:# unknown state. this is not possible but just in case
                x_left_boundary = None
                x_right_boundary = None

            if (x_left_boundary is not None) and (x_right_boundary is not None):
                # Center of corridor
                x_corridor_center = (x_left_boundary + x_right_boundary) / 2.0
                # Define Z extents relative to door
                z_min = self.minimum_depth_for_back_projection
                z_max = self.dynamic_door_depth + self.maximum_depth_beyond_door_depth_for_back_projection
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

                x_left_boundary_uv = int((x_left_boundary * self.fx) / self.dynamic_door_depth + self.cx)
                x_right_boundary_uv = int((x_right_boundary * self.fx) / self.dynamic_door_depth + self.cx)
                x_center_frame_dynamic_uv = int((self.x_center_frame_dynamic * self.fx) / self.dynamic_door_depth + self.cx)    

                # trigger traversal node if it is passable for the first time.
                #but in later if instantanous passability goes false, do not reset the trigger
                #trigger traverse node once we detect passability. but during the traversal passability
                # may be false, we do not want to retrigger the traversal node.
                if Passability_status:
                    self.trigger_traversal_node = True
            else:
                Passability_status = False
                front_clearance = 0.0
                passability_view = None
                x_corridor_center = 0.0


            get_duration_seconds.write_text_file()


            self.trigger_traversal_node_pub.publish(Bool(data=self.trigger_traversal_node))

            msg = Twist()
            msg.linear.x = float(front_clearance) if front_clearance is not None else 0.0
            msg.linear.y = float(x_corridor_center) if x_corridor_center is not None else 0.0
            msg.linear.z = 1.0 if Passability_status else 0.0


            #publish results
            self.Passability_pub.publish(msg)

            # Publish visualizations
            try:
                if self.enable_visualization and passability_view is not None:
                    cv2.line(passability_view, (x_left_boundary_uv, 0), (x_left_boundary_uv, passability_view.shape[0]), (0, 165, 255), 4)  # orange
                    cv2.line(passability_view, (x_right_boundary_uv, 0), (x_right_boundary_uv, passability_view.shape[0]), (0, 0, 255), 4)  # red
                    cv2.line(passability_view, (x_center_frame_dynamic_uv, 0), (x_center_frame_dynamic_uv, passability_view.shape[0]), (255, 0, 0), 4)  # blue

                    
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