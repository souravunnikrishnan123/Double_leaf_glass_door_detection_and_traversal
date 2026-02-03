#!/usr/bin/env python3
from check_if_passable import Passability_checker
from duration import get_duration_seconds
import rospy
from std_msgs.msg import String, Float32, Int32, Bool, UInt8
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
        self.x_center_frame_first_time_cam = None

       

        # Odometry
        self.start_pose_at_activation = None
        self.start_yaw_at_activation = None
        self.current_pose = None
        self.current_yaw = None

        #dynamic values
        self.dynamic_door_depth = None
        self.x_center_frame_dynamic_cam = None

        # Latest sensor data
        self.color_image = None
        self.depth_image = None
        self.latest_info = None 
        self.fx = self.fy = self.cx = self.cy = None

        # ---- Subscribers (conditional) ----
        self.color_sub = None
        self.depth_sub = None
        self.info_sub = None
        self.odom_sub = None    
        
        # -----------------------------
        # Backprojection parameters
        # -----------------------------
        self.minimum_depth_for_back_projection = rospy.get_param(f"{ns}/back_proj_params/minimum_depth", 0.05)  # meters
        self.maximum_depth_beyond_door_depth_for_back_projection = rospy.get_param(f"{ns}/back_proj_params/maximum_depth_beyond_door_depth", 1.5)  # meters
        self.subsample = rospy.get_param(f"{ns}/back_proj_params/subsample", 2)

        # local passability check parameters
        self.minimum_depth_for_back_projection_local_passability_check = rospy.get_param(f"{ns}/back_proj_params/minimum_depth_local_passability_check", 0.05)  # meters
        self.maximum_depth_for_back_projection_local_passability_check = rospy.get_param(f"{ns}/back_proj_params/maximum_depth_local_passability_check", 0.5)  # meters

        # traversal corridor parameter
        self.min_corridor_clearance_beyond_door_to_trigger_traversal_node = rospy.get_param(f"{ns}/traversal_params/min_corridor_clearance_beyond_door_to_trigger_traversal_node", 1.0)  # meters
        self.robot_width = rospy.get_param(f"{ns}/traversal_params/robot_width", 0.45)  # meters ,robot  width
        self.safety_margin_robot_width = rospy.get_param(f"{ns}/traversal_params/safety_margin_robot_width", 0.05)  # meters
        self.door_frame_margin = rospy.get_param(f"{ns}/traversal_params/door_frame_margin", 0.05)  # meters
        
        self.trigger_traversal_node = False
        self.requested_type_of_passability_check_from_traversal_node = 1 # 1 for corridor. because corridor is the default
        # ---- Always-on subscriber ---- or the trigger to activate
        self.door_state_sub = rospy.Subscriber(
            "/glass_door_detection/door_state",
            String,
            self._activate_node_cb,
            queue_size=1
        )

        # always on subscriber to trigger deactivation
        rospy.Subscriber("/door_traversal/door_traversal_done", 
                                                              Bool,   
                                                                self._deactivate_node_cb,
                                                                queue_size=1
                                                              )
        # other always on subscribers to get door depth and mid frame x px. these are low bandwidth topics, hence okay to keep them always on unlike color and depth images
        rospy.Subscriber(
                "/glass_door_detection/door_depth",
                Float32,
                self._door_depth_cb,
                queue_size=1
            )
        
        rospy.Subscriber(
                "/glass_door_detection/mid_frame_x_px_for_passability_check",
                Int32,
                self._mid_frame_x_px_cb,
                queue_size=1
            )
        rospy.Subscriber("/door_traversal/request_local_passability_check", 
                                                              UInt8,   
                                                                self._request_local_passability_check_cb,
                                                                queue_size=1
                                                              )




        #publishers.publish always because for rqt_graph stability and silence should never be a state
        #result publishers

        # to trigger traversal node
        self.trigger_traversal_node_pub = rospy.Publisher(
            "~trigger_traversal_node",
            Bool,
            queue_size=1
        )
        # passability result publisher to traversal node
        self.Passability_pub = rospy.Publisher(
            "~door_passability",
            Twist,
            queue_size=1
        )

        #to retrigger door detection node
        self.retrigger_door_detection_node_pub = rospy.Publisher(
            "~retrigger_door_detection_node",
            Bool,
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
    def _activate_node_cb(self, msg):
        if msg.data in ["open_left", "open_right"]:
            if not self.active:
                rospy.loginfo("PassabilityChecker: activated")
                #to create subscriptions
                self.activate_camera()
                self.get_odom()
            self.active = True
            self.open_side = msg.data

    def _deactivate_node_cb(self, msg):
        if msg.data:  # only deactivate on True
            if self.active:
                rospy.loginfo("PassabilityChecker: deactivated by traversal done signal")
                self.deactivate_camera()
                self.deactivate_odom()
            self.active = False
            self.trigger_traversal_node = False # to deactivate traversal node. because door is already traversed
            # Clear latest data, so that on the next activation we wait for fresh data
            self.x_center_frame_first_time_cam = None
            self.door_depth_first_time = None
            self.requested_type_of_passability_check_from_traversal_node = 1 # reset to default 1 for corridor
            self.mid_frame_x_px_from_frame_detection_node = None
            self.door_depth_from_frame_detection_node = None
    
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
        # checking self.active will make sure we get door depth and mid frame only when door state is open
        if self.mid_frame_x_px_from_frame_detection_node is None and self.active:
            self.mid_frame_x_px_from_frame_detection_node = msg.data

    def _request_local_passability_check_cb(self, msg):
        if msg.data in [0, 1, 2]: # 1 for corridor , 2 for local
            self.requested_type_of_passability_check_from_traversal_node = msg.data

    
    def _door_depth_cb(self, msg):
        if msg.data <= 0.0: # invalid depth
            return
        # only update if not set. which means take only first valid depth after activation
        # checking self.active will make sure we get door depth and mid frame only when door state is open
        if self.door_depth_from_frame_detection_node is None and self.active:
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

    def get_forward_displacement_since_start(self):
        """
        Computes displacement along initial heading.
        Robust to yaw corrections.
        """

        dx = self.current_pose[0] - self.start_pose_at_activation[0]
        dy = self.current_pose[1] - self.start_pose_at_activation[1]

        return dx * np.cos(self.start_yaw_at_activation) + dy * np.sin(self.start_yaw_at_activation)
    
    def get_lateral_displacement_since_start(self):
        """
        Computes lateral displacement perpendicular to initial heading.
        Robust to yaw corrections.
        """

        dx = self.current_pose[0] - self.start_pose_at_activation[0]
        dy = self.current_pose[1] - self.start_pose_at_activation[1]

        return -dx * np.sin(self.start_yaw_at_activation) + dy * np.cos(self.start_yaw_at_activation)
    
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

        #rotation_shift = (self.door_depth_first_time * np.tan(delta_yaw))
        rotation_shift = (self.dynamic_door_depth * np.tan(delta_yaw))
    
        
        rospy.loginfo(f"start pose is {self.start_pose_at_activation} and current pose is {self.current_pose}   Lateral displacement from start: {lateral_displacement:.3f} m   Rotation shift: {rotation_shift:.3f} m")
        
        self.x_center_frame_dynamic_cam = (
        self.x_center_frame_first_time_cam
        + lateral_displacement
        + rotation_shift
    )
  
        return self.x_center_frame_dynamic_cam
    


    def run(self):
        while not rospy.is_shutdown():
            if not self.active: # node inactive, skip processing. node will be activated when door state is open_left or open_right and 
                #de activated when traversal done signal is received
                self.rate.sleep()
                continue
            # because ros subscriber callbacks and main loop are in different threads, it is possible that during main loop execution 
            # new data could come in self.requested_type_of_passability_check_from_traversal_node. but it is okay because the new data will be used in next iteration of main loop.
            # but we cannot use self.requested_type_of_passability_check_from_traversal_node directly here because it may change during execution of this main loop iteration.
            # so we use a local variable to store its value at the start of this iteration.
            if self.trigger_traversal_node:# once traversal node is triggered, it should remain triggered until deactivation. and only makes sense to 
                # consider the value from call back if traversal node is triggered. its fine that, setting trigger_traversal_node to true doesnt mean value will be available at 
                # first iteration itself. because in that case we can wait for next iteration to get the value. because subscriber cb will only be called when new data arrives.
                # so this if check block avoid setting type_of_passability_check to default 0 when trigger_traversal_node is true and requested_type_of_passability_check_from_traversal_node is not yet set by callback.
                type_of_passability_check = self.requested_type_of_passability_check_from_traversal_node

            #need to wait for first time data to be ready and calculate x_center_frame_first_time_cam once
            if self.x_center_frame_first_time_cam is None:
                if not self.first_time_data_ready():
                    self.rate.sleep()
                    continue

                # Convert door mid-frame pixel to X (meters)
                self.x_center_frame_first_time_cam = (self.mid_frame_x_px_from_frame_detection_node - self.cx) * self.door_depth_from_frame_detection_node / self.fx
                self.door_depth_first_time = self.door_depth_from_frame_detection_node
                self.z_center_frame_first_time_cam = self.door_depth_first_time
                # x direction in camera frame is y direction in robot base frame
                # z direction in camera frame is x direction in robot base frame
                self.y_center_frame_first_time_robot_base = ( self.z_center_frame_first_time_cam * np.sin(self.start_yaw_at_activation) + self.x_center_frame_first_time_cam * np.cos(self.start_yaw_at_activation))

                type_of_passability_check = 1  # first do corridor passability check
                rospy.loginfo(f"Initial x_center_frame in camera frame: {self.x_center_frame_first_time_cam:.3f} m, door_depth: {self.door_depth_first_time:.3f} m, y_center_frame in robot base frame: {self.y_center_frame_first_time_robot_base:.3f} m")
            # ---------------------------------------
            # Dynamic update of center frame X and door_depth based on odometry
            # ---------------------------------------
            data_valid = True
            # ---- INITIALIZE ALL PUBLISHER OUTPUTS ----
            front_clearance = None
            x_corridor_center_cam = None
            y_corridor_center_start_yaw_frame = None
            door_depth = None
            corridoor_Passability_status = False  # not used in local check
            passability_view = None


            self.dynamic_door_depth = self.door_depth_first_time - self.get_forward_displacement_since_start()
            self.x_center_frame_dynamic_cam = self.calculate_dynamic_x_center_frame()
            self.y_center_frame_current_start_yaw_frame = self.y_center_frame_first_time_robot_base + self.get_lateral_displacement_since_start()
            door_depth = self.dynamic_door_depth

            # ---------------------------------------
            # Define robot-centric traversal corridor
            # ---------------------------------------
            if self.open_side == "open_left":
                x_left_boundary_cam   = self.x_center_frame_dynamic_cam - (self.robot_width + self.safety_margin_robot_width)
                x_right_boundary_cam  = self.x_center_frame_dynamic_cam - self.door_frame_margin
                x_corridor_center_cam = (x_left_boundary_cam + x_right_boundary_cam) / 2.0 

                y_left_boundary_start_yaw_frame = self.y_center_frame_current_start_yaw_frame - (self.robot_width + self.safety_margin_robot_width)
                y_right_boundary_start_yaw_frame = self.y_center_frame_current_start_yaw_frame - self.door_frame_margin
                y_corridor_center_start_yaw_frame = (y_left_boundary_start_yaw_frame + y_right_boundary_start_yaw_frame) / 2.0

            elif self.open_side == "open_right":
                x_left_boundary_cam   = self.x_center_frame_dynamic_cam + self.door_frame_margin
                x_right_boundary_cam  = self.x_center_frame_dynamic_cam + (self.robot_width + self.safety_margin_robot_width)
                x_corridor_center_cam = (x_left_boundary_cam + x_right_boundary_cam) / 2.0 

                y_left_boundary_start_yaw_frame = self.y_center_frame_current_start_yaw_frame + self.door_frame_margin
                y_right_boundary_start_yaw_frame = self.y_center_frame_current_start_yaw_frame + (self.robot_width + self.safety_margin_robot_width)
                y_corridor_center_start_yaw_frame = (y_left_boundary_start_yaw_frame + y_right_boundary_start_yaw_frame) / 2.0
            
            else:# unknown state. this is not possible but just in case
                rospy.logwarn("Unknown door open side state.")
                data_valid = False

            
            
            
            if data_valid and type_of_passability_check == 1: # corridor passability check
                rospy.loginfo("Performing corridor passability check.")
                # Define Z extents relative to door
                z_min = self.minimum_depth_for_back_projection
                z_max = self.dynamic_door_depth + self.maximum_depth_beyond_door_depth_for_back_projection

                x_left_limit = x_left_boundary_cam
                x_right_limit = x_right_boundary_cam

                reference_depth = self.dynamic_door_depth

            
            elif data_valid and type_of_passability_check == 2: # local passability check
                rospy.loginfo("Performing local passability check.")
                z_min = self.minimum_depth_for_back_projection_local_passability_check  # closer range for local passability
                z_max = self.maximum_depth_for_back_projection_local_passability_check  # only up to door

                x_left_limit = - (self.robot_width / 2.0 + self.safety_margin_robot_width)
                x_right_limit = (self.robot_width / 2.0 + self.safety_margin_robot_width)

                reference_depth = self.minimum_depth_for_back_projection_local_passability_check + self.maximum_depth_for_back_projection_local_passability_check / 2.0  # mid depth for local passability

            
            else:
                data_valid = False

            if data_valid:
                valid_points, uv, _ = backproject_depth_to_points(
                    self.depth_image, self.fx, self.fy, self.cx, self.cy,
                    max_depth=z_max, min_depth=z_min,
                    subsample=self.subsample, roi_polygon=None
                )
                # front clearance is none means, no points in image at all. hence passability cannot be determined
                front_clearance, passability_view = self.passability_checker.run(
                    self.depth_image,
                    self.color_image,
                    x_left_limit,
                    x_right_limit,
                    valid_points,
                    uv,
                    z_max
                )


              
            
            if type_of_passability_check == 1: # corridor passability check
                #executes only once
                if front_clearance is not None and self.trigger_traversal_node is False:
                    corridoor_Passability_status = front_clearance > self.dynamic_door_depth + self.min_corridor_clearance_beyond_door_to_trigger_traversal_node
                    if corridoor_Passability_status:
                        # trigger traversal node if it is passable for the first time.
                        #but in later if instantanous passability goes false, do not reset the trigger
                        #trigger traverse node once we detect passability. but during the traversal passability
                        # may be false, we do not want to retrigger the traversal node.
                        self.trigger_traversal_node = True

                #in local passability check we do not trigger traversal node, because it is already triggered in corridor passability check which will execute first

                             

            get_duration_seconds.write_text_file()


            self.trigger_traversal_node_pub.publish(Bool(data=self.trigger_traversal_node))

            msg = Twist()
            msg.linear.x = float(front_clearance) if front_clearance is not None else float('nan')
            msg.linear.y = float(y_corridor_center_start_yaw_frame) if y_corridor_center_start_yaw_frame is not None else float('nan')
            msg.linear.z = type_of_passability_check # 1 for corridor , 2 for local
            msg.angular.x = float(x_corridor_center_cam) if x_corridor_center_cam is not None else float('nan')
            msg.angular.y = float(door_depth) if door_depth is not None else float('nan')

            

            #publish results
            self.Passability_pub.publish(msg)

            # Publish visualizations
            try:
                if self.enable_visualization and passability_view is not None:
                    # Convert boundaries to pixel coordinates for visualization    
                    x_left_boundary_uv = int((x_left_limit * self.fx) / reference_depth + self.cx)
                    x_right_boundary_uv = int((x_right_limit * self.fx) / reference_depth + self.cx)
                    cv2.line(passability_view, (x_left_boundary_uv, 0), (x_left_boundary_uv, passability_view.shape[0]), (0, 165, 255), 4)  # orange
                    cv2.line(passability_view, (x_right_boundary_uv, 0), (x_right_boundary_uv, passability_view.shape[0]), (0, 0, 255), 4)  # red

                    
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