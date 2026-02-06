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
        self.define_new_corridor_req_from_frame_detection = False
        self.door_status_from_frame_detection_node = None
        self.mid_frame_x_px_from_frame_detection_node = None
        self.door_depth_from_frame_detection_node = None
        self.x_corridor_center_cam_first_time = None
        # Additional state for traversal-requested corridor definition and publishing
        self.define_new_corridor_req_from_traversal = False
        self.corridor_middle_point_depth_from_traversal = None
        self.virtual_corridor_definition_finished = False
        self.clearance_needed_beyond_mid_corridor = None
        self.corridor_middle_point_depth_first_time = None

       

        # Odometry
        self.start_pose_at_node_activation = None
        self.start_yaw_at_node_activation = None
        self.current_pose = None
        self.current_yaw = None

        #dynamic values
        self.corridor_middle_point_depth_dynamic = None
        self.x_corridor_center_cam_dynamic = None

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
        self.maximum_depth_beyond_corridor_center_point_for_back_projection = rospy.get_param(f"{ns}/back_proj_params/maximum_depth_beyond_corridor_center_point", 1.5)  # meters
        self.subsample = rospy.get_param(f"{ns}/back_proj_params/subsample", 2)

        # local passability check parameters
        self.minimum_depth_for_back_projection_local_passability_check = rospy.get_param(f"{ns}/back_proj_params/minimum_depth_local_passability_check", 0.05)  # meters
        self.maximum_depth_for_back_projection_local_passability_check = rospy.get_param(f"{ns}/back_proj_params/maximum_depth_local_passability_check", 0.5)  # meters
        #pass ability decision parameters
    
        # traversal corridor parameter
        self.min_corridor_clearance_beyond_door_to_trigger_traversal_node = rospy.get_param(f"{ns}/traversal_params/min_corridor_clearance_beyond_door_to_trigger_traversal_node", 1.0)  # meters
        self.robot_width = rospy.get_param(f"{ns}/traversal_params/robot_width", 0.45)  # meters ,robot  width
        self.safety_margin_robot_width = rospy.get_param(f"{ns}/traversal_params/safety_margin_robot_width", 0.05)  # meters
        self.door_frame_margin = rospy.get_param(f"{ns}/traversal_params/door_frame_margin", 0.05)  # meters
        
        self.trigger_traversal_node = False
        self.requested_type_of_passability_check_from_traversal_node = 1 # 1 for corridor. because corridor is the default
        
        # virtual corridor search
        self.slide_step_x_direction = rospy.get_param(f"{ns}/virtual_corridor_search/slide_step_x_direction", 0.1)  # meters
        
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
        
        rospy.Subscriber("/door_traversal/request_local_passability_check", 
                                                              UInt8,   
                                                                self._request_local_passability_check_cb,
                                                                queue_size=1
                                                              )
        
        rospy.Subscriber("/door_traversal/request_new_corridor_definition", 
                                                              Twist,   
                                                                self._request_new_corridor_definition_cb,
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




        #publishers.publish always because for rqt_graph stability and silence should never be a state
        #result publishers

        # to trigger traversal node
        self.trigger_traversal_node_pub = rospy.Publisher(
            "~trigger_traversal_node",
            Bool,
            queue_size=1
        )

        self.virtual_corridor_definition_finished_pub = rospy.Publisher(
            "~virtual_corridor_definition_finished",
            Bool,
            queue_size=1
        )
        # passability result publisher to traversal node
        self.Passability_pub = rospy.Publisher(
            "~door_passability",
            Twist,
            queue_size=1
        )

        #to retrigger door detection node. this is send once the complete traversal is done
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
        if msg.data in ["open_left", "open_right", "No_door_plane_detected", "no_frame_detected"]: # one time activation on these states
            if not self.active:
                rospy.loginfo("PassabilityChecker: activated")
                self.active = True
                #to create subscriptions
                self.activate_camera()
                self.get_odom()
                # store the door open status only once at activation
                self.door_status_from_frame_detection_node = msg.data 
                self.define_new_corridor_req_from_frame_detection = True  # to define new corridor on next run
                
            

    def _deactivate_node_cb(self, msg):
        if msg.data:  # only deactivate on True
            if self.active:
                rospy.loginfo("PassabilityChecker: deactivated by traversal done signal")
                self.deactivate_camera()
                self.deactivate_odom()
            self.active = False
            self.define_new_corridor_req_from_frame_detection = False
            self.trigger_traversal_node = False # to deactivate traversal node. because door is already traversed
            # Clear latest data, so that on the next activation we wait for fresh data
            self.x_corridor_center_cam_first_time = None
            self.door_status_from_frame_detection_node = None
            self.requested_type_of_passability_check_from_traversal_node = 1 # reset to default 1 for corridor
            self.mid_frame_x_px_from_frame_detection_node = None
            self.door_depth_from_frame_detection_node = None
            self.virtual_corridor_definition_finished = False
            self.retrigger_door_detection_node_pub.publish(True) # to retrigger door detection node for next detection and traversal cycle


            
    
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
        self.start_pose_at_node_activation = None
        self.start_yaw_at_node_activation = None
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
    
    def _request_new_corridor_definition_cb(self, msg):
        if msg.linear.x:  # only activate on True
            if self.active:
                rospy.loginfo("PassabilityChecker: New corridor definition requested by traversal node")
                self.define_new_corridor_req_from_traversal = True  # to define new corridor on next run
                self.corridor_middle_point_depth_from_traversal = msg.linear.y  # to define new corridor based on this depth


    
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
        if self.start_pose_at_node_activation is None:
            self.start_pose_at_node_activation = (p.x, p.y)
            self.start_yaw_at_node_activation = yaw
        

    
    # -----------------------------
    # Check if image data is available
    # -----------------------------

    def first_time_data_ready_after_node_activation(self):
        return (
            self.color_image is not None and
            self.depth_image is not None and
            self.fx is not None and
            self.door_status_from_frame_detection_node is not None and
            self.door_depth_from_frame_detection_node is not None and
            self.start_pose_at_node_activation is not None and
            self.start_yaw_at_node_activation is not None
        )

    def get_forward_displacement_since_start(self):
        """
        Computes displacement along initial heading.
        Robust to yaw corrections.
        """

        dx = self.current_pose[0] - self.start_pose_at_corridor_definition[0]
        dy = self.current_pose[1] - self.start_pose_at_corridor_definition[1]

        return dx * np.cos(self.start_yaw_at_corridor_definition) + dy * np.sin(self.start_yaw_at_corridor_definition)
    
    def get_lateral_displacement_since_start(self):
        """
        Computes lateral displacement perpendicular to initial heading.
        Robust to yaw corrections.
        """

        dx = self.current_pose[0] - self.start_pose_at_corridor_definition[0]
        dy = self.current_pose[1] - self.start_pose_at_corridor_definition[1]

        return -dx * np.sin(self.start_yaw_at_corridor_definition) + dy * np.cos(self.start_yaw_at_corridor_definition)
    
    def calculate_dynamic_x_corridor_center_in_cameraframe(self):
        """
        Calculates dynamic x_center_frame based on odometry.
        """

        dx = self.current_pose[0] - self.start_pose_at_corridor_definition[0]
        dy = self.current_pose[1] - self.start_pose_at_corridor_definition[1]
        
        #translation effect
        # Project displacement onto perpendicular to initial heading
        lateral_displacement = -dx * np.sin(self.start_yaw_at_corridor_definition) + dy * np.cos(self.start_yaw_at_corridor_definition)

        #rotation effect
        # rotation effect
        delta_yaw = self.current_yaw - self.start_yaw_at_corridor_definition
        delta_yaw = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))

        rotation_shift = (self.corridor_middle_point_depth_dynamic * np.tan(delta_yaw))
    
        
        rospy.loginfo(f"start pose is {self.start_pose_at_corridor_definition} and current pose is {self.current_pose}   Lateral displacement from start: {lateral_displacement:.3f} m   Rotation shift: {rotation_shift:.3f} m")
        
        
        return (
        lateral_displacement
        + rotation_shift
    )

    

    def find_best_corridor_center_based_on_final_points(self, final_points, z_max, min_clearance):
        # find best corridor center based on final points obtained from passability checker
        # final points are in camera frame
        x_min = np.min(final_points[:, 0])
        x_max = np.max(final_points[:, 0])
        x_centers = np.arange(x_min, x_max, self.slide_step_x_direction)

        valid_corridors = []

        for x_c in x_centers:
            x_left  = x_c - (self.robot_width / 2.0 + self.safety_margin_robot_width)
            x_right = x_c + (self.robot_width / 2.0 + self.safety_margin_robot_width)

            corridor_mask = (
                (final_points[:, 0] > x_left) &
                (final_points[:, 0] < x_right)
            )

            corridor_pts = final_points[corridor_mask]

            # no points in corridor means free of obstacles
            if len(corridor_pts) == 0:
                front_clearance = z_max
            else:
                front_clearance = np.min(corridor_pts[:, 2])

            if front_clearance >= min_clearance:
                valid_corridors.append({
                    "x_center": x_c,
                    "clearance": front_clearance,
                })

        def corridor_cost(c):
            lateral_cost   = abs(c["x_center"] - 0.0)  # prefer corridors near center of robot fov (x=0 in camera frame)

            return lateral_cost
        
        if len(valid_corridors) == 0:
            rospy.logwarn("Cannot find any valid virtual corridor for passability check.")
            # there is no virtual corridor available. consider virtual corridor is directly in front of robot
            return 0.0
        
        best_corridor = min(valid_corridors, key=corridor_cost) # it iterates over the list of dicts and passes each dict to corridor_cost

        rospy.loginfo(f"Virtual corridor center defined at x: {best_corridor['x_center']:.3f} m in camera frame")
        return best_corridor['x_center']
    

    def find_a_virtual_corridor(self, z_min, z_max, reference_depth, min_clearance):
        rospy.loginfo("Finding virtual corridor for passability check.")
       

        # run passability check for full width of image to find a virtual corridor
        x_left_limit = (0 - self.cx) * reference_depth / self.fx
        x_right_limit = (self.color_image.shape[1] - self.cx) * reference_depth / self.fx

        valid_points, uv, _ = backproject_depth_to_points(
            self.depth_image, self.fx, self.fy, self.cx, self.cy,
            max_depth=z_max, min_depth=z_min,
            subsample=self.subsample, roi_polygon=None
        )
        
        _, _, final_points = self.passability_checker.run(
            self.depth_image,
            self.color_image,
            x_left_limit,
            x_right_limit,
            valid_points,
            uv,
            z_max
        )

        # donot use the value of front clearance directly because corridor is not defined yet.  this front clerance is for full width of image
        # we need to find a virtual corridor center based on final points obtained from passability checker
        if final_points is not None and len(final_points) > 0:
            x_virtual_corridor_center_cam = self.find_best_corridor_center_based_on_final_points(final_points, z_max, min_clearance)

        else: # in this case, consider image center as virtual corridor center
            rospy.logwarn("Cannot find any valid points for virtual corridor definition. consider virtual corridor is directly in front of robot.")
            x_virtual_corridor_center_cam = 0.0
        
        return x_virtual_corridor_center_cam



    def run(self):
        while not rospy.is_shutdown():
            if not self.active: # node inactive, skip processing. node will be activated when door state is open_left or open_right and 
                #de activated when traversal done signal is received
                self.rate.sleep()
                continue
            
            # this is to wait until first time data is ready after activation
            if not self.first_time_data_ready_after_node_activation(): # if data not ready, wait
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

            

            # corridor definition stage. run only once when requested by frame detection node or traversal node. for frame detection node, it will be requested when door state is open. for traversal node, it can be requested anytime after activation when traversal node needs to redefine corridor based on updated robot pose
            if self.define_new_corridor_req_from_frame_detection:
                self.start_yaw_at_corridor_definition = self.current_yaw  # store the yaw at corridor definition time
                self.start_pose_at_corridor_definition = self.current_pose  # store the pose at corridor definition time
                clearance_needed_beyond_mid_corridor = self.min_corridor_clearance_beyond_door_to_trigger_traversal_node #value is considering robot should pass through the door. this info is not from traversal node but from the parameter server because it is related to the door and corridor, not related to the robot pose which is updated in real time. so it should be a fixed value instead of dynamic value updated from traversal node
                
                if self.door_status_from_frame_detection_node in ["open_left", "open_right"]: # fixed corridor. and corridor is defined only when door is open
                    # find the corridor based on frame detection node data(x_center_frame_cam_first_time)
                    self.corridor_middle_point_depth_first_time = self.door_depth_from_frame_detection_node
                    # Convert door mid-frame pixel to X (meters)
                    x_center_frame_cam_first_time = (self.mid_frame_x_px_from_frame_detection_node - self.cx) * self.corridor_middle_point_depth_first_time / self.fx
                    # x direction in camera frame is y direction in robot base frame
                    # z direction in camera frame is x direction in robot base frame
                    x_center_frame_start_yaw_frame_first_time = ( self.corridor_middle_point_depth_first_time * np.sin(self.start_yaw_at_corridor_definition) + x_center_frame_cam_first_time * np.cos(self.start_yaw_at_corridor_definition))

                    # ---------------------------------------
                    # Define robot-centric traversal corridor for corridor passability check
                    # ---------------------------------------
                    if self.door_status_from_frame_detection_node == "open_left":
                        x_left_limit_corridor_cam_first_time   = x_center_frame_cam_first_time - (self.door_frame_margin + self.robot_width + self.safety_margin_robot_width)
                        x_right_limit_corridor_cam_first_time  = x_center_frame_cam_first_time - self.door_frame_margin
                        self.x_corridor_center_cam_first_time = (x_left_limit_corridor_cam_first_time + x_right_limit_corridor_cam_first_time) / 2.0 

                        x_left_boundary_start_yaw_frame_first_time = x_center_frame_start_yaw_frame_first_time - (self.door_frame_margin + self.robot_width + self.safety_margin_robot_width)
                        x_right_boundary_start_yaw_frame_first_time = x_center_frame_start_yaw_frame_first_time - self.door_frame_margin
                        self.x_corridor_center_start_yaw_frame_first_time = (x_left_boundary_start_yaw_frame_first_time + x_right_boundary_start_yaw_frame_first_time) / 2.0

                    else: # means "open_right":
                        x_left_limit_corridor_cam_first_time   = x_center_frame_cam_first_time + self.door_frame_margin
                        x_right_limit_corridor_cam_first_time  = x_center_frame_cam_first_time + (self.door_frame_margin + self.robot_width + self.safety_margin_robot_width)
                        self.x_corridor_center_cam_first_time = (x_left_limit_corridor_cam_first_time + x_right_limit_corridor_cam_first_time) / 2.0 
                        rospy.loginfo(f"Door frame based corridor definition. x_left_limit_corridor_cam_first_time: {x_left_limit_corridor_cam_first_time:.3f} m, x_right_limit_corridor_cam_first_time: {x_right_limit_corridor_cam_first_time:.3f} m, x_corridor_center_cam_first_time: {self.x_corridor_center_cam_first_time:.3f} m")
                        
                        x_left_boundary_start_yaw_frame_first_time = x_center_frame_start_yaw_frame_first_time + self.door_frame_margin
                        x_right_boundary_start_yaw_frame_first_time = x_center_frame_start_yaw_frame_first_time + (self.door_frame_margin + self.robot_width + self.safety_margin_robot_width)
                        self.x_corridor_center_start_yaw_frame_first_time = (x_left_boundary_start_yaw_frame_first_time + x_right_boundary_start_yaw_frame_first_time) / 2.0
                    
                    rospy.loginfo(f"Initial x corridor center in camera frame: {self.x_corridor_center_cam_first_time:.3f} m, corridor mid depth: {self.corridor_middle_point_depth_first_time:.3f} m, Initial x corridor center in world frame: {self.x_corridor_center_start_yaw_frame_first_time:.3f} m")
                
                elif self.door_status_from_frame_detection_node in ["No_door_plane_detected", "no_frame_detected"]:
                    self.corridor_middle_point_depth_first_time = self.door_depth_from_frame_detection_node# only useful information from frame detection node in this case. they cannot provide mid frame x px because no door frame detected
                    rospy.logwarn("Door state is No_door_plane_detected or no_frame_detected, cannot define fixed corridor. need to find a virtual fixed corridor.")
                    # virtual corridor finding. corridor is not fixed yet, can be anywhere in front of robot. no need of temporal smoothing for virtual corridor can be changed not a fixed value
                    z_min = self.minimum_depth_for_back_projection
                    z_max = self.corridor_middle_point_depth_first_time + self.maximum_depth_beyond_corridor_center_point_for_back_projection
                    reference_depth = self.corridor_middle_point_depth_first_time
                    min_clearance =  self.corridor_middle_point_depth_first_time + self.min_corridor_clearance_beyond_door_to_trigger_traversal_node
                    self.x_corridor_center_cam_first_time = self.find_a_virtual_corridor(z_min, z_max, reference_depth, min_clearance)
                    # once virtual corridor is found, we can consider corridor defined for next iterations
                    # calculate x coordinate of corridor center in robot base frame at start yaw frame
                    
                    self.x_corridor_center_start_yaw_frame_first_time =  ( reference_depth * np.sin(self.start_yaw_at_corridor_definition) + self.x_corridor_center_cam_first_time * np.cos(self.start_yaw_at_corridor_definition))
                
                else:
                    rospy.logwarn("Unknown door open side state during corridor definition.")
                    self.rate.sleep()
                    continue
                
                
                type_of_passability_check = 1  # after corridor definiton, first do corridor passability check
                self.define_new_corridor_req_from_frame_detection = False  # corridor defined, no need to define again until next activation or next corridor definition request
            
            
            elif self.define_new_corridor_req_from_traversal: #need to find a virtual corridor because, there is no fixed corridor. fixed corridor will be defined only when door is open left or open right (i.e frame detection has data)
                self.start_yaw_at_corridor_definition = self.current_yaw  # store the yaw at corridor definition time
                self.start_pose_at_corridor_definition = self.current_pose  # store the pose at corridor definition time
                clearance_needed_beyond_mid_corridor = self.corridor_middle_point_depth_first_time # by default half of needed corridor length is received as door_middle_depth from traversal node
                # similar to no frame detected case above, we need to find a virtual corridor because new corridor is requested by traversal node
                rospy.logwarn(" traversal node has requested for new corridor definition.  need to find a virtual fixed corridor.")
                # virtual corridor finding. corridor is not fixed yet, can be anywhere in front of robot. no need of temporal smoothing for virtual corridor can be changed not a fixed value
                z_min = self.minimum_depth_for_back_projection
                z_max = self.corridor_middle_point_depth_first_time + self.maximum_depth_beyond_corridor_center_point_for_back_projection
                reference_depth = self.corridor_middle_point_depth_first_time
                min_clearance = self.corridor_middle_point_depth_first_time + clearance_needed_beyond_mid_corridor # door traversal node request this value based on the total corridor length it is trying to traverse. so always publish half of the value
                self.x_corridor_center_cam_first_time = self.find_a_virtual_corridor(z_min, z_max, reference_depth, min_clearance)
                # once virtual corridor is found, we can consider corridor defined for next iterations
                # calculate x coordinate of corridor center in robot base frame at start yaw frame
                
                self.x_corridor_center_start_yaw_frame_first_time =  ( reference_depth * np.sin(self.start_yaw_at_corridor_definition) + self.x_corridor_center_cam_first_time * np.cos(self.start_yaw_at_corridor_definition))
                
                type_of_passability_check = 1  # after corridor definiton, first do corridor passability check
                # no need to publish this in door frame detection node based corridor definition. because at that stage we also checking the corridor passability and then only triggering the traversal node.
                #  but in traversal node requested corridor definition,  we need to send an ack back to traversal node that new corridor is defined and info received by traversal node is for this new virtual corridor
                self.virtual_corridor_definition_finished_pub.publish(True)
                self.define_new_corridor_req_from_traversal = False  # reset the request flag



            # ---- INITIALIZE ALL PUBLISHER OUTPUTS ----
            front_clearance = None
            x_corridor_center_cam = None
            x_corridor_center_start_yaw_frame = None
            corridor_middle_point_depth = None
            corridor_Passability_status = False  # not used in local check
            passability_view = None



            # always calculate dynamic door depth and x center frame based on odometry referenced to start of corridor definition
            # even if passability requested is local,  we still need to calculate corridor center in camera and world frame because we will use it later.
            # corridor or local passability. corridor is already defined
            # ---------------------------------------
            # Dynamic update of corridor based on odometry once the corridor is defined
            # ---------------------------------------
            self.corridor_middle_point_depth_dynamic = self.corridor_middle_point_depth_first_time - self.get_forward_displacement_since_start()
            self.x_corridor_center_cam_dynamic = self.x_corridor_center_cam_first_time + self.calculate_dynamic_x_corridor_center_in_cameraframe()
            self.x_corridor_center_start_yaw_frame_dynamic = self.x_corridor_center_start_yaw_frame_first_time + self.get_lateral_displacement_since_start()
            
            corridor_middle_point_depth = self.corridor_middle_point_depth_dynamic
            x_corridor_center_cam = self.x_corridor_center_cam_dynamic
            x_corridor_center_start_yaw_frame = self.x_corridor_center_start_yaw_frame_dynamic

            #once corridor is defined, we can do corridor passability check or local passability check based on request
       
            if type_of_passability_check == 1: # corridor passability check. request could come from traversal node or by default.
                rospy.loginfo("Performing corridor passability check.")
                # Define Z extents relative to door
                z_min = self.minimum_depth_for_back_projection
                z_max = self.corridor_middle_point_depth_dynamic + self.maximum_depth_beyond_corridor_center_point_for_back_projection
                reference_depth = self.corridor_middle_point_depth_dynamic

                # since we track corridor center dynamically.
                x_left_limit   = self.x_corridor_center_cam_dynamic - ((self.robot_width + self.safety_margin_robot_width)/2.0)
                x_right_limit  = self.x_corridor_center_cam_dynamic + ((self.robot_width + self.safety_margin_robot_width)/2.0)
                data_valid = True
            

            
            elif type_of_passability_check == 2: # local passability check
                rospy.loginfo("Performing local passability check.")
                z_min = self.minimum_depth_for_back_projection_local_passability_check  # closer range for local passability
                z_max = self.maximum_depth_for_back_projection_local_passability_check  # only up to door
                reference_depth = (z_max - z_min) / 2.0  # mid depth for local passability
                # ---------------------------------------
                # Define robot-centric traversal corridor for local passability check
                # ---------------------------------------
                x_left_limit = - (self.robot_width / 2.0 + self.safety_margin_robot_width)
                x_right_limit = (self.robot_width / 2.0 + self.safety_margin_robot_width)
                data_valid = True 

            
            else:
                data_valid = False


            if data_valid:
                valid_points, uv, _ = backproject_depth_to_points(
                    self.depth_image, self.fx, self.fy, self.cx, self.cy,
                    max_depth=z_max, min_depth=z_min,
                    subsample=self.subsample, roi_polygon=None
                )
                
                # not using final points here as the corridor is already defined
                front_clearance, passability_view, _ = self.passability_checker.run(
                    self.depth_image,
                    self.color_image,
                    x_left_limit,
                    x_right_limit,
                    valid_points,
                    uv,
                    z_max
                )

            #to trigger the traversal node based on corridor passability result
            if self.trigger_traversal_node is False and type_of_passability_check == 1 and front_clearance is not None: # if traversal node is not yet triggered, we can do passability check and trigger traversal node based on the result. but once traversal node is triggered, we should not reset the trigger even if passability goes false in later iterations. because it only makes sense to trigger traversal node once when we detect passability for the first time. and during traversal passability may go false but we do not want to retrigger traversal node.
                #executes only once
                corridor_Passability_status = front_clearance > self.corridor_middle_point_depth_dynamic + self.min_corridor_clearance_beyond_door_to_trigger_traversal_node
                if corridor_Passability_status:
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
            msg.linear.y = float(x_corridor_center_start_yaw_frame) if x_corridor_center_start_yaw_frame is not None else float('nan')
            msg.linear.z = type_of_passability_check # 1 for corridor , 2 for local
            msg.angular.x = float(x_corridor_center_cam) if x_corridor_center_cam is not None else float('nan')
            msg.angular.y = float(corridor_middle_point_depth) if corridor_middle_point_depth is not None else float('nan')
            msg.angular.z = float(clearance_needed_beyond_mid_corridor) if clearance_needed_beyond_mid_corridor is not None else float('nan')
            

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