#!/usr/bin/env python3
"""
door_traversal_controller.py

Closed-loop door traversal controller for Unitree Go1.

States:
    - TRAVERSE_DOOR
    - ABORT
    - DONE

Responsibilities:
    - Subscribe to odometry
    - Consume door passability results
    - Generate cmd_vel
    - Decide when door traversal is complete
    - Abort safely if corridor becomes unsafe
"""


import rospy
import numpy as np
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool , String, UInt8
from robodog_glass_door_detection.msg import Robot_passability
from tf.transformations import euler_from_quaternion
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState


# -------------------------------
# FSM states
# -------------------------------
IDLE = "IDLE"
PRE_ALIGN_HEADING_TO_CORRIDOR = "PRE_ALIGN_HEADING_TO_CORRIDOR"
PRE_ALIGN_READJUST_HEADING_TO_CORRIDOR = "PRE_ALIGN_READJUST_HEADING_TO_CORRIDOR"
PRE_ALIGN_POSITION_TO_CORRIDOR = "PRE_ALIGN_POSITION_TO_CORRIDOR"
ALIGN_TO_CORRIDOR = "ALIGN_TO_CORRIDOR"
TRAVERSE_DOOR = "TRAVERSE_DOOR" 
MOVE_AFTER_CROSSING_CORRIDOR_MIDPOINT = "MOVE_AFTER_CROSSING_CORRIDOR_MIDPOINT"
LOOK_FOR_A_VIRTUAL_CORRIDOR = "LOOK_FOR_A_VIRTUAL_CORRIDOR"
ABORT = "ABORT"
DONE = "DONE"


class DoorTraversalController:
    def __init__(self):
        rospy.init_node("door_traversal_controller")
        # =========================================================
        # Parameters (tuned for Unitree Go1)
        # =========================================================
        ns = "~door_traversal_controller"
        # Control loop
        self.control_rate_hz = rospy.get_param(f"{ns}/control_rate_hz", 5.0)
        self.rate = rospy.Rate(self.control_rate_hz)

        self.min_front_clearance_local_passability_check = rospy.get_param(f"{ns}/min_front_clearance_local_passability_check", 0.2)  # meters
        self.robot_length = rospy.get_param(f"{ns}/robot_length", 0.7)  # meters

        # Pre-alignment state
        sph = "pre_align_heading_state"
        self.lateral_error_tolerance_pre_align_heading_state = rospy.get_param(f"{ns}/{sph}/lateral_error_tolerance", 0.05)  # meters
        self.omega_max_pre_align_heading_state = rospy.get_param(f"{ns}/{sph}/omega_max", 0.4)   # rad/s
        self.kp_lateral_movement_pre_align_heading_state = rospy.get_param(f"{ns}/{sph}/kp_lateral_movement", -1.2)  # rad/s per meter lateral error
        
        # pre-align readjust heading state
        spr = "pre_align_readjust_heading_state"
        self.fraction_distance_to_corridor_mid_point_for_readjust_heading_state = rospy.get_param(f"{ns}/{spr}/fraction_distance_to_corridor_mid_point_for_readjust_heading", 0.3)  # fraction of distance to door
        self.omega_max_pre_align_readjust_heading_state = rospy.get_param(f"{ns}/{spr}/omega_max", 0.4)   # rad/s
        self.kp_lateral_movement_pre_align_readjust_heading_state = rospy.get_param(f"{ns}/{spr}/kp_lateral_movement", -1.2)  # rad/s per meter lateral error

        # pre-align position state
        spp = "pre_align_position_state"
        self.pre_align_position_heading_ref = 0.0
        self.lateral_error_tolerance_pre_align_position_state = rospy.get_param(f"{ns}/{spp}/lateral_error_tolerance", 0.35)  # half of the width of robot in meters
        self.velocity_pre_align_position_state = rospy.get_param(f"{ns}/{spp}/velocity", 0.03)  # m/s
        self.omega_max_pre_align_position_state = rospy.get_param(f"{ns}/{spp}/omega_max", 0.4)   # rad/s
        self.kp_lateral_movement_pre_align_position_state = rospy.get_param(f"{ns}/{spp}/kp_lateral_movement", -1.2)  # rad/s per meter lateral error

        # LOOK_FOR_OTHER_WAY state
        sl = "look_for_other_way_state"


        # Align state
        sa = "align_state"
        self.heading_error_tolerance_align_state = rospy.get_param(f"{ns}/{sa}/heading_error_tolerance", 0.02)  # rad
        self.lateral_error_tolerance_align_state = rospy.get_param(f"{ns}/{sa}/lateral_error_tolerance", 0.35)  # half of the width of robot in meters
        self.omega_max_align_state = rospy.get_param(f"{ns}/{sa}/omega_max", 0.5)   # rad/s
        # Heading control gains
        self.kp_heading_align_state = rospy.get_param(f"{ns}/{sa}/kp_heading_align_state", -1.2)

        # TRAVERSAL state
        st = "traverse_state"
        self.prev_corridor_center_x = None
        self.prev_time = None
        # Linear velocity limits (Go1-safe indoors)
        self.v_max_traverse_state = rospy.get_param(f"{ns}/{st}/v_max", 0.20)          # m/s
        self.v_min_traverse_state = rospy.get_param(f"{ns}/{st}/v_min", 0.05)          # m/s
        # Angular velocity limits
        self.omega_max_traverse_state = rospy.get_param(f"{ns}/{st}/omega_max", 0.6)   # rad/s
        # Control gains
        self.k_clearance_traverse_state = rospy.get_param(f"{ns}/{st}/k_clearance", 0.3)
        self.kp_corridor_center_traverse_state = rospy.get_param(f"{ns}/{st}/kp_corridor_center", -1.5)
        self.kp_heading_traverse_state = rospy.get_param(f"{ns}/{st}/kp_heading", -0.8)
        self.kd_corridor_center_traverse_state = rospy.get_param(f"{ns}/{st}/kd_corridor_center", 0.8)
  
        # MOVE_AFTER_CROSSING_CORRIDOR_MIDPOINT state
        sd = "move_after_crossing_corridor_midpoint_state"
        self.start_pose_at_corridor_mid_point = None
        self.start_yaw_at_corridor_mid_point = None
        self.kp_heading_adjustment_move_after_crossing_corridor_midpoint_state = rospy.get_param(f"{ns}/{sd}/kp_heading_adjustment", -1.2)  # rad/s per meter lateral error
        self.velocity_move_after_crossing_corridor_midpoint_state = rospy.get_param(f"{ns}/{sd}/velocity", 0.03)  # m/s
        self.omega_max_move_after_crossing_corridor_midpoint_state = rospy.get_param(f"{ns}/{sd}/omega_max", 0.4)   # rad/s
   
        # LOOK_FOR_A_VIRTUAL_CORRIDOR state
        slm = "LOOK_FOR_A_VIRTUAL_CORRIDOR_state"


        # Abort back-off
        sa = "abort"
        self.abort_backoff_distance = rospy.get_param(f"{ns}/{sa}/abort_backoff_distance", 0.25)
        self.abort_backoff_speed = rospy.get_param(f"{ns}/{sa}/abort_backoff_speed", -0.08)



        #heading error calculation
        # Low-pass filter for heading estimate
        he = "heading_error_estimation"
        self.heading_filter_alpha = rospy.get_param(f"~{he}/heading_filter_alpha", 0.3)
        self.heading_error_tolerance = rospy.get_param(f"~{he}/heading_error_tolerance", 0.03)
        self.min_forward_motion_for_heading = rospy.get_param(f"~{he}/min_forward_motion_for_heading", 0.02)  # meters
        self.required_heading_samples = rospy.get_param(f"~{he}/required_heading_samples", 2)
        self.heading_estimate_samples = 0
        self.heading_estimate_valid = False
        self.prev_pose_for_heading = None

        # Passability hysteresis (to avoid glitch-induced stops)
        ph = "passability_hysteresis"
        self.passability_true_hysteresis = rospy.get_param(
            f"~{ph}/consecutive_passable_to_true", 5
        )
        self.passability_false_hysteresis = rospy.get_param(
            f"~{ph}/consecutive_unpassable_to_false", 4
        )
        # Hysteresis counters
        self.safe_seq_corridor = 0
        self.unsafe_seq_corridor = 0 
        self.safe_seq_local = 0
        self.unsafe_seq_local = 0   

        # =========================================================
        # Internal state
        # =========================================================
        self.state = IDLE
        self.start_pose = None
        self.start_yaw = None
        self.virtual_corridor_definition_finished = False

        # Odometry
        self.current_pose = None
        self.current_yaw = None
        self.gazebo_pose = None

        # Passability 
        self.corridor_passable = False
        self.local_passable = False
        self.front_clearance = None
        self.x_corridor_center_cam = None
        self.distance_to_corridor_mid_point = None
        self.minimum_clearance_beyond_corridor_mid_point = None
        self.corridor_center_x_robot_base = 0.0
        self.heading_error_to_corridor = None


        # heading estimation
        self.prev_corridor_center_x_robot_base = None
        self.heading_error_est = 0.0
        # =========================================================
        # ROS interfaces
        # =========================================================
        rospy.loginfo("Waiting for /gazebo/set_model_state service...")
        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_model_state = rospy.ServiceProxy(
            "/gazebo/set_model_state",
            SetModelState
        )
        rospy.loginfo("Connected to /gazebo/set_model_state service")

        self.modelstate = ModelState()


        self.done_pub = rospy.Publisher(
            "~door_traversal_done", Bool, queue_size=1
        )

        self.request_local_passability_check_pub = rospy.Publisher(
            "~request_local_passability_check", UInt8, queue_size=1
        )

        self.request_new_corridor_definition_pub = rospy.Publisher(
            "~request_new_corridor_definition", Twist, queue_size=1
        )


        self.start_movement = False
        self.movement_is_started = False

        rospy.Subscriber(
            "/check_if_corridor_is_passable/trigger_traversal_node", Bool, self.trigger_traversal_node_callback
        )

        rospy.Subscriber(
            "/check_if_corridor_is_passable/virtual_corridor_definition_finished", Bool, self.virtual_corridor_definition_finished_callback
        )


        #  passability results
        rospy.Subscriber(
            "/check_if_corridor_is_passable/door_passability", Robot_passability, self.passability_callback
        )

        # odometry
        rospy.Subscriber("/odom_bridge_output", Odometry, self.odom_callback)

        # pose from gazebo just for sending current pose info to /gazebo/set_model_state service
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_callback)


        rospy.loginfo("DoorTraversalController initialized")
        

    # =========================================================
    # Callbacks
    # =========================================================


    def trigger_traversal_node_callback(self, msg):
        if msg.data:
            if not self.start_movement:
                self.start_movement = True

    def virtual_corridor_definition_finished_callback(self, msg):
        self.virtual_corridor_definition_finished = msg.data


    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.current_pose = (p.x, p.y)

        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)
        self.current_yaw = yaw

    def passability_callback(self, msg):
        """
        EXPECTED CONVENTION (example):
            msg.front_clearance  -> front_clearance (m)
            msg.x_corridor_center_start_yaw_frame  -> corridor_center_x_robot_base (m)
            msg.type_of_passability_check  -> type_of_passability_check (1.0 or 2.0)
            msg.x_corridor_center_cam -> x_corridor_center_cam (m)
            msg.corridor_middle_point_depth -> distance_to_corridor_mid_point (m)
            msg.clearance_needed_beyond_mid_corridor -> clearance_needed_beyond_mid_corridor (m)
            msg.heading_error -> heading_error (rad)


        """
        self.front_clearance = msg.front_clearance
        self.corridor_center_x_robot_base = msg.x_corridor_center_start_yaw_frame
        self.x_corridor_center_cam = msg.x_corridor_center_cam
        self.distance_to_corridor_mid_point = msg.corridor_middle_point_depth
        self.minimum_clearance_beyond_corridor_mid_point = msg.clearance_needed_beyond_mid_corridor
        self.heading_error_to_corridor = msg.heading_error

        # Raw passability from detector
        if msg.type_of_passability_check == 1.0: # corridor passability check
            self.local_passable = False  # reset local passability. because we are doing corridor passability check now
            self.safe_seq_local = 0
            self.unsafe_seq_local = 0
            
            if self.front_clearance is not None:
                if self.front_clearance > self.distance_to_corridor_mid_point +self.minimum_clearance_beyond_corridor_mid_point: # ensure there is enough clearance beyond the corridor mid point, not just at the current position, to avoid the case where the robot is currently in a safe spot but will soon collide with the door frame when it moves forward:
                    raw_corridor_passable = True
                else: 
                    raw_corridor_passable = False
            else:
                raw_corridor_passable = False

            # Apply hysteresis: require consecutive confirmations before toggling
            if raw_corridor_passable:
                self.safe_seq_corridor += 1
                self.unsafe_seq_corridor = 0
                if self.safe_seq_corridor >= self.passability_true_hysteresis:
                    self.corridor_passable = True
                        
            else:
                self.unsafe_seq_corridor += 1
                self.safe_seq_corridor = 0
                if self.unsafe_seq_corridor >= self.passability_false_hysteresis:
                    self.corridor_passable = False
            
             
        elif msg.type_of_passability_check == 2.0: # local passability check
            self.corridor_passable = False  # reset corridor passability. because we are doing local passability check now
            self.safe_seq_corridor = 0
            self.unsafe_seq_corridor = 0
            if self.front_clearance is not None:
                if self.front_clearance > self.min_front_clearance_local_passability_check:
                    raw_local_passable = True
                else: 
                    raw_local_passable = False
            else:
                raw_local_passable = False
            
            # Apply hysteresis: require consecutive confirmations before toggling                  
            if raw_local_passable:
                self.safe_seq_local += 1
                self.unsafe_seq_local = 0
                if self.safe_seq_local >= self.passability_true_hysteresis:
                    self.local_passable = True
            else:
                self.unsafe_seq_local += 1
                self.safe_seq_local = 0
                if self.unsafe_seq_local >= self.passability_false_hysteresis:
                    self.local_passable = False





    def model_states_callback(self, msg):
        if "go1_gazebo" not in msg.name:
            return
        index = msg.name.index("go1_gazebo")
        self.gazebo_pose = msg.pose[index]


    def data_ready(self):        return (
            self.current_pose is not None
            and self.current_yaw is not None
            and self.front_clearance is not None
        )
    # =========================================================
    # Utility functions
    # =========================================================

    def stop_robot(self):
        self.publish_cmd_vel(0.0, 0.0, 0.0)

    def wrap_angle(self, angle):
        """Normalize an angle in radians to the range [-pi, pi]."""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi


    def publish_cmd_vel(self, v_forward, v_lateral, omega):

        if self.gazebo_pose is None:
            rospy.logwarn("Gazebo pose not yet received; cannot publish velocity and angular velocity to /set_model_state service")
            return

       # DO NOT TOUCH pose at all
        self.modelstate.model_name = "go1_gazebo"
        self.modelstate.reference_frame = "world"

        # always use current gazebo pose
        self.modelstate.pose = self.gazebo_pose

        # Convert body-frame forward velocity to world frame
        # only forward velocity v is given in robot base frame no lateral velocity
        #v_lateral is always zero.
        vx_world = v_forward * np.cos(self.current_yaw) - v_lateral * np.sin(self.current_yaw)
        vy_world = v_forward * np.sin(self.current_yaw) + v_lateral * np.cos(self.current_yaw)

        self.modelstate.twist.linear.x = vx_world
        self.modelstate.twist.linear.y = vy_world
        self.modelstate.twist.linear.z = 0.0

        self.modelstate.twist.angular.x = 0.0
        self.modelstate.twist.angular.y = 0.0
        self.modelstate.twist.angular.z = omega

        rospy.loginfo("Publishing cmd_vel: v_forward=%.3f m/s, v_lateral=%.3f m/s, omega=%.3f rad/s", v_forward, v_lateral, omega)

        response = self.set_model_state(self.modelstate)
        rospy.loginfo(
            "Service response: success=%s message=%s",
            response.success,
            response.status_message
        )

    def get_forward_displacement(self, ref_pose, ref_yaw):
        """
        Computes displacement along initial heading.
        Robust to yaw corrections.
        """
        if self.current_pose is None:
            return 0.0

        dx = self.current_pose[0] - ref_pose[0]
        dy = self.current_pose[1] - ref_pose[1]

        return dx * np.cos(ref_yaw) + dy * np.sin(ref_yaw)
    
    def find_heading_error_estimate(self):
        # Estimate heading error based on change in corridor center x
        # =================================================
        # Update heading error estimate (from corridor drift and motion-gated)
        # =================================================
        if (
            self.prev_corridor_center_x_robot_base is not None and
            self.prev_pose_for_heading is not None and
            self.current_pose is not None
            ):
            # Compute actual forward motion
            dx_world = self.current_pose[0] - self.prev_pose_for_heading[0]
            dy_world = self.current_pose[1] - self.prev_pose_for_heading[1]

            forward_motion = (
                dx_world * np.cos(self.current_yaw) +
                dy_world * np.sin(self.current_yaw)
            )

            # Only update heading if robot actually moved forward
            if abs(forward_motion) > self.min_forward_motion_for_heading:
                lateral_drift = self.corridor_center_x_robot_base - self.prev_corridor_center_x_robot_base
                heading_raw = lateral_drift / forward_motion

                self.heading_error_est = (
                    self.heading_filter_alpha * heading_raw +
                    (1.0 - self.heading_filter_alpha) * self.heading_error_est
                )
                rospy.loginfo(f"Heading error estimate: {self.heading_error_est:.3f} rad")
                
                self.heading_estimate_samples += 1
                if self.heading_estimate_samples >= self.required_heading_samples:
                    self.heading_estimate_valid = True

                self.prev_corridor_center_x_robot_base = self.corridor_center_x_robot_base
                self.prev_pose_for_heading = self.current_pose


    # =========================================================
    # FSM control loop
    # =========================================================

    def control_loop(self):
             

        while not rospy.is_shutdown():
            if not self.start_movement:# node is activated as there is a corridor to traverse
                self.rate.sleep()
                continue
            
            if not self.data_ready():
                self.rate.sleep()
                continue
            
        
            # =================================================
            # IDLE - to wait until temporal smoothing for passability is done
            # ================================================= 

            if self.state == IDLE:
                # Initialize traversal start pose
                self.start_pose = self.current_pose
                self.start_yaw = self.current_yaw
                if not self.corridor_passable:
                    rospy.loginfo("IDLE: waiting for temporal smoothing to finish")
                    #stay where you are
                    self.stop_robot()
                    continue
                else:#now we are sure the corridor is passable by temporal smoothing
                    rospy.loginfo("IDLE complete → PRE-ALIGN_HEADING_TO_CORRIDOR")
                    self.state = PRE_ALIGN_HEADING_TO_CORRIDOR
                    self.request_local_passability_check_pub.publish(UInt8(data=0)) # disable both until we reach pre-align position state
                    self.stop_robot()
                    continue
            # =================================================
            # PRE_ALIGN_HEADING_TO_CORRIDOR
            # =================================================    
            if self.state == PRE_ALIGN_HEADING_TO_CORRIDOR:
                """
                PRE_ALIGN_HEADING_TO_CORRIDOR:
                to make sure robot heading is roughly toward corridor direction.
                Uses slow forward motion + yaw bias.
                """ 
                rospy.loginfo(" PRE-ALIGN_HEADING_TO_CORRIDOR state")

                # no need to check local passability here because it is a rotation state no translation involved

                # Lateral error (meters)
                lateral_error_in_robot_base = self.corridor_center_x_robot_base
                # x_corridor_center_cam is used only once and used in pre-align heading state for aligning heading toward corridor direction
                lateral_error_in_cam =  self.x_corridor_center_cam

                rospy.loginfo(f"PRE_ALIGN_HEADING: lateral_error in robot base frame ={lateral_error_in_robot_base:.3f}m, lateral_error in camera frame={lateral_error_in_cam:.3f}m")
                

                 #If we are sufficiently centered, move to PRE_ALIGN_READJUST_HEADING_TO_CORRIDOR
                if abs(lateral_error_in_cam) < self.lateral_error_tolerance_pre_align_heading_state:  # 5 cm tolerance
                    rospy.loginfo("PRE_ALIGN_HEADING complete → PRE_ALIGN_READJUST_HEADING_TO_CORRIDOR")
                    self.stop_robot()
                    self.state = PRE_ALIGN_READJUST_HEADING_TO_CORRIDOR
                    continue



                # Yaw bias steers robot toward corridor center
                omega = self.kp_lateral_movement_pre_align_heading_state * lateral_error_in_cam
                omega = np.clip(omega, -self.omega_max_pre_align_heading_state, self.omega_max_pre_align_heading_state)

                
                self.publish_cmd_vel(0.0 , 0.0, omega)

            # =================================================
            # PRE_ALIGN_READJUST_HEADING_TO_CORRIDOR
            # =================================================  
            if self.state == PRE_ALIGN_READJUST_HEADING_TO_CORRIDOR:
                """
                 to adjust robot's orientation so that the robot will be within the corridor axis, within traveling a particular fraction of distance to the door. 
                """
                rospy.loginfo(" PRE_ALIGN_READJUST_HEADING_TO_CORRIDOR state")

                # no need to check local passability here because it is a rotation state no translation involved

                lateral_error_in_robot_base = self.corridor_center_x_robot_base
                estimated_forward_distance_to_reach_corridor = lateral_error_in_robot_base / np.tan(self.current_yaw)
                maximum_allowable_forward_distance_to_reach_corridor = abs(self.distance_to_corridor_mid_point * self.fraction_distance_to_corridor_mid_point_for_readjust_heading_state) 
                distance_offset = abs(estimated_forward_distance_to_reach_corridor) - maximum_allowable_forward_distance_to_reach_corridor
                rospy.loginfo(f"PRE_ALIGN_READJUST_HEADING: lateral_error={lateral_error_in_robot_base:.3f} m, estimated_forward_distance_to_reach_corridor={estimated_forward_distance_to_reach_corridor:.3f} m, maximum_allowable_forward_distance_to_reach_corridor={maximum_allowable_forward_distance_to_reach_corridor:.3f} m, distance_offset={distance_offset:.3f} m")
                if distance_offset <= 0.1: # will reach corridor within allowable distance
                    rospy.loginfo("PRE_ALIGN_READJUST_HEADING complete → PRE-ALIGN_POSITION_TO_CORRIDOR")
                    self.stop_robot()
                    self.state = PRE_ALIGN_POSITION_TO_CORRIDOR
                    self.movement_is_started = False
                    # set reference heading for position pre-alignment
                    self.pre_align_position_heading_ref = self.current_yaw
                    self.request_local_passability_check_pub.publish(UInt8(data=2)) # request local passability check
                    continue
                

                # Yaw bias steers robot toward corridor center
                omega = self.kp_lateral_movement_pre_align_readjust_heading_state * distance_offset
                omega = np.clip(omega, -self.omega_max_pre_align_readjust_heading_state, self.omega_max_pre_align_readjust_heading_state)
                
                self.publish_cmd_vel(0.0 , 0.0, omega)
                




            # =================================================
            # PRE_ALIGN_POSITION_TO_CORRIDOR
            # =================================================  
            if self.state == PRE_ALIGN_POSITION_TO_CORRIDOR:
                """
                 to make sure that robot's body is roughly inside in the corridor.
                """
                rospy.loginfo(" PRE-ALIGN_POSITION_TO_CORRIDOR state")

                if not self.local_passable: 
                    if not self.movement_is_started: 
                        rospy.loginfo("PRE-ALIGN_POSITION_TO_CORRIDOR: waiting for temporal smoothing to finish")
                        #stay where you are
                        self.stop_robot()
                        continue
                    else:
                        # if passability lost here means, movement were already started and then passabililty lost( that is self.local_passable is False not due to temporal smoothing)
                        # temporal smoothing already done and also hysteresis done in passability callback
                        # so if the passability is lost here, means it is not safe to move forward( it is not due to glitch )
                        # but since we are at pre-align position state, we dont have to abort immediately. robot can still go to align state and then check passability again.
                        rospy.logwarn("Passability lost → going to ALIGN_TO_CORRIDOR")
                        self.stop_robot()
                        self.pre_align_position_heading_ref = 0.0
                        self.state = ALIGN_TO_CORRIDOR
                        self.movement_is_started = False
                        continue
                        

                self.movement_is_started = True

                lateral_error_in_robot_base = self.corridor_center_x_robot_base
                rospy.loginfo(f"PRE-ALIGN_POSITION_TO_CORRIDOR: lateral_error={lateral_error_in_robot_base:.3f} m")
                # If we are sufficiently centered, move to ALIGN
                if abs(lateral_error_in_robot_base) < self.lateral_error_tolerance_pre_align_position_state:  # 5 cm tolerance
                    rospy.loginfo("PRE-ALIGN_POSITION_TO_CORRIDOR complete → ALIGN_TO_CORRIDOR")
                    self.stop_robot()
                    self.state = ALIGN_TO_CORRIDOR
                    self.pre_align_position_heading_ref = 0.0
                    self.movement_is_started = False
                    self.request_local_passability_check_pub.publish(UInt8(data=0)) # disable both until we reach traverse state
                    continue

                # Heading hold (NOT lateral correction)
                heading_error = self.wrap_angle(self.current_yaw - self.pre_align_position_heading_ref)

                # Yaw bias steers robot toward corridor center
                omega = self.kp_lateral_movement_pre_align_position_state * heading_error
                omega = np.clip(omega, -self.omega_max_pre_align_position_state, self.omega_max_pre_align_position_state)

                
                self.publish_cmd_vel(self.velocity_pre_align_position_state, 0.0, omega)

            # =================================================
            # ALIGN_TO_CORRIDOR
            # =================================================    
            elif self.state == ALIGN_TO_CORRIDOR:
                """
                ALIGN:
                Correct heading so robot faces corridor direction.
                Uses rotation + minimal creep.
                """
                rospy.loginfo("  ALIGN_TO_CORRIDOR state")

                # no need to check local passability here because it is a rotation state no translation involved
                

                required_heading_correction = 0 - self.heading_error_to_corridor
                rospy.loginfo(f"ALIGN: current yaw is {self.current_yaw:.3f} rad, required_heading_correction={required_heading_correction:.3f} rad")
            
                # If heading is good enough, start traversal
                if abs(required_heading_correction) < self.heading_error_tolerance_align_state: # heading error tolerance
                    # but the lateral error may be still high, if the align state is reached from pre-align position state due to passability loss
                    lateral_error_in_robot_base = self.corridor_center_x_robot_base
                    #to check if the lateral error is also within tolerance. if not means we are not means robot body is not within corridor axis
                    if abs(lateral_error_in_robot_base) <= self.lateral_error_tolerance_align_state:  # half of the width of robot in meters
                        rospy.loginfo("ALIGN complete → TRAVERSE_DOOR")
                        self.stop_robot()

                        # Reset traversal reference
                        self.movement_is_started = False
                        self.state = TRAVERSE_DOOR
                        #exit action of ALIGN state
                        self.request_local_passability_check_pub.publish(UInt8(data=1)) # check corridor passability
                        continue
                    else:
                        rospy.loginfo("ALIGN_TO_CORRIDOR: lateral error too high → PRE_ALIGN_HEADING_TO_CORRIDOR")
                        self.stop_robot()
                        self.state = PRE_ALIGN_HEADING_TO_CORRIDOR
                        #restart from pre-align heading state
                        self.pre_align_position_heading_ref = 0.0
                        self.movement_is_started = False
                        self.request_local_passability_check_pub.publish(UInt8(data=0)) # disable both 
                        continue

                # Rotate to reduce lateral error
                omega = self.kp_heading_align_state * required_heading_correction
                omega = np.clip(omega, -self.omega_max_align_state, self.omega_max_align_state)

                # Very small forward creep to stabilize yaw estimation

                self.publish_cmd_vel(0.0, 0.0, omega)

            # =================================================
            # TRAVERSE_DOOR
            # =================================================
            elif self.state == TRAVERSE_DOOR:
                rospy.loginfo(" TRAVERSE_DOOR state")
                
                if not self.corridor_passable:
                    if not self.movement_is_started: 
                        rospy.loginfo("TRAVERSE_DOOR: waiting for temporal smoothing to finish")
                        #stay where you are
                        self.stop_robot()
                        continue
                    else:
                        # if passability lost here means, traversal were already started and then passabililty lost( that is self.corridor_passable is False not due to temporal smoothing)
                        # temporal smoothing already done and also hysteresis done in passability callback
                        # so if the passability is lost here, we abort immediately( it is not due to glitch )
                        rospy.logwarn("Passability lost → ABORT")
                        self.stop_robot()
                        self.state = ABORT
                        self.movement_is_started = False
                        continue
                        

                self.movement_is_started = True
                
                # -----------------------------
                # Completion check
                # -----------------------------
                if self.distance_to_corridor_mid_point <= 0.0: # within 5 cm of door plane. now no chance to collide with closed glass door half
                    #at this point we also ensured that corridor in front has a clearance requested from passability node
                    rospy.loginfo("Door traversal DONE")
                    self.stop_robot()
                    self.movement_is_started = False
                    distance_to_move_beyond_corridor_mid_point = self.minimum_clearance_beyond_corridor_mid_point + self.robot_length #because if the door open outward, we need to move sufficiently forward to be beyond the door swing area. also since the camera is on the head, we need to make sure the body ( behind) is beyond the door frame
                    self.request_local_passability_check_pub.publish(UInt8(data=2)) # request local passability check
                    self.state = MOVE_AFTER_CROSSING_CORRIDOR_MIDPOINT
                    
                    # set reference  for move after crossing corridor midpoint state.
                    #until now the traversal 
                    self.start_pose_at_corridor_mid_point = self.current_pose
                    self.start_yaw_at_corridor_mid_point = self.current_yaw
                    continue
                    #self.state = DONE
                    #self.done_pub.publish(Bool(data=True))

                now = rospy.Time.now().to_sec()
                #guard
                if self.prev_time is None:
                    self.prev_time = now
                    self.prev_corridor_center_x = self.corridor_center_x_robot_base
                    continue
                
                dt = now - self.prev_time
                if dt <= 0.0:
                    continue
                
                # lateral offset from corridor center
                x_c = self.corridor_center_x_robot_base
                heading_error_correction = 0.0 - self.heading_error_to_corridor
                x_p_error = 0.0 - x_c  # desired corridor center is at x=0 in robot base frame

                # ------------------------------------------------
                # Lateral rate ( yaw damping)
                # ------------------------------------------------
                x_dot = (x_c - self.prev_corridor_center_x) / dt
                # ------------------------------------------------
                # Angular velocity command (NO heading estimation)
                # ------------------------------------------------
                rospy.loginfo(f"TRAVERSE_DOOR: lateral error={x_c:.3f} m, lateral_kp_error={x_p_error:.3f} m, lateral error rate={x_dot:.3f} m/s")
                omega = (
                    self.kp_corridor_center_traverse_state * x_p_error + 
                    self.kp_heading_traverse_state * heading_error_correction +
                    self.kd_corridor_center_traverse_state * x_dot
                )   
                omega = np.clip(omega, -self.omega_max_traverse_state, self.omega_max_traverse_state)
                # -----------------------------
                # Compute velocities
                # -----------------------------
                velocity_forward = min(self.v_max_traverse_state,
                        self.k_clearance_traverse_state * self.front_clearance)
                velocity_forward = max(self.v_min_traverse_state, velocity_forward)
            

                
                self.publish_cmd_vel(velocity_forward, 0.0, omega)
                # ------------------------------------------------
                # State update
                # ------------------------------------------------
                self.prev_corridor_center_x = x_c
                self.prev_time = now


                    
            # =================================================
            # MOVE_AFTER_CROSSING_CORRIDOR_MIDPOINT
            # =================================================
            elif self.state == MOVE_AFTER_CROSSING_CORRIDOR_MIDPOINT:
                rospy.loginfo(" MOVE_AFTER_CROSSING_CORRIDOR_MIDPOINT state")
                # donot use the heading error from passability node. because we requested for local passability check just for checking the clearance in front after crossing the door plane. hence heading error estimation is no longer available.
                # so we use the last heading and pose from traverse state as reference 
                # calculating forward displacment along the heading direction at the time when we just crossed the corridor mid point (door plane). i.e in world frame( corridor frame no longer exists)
                distance_travelled_beyond_corridor_mid_point = self.get_forward_displacement(self.start_pose_at_corridor_mid_point, self.start_yaw_at_corridor_mid_point)
                
                if not self.local_passable:
                    if not self.movement_is_started: 
                        rospy.loginfo("MOVE_AFTER_CROSSING_CORRIDOR_MIDPOINT: waiting for temporal smoothing to finish")
                        #stay where you are
                        self.stop_robot()
                        continue
                    else:
                        # if passability lost here means, traversal were already started and then passabililty lost( that is self.local_passable is False not due to temporal smoothing)
                        # temporal smoothing already done and also hysteresis done in passability callback
                        # so if the passability is lost here, we abort immediately( it is not due to glitch )
                        rospy.logwarn("local Passability lost → LOOK_FOR_A_VIRTUAL_CORRIDOR")
                        self.stop_robot()
                        self.state = LOOK_FOR_A_VIRTUAL_CORRIDOR
                        self.request_local_passability_check_pub.publish(UInt8(data=0)) # disable both until new corridor is defined

                        # request new corridor definition. its not like whenever we cross the door we will ask for new corridor. instead, whenever we cross the door and then find that local passability is lost, only then we will ask for new corridor definition.
                        msg = Twist()
                        msg.linear.x = True # to request new corridor definition
                        #set the middle depth of requested corridor for corridor definition
                        if distance_to_move_beyond_corridor_mid_point - distance_travelled_beyond_corridor_mid_point <= 0.25:
                            msg.linear.y = 0.25  # corridor is of 0.5m
                            # because we cannot give very low value as corridor as defining a new corridor use back projection and it has some minimum depth limit
                        else:
                            msg.linear.y = (distance_to_move_beyond_corridor_mid_point - distance_travelled_beyond_corridor_mid_point)/2.0

                        self.request_new_corridor_definition_pub.publish(msg) # request new corridor definition
                        self.movement_is_started = False
                        continue
                        

                self.movement_is_started = True

                # -----------------------------
                # Completion check
                # -----------------------------
                
                rospy.loginfo(f"MOVE_AFTER_CROSSING_CORRIDOR_MIDPOINT: moved distance = {distance_travelled_beyond_corridor_mid_point:.3f} m, distance to move {distance_to_move_beyond_corridor_mid_point:.3f}")
                
                if distance_travelled_beyond_corridor_mid_point >= distance_to_move_beyond_corridor_mid_point:
                    rospy.loginfo("move after Door frame crossed state --> DONE")
                    self.stop_robot()
                    self.movement_is_started = False
                    self.state = DONE
                    continue
               
                # Heading hold (NOT lateral correction)
                # Yaw bias steers robot toward corridor center
                # cannot use  self.heading_error_to_corridor, because in this state we are in local passability check. hence the heading error estimation from passability node is no longer available. but we can still use the current yaw and the yaw at the time when we just crossed the corridor mid point to do heading hold to make sure the robot is moving along the right direction after crossing the door plane. because if the robot drifts too much in heading after crossing the door plane, it may collide with the door frame half or it may go out of the new corridor after crossing the door plane.
                #heading_error = 0 - self.heading_error_to_corridor
                heading_error = self.wrap_angle(self.current_yaw - self.start_yaw_at_corridor_mid_point)

                omega = self.kp_heading_adjustment_move_after_crossing_corridor_midpoint_state * heading_error

                omega = np.clip(omega, -self.omega_max_move_after_crossing_corridor_midpoint_state, self.omega_max_move_after_crossing_corridor_midpoint_state)

                
                self.publish_cmd_vel(self.velocity_move_after_crossing_corridor_midpoint_state, 0.0, omega)
            
            # =================================================
            # LOOK_FOR_A_VIRTUAL_CORRIDOR
            # =================================================
            elif self.state == LOOK_FOR_A_VIRTUAL_CORRIDOR:
                rospy.loginfo("LOOK_FOR_A_VIRTUAL_CORRIDOR:  find new corridor")
                # wait unitl new corridor is defined
                if not self.virtual_corridor_definition_finished:
                    rospy.loginfo("LOOK_FOR_A_VIRTUAL_CORRIDOR: waiting for new corridor to be defined")
                    #stay where you are
                    self.stop_robot()
                    continue
                else: # new corridor is defined
                    rospy.loginfo("LOOK_FOR_A_VIRTUAL_CORRIDOR complete → IDLE")
                    self.state = IDLE
                    self.stop_robot()
                    continue


            # =================================================
            # ABORT
            # =================================================
            elif self.state == ABORT:
                rospy.logwarn("ABORT: backing off")

                # Back off a small distance
                backoff_start = self.current_pose
                while not rospy.is_shutdown():
                    self.publish_cmd_vel(self.abort_backoff_speed, 0.0, 0.0)

                    dx = self.current_pose[0] - backoff_start[0]
                    dy = self.current_pose[1] - backoff_start[1]
                    back_dist = np.hypot(dx, dy)

                    if back_dist >= self.abort_backoff_distance:
                        break

                    self.rate.sleep()

                self.stop_robot()
                rospy.logwarn("Abort completed")
                return

            # =================================================
            # DONE
            # =================================================
            elif self.state == DONE:
                self.stop_robot()
                self.done_pub.publish(Bool(data=True))
                # publish done signal in the DONE state  
                return

            self.rate.sleep()


if __name__ == "__main__":
    try:
        move_through_coridoor = DoorTraversalController()
        move_through_coridoor.control_loop()

    except rospy.ROSInterruptException:
        pass
