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
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool , String
from tf.transformations import euler_from_quaternion
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
import math

# -------------------------------
# FSM states
# -------------------------------
IDLE = "IDLE"
PRE_ALIGN_TO_CORRIDOR = "PRE_ALIGN_TO_CORRIDOR"
ALIGN_TO_CORRIDOR = "ALIGN_TO_CORRIDOR"
TRAVERSE_DOOR = "TRAVERSE_DOOR"
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



        # Pre-alignment state
        sp = "pre_align_state"
        self.lateral_error_tolerance_pre_align_state = rospy.get_param(f"{ns}/{sp}/lateral_error_tolerance", 0.05)  # meters
        self.velocity_pre_align_state = rospy.get_param(f"{ns}/{sp}/velocity", 0.03)  # m/s
        self.omega_max_pre_align_state = rospy.get_param(f"{ns}/{sp}/omega_max", 0.4)   # rad/s
        self.kp_lateral_movement_pre_align_state = rospy.get_param(f"{ns}/{sp}/kp_lateral_movement", -1.2)  # rad/s per meter lateral error
        
        # Align state
        sa = "align_state"
        self.lateral_error_tolerance_align_state = rospy.get_param(f"{ns}/{sa}/lateral_error_tolerance", 0.02)  # meters
        self.velocity_align_state = rospy.get_param(f"{ns}/{sa}/velocity", 0.02)  # m/s
        self.omega_max_align_state = rospy.get_param(f"{ns}/{sa}/omega_max", 0.5)   # rad/s
        self.kp_lateral_movement_align_state = rospy.get_param(f"{ns}/{sa}/kp_lateral_movement", -1.8)  # rad/s per meter lateral error
        # Heading control gains
        self.kp_heading_align_state = rospy.get_param(f"{ns}/{sa}/kp_heading_align_state", 1.2)

        # TRAVERSAL state
        st = "traverse_state"
        self.distance_threshold_traverse_state = rospy.get_param(f"{ns}/{st}/distance_threshold", 1.0)
        # Linear velocity limits (Go1-safe indoors)
        self.v_max_traverse_state = rospy.get_param(f"{ns}/{st}/v_max", 0.20)          # m/s
        self.v_min_traverse_state = rospy.get_param(f"{ns}/{st}/v_min", 0.05)          # m/s
        # Angular velocity limits
        self.omega_max_traverse_state = rospy.get_param(f"{ns}/{st}/omega_max", 0.6)   # rad/s
        # Control gains
        self.k_clearance_traverse_state = rospy.get_param(f"{ns}/{st}/k_clearance", 0.3)
        self.kp_corridor_center_traverse_state = rospy.get_param(f"{ns}/{st}/k_center", 1.5)
        self.scale_factor_traverse_state = rospy.get_param(f"{ns}/{st}/scale_factor", 0.1)  # for tanh scaling
        self.kp_heading_traverse_state = rospy.get_param(f"{ns}/{st}/kp_heading_traverse_state", 0.8)

        # Abort back-off
        sa = "abort"
        self.abort_backoff_distance = rospy.get_param(f"{ns}/{sa}/abort_backoff_distance", 0.25)
        self.abort_backoff_speed = rospy.get_param(f"{ns}/{sa}/abort_backoff_speed", -0.08)

        # Temporal filtering
        self.required_consecutive_safe = rospy.get_param(
            "~required_consecutive_safe", 3
        )

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

        # =========================================================
        # Internal state
        # =========================================================
        self.state = IDLE
        self.start_pose = None
        self.start_yaw = None
        self.consecutive_safe = 0

        # Odometry
        self.current_pose = None
        self.current_yaw = None
        self.gazebo_pose = None

        # Passability 
        self.passable = False
        self.front_clearance = None
        self.corridor_center_x = 0.0
        # Hysteresis counters
        self.safe_seq = 0
        self.unsafe_seq = 0

        # heading estimation
        self.prev_corridor_center_x = None
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
            "/door_traversal_done", Bool, queue_size=1, latch=True
        )


        self.start_movement = False

        rospy.Subscriber(
            "/check_if_corridor_is_passable/trigger_traversal_node", Bool, self.trigger_traversal_node_callback
        )

        #  passability results
        rospy.Subscriber(
            "/check_if_corridor_is_passable/door_passability", Twist, self.passability_callback
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
            msg.linear.x  -> front_clearance (m)
            msg.linear.y  -> corridor_center_x (m)
            msg.linear.z  -> passable (1.0 or 0.0)

        Adjust this mapping to your actual message type.
        """
        self.front_clearance = msg.linear.x
        self.corridor_center_x = msg.linear.y

        # Raw passability from detector
        raw_passable = msg.linear.z > 0.5

        # Apply hysteresis: require consecutive confirmations before toggling
        if raw_passable:
            self.safe_seq += 1
            self.unsafe_seq = 0
            if self.safe_seq >= self.passability_true_hysteresis:
                self.passable = True
        else:
            self.unsafe_seq += 1
            self.safe_seq = 0
            if self.unsafe_seq >= self.passability_false_hysteresis:
                self.passable = False

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
        self.publish_cmd_vel(0.0, 0.0)


    def publish_cmd_vel(self, v, omega):

        if self.gazebo_pose is None:
            rospy.logwarn("Gazebo pose not yet received; cannot publish velocity and angular velocity to /set_model_state service")
            return

       # DO NOT TOUCH pose at all
        self.modelstate.model_name = "go1_gazebo"
        self.modelstate.reference_frame = "world"

        # always use current gazebo pose
        self.modelstate.pose = self.gazebo_pose

        self.modelstate.twist.linear.x = v
        self.modelstate.twist.linear.y = 0.0
        self.modelstate.twist.linear.z = 0.0

        self.modelstate.twist.angular.x = 0.0
        self.modelstate.twist.angular.y = 0.0
        self.modelstate.twist.angular.z = omega

        rospy.loginfo("Publishing cmd_vel: v=%.3f m/s, omega=%.3f rad/s", v, omega)

        response = self.set_model_state(self.modelstate)
        rospy.loginfo(
            "Service response: success=%s message=%s",
            response.success,
            response.status_message
        )

    def get_forward_displacement(self):
        """
        Computes displacement along initial heading.
        Robust to yaw corrections.
        """
        if self.current_pose is None:
            return 0.0

        dx = self.current_pose[0] - self.start_pose[0]
        dy = self.current_pose[1] - self.start_pose[1]

        return dx * np.cos(self.start_yaw) + dy * np.sin(self.start_yaw)
    
    def find_heading_error_estimate(self):
        # Estimate heading error based on change in corridor center x
        # =================================================
        # Update heading error estimate (from corridor drift and motion-gated)
        # =================================================
        if (
            self.prev_corridor_center_x is not None and
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
                lateral_drift = self.corridor_center_x - self.prev_corridor_center_x
                heading_raw = lateral_drift / forward_motion

                self.heading_error_est = (
                    self.heading_filter_alpha * heading_raw +
                    (1.0 - self.heading_filter_alpha) * self.heading_error_est
                )
                rospy.loginfo(f"Heading error estimate: {self.heading_error_est:.3f} rad")
                
                self.heading_estimate_samples += 1
                if self.heading_estimate_samples >= self.required_heading_samples:
                    self.heading_estimate_valid = True

                self.prev_corridor_center_x = self.corridor_center_x
                self.prev_pose_for_heading = self.current_pose


    # =========================================================
    # FSM control loop
    # =========================================================

    def control_loop(self):
             

        while not rospy.is_shutdown():
            if not self.start_movement:
                self.rate.sleep()
                continue
            
            if not self.data_ready():
                self.rate.sleep()
                continue

            
            
            # Initialize traversal start pose
            self.start_pose = self.current_pose
            self.start_yaw = self.current_yaw
            
        
            # =================================================
            # IDLE - to wait until temporal smoothing for passability is done
            # ================================================= 

            if self.state == IDLE:
                if not self.passable:
                    rospy.loginfo("IDLE: waiting for temporal smoothing to finish")
                    #stay where you are
                    self.stop_robot()
                    continue
                else:#now we are sure the corridor is passable by temporal smoothing
                    rospy.loginfo("IDLE complete → PRE-ALIGN_TO_CORRIDOR")
                    self.state = PRE_ALIGN_TO_CORRIDOR
                    continue
            # =================================================
            # PRE_ALIGN_TO_CORRIDOR
            # =================================================    
            if self.state == PRE_ALIGN_TO_CORRIDOR:
                """
                PRE-ALIGN:
                Bring robot footprint inside corridor.
                Uses slow forward motion + yaw bias.
                """ 
                rospy.loginfo("Entering PRE-ALIGN_TO_CORRIDOR state")
                if not self.passable: # temporal smoothing already done in IDLE state and also hysterisis done in passability callback
                    # so if the passability is lost here, we abort immediately( it is not due to glitch )
                    rospy.logwarn("PRE_ALIGN: corridor unsafe → ABORT")
                    self.stop_robot()
                    self.state = ABORT
                    continue

                # Lateral error (meters)
                lateral_error = self.corridor_center_x
                rospy.loginfo(f"PRE_ALIGN: lateral_error={lateral_error:.3f}")
                # If we are sufficiently centered, move to ALIGN
                if abs(lateral_error) < self.lateral_error_tolerance_pre_align_state:  # 5 cm tolerance
                    rospy.loginfo("PRE_ALIGN complete → ALIGN_TO_CORRIDOR")
                    self.stop_robot()
                    self.state = ALIGN_TO_CORRIDOR
                    # reset heading estimation variables for ALIGN state
                    self.prev_corridor_center_x = self.corridor_center_x
                    self.prev_pose_for_heading = self.current_pose
                    continue

                # Small forward creep
                v = self.velocity_pre_align_state

                # Yaw bias steers robot toward corridor center
                omega = self.kp_lateral_movement_pre_align_state * lateral_error
                omega = np.clip(omega, -self.omega_max_pre_align_state, self.omega_max_pre_align_state)

                
                self.publish_cmd_vel(v, omega)

            # =================================================
            # ALIGN_TO_CORRIDOR
            # =================================================    
            elif self.state == ALIGN_TO_CORRIDOR:
                """
                ALIGN:
                Correct heading so robot faces corridor direction.
                Uses rotation + minimal creep.
                """

                if not self.passable:
                    # temporal smoothing already done in IDLE state and also hysterisis done in passability callback
                    # so if the passability is lost here, we abort immediately( it is not due to glitch )
                    rospy.logwarn("ALIGN: corridor unsafe → ABORT")
                    self.stop_robot()
                    self.state = ABORT
                    continue
                
                self.find_heading_error_estimate()
                # If heading is good enough, start traversal
                if self.heading_estimate_valid and abs(self.heading_error_est) < self.heading_error_tolerance: # heading error tolerance
                    rospy.loginfo("ALIGN complete → TRAVERSE_DOOR")
                    self.stop_robot()

                    # Reset traversal reference
                    self.start_pose = self.current_pose
                    self.start_yaw = self.current_yaw
                    self.consecutive_safe = 0

                    self.state = TRAVERSE_DOOR
                    # reset heading estimation variables for TRAVERSE_DOOR state
                    #exit action of ALIGN state
                    self.prev_corridor_center_x = self.corridor_center_x
                    self.prev_pose_for_heading = self.current_pose
                    #PRE_ALIGN and ALIGN motions contaminate drift history
                    #TRAVERSE should start with a clean estimator
                    self.heading_error_est = 0.0
                    self.heading_estimate_valid = False
                    self.heading_estimate_samples = 0
                    continue

                # Rotate to reduce lateral error
                omega = self.kp_heading_align_state * self.heading_error_est
                omega = np.clip(omega, -self.omega_max_align_state, self.omega_max_align_state)

                # Very small forward creep to stabilize yaw estimation

                self.publish_cmd_vel(self.velocity_align_state, omega)

            # =================================================
            # TRAVERSE_DOOR
            # =================================================
            elif self.state == TRAVERSE_DOOR:

                if not self.passable:
                    # temporal smoothing already done in IDLE state and also hysterisis done in passability callback
                    # so if the passability is lost here, we abort immediately( it is not due to glitch )
                    rospy.logwarn("Passability lost → ABORT")
                    self.stop_robot()
                    self.state = ABORT
                    continue

                self.consecutive_safe += 1
                if self.consecutive_safe < self.required_consecutive_safe:
                    self.stop_robot()
                    self.rate.sleep()
                    continue

                # -----------------------------
                # Compute velocities
                # -----------------------------
                v = min(self.v_max_traverse_state,
                        self.k_clearance_traverse_state * self.front_clearance)
                v = max(self.v_min_traverse_state, v)
                
                self.find_heading_error_estimate()
                 # ---------------------------------------------
                # Combined lateral + heading correction
                # ---------------------------------------------

                # Lateral position correction
                omega_pos = -self.kp_corridor_center_traverse_state * self.corridor_center_x

                # Heading correction (damps oscillations)
                omega_heading = -self.kp_heading_traverse_state * self.heading_error_est

                omega = omega_pos + omega_heading
                rospy.loginfo(f"TRAVERSE: omega_pos={omega_pos:.3f}, omega_heading={omega_heading:.3f}, total_omega={omega:.3f}")

                # Deadband to prevent sign flipping
                if abs(omega) < 0.01:
                    omega = 0.0

                omega = np.clip(omega, -self.omega_max_traverse_state, self.omega_max_traverse_state)
                self.publish_cmd_vel(v, omega)

                # -----------------------------
                # Completion check
                # -----------------------------
                dist = self.get_forward_displacement()
                if dist >= self.distance_threshold_traverse_state:
                    rospy.loginfo("Door traversal DONE")
                    self.stop_robot()
                    self.state = DONE
                    self.done_pub.publish(Bool(data=True))
                    continue

            # =================================================
            # ABORT
            # =================================================
            elif self.state == ABORT:
                rospy.logwarn("ABORT: backing off")

                # Back off a small distance
                backoff_start = self.current_pose
                while not rospy.is_shutdown():
                    self.publish_cmd_vel(self.abort_backoff_speed, 0.0)

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
                return

            self.rate.sleep()


if __name__ == "__main__":
    try:
        move_through_coridoor = DoorTraversalController()
        move_through_coridoor.control_loop()

    except rospy.ROSInterruptException:
        pass
