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
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion

# -------------------------------
# FSM states
# -------------------------------
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
        
        
        # Abort back-off
        sa = "abort"
        self.abort_backoff_distance = rospy.get_param(f"{ns}/{sa}/abort_backoff_distance", 0.25)
        self.abort_backoff_speed = rospy.get_param(f"{ns}/{sa}/abort_backoff_speed", -0.08)

        # Temporal filtering
        self.required_consecutive_safe = rospy.get_param(
            "~required_consecutive_safe", 3
        )

        # =========================================================
        # Internal state
        # =========================================================
        self.state = PRE_ALIGN_TO_CORRIDOR
        self.start_pose = None
        self.start_yaw = None
        self.consecutive_safe = 0

        # Odometry
        self.current_pose = None
        self.current_yaw = None

        # Passability (to be updated externally)
        self.passable = False
        self.front_clearance = None
        self.corridor_center_x = 0.0

        # =========================================================
        # ROS interfaces
        # =========================================================

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        self.done_pub = rospy.Publisher(
            "/door_traversal_done", Bool, queue_size=1, latch=True
        )

        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        # You can adapt this subscriber to match how you expose
        # your passability results (topic, service, or direct call)
        rospy.Subscriber(
            "/door_passability", Twist, self.passability_callback
        )

        rospy.loginfo("DoorTraversalController initialized")
        self.control_loop()

    # =========================================================
    # Callbacks
    # =========================================================

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
        self.passable = msg.linear.z > 0.5 # Convert to boolean

    # =========================================================
    # Utility functions
    # =========================================================

    def stop_robot(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def publish_cmd_vel(self, v, omega):
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        self.cmd_pub.publish(cmd)

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

    # =========================================================
    # FSM control loop
    # =========================================================

    def control_loop(self):
        rate = rospy.Rate(self.control_rate_hz)

        # Wait for odometry
        rospy.loginfo("Waiting for odometry...")
        while not rospy.is_shutdown() and self.current_pose is None:
            rate.sleep()

        # Initialize traversal start pose
        self.start_pose = self.current_pose
        self.start_yaw = self.current_yaw

        rospy.loginfo("Entering ALIGN_TO_CORRIDOR state")

        while not rospy.is_shutdown():
            # =================================================
            # PRE_ALIGN_TO_CORRIDOR
            # =================================================    
            if self.state == PRE_ALIGN_TO_CORRIDOR:
                """
                PRE-ALIGN:
                Bring robot footprint inside corridor.
                Uses slow forward motion + yaw bias.
                """

                if not self.passable:
                    rospy.logwarn("PRE_ALIGN: corridor unsafe → ABORT")
                    self.stop_robot()
                    self.state = ABORT
                    continue

                # Lateral error (meters)
                lateral_error = self.corridor_center_x

                # If we are sufficiently centered, move to ALIGN
                if abs(lateral_error) < self.lateral_error_tolerance_pre_align_state:  # 5 cm tolerance
                    rospy.loginfo("PRE_ALIGN complete → ALIGN_TO_CORRIDOR")
                    self.stop_robot()
                    self.state = ALIGN_TO_CORRIDOR
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
                    rospy.logwarn("ALIGN: corridor unsafe → ABORT")
                    self.stop_robot()
                    self.state = ABORT
                    continue

                lateral_error = self.corridor_center_x

                # If heading is good enough, start traversal
                if abs(lateral_error) < self.lateral_error_tolerance_align_state:  # tighter tolerance (2 cm)
                    rospy.loginfo("ALIGN complete → TRAVERSE_DOOR")
                    self.stop_robot()

                    # Reset traversal reference
                    self.start_pose = self.current_pose
                    self.start_yaw = self.current_yaw
                    self.consecutive_safe = 0

                    self.state = TRAVERSE_DOOR
                    continue

                # Rotate to reduce lateral error
                omega = self.kp_lateral_movement_align_state * lateral_error
                omega = np.clip(omega, -self.omega_max_align_state, self.omega_max_align_state)

                # Very small forward creep to stabilize yaw estimation
                v = self.velocity_align_state

                self.publish_cmd_vel(v, omega)

            # =================================================
            # TRAVERSE_DOOR
            # =================================================
            elif self.state == TRAVERSE_DOOR:

                if not self.passable:
                    rospy.logwarn("Passability lost → ABORT")
                    self.stop_robot()
                    self.state = ABORT
                    continue

                self.consecutive_safe += 1
                if self.consecutive_safe < self.required_consecutive_safe:
                    self.stop_robot()
                    rate.sleep()
                    continue

                # -----------------------------
                # Compute velocities
                # -----------------------------
                v = min(self.v_max_traverse_state,
                        self.k_clearance_traverse_state * self.front_clearance)
                v = max(self.v_min_traverse_state, v)

                #tanh for smooth saturation before omega max clipping
                omega = -self.kp_corridor_center_traverse_state * np.tanh(self.corridor_center_x / self.scale_factor_traverse_state)
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

                    rate.sleep()

                self.stop_robot()
                rospy.logwarn("Abort completed")
                return

            # =================================================
            # DONE
            # =================================================
            elif self.state == DONE:
                self.stop_robot()
                return

            rate.sleep()


if __name__ == "__main__":
    try:
        DoorTraversalController()
    except rospy.ROSInterruptException:
        pass
