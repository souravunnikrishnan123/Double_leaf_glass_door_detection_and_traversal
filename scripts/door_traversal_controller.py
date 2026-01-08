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
TRAVERSE_DOOR = "TRAVERSE_DOOR"
ABORT = "ABORT"
DONE = "DONE"


class DoorTraversalController:
    def __init__(self):
        rospy.init_node("door_traversal_controller")

        # =========================================================
        # Parameters (tuned for Unitree Go1)
        # =========================================================

        # Control loop
        self.control_rate_hz = rospy.get_param("~control_rate_hz", 20.0)

        # Linear velocity limits (Go1-safe indoors)
        self.v_max = rospy.get_param("~v_max", 0.20)          # m/s
        self.v_min = rospy.get_param("~v_min", 0.05)          # m/s

        # Angular velocity limits
        self.omega_max = rospy.get_param("~omega_max", 0.6)   # rad/s

        # Control gains
        self.k_clearance = rospy.get_param("~k_clearance", 0.3)
        self.k_center = rospy.get_param("~k_center", 1.5)

        # Traversal completion
        self.traversal_distance_threshold = rospy.get_param(
            "~traversal_distance_threshold", 1.0
        )

        # Abort back-off
        self.abort_backoff_distance = rospy.get_param(
            "~abort_backoff_distance", 0.25
        )
        self.abort_backoff_speed = rospy.get_param(
            "~abort_backoff_speed", -0.08
        )

        # Temporal filtering
        self.required_consecutive_safe = rospy.get_param(
            "~required_consecutive_safe", 3
        )

        # =========================================================
        # Internal state
        # =========================================================
        self.state = TRAVERSE_DOOR
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
        self.passable = msg.linear.z > 0.5

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

        rospy.loginfo("Entering TRAVERSE_DOOR state")

        while not rospy.is_shutdown():

            # =================================================
            # TRAVERSE_DOOR
            # =================================================
            if self.state == TRAVERSE_DOOR:

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
                v = min(self.v_max,
                        self.k_clearance * self.front_clearance)
                v = max(self.v_min, v)

                omega = -self.k_center * self.corridor_center_x
                omega = np.clip(omega, -self.omega_max, self.omega_max)

                self.publish_cmd_vel(v, omega)

                # -----------------------------
                # Completion check
                # -----------------------------
                dist = self.get_forward_displacement()
                if dist >= self.traversal_distance_threshold:
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
