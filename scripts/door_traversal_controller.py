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
# Alignment is split into heading, intercept, and final-heading phases. Trying
# to correct all three errors with one controller was too ambiguous near a door.
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
    """
    Control the robot while it aligns with and traverses a door corridor.

    The controller consumes corridor geometry and passability updates, applies
    confirmation hysteresis, aligns the robot in stages, and continuously
    checks clearance while crossing and moving beyond the door. Recovery states
    either back the robot away from an unsafe corridor or ask the passability
    node to define a temporary corridor around a newly observed obstruction.

    In the current Gazebo integration, velocity commands are converted from
    the robot frame to world-frame twists and sent through
    ``/gazebo/set_model_state``.

    Attributes:
        state:
            Name of the active finite-state-machine state.

        current_pose:
            Latest planar odometry position as ``(x, y)`` in meters.

        current_yaw:
            Latest robot yaw in radians.

        corridor_passable:
            Hysteresis-filtered result for the complete door corridor.

        local_passable:
            Hysteresis-filtered result for the space immediately ahead.

        front_clearance:
            Measured obstacle-free distance in front of the robot, in meters.

        corridor_center_x_robot_base:
            Lateral corridor-center offset in the robot's initial-yaw frame.

        heading_error_to_corridor:
            Angular error between the robot and corridor headings, in radians.

        start_movement:
            Whether an external trigger has enabled the control loop.

        movement_is_started:
            Whether translation has begun in the current movement state.

        set_model_state:
            ROS service proxy used to apply Gazebo model velocities.
    """

    def __init__(self):
        """
        Initialize control parameters, state, and ROS interfaces.

        Parameters are loaded from
        ``~door_traversal_controller`` and the related heading-estimation and
        hysteresis namespaces. The constructor then waits for Gazebo's
        ``/gazebo/set_model_state`` service, creates publishers, and subscribes
        to traversal triggers, passability results, odometry, and model poses.

        Raises:
            rospy.ROSInterruptException:
                If ROS shuts down while the constructor is waiting for a
                required service.

        Notes:
            Constructing this object initializes the ROS node and may block
            until the Gazebo model-state service becomes available.
        """
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
        self.kp_lateral_movement_pre_align_readjust_heading_state = rospy.get_param(f"{ns}/{spr}/kp_lateral_movement", 1.2)  # rad/s per meter lateral error

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
        self.kp_heading_align_state = rospy.get_param(f"{ns}/{sa}/kp_heading_align_state", 1.2)

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

        # Safety decisions arrive from noisy depth. Separate on/off counts let
        # the controller demand evidence without chattering at the threshold.
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
        """
        Latch an external request to begin a traversal cycle.

        Args:
            msg:
                ROS ``Bool`` message. A true value enables processing; false
                messages do not cancel an already latched request.
        """
        # This is a latch, not a joystick command; false messages do not cancel
        # a traversal already handed over to the controller.
        if msg.data:
            if not self.start_movement:
                self.start_movement = True

    def virtual_corridor_definition_finished_callback(self, msg):
        """
        Store the completion state of a requested virtual corridor.

        Args:
            msg:
                ROS ``Bool`` message published by the passability node.
        """
        self.virtual_corridor_definition_finished = msg.data


    def odom_callback(self, msg):
        """
        Cache the robot's planar pose from an odometry message.

        Args:
            msg:
                ROS ``Odometry`` message containing position and quaternion
                orientation.

        Notes:
            Roll, pitch, altitude, covariance, and twist are intentionally
            ignored because the controller operates in the ground plane.
        """
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.current_pose = (p.x, p.y)

        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)
        self.current_yaw = yaw

    def passability_callback(self, msg):
        """
        Update corridor geometry and filter the latest passability result.

        ``type_of_passability_check`` selects one of two independent results.
        Type ``1`` verifies clearance through and beyond the corridor midpoint;
        type ``2`` verifies only the minimum local clearance. Consecutive safe
        or unsafe samples are required before either stored result changes.

        Args:
            msg:
                ``Robot_passability`` message. Distances are expected in
                meters, lateral offsets use the camera or initial-yaw frames
                named by their fields, and ``heading_error`` is in radians.

        Notes:
            Receiving one check type clears the counters and result belonging
            to the other type. This prevents a stale local result from being
            treated as a current corridor result, or vice versa.
        """
        self.front_clearance = msg.front_clearance
        self.corridor_center_x_robot_base = msg.x_corridor_center_start_yaw_frame
        self.x_corridor_center_cam = msg.x_corridor_center_cam
        self.distance_to_corridor_mid_point = msg.corridor_middle_point_depth
        self.minimum_clearance_beyond_corridor_mid_point = msg.clearance_needed_beyond_mid_corridor
        self.heading_error_to_corridor = msg.heading_error

        # Raw passability from detector
        # Corridor and local checks answer different questions, so never let a
        # previously true result from one mode satisfy the other mode.
        if msg.type_of_passability_check == 1.0: # corridor passability check
            self.local_passable = False  # reset local passability. because we are doing corridor passability check now
            self.safe_seq_local = 0
            self.unsafe_seq_local = 0
            
            if self.front_clearance is not None:
                # Require sight past the midpoint, not merely free space up to
                # the door plane where the robot would have no room to recover.
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
        """
        Cache the current Gazebo pose of the Go1 model.

        Args:
            msg:
                ROS ``ModelStates`` message containing parallel model-name and
                pose arrays.

        Notes:
            Messages without a model named ``go1_gazebo`` are ignored.
        """
        if "go1_gazebo" not in msg.name:
            return
        index = msg.name.index("go1_gazebo")
        self.gazebo_pose = msg.pose[index]


    def data_ready(self):
        """
        Check whether the minimum feedback needed by the controller is ready.

        Returns:
            ``True`` after position, yaw, and front-clearance values have all
            been received; otherwise ``False``.
        """
        return (
            self.current_pose is not None
            and self.current_yaw is not None
            and self.front_clearance is not None
        )
    # =========================================================
    # Utility functions
    # =========================================================

    def stop_robot(self):
        """
        Command zero forward, lateral, and angular velocity.

        Notes:
            The stop command uses the same Gazebo model-state service as normal
            movement commands. If no Gazebo pose has arrived yet,
            :meth:`publish_cmd_vel` logs a warning and returns.
        """
        self.publish_cmd_vel(0.0, 0.0, 0.0)

    def wrap_angle(self, angle):
        """
        Normalize an angle to the principal signed range.

        Args:
            angle:
                Angle in radians. Scalars and NumPy-compatible arrays are
                accepted.

        Returns:
            The equivalent angle in the half-open range ``[-pi, pi)``.
        """
        return (angle + np.pi) % (2.0 * np.pi) - np.pi


    def publish_cmd_vel(self, v_forward, v_lateral, omega):
        """
        Apply a robot-frame velocity command to the Gazebo model.

        Linear velocity is rotated into the world frame while angular velocity
        remains a yaw rate about the world z axis. The latest Gazebo pose is
        copied into the request so the service changes velocity without
        teleporting the robot.

        Args:
            v_forward:
                Forward velocity in the robot frame, in meters per second.

            v_lateral:
                Lateral velocity in the robot frame, in meters per second.

            omega:
                Yaw rate in radians per second.

        Raises:
            rospy.ServiceException:
                If the Gazebo service call fails.

        Notes:
            The command is skipped when no Gazebo pose has been received. The
            world-frame conversion uses the latest odometry yaw.
        """

        if self.gazebo_pose is None:
            rospy.logwarn("Gazebo pose not yet received; cannot publish velocity and angular velocity to /set_model_state service")
            return

        # SetModelState replaces the whole state. Copying the latest pose avoids
        # accidentally teleporting the model while changing only its velocity.
       # DO NOT TOUCH pose at all
        self.modelstate.model_name = "go1_gazebo"
        self.modelstate.reference_frame = "world"

        # always use current gazebo pose
        self.modelstate.pose = self.gazebo_pose

        # Convert body-frame forward velocity to world frame
        # only forward velocity v is given in robot base frame no lateral velocity
        #v_lateral is always zero.
        # Gazebo expects world-frame linear velocity, whereas every controller
        # in this file produces commands in the robot body frame.
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
        Measure displacement along a fixed reference heading.

        The current world-frame translation is projected onto the heading that
        was active when the reference pose was recorded. Later yaw corrections
        therefore do not change the meaning of the travelled distance.

        Args:
            ref_pose:
                Reference planar position as ``(x, y)`` in meters.

            ref_yaw:
                Reference heading in radians.

        Returns:
            Signed forward displacement in meters, or ``0.0`` when current
            odometry is not available.
        """
        if self.current_pose is None:
            return 0.0

        dx = self.current_pose[0] - ref_pose[0]
        dy = self.current_pose[1] - ref_pose[1]

        return dx * np.cos(ref_yaw) + dy * np.sin(ref_yaw)
    
    def find_heading_error_estimate(self):
        """
        Update the motion-based estimate of corridor heading error.

        Lateral corridor-center drift is divided by measured forward motion and
        passed through a first-order low-pass filter. Updates are accepted only
        after enough forward displacement to avoid amplifying stationary
        measurement noise.

        Notes:
            The method updates ``heading_error_est``,
            ``heading_estimate_samples``, and ``heading_estimate_valid`` in
            place. It does not return the estimate.
        """
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
            # Dividing by near-zero motion would turn pixel/depth jitter into a
            # huge heading estimate, so update only after measurable travel.
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
        """
        Run the door-alignment and traversal finite-state machine.

        The loop waits for a trigger and valid feedback, performs two heading
        alignment stages and one position-alignment stage, turns to the final
        corridor heading, traverses the corridor, and continues beyond its
        midpoint. Unsafe feedback causes either an abort/back-off or a request
        for a virtual corridor, depending on where the robot is in the
        sequence.

        Raises:
            rospy.ServiceException:
                If a Gazebo model-state command cannot be delivered.

        Notes:
            This is a blocking loop intended to run for the lifetime of the
            node. It returns after the ``ABORT`` or ``DONE`` state completes,
            or when ROS shuts down.
        """
             

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

            # IDLE also gives passability hysteresis time to settle before the
            # first nonzero command is sent.
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
                # At this distance the camera-frame offset is the most direct
                # bearing cue to the opening.
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
                # Estimate where the present heading intersects the corridor
                # centerline; rotate until that intercept is comfortably nearby.
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
                omega = self.kp_lateral_movement_pre_align_readjust_heading_state * distance_offset * np.sign(lateral_error_in_robot_base)
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

                # This phase translates, so it uses the short-range robot-centric
                # safety check rather than the door corridor check.
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
                # Hold the intercept heading while forward motion removes lateral error.
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
                

                rospy.loginfo(f"ALIGN: current yaw is {self.current_yaw:.3f} rad, heading_error_to_corridor={self.heading_error_to_corridor:.3f} rad")
            
                # If heading is good enough, start traversal
                # Heading alone is insufficient: confirm the body also fits
                # laterally before committing to the narrow traversal phase.
                if abs(self.heading_error_to_corridor) < self.heading_error_tolerance_align_state: # heading error tolerance
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
                omega = self.kp_heading_align_state * self.heading_error_to_corridor
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
                        if self.corridor_mid_point_was_reached_for_previous_corridor_in_the_same_traversal_cycle is False:
                            rospy.logwarn("Passability lost → ABORT")
                            self.stop_robot()
                            self.state = ABORT
                            self.movement_is_started = False
                            continue
                        else:
                            # because instead of failing we can check other virtual corridor definition.because we already crossed door plane with glass so going any direction should be fine.  so we dont abort here. instead we will try to find another virtual corridor.
                            # mid corridor point is reached and go to MOVE_AFTER_CROSSING_CORRIDOR_MIDPOINT. so for that we are ading self.distance_to_corridor_mid_point to the distance_to_move_beyond_corridor_mid_point. so that we can go to  MOVE_AFTER_CROSSING_CORRIDOR_MIDPOINT state. and then in that state we will request for new corridor definition.
                            rospy.logwarn("Passability lost but virtual corridor definition already requested hence donot abort, go to MOVE_AFTER_CROSSING_CORRIDOR_MIDPOINT state and request new corridor definition")
                            self.stop_robot()
                            self.movement_is_started = False
                            distance_to_move_beyond_corridor_mid_point = self.distance_to_corridor_mid_point + self.minimum_clearance_beyond_corridor_mid_point + self.robot_length #because if the door open outward, we need to move sufficiently forward to be beyond the door swing area. also since the camera is on the head, we need to make sure the body ( behind) is beyond the door frame
                            self.request_local_passability_check_pub.publish(UInt8(data=2)) # request local passability check
                            self.state = MOVE_AFTER_CROSSING_CORRIDOR_MIDPOINT
                            # set reference  for move after crossing corridor midpoint state.
                            # here virtual corridor is not passable now. but we consider it as mid point is reached so that we can go the next state
                            # here actually we didnt reach the mid point thats why value of self.distance_to_corridor_mid_point is added to distance_to_move_beyond_corridor_mid_point. 
                            #until now the traversal 
                            self.start_pose_at_corridor_mid_point = self.current_pose
                            self.start_yaw_at_corridor_mid_point = self.current_yaw
                            continue


                self.movement_is_started = True
                
                # -----------------------------
                # Completion check
                # -----------------------------
                # Crossing the midpoint changes the safety model: from here the
                # controller watches immediate forward clearance and clears the
                # trailing body past the swing area.
                if self.distance_to_corridor_mid_point <= 0.3: # within 5 cm of door plane. now no chance to collide with closed glass door half
                    #at this point we also ensured that corridor in front has a clearance requested from passability node
                    rospy.loginfo("Door traversal DONE")
                    self.stop_robot()
                    self.movement_is_started = False
                    distance_to_move_beyond_corridor_mid_point = self.distance_to_corridor_mid_point + self.minimum_clearance_beyond_corridor_mid_point + self.robot_length #because if the door open outward, we need to move sufficiently forward to be beyond the door swing area. also since the camera is on the head, we need to make sure the body ( behind) is beyond the door frame
                    self.request_local_passability_check_pub.publish(UInt8(data=2)) # request local passability check
                    self.corridor_mid_point_was_reached_for_previous_corridor_in_the_same_traversal_cycle = True
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
                # The derivative damps rapid lateral drift; proportional terms
                # center the robot and align it with the corridor axis.
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
                # Slow down as measured clearance shrinks, while keeping enough
                # speed for the simulated gait/controller to make progress.
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
                # Measure against the crossing heading so subsequent yaw
                # corrections do not inflate the distance already travelled.
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

                        # Hand corridor selection back to perception; steering
                        # blindly around the new obstacle would be unsafe.
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
                # Remain stopped until the passability node acknowledges that
                # all following measurements refer to the replacement corridor.
                if not self.virtual_corridor_definition_finished:
                    rospy.loginfo("LOOK_FOR_A_VIRTUAL_CORRIDOR: waiting for new corridor to be defined")
                    #stay where you are
                    self.stop_robot()
                    continue
                else: # new corridor is defined
                    rospy.loginfo("LOOK_FOR_A_VIRTUAL_CORRIDOR complete → IDLE")
                    self.request_local_passability_check_pub.publish(UInt8(data=1)) # check corridor passability for the new corridor
                    self.state = IDLE
                    self.stop_robot()
                    continue


            # =================================================
            # ABORT
            # =================================================
            elif self.state == ABORT:
                rospy.logwarn("ABORT: backing off")

                # Back off a small distance
                # Back off along the current body heading to recreate stopping
                # distance before ending this controller cycle.
                backoff_start = self.current_pose
                self.corridor_mid_point_was_reached_for_previous_corridor_in_the_same_traversal_cycle = False 
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
                self.corridor_mid_point_was_reached_for_previous_corridor_in_the_same_traversal_cycle = False 
                # publish done signal in the DONE state  
                return

            self.rate.sleep()


if __name__ == "__main__":
    try:
        move_through_coridoor = DoorTraversalController()
        move_through_coridoor.control_loop()

    except rospy.ROSInterruptException:
        pass
