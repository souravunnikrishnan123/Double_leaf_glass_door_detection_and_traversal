#!/usr/bin/env python3
"""ROS node for defining a traversable corridor and monitoring its clearance.

The node activates after the door detector reports an opening. It projects
aligned depth into the robot frame, defines either a frame-anchored or virtual
corridor, updates that corridor from odometry, and publishes clearance and
heading information to the traversal controller.
"""
import os
import sys

import rospkg

# Ensure Python can import modules from this package's scripts directory
_pkg_path = rospkg.RosPack().get_path('robodog_glass_door_detection')
_scripts_path = os.path.join(_pkg_path, 'scripts')
if _scripts_path not in sys.path:
    sys.path.insert(0, _scripts_path)

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
from robodog_glass_door_detection.msg import Robot_passability
from processing_classes import backproject_depth_to_points
import cv2
import tf2_ros
from scipy.spatial.transform import Rotation as R



class PassabilityCheckerNode:
    """
    Define traversable corridors and monitor their obstacle clearance.

    The node is activated by a door-opening result. It combines the detected
    door geometry, aligned RGB-D data, camera-to-robot extrinsics, and odometry
    to build a corridor in a fixed reference frame. As the robot moves, the
    stored corridor is propagated into the current robot frame before the next
    clearance measurement is published to the traversal controller.

    The same processing loop can also perform short-range, robot-centric
    checks and can search the visible scene for a virtual corridor when the
    original door corridor is no longer usable.

    Attributes:
        active:
            Whether image and odometry processing is enabled.

        color_image:
            Latest BGR color frame, or ``None`` before one is received.

        depth_image:
            Latest aligned depth image in meters.

        fx:
            Camera focal length along the image x axis, in pixels.

        R_rc:
            Rotation matrix that maps camera-frame vectors into the robot
            ``trunk`` frame.

        t_rc:
            Camera origin expressed in the robot frame, in meters.

        requested_type_of_passability_check_from_traversal_node:
            Requested mode: ``0`` disables checks, ``1`` checks the stored
            corridor, and ``2`` checks the space directly ahead.

        corridor_middle_point_depth_dynamic_along_corridor_axis:
            Current signed distance to the corridor midpoint, measured along
            the corridor axis in meters.

        passability_checker:
            Point-cloud filtering pipeline used to find the nearest obstacle.
    """

    def __init__(self):
        """
        Initialize corridor state, parameters, and ROS interfaces.

        The constructor creates the always-on control subscribers and result
        publishers. High-bandwidth image topics and odometry remain
        unsubscribed until a suitable door state activates the node. Finally,
        the camera extrinsics are read from TF.

        Raises:
            tf2_ros.LookupException:
                If the requested camera transform is unavailable.

            tf2_ros.ConnectivityException:
                If TF cannot connect the two requested frames.

            tf2_ros.ExtrapolationException:
                If TF cannot provide a transform at the requested time.

        Notes:
            The ROS node itself is initialized by :func:`main`, before this
            class is constructed.
        """
        self.enable_visualization = rospy.get_param("~enable_visualization", False)
        self.color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        self.profile_enabled = rospy.get_param("~profiling_enabled", False)
        self.bridge = CvBridge()
        ns = "~passabilility_check"
        # ---- State ----
        self.active = False
        self.define_new_corridor_req_from_frame_detection = False
        self.door_status_from_frame_detection_node = None
        self.mid_frame_x_px_from_frame_detection_node = None
        self.door_depth_from_frame_detection_node = None
        self.plane_info_distance_m = None
        self.plane_info_norm_camera_frame = None
        self.x_corridor_center_cam_first_time = None
        # Additional state for traversal-requested corridor definition and publishing
        self.define_new_corridor_req_from_traversal = False
        self.corridor_middle_point_depth_from_traversal = None
        self.virtual_corridor_definition_finished = False
        self.clearance_needed_beyond_mid_corridor = None
        self.corridor_middle_point_depth_first_time_camera_frame = None


        # Odometry
        self.start_pose_at_node_activation = None
        self.start_yaw_at_node_activation = None
        self.current_pose = None
        self.current_yaw = None

        #dynamic values
        self.corridor_middle_point_depth_dynamic_along_corridor_axis = None
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
        self.robot_height_above_camera_level = rospy.get_param(f"{ns}/traversal_params/robot_height_above_camera_level", 0.5)  # meters
        #pass ability decision parameters
    
        # traversal corridor parameter
        self.min_corridor_clearance_beyond_door_to_trigger_traversal_node = rospy.get_param(f"{ns}/traversal_params/min_corridor_clearance_beyond_door_to_trigger_traversal_node", 0.7)  # meters
        self.robot_width = rospy.get_param(f"{ns}/traversal_params/robot_width", 0.45)  # meters ,robot  width
        self.robot_length = rospy.get_param(f"{ns}/traversal_params/robot_length", 0.7)  # meters
        self.safety_margin_robot_width = rospy.get_param(f"{ns}/traversal_params/safety_margin_robot_width", 0.05)  # meters
        self.door_frame_margin = rospy.get_param(f"{ns}/traversal_params/door_frame_margin", 0.05)  # meters
        
        # Every corridor is sized for the robot body plus clearance on one side;
        # using a half-width keeps the later symmetric bounds readable.
        self.half_width = (self.robot_width /2.0) + self.safety_margin_robot_width

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


        # Keep small control topics alive so the activation handshake cannot miss
        # them. High-bandwidth image subscriptions are created only while active.
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

        rospy.Subscriber(
                "/glass_door_detection/plane_info",
                Twist,
                self._plane_info_cb,
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
            Robot_passability,
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

        self.R_rc, self.t_rc = self.get_camera_extrinsics_tf()
        # Single-precision copies for the bulk point-cloud transform. The clouds are
        # float32, so using the float64 originals would widen the whole array and
        # double the memory traffic of the hottest transform in the node.
        self.R_rc_f32 = np.asarray(self.R_rc, dtype=np.float32)
        self.t_rc_f32 = np.asarray(self.t_rc, dtype=np.float32)

        # Stage timings are flushed on a slow timer instead of once per control
        # iteration; the writer rewrites the whole file on every call.
        if self.profile_enabled:
            rospy.Timer(rospy.Duration(5.0), self._write_durations)

    def _write_durations(self, _event=None):
        """
        Flush stage timing averages to disk.

        Notes:
            Driven by a low-rate timer. Failures are logged rather than raised so
            a profiling problem cannot stop the passability loop.
        """
        try:
            get_duration_seconds.write_text_file()
        except Exception as exc:
            rospy.logwarn_throttle(30.0, "Could not write durations file: %s", exc)



    # -----------------------------
    # Activation logic called upon door state changes
    # only update states, store data and activation/deactivation of camera subscriptions
    # call backs shall be non blocking and lightweight
    # -----------------------------

    def get_camera_extrinsics_tf(self,
                                robot_frame="trunk",
                                camera_frame="realsense_d455_optical_frame"
                                ):
        """
        Read the camera pose in the robot frame from TF.

        The returned transform maps an optical-frame point ``p_c`` to the
        robot frame as ``p_r = R_rc @ p_c + t_rc``. The camera optical axes are
        right, down, and forward; the robot axes are forward, left, and up.

        Args:
            robot_frame:
                Target robot frame in which points should be expressed.

            camera_frame:
                Source optical frame attached to the depth camera.

        Returns:
            A tuple ``(R_rc, t_rc)`` containing a ``3 x 3`` rotation matrix and
            a three-element translation vector in meters.

        Raises:
            tf2_ros.LookupException:
                If no transform is available for the named frames.

            tf2_ros.ConnectivityException:
                If the TF tree cannot connect the frames.

            tf2_ros.ExtrapolationException:
                If TF cannot satisfy the requested timestamp.

        Notes:
            This method is intended to run once after the robot and camera
            frames have been published. It waits up to five seconds in the
            initial availability check.

        """


        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)

        rospy.loginfo("Waiting for TF %s -> %s",
                    robot_frame, camera_frame)

        # Wait until TF is available
        tf_buffer.can_transform(
            robot_frame,
            camera_frame,
            rospy.Time(0),
            rospy.Duration(5.0)
        )

        # Asking for robot <- camera makes the returned matrix directly usable
        # as p_robot = R_rc * p_camera + t_rc throughout this node.
        transform = tf_buffer.lookup_transform(
            robot_frame,
            camera_frame,
            rospy.Time(0)
        )

        # Translation
        t = transform.transform.translation
        t_rc = np.array([t.x, t.y, t.z])

        # Rotation (quaternion → matrix)
        q = transform.transform.rotation
        quat = [q.x, q.y, q.z, q.w]
        R_rc = R.from_quat(quat).as_matrix()

        rospy.loginfo("Camera extrinsics loaded from TF")
        print("R_rc =\n", R_rc)
        test = np.array([0, 0, 1])  # camera forward
        print("Camera forward in robot frame:", R_rc @ test)



        return R_rc, t_rc


    def _activate_node_cb(self, msg):
        """
        Activate passability processing for an eligible door state.

        Args:
            msg:
                ROS ``String`` containing the detected door state.

        Notes:
            ``open_left``, ``open_right``, ``No_door_plane_detected``, and
            ``no_frame_detected`` start a new cycle. The first accepted state is
            retained as the basis for corridor definition.
        """
        # The two failure labels are accepted because a fully open doorway may
        # provide no pane or center frame even though free space is visible.
        
        if msg.data in ["open_left", "open_right", "No_door_plane_detected", "no_frame_detected"]: # one time activation on these states
            if not self.active:
                rospy.loginfo("PassabilityChecker: activated and door status is %s", msg.data)
                self.active = True
                #to create subscriptions
                self.activate_camera()
                self.get_odom()
                # store the door open status only once at activation
                self.door_status_from_frame_detection_node = msg.data 
                self.define_new_corridor_req_from_frame_detection = True  # to define new corridor on next run
                
            

    def _deactivate_node_cb(self, msg):
        """
        End the active passability cycle after traversal completes.

        Args:
            msg:
                ROS ``Bool`` message. Only a true value causes deactivation.

        Notes:
            The callback unregisters conditional subscribers, clears
            cycle-specific geometry, and asks the door detector to start its
            next detection cycle.
        """
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
            self.plane_info_distance_m = None
            self.plane_info_norm_camera_frame = None
            self.virtual_corridor_definition_finished = False
            self.corridor_middle_point_depth_from_traversal = None
            # This acknowledgement closes the cycle and lets the detector clear
            # its latched result before looking for the next door.
            self.retrigger_door_detection_node_pub.publish(True) # to retrigger door detection node for next detection and traversal cycle


            
    
    # -----------------------------
    # Conditional subscriptions
    # -----------------------------
    def activate_camera(self):
        """
        Subscribe to the color, aligned-depth, and camera-info topics.

        Notes:
            Each subscriber is created only if it is not already active, so
            repeated activation requests are safe.
        """
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
        """
        Unregister camera subscribers and discard their cached data.

        Notes:
            Clearing the cached frames forces a later activation cycle to wait
            for fresh camera data instead of reusing an old observation.
        """
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
        """
        Subscribe to robot odometry when it is not already active.

        Notes:
            The first odometry sample received after subscription establishes
            the node-activation reference pose.
        """
        if self.odom_sub is None:
            self.odom_sub = rospy.Subscriber(
                "/odom_bridge_output",
                Odometry,
                self._odom_callback,
                queue_size=1
            )

    def deactivate_odom(self):
        """
        Unregister odometry and clear all cached planar poses.

        Notes:
            Both the current pose and the activation reference are reset so
            that the next cycle starts in a new local reference frame.
        """
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
        """
        Convert and cache the latest ROS color image.

        Args:
            color_msg:
                ROS ``Image`` converted to an OpenCV BGR image with
                ``CvBridge``.

        Raises:
            cv_bridge.CvBridgeError:
                If the message cannot be converted as ``bgr8``.
        """
        self.color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

    def _depth_cb(self, depth_msg):
        """
        Convert and cache the latest depth image in meters.

        Args:
            depth_msg:
                ROS ``Image`` containing millimeter ``16UC1``/``mono16`` depth
                or a floating-point depth representation.

        Raises:
            cv_bridge.CvBridgeError:
                If the ROS image cannot be converted.
        """
        img = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        if depth_msg.encoding in ("16UC1", "mono16"):
            # The intermediate uint16 copy is a no-op for an already-16-bit image;
            # convert once and scale by a reciprocal instead.
            depth_m = img.astype(np.float32) * np.float32(0.001)
        else:
            depth_m = img.astype(np.float32)

        self.depth_image = depth_m

    def _info_cb(self, info_msg: CameraInfo):
        """
        Cache camera calibration from the first camera-info message.

        Args:
            info_msg:
                ROS ``CameraInfo`` whose intrinsic matrix follows the standard
                row-major ``K`` layout.

        Notes:
            Calibration is treated as constant during a run; later messages
            are ignored.
        """
        # storing the camera intrinsics only once. it will not change during runtime
        if self.latest_info is None:
            self.latest_info = info_msg
            K = info_msg.K
            self.fx, self.fy = K[0], K[4]
            self.cx, self.cy = K[2], K[5]
            rospy.loginfo("Camera intrinsics received and stored.")
    
    def _mid_frame_x_px_cb(self, msg):
        """
        Store the first valid door-frame midpoint pixel of an active cycle.

        Args:
            msg:
                ROS ``Int32`` containing the horizontal image coordinate.

        Notes:
            Non-positive values and samples received while inactive are
            ignored.
        """
        if msg.data <= 0: # invalid pixel
            return
        # only update if not set. which means take only first valid pixel after activation
        # checking self.active will make sure we get door depth and mid frame only when door state is open, No_door_plane_detected", "no_frame_detected
        # Latch the value that belongs to activation; later detector frames may
        # refer to a different instantaneous ROI while the robot is moving.
        if self.mid_frame_x_px_from_frame_detection_node is None and self.active:
            self.mid_frame_x_px_from_frame_detection_node = msg.data

    def _plane_info_cb(self, msg):
        """
        Store the first valid detected door-plane measurement.

        Args:
            msg:
                ROS ``Twist`` used as a compact container. ``linear.x`` is
                plane distance in meters, while ``angular.(x, y, z)`` contains
                the camera-frame normal.

        Notes:
            A non-positive distance is invalid. Only the first valid sample in
            an active cycle is retained.
        """
        # plane info is published as Twist for simplicity, where:
        # linear.x = distance_m, angular.x = plane_norm_x, angular.y = plane_norm_y, angular.z = plane_norm_z
        if msg.linear.x <= 0.0: # invalid distance
            return
        if self.plane_info_distance_m is None and self.active:
            self.plane_info_distance_m = msg.linear.x
            self.plane_info_norm_camera_frame = np.array([msg.angular.x, msg.angular.y, msg.angular.z])
            rospy.loginfo(f"Received plane info from frame detection node: distance {self.plane_info_distance_m:.2f} m, normal vector {self.plane_info_norm_camera_frame}")

    def _request_local_passability_check_cb(self, msg):
        """
        Store the passability mode requested by the traversal controller.

        Args:
            msg:
                ROS ``UInt8``: ``0`` disables checking, ``1`` requests the
                corridor mode, and ``2`` requests the local mode.
        """
        if msg.data in [0, 1, 2]: # 1 for corridor , 2 for local
            self.requested_type_of_passability_check_from_traversal_node = msg.data
    
    def _request_new_corridor_definition_cb(self, msg):
        """
        Record a traversal-side request for a virtual corridor.

        Args:
            msg:
                ROS ``Twist`` used as a request container. A truthy
                ``linear.x`` enables the request and ``linear.y`` supplies the
                desired corridor-midpoint depth in meters.

        Notes:
            Requests are accepted only while the node is active.
        """
        if msg.linear.x:  # only activate on True
            if self.active:
                rospy.loginfo("PassabilityChecker: New corridor definition requested by traversal node")
                self.define_new_corridor_req_from_traversal = True  # to define new corridor on next run
                self.corridor_middle_point_depth_from_traversal = msg.linear.y  # to define new corridor based on this depth


    
    def _door_depth_cb(self, msg):
        """
        Store the first valid detected door depth of an active cycle.

        Args:
            msg:
                ROS ``Float32`` containing depth in meters.

        Notes:
            Non-positive samples and later samples from the same cycle are
            ignored.
        """
        if msg.data <= 0.0: # invalid depth
            return
        # only update if not set. which means take only first valid depth after activation
        # checking self.active will make sure we get door depth and mid frame only when door state is open, No_door_plane_detected", "no_frame_detected
        if self.door_depth_from_frame_detection_node is None and self.active:
            self.door_depth_from_frame_detection_node = msg.data

    def _odom_callback(self, msg):
        """
        Cache planar odometry and establish the activation reference pose.

        Args:
            msg:
                ROS ``Odometry`` containing the robot position and quaternion
                orientation.

        Notes:
            The first sample after activation is stored separately so later
            displacement can be measured relative to the start of the cycle.
        """
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
        """
        Check whether initial corridor-definition inputs are available.

        Returns:
            ``True`` when color, depth, intrinsics, door state, door depth, and
            the activation pose have all been received; otherwise ``False``.
        """
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
        Measure motion along the corridor-definition heading.

        Returns:
            Signed displacement in meters along the yaw recorded when the
            current corridor was defined.

        Notes:
            Projecting world translation onto a fixed heading keeps the result
            independent of later yaw corrections.
        """

        dx = self.current_pose[0] - self.start_pose_at_corridor_definition[0]
        dy = self.current_pose[1] - self.start_pose_at_corridor_definition[1]

        return dx * np.cos(self.start_yaw_at_corridor_definition) + dy * np.sin(self.start_yaw_at_corridor_definition)
    
    def get_lateral_displacement_since_start(self):
        """
        Measure motion perpendicular to the corridor-definition heading.

        Returns:
            Signed lateral displacement in meters in the reference frame that
            was active when the corridor was defined.

        Notes:
            The sign follows the left-handed lateral basis produced by
            ``(-sin(yaw), cos(yaw))``.
        """

        dx = self.current_pose[0] - self.start_pose_at_corridor_definition[0]
        dy = self.current_pose[1] - self.start_pose_at_corridor_definition[1]

        return -dx * np.sin(self.start_yaw_at_corridor_definition) + dy * np.cos(self.start_yaw_at_corridor_definition)
    
    def calculate_dynamic_x_corridor_center_in_cameraframe(self):
        """
        Estimate the apparent lateral corridor shift from odometry.

        Translation perpendicular to the initial heading and the image shift
        caused by accumulated yaw are added together.

        Returns:
            Estimated horizontal corridor-center displacement in meters.

        Notes:
            The rotational term uses the current distance to the corridor
            midpoint, so the estimate grows as heading error acts over a longer
            viewing distance.
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

        rotation_shift = (self.corridor_middle_point_depth_dynamic_along_corridor_axis * np.tan(delta_yaw))
    
        
        rospy.loginfo(f"start pose is {self.start_pose_at_corridor_definition} and current pose is {self.current_pose}   Lateral displacement from start: {lateral_displacement:.3f} m   Rotation shift: {rotation_shift:.3f} m")
        
        
        return (
        lateral_displacement
        + rotation_shift
    )

    

    def find_best_corridor_center_based_on_final_points(self, final_points_robot_frame, z_max, min_clearance):
        """
        Slide a robot-width window and choose a clear centerline.

        Candidate centers span the camera field of view in corridor
        coordinates. A candidate is valid when its nearest obstacle is beyond
        ``min_clearance``. Among valid candidates, the center closest to the
        robot's current forward axis is preferred.

        Args:
            final_points_robot_frame:
                ``N x 2`` array of filtered obstacle points on the robot-frame
                ground plane, in meters.

            z_max:
                Far search distance in meters. It defines the visible lateral
                span and is used as clearance when a candidate contains no
                obstacle points.

            min_clearance:
                Minimum acceptable forward clearance in meters.

        Returns:
            Lateral center coordinate of the best corridor in meters, or
            ``None`` when no robot-width candidate meets the clearance
            requirement.

        Notes:
            Candidate width includes the configured robot half-width and safety
            margin.
        """
        # find best corridor center based on final points obtained from passability checker
        # final points are in robot frame


        # Work in the corridor basis, where each candidate becomes a simple
        # lateral interval and its clearance is the minimum forward coordinate.
        forward_coordinate_of_points_in_corridor_frame = final_points_robot_frame@self.corridor_forward_2d_unit_vector_first_time_robot_frame
        lateral_coordinate_of_points_in_corridor_frame = final_points_robot_frame@self.corridor_lateral_2d_unit_vector_first_time_robot_frame


        H, W = self.depth_image.shape
        x_edge_camera_frame = (W / 2.0) * z_max / self.fx

        left_edge_camera_frame  = np.array([-x_edge_camera_frame, 0.0, z_max])
        right_edge_camera_frame = np.array([ x_edge_camera_frame, 0.0, z_max])

        left_edge_robot_frame = self.R_rc @ left_edge_camera_frame + self.t_rc
        right_edge_robot_frame = self.R_rc @ right_edge_camera_frame + self.t_rc

        left_edge_corridor_frame = left_edge_robot_frame[:2] @ self.corridor_lateral_2d_unit_vector_first_time_robot_frame
        right_edge_corridor_frame = right_edge_robot_frame[:2] @ self.corridor_lateral_2d_unit_vector_first_time_robot_frame

        lat_min_corridor_frame = min(left_edge_corridor_frame, right_edge_corridor_frame)
        lat_max_corridor_frame = max(left_edge_corridor_frame, right_edge_corridor_frame)
        # Keep the full robot footprint inside the visible field of view.
        slide_centers_corridor_frame = np.arange(lat_min_corridor_frame + self.half_width, lat_max_corridor_frame - self.half_width, self.slide_step_x_direction)
        # because lateral min and max depends on entire image,not just on the obstacle points. 

        # A candidate corridor is accepted when its nearest obstacle is at least
        # min_clearance away. Rather than slicing the cloud once per candidate,
        # note that this is equivalent to asking whether any point closer than
        # min_clearance falls inside the candidate's lateral window. Sorting the
        # lateral coordinates once turns every candidate into two binary searches.
        x_left_all = slide_centers_corridor_frame - self.half_width
        x_right_all = slide_centers_corridor_frame + self.half_width

        lateral_sorted = np.sort(lateral_coordinate_of_points_in_corridor_frame)
        blocking_lateral_sorted = np.sort(
            lateral_coordinate_of_points_in_corridor_frame[
                forward_coordinate_of_points_in_corridor_frame < min_clearance
            ]
        )

        # Counts over the open interval (x_left, x_right), matching the strict
        # comparisons the per-candidate test used.
        def count_within(sorted_values):
            return (
                np.searchsorted(sorted_values, x_right_all, side="left")
                - np.searchsorted(sorted_values, x_left_all, side="right")
            )

        blocking_count = count_within(blocking_lateral_sorted)
        total_count = count_within(lateral_sorted)

        # An empty window is treated as clear out to z_max, so it qualifies only
        # if z_max itself satisfies the clearance requirement.
        valid_corridor_mask = (blocking_count == 0) & ((total_count > 0) | (z_max >= min_clearance))

        if not np.any(valid_corridor_mask):
            rospy.logwarn("Cannot find any valid virtual corridor for passability check.")
            # there is no virtual corridor available. you cannot consider in front of the robot as coridor. because there may be no passable corridor at all
            return None

        # Prefer the smallest steering correction once safety requirements are met.
        # argmin returns the first minimum, preserving the original tie-breaking
        # towards the earliest candidate in the sweep.
        valid_centers = slide_centers_corridor_frame[valid_corridor_mask]
        best_x_center = valid_centers[np.argmin(np.abs(valid_centers))]

        rospy.loginfo(f"Virtual corridor center defined at x: {best_x_center:.3f} m in camera frame")
        return best_x_center
    

    def find_a_virtual_corridor(self, z_min, z_max, min_clearance):
        """
        Search the full depth image for a clear robot-width corridor.

        Depth pixels in the requested range are back-projected, filtered by the
        normal/outlier/connected-component pipeline, transformed to the robot
        frame, and passed to the sliding-window corridor search.

        Args:
            z_min:
                Nearest depth included in the search, in meters.

            z_max:
                Farthest depth included in the search, in meters.

            min_clearance:
                Required obstacle-free distance in meters.

        Returns:
            A tuple ``(center, image)``. ``center`` is the selected lateral
            coordinate in the corridor frame, ``0.0`` when filtering finds no
            obstacles, or ``None`` when obstacles leave no valid candidate.
            ``image`` is the passability visualization.

        Notes:
            An empty point cloud is interpreted as open space within the
            requested depth interval, so the forward axis is selected.
        """
        rospy.loginfo("Finding virtual corridor for passability check.")
       

        # run passability check for full width of image to find a virtual corridor

        # Search the whole image here; there is no trusted frame edge from which
        # to build a narrower ROI.
        valid_points_3d_for_virtual_corridor_definition_camera_frame, uv_for_virtual_corridor_definition_camera_frame, _ = backproject_depth_to_points(
            self.depth_image, self.fx, self.fy, self.cx, self.cy,
            max_depth=z_max, min_depth=z_min,
            subsample=self.subsample, roi_polygon=None
        )
        
        if len(valid_points_3d_for_virtual_corridor_definition_camera_frame) == 0:
            rospy.logwarn("No valid depth points after back-projection. Cannot define virtual corridor. but it is also possible that there is no points at all because entire region in front is free because of small depth range. consider virtual corridor is directly in front of robot.")
            x_virtual_corridor_center_corridor_frame = 0.0
            return x_virtual_corridor_center_corridor_frame, self.color_image
        else:
            corridor_mask_camera_frame = (
                valid_points_3d_for_virtual_corridor_definition_camera_frame[:, 1] > -self.robot_height_above_camera_level
            )
            #we are in camera frame so y down is positive. so we check if points are below the camera height (which means above the ground) by checking if y is greater than -robot_height_above_camera_level. level of camera is 0 in camera frame, and ground is at positive in camera frame. 
            # no image cropping. we need to check entire image for passability. 
            corridor_pts_for_virtual_corridor_definition_camera_frame = valid_points_3d_for_virtual_corridor_definition_camera_frame[corridor_mask_camera_frame]
            corridor_uv_for_virtual_corridor_definition_camera_frame = uv_for_virtual_corridor_definition_camera_frame[corridor_mask_camera_frame]
            
            _, image, final_points_3d_for_virtual_corridor_definition_camera_frame = self.passability_checker.run(
                self.depth_image,
                self.color_image,
                corridor_pts_for_virtual_corridor_definition_camera_frame,
                corridor_uv_for_virtual_corridor_definition_camera_frame,
                z_max
            )

            if final_points_3d_for_virtual_corridor_definition_camera_frame is not None and len(final_points_3d_for_virtual_corridor_definition_camera_frame) > 0:
                # Equivalent to (R @ p.T).T + t, but without the two transposes and
                # the float64 promotion of the whole cloud.
                final_points_3d_for_virtual_corridor_definition_robot_frame = (
                    final_points_3d_for_virtual_corridor_definition_camera_frame @ self.R_rc_f32.T + self.t_rc_f32
                )
                # Keep only ground-plane components. Project to ground plane (robot X–Y)
                final_points_2d_for_virtual_corridor_definition_robot_frame = final_points_3d_for_virtual_corridor_definition_robot_frame[:, :2]  # Extract 2D points (x, y)
                # donot use the value of front clearance directly because corridor is not defined yet.  this front clerance is for full width of image
                # we need to find a virtual corridor center based on final points obtained from passability checker
                

                x_virtual_corridor_center_corridor_frame = self.find_best_corridor_center_based_on_final_points(final_points_2d_for_virtual_corridor_definition_robot_frame, z_max, min_clearance)
                #x_virtual_corridor_center_corridor_frame value can be None or a float value. if it is None, it means no valid corridor is found. if it is a float value, it means the x center of the virtual corridor in the corridor frame is at that value. positive value means right side of the robot and negative value means left side of the robot. and 0 means directly in front of the robot.
            else: # in this case, consider image center as virtual corridor center
                rospy.logwarn("Cannot find any valid points after filtering for virtual corridor definition. This is possible that there is no points left after filtering, means entire region in front is free. consider virtual corridor is directly in front of robot.")
                x_virtual_corridor_center_corridor_frame = 0.0
            
            return x_virtual_corridor_center_corridor_frame, image

    def relative_robot_motion(self):
        """
        Compute robot motion since the current corridor was defined.

        Returns:
            A tuple ``(R, t)``. ``R`` is the planar rotation from the corridor
            definition orientation to the current orientation, and ``t`` is
            current translation expressed in the definition frame.

        Notes:
            Yaw difference is wrapped to ``[-pi, pi]`` before constructing the
            rotation matrix.
        """

        dx_world = self.current_pose[0] - self.start_pose_at_corridor_definition[0]
        dy_world = self.current_pose[1] - self.start_pose_at_corridor_definition[1]
        delta_yaw = self.current_yaw - self.start_yaw_at_corridor_definition
        delta_yaw = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))

        # Rotation matrix (start -> current)
        R = np.array([
            [ np.cos(delta_yaw), -np.sin(delta_yaw)],
            [ np.sin(delta_yaw),  np.cos(delta_yaw)]
        ])

        R_world_to_R0 = np.array([
            [ np.cos(self.start_yaw_at_corridor_definition),  np.sin(self.start_yaw_at_corridor_definition)],
            [-np.sin(self.start_yaw_at_corridor_definition),  np.cos(self.start_yaw_at_corridor_definition)]
        ])

        # Express translation in the definition frame before using it to move a
        # fixed corridor point into the robot's current frame.
        t = R_world_to_R0 @ np.array([dx_world, dy_world])
        return R, t

    def propagate_corridor_point(self, corridor_point_R0,R, t):
        """
        Propagate a fixed corridor point into the current robot frame.

        Args:
            corridor_point_R0:
                Two-element corridor point expressed in the robot frame at
                corridor-definition time.

            R:
                Current planar rotation relative to the definition frame, as
                returned by :meth:`relative_robot_motion`.

            t:
                Current planar translation in the definition frame.

        Returns:
            Two-element ``[forward, lateral]`` point in the current robot
            frame.
        """

        # Extract x,z (planar)
        p0 = np.array([
            corridor_point_R0[0],
            corridor_point_R0[1]
        ])

        # Inverse robot motion
        # The corridor is fixed in the world, so apply the inverse of robot motion.
        p_now = R.T @ (p0 - t)

        return p_now  # [X_forward, Y_lateral] in robot frame
    
    def project_to_pixel(self, P_cam):
        """
        Project one camera-frame point into image coordinates.

        Args:
            P_cam:
                Three-element ``(X, Y, Z)`` camera-frame point in meters.

        Returns:
            Integer ``(u, v)`` pixel coordinates, or ``None`` when the point is
            on or behind the camera plane.
        """
        X, Y, Z = P_cam
        if Z <= 0:
            return None
        u = int(self.fx * X / Z + self.cx)
        v = int(self.fy * Y / Z + self.cy)
        return (u, v)


    def run(self):
        """
        Define, propagate, and evaluate corridors until ROS shuts down.

        On activation, the loop first defines a corridor from detected door
        geometry. A later traversal request can replace it with a virtual
        corridor selected from the visible free space. Each iteration uses
        odometry to express the fixed corridor in the current robot frame,
        chooses either corridor or local passability mode, filters the
        relevant RGB-D points, and publishes clearance and alignment data.

        The traversal trigger is latched after the original corridor first
        provides enough clearance through and beyond the door. Optional
        visualization projects the evaluated corridor boundaries back into the
        color image.

        Raises:
            cv_bridge.CvBridgeError:
                If a generated visualization cannot be converted to a ROS
                image outside the guarded visualization-publish block.

        Notes:
            This is a blocking node loop. Subscriber callbacks only cache
            state; all expensive point-cloud work is performed here at the
            configured five-hertz rate.
        """
        while not rospy.is_shutdown():
            if not self.active: # node inactive, skip processing. node will be activated when door state is open_left or open_right and No_door_plane_detected", "no_frame_detected
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
            # Snapshot callback-owned mode once per loop; a mid-cycle update is
            # intentionally applied on the next 5 Hz iteration.
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
                
                # find the corridor based on frame detection node data(x_center_frame_first_time_camera_frame)
                self.corridor_middle_point_depth_first_time_camera_frame = self.door_depth_from_frame_detection_node
                    
                # The ground-plane projection of the door normal defines the
                # corridor forward axis; its perpendicular defines lateral error.
                self.plane_info_norm_camera_frame /= np.linalg.norm(self.plane_info_norm_camera_frame) # normalize plane normal vector
                plane_info_norm_robot_frame = self.R_rc @ self.plane_info_norm_camera_frame 
                rospy.loginfo(f"Plane normal in robot base frame: {plane_info_norm_robot_frame}")
                self.corridor_forward_2d_unit_vector_first_time_robot_frame = np.array([
                    plane_info_norm_robot_frame[0],   # X ( forward in robot base )
                    plane_info_norm_robot_frame[1]    # Y ( left in robot base )
                ]) # y component of the normal is ignored. because that describe the tilt of the plane which is not relevant for corridor definition. we only care about the normal direction parallel to the ground plane which is defined by x and z in robot base frame
                
                norm = np.linalg.norm(self.corridor_forward_2d_unit_vector_first_time_robot_frame)
                if norm < 1e-6:
                    raise ValueError("Plane normal is nearly vertical — cannot define corridor axis")
                self.corridor_forward_2d_unit_vector_first_time_robot_frame /= norm

                

                """
                # Ensure corridor axis points away from robot. After this: forward means “towards the corridor”
                if np.dot(self.corridor_forward_2d_unit_vector_first_time_robot_frame, Point_mid_frame_first_time_2d_robot_frame) < 0:
                    self.corridor_forward_2d_unit_vector_first_time_robot_frame *= -1.0
                """

                self.corridor_lateral_2d_unit_vector_first_time_robot_frame = np.array([
                                            -self.corridor_forward_2d_unit_vector_first_time_robot_frame[1],
                                            self.corridor_forward_2d_unit_vector_first_time_robot_frame[0]
                                        ])
                self.corridor_lateral_2d_unit_vector_first_time_robot_frame /= np.linalg.norm(self.corridor_lateral_2d_unit_vector_first_time_robot_frame)  # left or right direction parallel to the ground plane and perpendicular to the corridor forward direction
                
                # convert to 3d by adding z component. this is because we need to calculate the offset_3d_robot_frame in 3d space and then add it to the mid frame point in 3d space to get the corridor center point in 3d space. if we do the calculation in 2d space, we will lose the z component which is important for accurate corridor center point calculation in 3d space
                #lift axes to 3d just to calculate the offset in 3d space. because we need to consider the z component of the plane normal for accurate corridor center point calculation in 3d space. if we do the calculation in 2d space, we will lose the z component which is important for accurate corridor center point calculation in 3d space
                # but for the calculation of lateral error and forward error for passability check, we will consider only the x and y component of the corridor center point in robot frame because we are interested in the ground plane distance of the robot from the corridor center line. and for that we will project the robot position and corridor center point to the ground plane by ignoring the z component. because the z component may have noise and errors which can affect the accuracy of the forward and lateral error calculation for passability check. so for corridor center point calculation we will consider z component but for error calculation we will ignore z component by projecting to ground plane. this is a design choice to improve the robustness of the passability check against noise in depth data which can affect the accuracy of z component of corridor center point in robot frame.

                
                corridor_lateral_3d_unit_vector_robot_frame = np.array([
                    self.corridor_lateral_2d_unit_vector_first_time_robot_frame[0],
                    self.corridor_lateral_2d_unit_vector_first_time_robot_frame[1],
                    0.0
                ]) # convert to 3d by adding z component. this is because we need to calculate the offset_3d_robot_frame in 3d space and then add it to the mid frame point in 3d space to get the corridor center point in 3d space. if we do the calculation in 2d space, we will lose the z component which is important for accurate corridor center point calculation in 3d space
                

                rospy.loginfo(f"Corridor forward 2D unit vector: {self.corridor_forward_2d_unit_vector_first_time_robot_frame}, Corridor lateral 2D unit vector: {self.corridor_lateral_2d_unit_vector_first_time_robot_frame}")
                

                if self.door_status_from_frame_detection_node in ["open_left", "open_right"]: # fixed corridor. and corridor is defined only when door is open
                    # Convert door mid-frame pixel to X (meters)
                    # Backproject only X because depth and the relevant pixel row
                    # are already known; height is discarded for ground navigation.
                    x_center_frame_first_time_camera_frame = (self.mid_frame_x_px_from_frame_detection_node - self.cx) * self.corridor_middle_point_depth_first_time_camera_frame / self.fx
                    Point_mid_frame_first_time_3d_camera_frame = np.array([x_center_frame_first_time_camera_frame, 0.0, self.corridor_middle_point_depth_first_time_camera_frame]) # in camera frame
                    # transform mid frame point from camera frame to robot base frame at start yaw frame. 
                    Point_mid_frame_first_time_3d_robot_frame = self.R_rc @ Point_mid_frame_first_time_3d_camera_frame + self.t_rc
                    rospy.loginfo(f"mid frame point in camera frame  : {Point_mid_frame_first_time_3d_camera_frame} Mid frame point in robot base frame at start yaw frame: {Point_mid_frame_first_time_3d_robot_frame}")
                    # vector from robot to mid frame point in robot base frame at start yaw frame. because we are interested in the relative position of mid frame point with respect to robot, so we need to subtract the camera translation in robot base frame
                    # projection to ground plane
                    #Point_mid_frame_first_time_2d_robot_frame = np.array([Point_mid_frame_first_time_3d_robot_frame[0], Point_mid_frame_first_time_3d_robot_frame[1]]) # only consider x(forward) and y(left) because we are interested in the ground plane
                    
                    #x_center_frame_start_yaw_frame_first_time = x_center_frame_first_time_camera_frame

                    # ---------------------------------------
                    # Define robot-centric traversal corridor for corridor passability check
                    # ---------------------------------------
                    if self.door_status_from_frame_detection_node == "open_left":

                        sign = 1.0 # for open_left, corridor center is on the left side of mid frame point in camera frame, which means negative x direction in camera frame

                    else: # means "open_right":

                        sign = -1.0 # for open_right, corridor center is on the right side of mid frame point in camera frame, which means positive x direction in camera frame

                    # Shift one half robot-width into the detected opening. The
                    # central frame edge itself is not a safe centerline.
                    offset_3d_robot_frame = sign * self.half_width * corridor_lateral_3d_unit_vector_robot_frame
                    self.Point_corridor_center_first_time_3d_robot_frame = Point_mid_frame_first_time_3d_robot_frame + offset_3d_robot_frame
                    self.Point_corridor_center_first_time_2d_robot_frame =self.Point_corridor_center_first_time_3d_robot_frame[:2]

                    rospy.loginfo(f"offset_3d_robot_frame is  {offset_3d_robot_frame},  Point_mid_frame_first_time_3d_robot_frame is  {Point_mid_frame_first_time_3d_robot_frame}, Point_corridor_center_first_time_3d_robot_frame value is {self.Point_corridor_center_first_time_3d_robot_frame}")
                    
                    
                    Point_corridor_center_first_time_3d_camera_frame = self.R_rc.T @ (self.Point_corridor_center_first_time_3d_robot_frame - self.t_rc)

                    rospy.loginfo(f"Point_corridor_center_first_time_3d_robot_frame: {self.Point_corridor_center_first_time_3d_robot_frame}, Point_corridor_center_first_time_3d_camera_frame: {Point_corridor_center_first_time_3d_camera_frame}")
            
                    self.x_corridor_center_cam_first_time = Point_corridor_center_first_time_3d_camera_frame[0]
            
                    # Store FIXED distance ( starting distance) of corridor center along corridor axis
                    self.corridor_center_distance_along_corridor_axis_first_time = np.dot(
                        self.Point_corridor_center_first_time_2d_robot_frame,
                        self.corridor_forward_2d_unit_vector_first_time_robot_frame
                    )#Distance of corridor center along corridor axis at start time

                    self.corridor_center_distance_perpendicular_to_corridor_axis_first_time = np.dot(
                        self.Point_corridor_center_first_time_2d_robot_frame,
                        self.corridor_lateral_2d_unit_vector_first_time_robot_frame
                    )#Distance of corridor center perpendicular to corridor axis at start time. this value should be close to 0 because we are considering the corridor center is on the corridor axis. but it may not be exactly 0 because of noise in the data from frame detection node and also because of the fact that we are considering the mid frame point as reference for corridor definition which may not be exactly on the corridor axis because of noise and errors in the data from frame detection node. but it should be close to 0 ideally.
                    rospy.loginfo(f"corridor_center_distance_along_corridor_axis_first_time: {self.corridor_center_distance_along_corridor_axis_first_time:.3f} m, corridor_center_distance_perpendicular_to_corridor_axis_first_time: {self.corridor_center_distance_perpendicular_to_corridor_axis_first_time:.3f} m")

                    self.start_yaw_at_corridor_definition = self.current_yaw  # store the yaw at corridor definition time
                    self.start_pose_at_corridor_definition = self.current_pose  # store the pose at corridor definition time
                    rospy.loginfo(f"Initial x corridor center in camera frame: {self.x_corridor_center_cam_first_time} m, corridor mid depth: {self.corridor_middle_point_depth_first_time_camera_frame} m, Initial x corridor center in world frame: {self.Point_corridor_center_first_time_2d_robot_frame} m")
                    type_of_passability_check = 1 # set to 1 for corridor passability check when corridor is defined based on frame detection node. because frame detection node provide the most accurate corridor definition based on the door state and door geometry. so we can consider corridor is defined and we can trigger corridor passability check. and we will update the type_of_passability_check variable in next iterations if there is any request from traversal node to change the type of passability check. but for now we will set it to 1 for corridor passability check because we have just defined the corridor based on frame detection node data.
                    self.define_new_corridor_req_from_frame_detection = False  # corridor defined, no need to define again until next activation or next corridor definition request
                    

                    # Numerically robust angle computation: clip dot to [-1, 1]
                    robot_forward_2d = np.array([
                        np.cos(self.start_yaw_at_corridor_definition),
                        np.sin(self.start_yaw_at_corridor_definition)
                    ])
                    robot_forward_2d = np.array([1.0, 0.0])

                    dot_val = float(np.clip(
                        np.dot(self.corridor_forward_2d_unit_vector_first_time_robot_frame, robot_forward_2d),
                        -1.0,
                        1.0
                    ))

                    angle_deg = float(np.degrees(np.arccos(dot_val)))

                    rospy.loginfo(
                        f"angle between robot forward (at corridor definition) "
                        f"and corridor axis (deg): {angle_deg:.3f}"
                    )

                elif self.door_status_from_frame_detection_node in ["no_frame_detected", "No_door_plane_detected"]:
                    # door_depth_from_frame_detection_node and door plane normal are the useful information from frame detection node in this case. they cannot provide mid frame x px because no door frame detected
                    # depending on the state, normal will be different. if state is no_frame_detected, actual door plane normal at self.plane_info_norm_camera_frame. if door state is No_door_plane_detected, we assume a vertical plane and give normal as [0,0,1] in camera frame. but in both cases, we can use the plane normal and door depth to find a virtual corridor for passability check. because we know the plane normal and depth of the plane, we can find the point on the plane which is closest to the robot and consider that as the mid point of the corridor. and then we can find the corridor center point by shifting from the mid point in the direction perpendicular to the plane normal by half of the robot width plus safety margin. this way we can define a virtual corridor even when there is no frame detected or no door plane detected by frame detection node. and then we can use this virtual corridor for passability check until next corridor definition request from frame detection node or traversal node.
                    rospy.logwarn("Door state is no_frame_detected or No_door_plane_detected, cannot define fixed corridor. need to find a virtual fixed corridor.")
                    # virtual corridor finding. corridor is not fixed yet, can be anywhere in front of robot. no need of temporal smoothing for virtual corridor can be changed not a fixed value
                    z_min = self.minimum_depth_for_back_projection
                    z_max = self.corridor_middle_point_depth_first_time_camera_frame + self.maximum_depth_beyond_corridor_center_point_for_back_projection
                    min_clearance =  self.corridor_middle_point_depth_first_time_camera_frame + clearance_needed_beyond_mid_corridor
                    # With no reliable frame edge, choose a robot-width opening
                    # directly from the filtered obstacle cloud.
                    self.x_corridor_center_first_time_corridor_frame , passability_view = self.find_a_virtual_corridor(z_min, z_max, min_clearance)
                    # once virtual corridor is found, we can consider corridor defined for next iterations
                    # calculate x coordinate of corridor center in robot base frame at start yaw frame
                    if self.x_corridor_center_first_time_corridor_frame is not None:# can be corridor center or zero( means consider corridor directly in front of the robot asw there were not enough points to check corridor.)
                        Point_corridor_center_first_time_3d_camera_frame_temp = np.array([0.0, 0.0, self.corridor_middle_point_depth_first_time_camera_frame]) # y is 0 because we are considering corridor center point which is at the same height as camera, and we are only interested in x and z for corridor center point. and we will transform this point to robot frame to get x coordinate of corridor center in robot frame.
                        self.Point_corridor_center_first_time_3d_robot_frame_temp = self.R_rc @ Point_corridor_center_first_time_3d_camera_frame_temp + self.t_rc
                        self.Point_corridor_center_first_time_2d_robot_frame_temp = self.Point_corridor_center_first_time_3d_robot_frame_temp[:2] # we only care about x and z coordinates in robot frame for corridor center point, and we will consider this as the corridor center point in robot frame for control. and we will ignore the y coordinate in robot frame because we are considering corridor center point which is at the same height as camera, so the y coordinate in robot frame should be 0 or close to 0.

                        # Store FIXED distance ( starting distance) of corridor center along corridor axis
                        self.corridor_center_distance_along_corridor_axis_first_time = np.dot(
                            self.Point_corridor_center_first_time_2d_robot_frame_temp,
                            self.corridor_forward_2d_unit_vector_first_time_robot_frame
                        )#Distance of corridor center along corridor axis at start time

                        self.corridor_center_distance_perpendicular_to_corridor_axis_first_time = self.x_corridor_center_first_time_corridor_frame #Distance of corridor center perpendicular to corridor axis at start time. this value should be close to 0 because we are considering the corridor center is on the corridor axis. but it may not be exactly 0 because of noise in the data from frame detection node and also because of the fact that we are considering the mid frame point as reference for corridor definition which may not be exactly on the corridor axis because of noise and errors in the data from frame detection node. but it should be close to 0 ideally.
                        
                        self.Point_corridor_center_first_time_2d_robot_frame = (
                            self.corridor_center_distance_along_corridor_axis_first_time * self.corridor_forward_2d_unit_vector_first_time_robot_frame
                            + self.corridor_center_distance_perpendicular_to_corridor_axis_first_time  * self.corridor_lateral_2d_unit_vector_first_time_robot_frame
                        )
                        rospy.loginfo(f"corridor_center_distance_along_corridor_axis_first_time: {self.corridor_center_distance_along_corridor_axis_first_time:.3f} m, corridor_center_distance_perpendicular_to_corridor_axis_first_time: {self.corridor_center_distance_perpendicular_to_corridor_axis_first_time:.3f} m")
                        rospy.loginfo(f"corridor center point in robot frame temp is {self.Point_corridor_center_first_time_2d_robot_frame_temp} m, corridor center point in robot frame calculated from corridor frame is {self.Point_corridor_center_first_time_2d_robot_frame} m")
                        self.start_yaw_at_corridor_definition = self.current_yaw  # store the yaw at corridor definition time
                        self.start_pose_at_corridor_definition = self.current_pose  # store the pose at corridor definition time

                        type_of_passability_check = 1  # after corridor definiton, first do corridor passability check
                        self.define_new_corridor_req_from_frame_detection = False  # corridor defined, no need to define again until next activation or next corridor definition request
                       

                    else: # in this case, we cannot find a valid virtual corridor. this can happen when there was points in front of the robot but couldnt find a virtual corridor or all the points in front of the robot are occupied by obstacles. in this case, we cannot define a corridor for passability check. so we will skip passability check and wait for next iteration when new data comes in and we can try to find a virtual corridor again. because new data may help to find a valid virtual corridor center. and during this time, we will set type_of_passability_check to 0 which means no passability check because corridor is not defined yet.
                        self.define_new_corridor_req_from_frame_detection = True  # corridor is not yet defined, so need to run corridor definition again in next iteration. this can happen when virtual corridor finding fails to find a valid corridor center. in that case we can try again in next iteration because new data may come in and it may help to find a valid virtual corridor center. 
                        self.Point_corridor_center_first_time_2d_robot_frame = None # because corridor is not defined yet, we cannot calculate Point_corridor_center_first_time_2d_robot_frame. so set it to None. it will be calculated in next iteration once corridor is defined. and during this iteration, since corridor is not defined, we will skip passability check and wait for next iteration when corridor is defined.
                

                else:
                    rospy.logwarn("Unknown door open side state during corridor definition.")
                    self.rate.sleep()
                    continue
                
                rospy.loginfo(f"corridor definition is {self.define_new_corridor_req_from_frame_detection} and door status from frame detection node is {self.door_status_from_frame_detection_node}, corridor middle point depth at first time is {self.corridor_middle_point_depth_first_time_camera_frame} m, x corridor center in camera frame at first time is {self.x_corridor_center_cam_first_time} m, x corridor center in start yaw frame at first time is {self.Point_corridor_center_first_time_2d_robot_frame} m")
                
        
            elif self.define_new_corridor_req_from_traversal: #need to find a virtual corridor because, there is no fixed corridor. fixed corridor will be defined only when door is open left or open right (i.e frame detection has data)
                self.start_yaw_at_corridor_definition = self.current_yaw  # store the yaw at corridor definition time
                self.start_pose_at_corridor_definition = self.current_pose  # store the pose at corridor definition time
                self.corridor_middle_point_depth_first_time_camera_frame = self.corridor_middle_point_depth_from_traversal # traversal node will provide the depth of the middle point of the corridor based on the robot pose and the door geometry. this is because traversal node has the most up to date information about the robot pose and door geometry, so it can provide the most accurate depth of the middle point of the corridor for virtual corridor definition. and this depth value will be used as reference depth for virtual corridor finding in case there is no frame detected by frame detection node or no door plane detected by frame detection node. because in that case we cannot rely on frame detection node data for corridor definition, so we will rely on traversal node data for virtual corridor definition. and this value will be updated in real time as robot moves and new data comes in from traversal node, so it can help to update the virtual corridor definition in real time based on the latest robot pose and door geometry information from traversal node.
                clearance_needed_beyond_mid_corridor = self.corridor_middle_point_depth_first_time_camera_frame # by default half of needed corridor length is received as door_middle_depth from traversal node
                # similar to no frame detected case above, we need to find a virtual corridor because new corridor is requested by traversal node
                rospy.logwarn(" traversal node has requested for new corridor definition.  need to find a virtual fixed corridor.")
                # virtual corridor finding. corridor is not fixed yet, can be anywhere in front of robot. no need of temporal smoothing for virtual corridor can be changed not a fixed value
                z_min = self.minimum_depth_for_back_projection
                z_max = self.corridor_middle_point_depth_first_time_camera_frame + self.maximum_depth_beyond_corridor_center_point_for_back_projection
                min_clearance = self.corridor_middle_point_depth_first_time_camera_frame + clearance_needed_beyond_mid_corridor # door traversal node request this value based on the total corridor length it is trying to traverse. so always publish half of the value
                self.x_corridor_center_first_time_corridor_frame , passability_view= self.find_a_virtual_corridor(z_min, z_max, min_clearance)
                # once virtual corridor is found, we can consider corridor defined for next iterations
                # calculate x coordinate of corridor center in robot base frame at start yaw frame
                if self.x_corridor_center_first_time_corridor_frame is not None: #can be corridor center or zero( means consider corridor directly in front of the robot asw there were not enough points to check corridor.)
                    Point_corridor_center_first_time_3d_camera_frame_temp = np.array([0.0, 0.0, self.corridor_middle_point_depth_first_time_camera_frame])
                    self.Point_corridor_center_first_time_3d_robot_frame_temp = self.R_rc @ Point_corridor_center_first_time_3d_camera_frame_temp + self.t_rc
                    self.Point_corridor_center_first_time_2d_robot_frame_temp = self.Point_corridor_center_first_time_3d_robot_frame_temp[:2] # we only care about x and z coordinates in robot frame for corridor center point, and we will consider this as the corridor center point in robot frame for control. and we will ignore the y coordinate in robot frame because we are considering corridor center point which is at the same height as camera, so the y coordinate in robot frame should be 0 or close to 0.

                    # Store FIXED distance ( starting distance) of corridor center along corridor axis
                    self.corridor_center_distance_along_corridor_axis_first_time = np.dot(
                        self.Point_corridor_center_first_time_2d_robot_frame_temp,
                        self.corridor_forward_2d_unit_vector_first_time_robot_frame
                    )#Distance of corridor center along corridor axis at start time. use the value of self.corridor_forward_2d_unit_vector_first_time_robot_frame which is calculated based on the plane normal from frame detection node because we are using the same plane normal for virtual corridor definition in traversal node requested corridor definition. and we are using the same reference frame for corridor definition which is the start yaw frame. so it is consistent to use the same corridor forward unit vector for calculating the distance along corridor axis for both frame detection node requested corridor definition and traversal node requested corridor definition. 

                    self.corridor_center_distance_perpendicular_to_corridor_axis_first_time = self.x_corridor_center_first_time_corridor_frame#Distance of corridor center perpendicular to corridor axis at start time. this value should be close to 0 because we are considering the corridor center is on the corridor axis. but it may not be exactly 0 because of noise in the data from frame detection node and also because of the fact that we are considering the mid frame point as reference for corridor definition which may not be exactly on the corridor axis because of noise and errors in the data from frame detection node. but it should be close to 0 ideally.
                    
                    self.Point_corridor_center_first_time_2d_robot_frame = (
                            self.corridor_center_distance_along_corridor_axis_first_time * self.corridor_forward_2d_unit_vector_first_time_robot_frame
                            + self.corridor_center_distance_perpendicular_to_corridor_axis_first_time  * self.corridor_lateral_2d_unit_vector_first_time_robot_frame
                        )
                    
                    rospy.loginfo(f"Point_corridor_center_first_time_2d_robot_frame: {self.Point_corridor_center_first_time_2d_robot_frame}, corridor center point in robot frame temp is {self.Point_corridor_center_first_time_2d_robot_frame_temp} m, corridor center point in robot frame calculated from corridor frame is {self.Point_corridor_center_first_time_2d_robot_frame} m")
                    rospy.loginfo(f"corridor_center_distance_along_corridor_axis_first_time: {self.corridor_center_distance_along_corridor_axis_first_time:.3f} m, corridor_center_distance_perpendicular_to_corridor_axis_first_time: {self.corridor_center_distance_perpendicular_to_corridor_axis_first_time:.3f} m")


                    self.start_yaw_at_corridor_definition = self.current_yaw  # store the yaw at corridor definition time
                    self.start_pose_at_corridor_definition = self.current_pose  # store the pose at corridor definition time
                    self.define_new_corridor_req_from_traversal = False  # corridor defined, no need to define again until next request
                    type_of_passability_check = 1  # after corridor definiton, first do corridor passability check
                else:# in this case, we cannot find a valid virtual corridor. this can happen when there was points in front of the robot but couldnt find a virtual corridor or all the points in front of the robot are occupied by obstacles. in this case, we cannot define a corridor for passability check. so we will skip passability check and wait for next iteration when new data comes in and we can try to find a virtual corridor again. because new data may help to find a valid virtual corridor center. and during this time, we will set type_of_passability_check to 0 which means no passability check because corridor is not defined yet.
                    self.define_new_corridor_req_from_traversal = True  # corridor is not yet defined, so need to run corridor definition again in next iteration. this can happen when virtual corridor finding fails to find a valid corridor center. in that case we can try again in next iteration because new data may come in and it may help to find a valid virtual corridor center.
                    self.Point_corridor_center_first_time_2d_robot_frame = None # because corridor is not defined yet, we cannot calculate Point_corridor_center_first_time_2d_robot_frame. so set it to None. it will be calculated in next iteration once corridor is defined. and during this iteration, since corridor is not defined, we will skip passability check and wait for next iteration when corridor is defined.
                
                # no need to publish this in door frame detection node based corridor definition. because at that stage we also checking the corridor passability and then only triggering the traversal node.
                #  but in traversal node requested corridor definition,  we need to send an ack back to traversal node that new corridor is defined and info received by traversal node is for this new virtual corridor
                self.virtual_corridor_definition_finished_pub.publish(not self.define_new_corridor_req_from_traversal)
                rospy.loginfo(f"New corridor definition requested by traversal node. corridor definition is {not self.define_new_corridor_req_from_traversal} and corridor middle point depth at first time is {self.corridor_middle_point_depth_first_time_camera_frame} m, x corridor center in camera frame at first time is {self.x_corridor_center_cam_first_time} m, x corridor center in start yaw frame at first time is {self.Point_corridor_center_first_time_2d_robot_frame} m")

            if self.define_new_corridor_req_from_frame_detection or self.define_new_corridor_req_from_traversal:
                # if corridor is not yet defined, we cannot do passability check. so skip the rest of the loop and wait for next iteration when corridor is defined
                # Serializing a full-resolution image on this retry path is pure
                # diagnostic cost, and this branch repeats for as long as corridor
                # definition keeps failing.
                if self.enable_visualization and passability_view is not None:
                    self.passability_view_pub.publish(self.bridge.cv2_to_imgmsg(passability_view, encoding="bgr8"))
                self.rate.sleep()
                continue

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
            # ---- Every control cycle ----
            # Propagate the original corridor instead of re-detecting it from a
            # changing camera view as the robot approaches the door.
            R, t = self.relative_robot_motion()#t = robot displacement since corridor definition # robot displacement in start frame

            #t is in the same frame as self.corridor_forward_2d_unit_vector_first_time_robot_frame
            # corridor center expressed in CURRENT robot frame
            # only calculate in 2d ground plane. we dont want to use noisy z component for corridor center point calculation in robot frame. because we are interested in the ground plane distance of the robot from the corridor center line for passability check. and for that we will project the robot position and corridor center point to the ground plane by ignoring the z component. because the z component may have noise and errors which can affect the accuracy of the forward and lateral error calculation for passability check. so for corridor center point calculation we will consider z component but for error calculation we will ignore z component by projecting to ground plane. this is a design choice to improve the robustness of the passability check against noise in depth data which can affect the accuracy of z component of corridor center point in robot frame.
            Pr_now = self.propagate_corridor_point(self.Point_corridor_center_first_time_2d_robot_frame, R, t) # corridor point now in robot frame
            
            #lift corridor point to 3d just for visualization
            Pr_now_3d = np.array([Pr_now[0], Pr_now[1], 0.0])



            corridor_forward_2d_unit_vector_robot_frame_now = R.T @ self.corridor_forward_2d_unit_vector_first_time_robot_frame
            corridor_lateral_2d_unit_vector_robot_frame_now = R.T @ self.corridor_lateral_2d_unit_vector_first_time_robot_frame  

            # normalize the vectors to ensure they remain unit vectors after rotation. because due to numerical errors, the length of the vectors may change after rotation, which can affect the accuracy of the forward and lateral motion calculation. by normalizing them, we ensure that they remain unit vectors and the motion calculation remains accurate.
            corridor_forward_2d_unit_vector_robot_frame_now /= np.linalg.norm(corridor_forward_2d_unit_vector_robot_frame_now)
            corridor_lateral_2d_unit_vector_robot_frame_now /= np.linalg.norm(corridor_lateral_2d_unit_vector_robot_frame_now)
            
            #lift axes to 3d just to calculate the offset in 3d space. because we need to consider the z component of the plane normal for accurate corridor center point calculation in 3d space. if we do the calculation in 2d space, we will lose the z component which is important for accurate corridor center point calculation in 3d space
            # but for the calculation of lateral error and forward error for passability check, we will consider only the x and y component of the corridor center point in robot frame because we are interested in the ground plane distance of the robot from the corridor center line. and for that we will project the robot position and corridor center point to the ground plane by ignoring the z component. because the z component may have noise and errors which can affect the accuracy of the forward and lateral error calculation for passability check. so for corridor center point calculation we will consider z component but for error calculation we will ignore z component by projecting to ground plane. this is a design choice to improve the robustness of the passability check against noise in depth data which can affect the accuracy of z component of corridor center point in robot frame.
            # corridor_forward_3d_unit_vector_robot_frame is created just for visualization
            corridor_forward_3d_unit_vector_robot_frame_now = np.array([
                corridor_forward_2d_unit_vector_robot_frame_now[0],
                corridor_forward_2d_unit_vector_robot_frame_now[1],
                0.0
            ])  
            
            corridor_lateral_3d_unit_vector_robot_frame_now = np.array([
                corridor_lateral_2d_unit_vector_robot_frame_now[0],
                corridor_lateral_2d_unit_vector_robot_frame_now[1],
                0.0
            ]) 

            heading_error = np.arctan2(
                    corridor_forward_2d_unit_vector_robot_frame_now[1],
                    corridor_forward_2d_unit_vector_robot_frame_now[0]
                )

            # this is value increase as robot go towards corridor. not a distance measure
            #forward_motion = np.dot(Pr_now, corridor_forward_2d_unit_vector_robot_frame_now)


            # Rotate robot displacement into corridor frame. ow much the robot has moved forward along the corridor axis

            # Distance to corridor midpoint ALONG corridor axis
            """
            distance_along_corridor_axis = (
                self.corridor_center_distance_along_corridor_axis_first_time
                - np.dot(t, self.corridor_forward_2d_unit_vector_first_time_robot_frame)
            )

            lateral_distance_perpendicular_to_corridor_axis = (
                self.corridor_center_distance_perpendicular_to_corridor_axis_first_time
                - np.dot(t, self.corridor_lateral_2d_unit_vector_first_time_robot_frame)
            )
            """
            
            
                        # Compute distances using CURRENT robot frame to avoid frame mismatches
            # Project the propagated corridor center onto the CURRENT corridor axes
            
            distance_along_corridor_axis = np.dot(
                Pr_now,
                corridor_forward_2d_unit_vector_robot_frame_now
            )

            lateral_distance_perpendicular_to_corridor_axis = np.dot(
                Pr_now,
                corridor_lateral_2d_unit_vector_robot_frame_now
            )
            
            
            #lateral_motion = np.dot(Pr_now, corridor_lateral_2d_unit_vector_robot_frame_now)
            
            self.corridor_middle_point_depth_dynamic_along_corridor_axis =  distance_along_corridor_axis
            self.x_corridor_center_start_yaw_frame_dynamic = lateral_distance_perpendicular_to_corridor_axis 

            
            Pc_now = self.R_rc.T @ (np.array([
                    Pr_now[0],
                    Pr_now[1],
                    0.0
                    
                ]) - self.t_rc)
            
            self.x_corridor_center_cam_dynamic = Pc_now[0]
            
            
            rospy.loginfo(f"corridor_middle_point_depth_dynamic_along_corridor_axis is {self.corridor_middle_point_depth_dynamic_along_corridor_axis}, x_corridor_center_cam_dynamic is {self.x_corridor_center_cam_dynamic}, x_corridor_center_start_yaw_frame_dynamic is {self.x_corridor_center_start_yaw_frame_dynamic}, heading error is {heading_error}")
            corridor_middle_point_depth = self.corridor_middle_point_depth_dynamic_along_corridor_axis
            x_corridor_center_cam = self.x_corridor_center_cam_dynamic
            x_corridor_center_start_yaw_frame = self.x_corridor_center_start_yaw_frame_dynamic # because lateral motion is the error in lateral direction, which means how much the corridor center has shifted in lateral direction in world frame at start yaw frame. and x direction in camera frame is y direction in robot base frame, so it is also the shift of corridor center in camera frame. so we can use this value for both x_corridor_center_cam and x_corridor_center_start_yaw_frame because they are parallel to each other at start yaw frame and we are considering only the shift of corridor center from initial position, so the value will be same for both frames.

            #once corridor is defined, we can do corridor passability check or local passability check based on request

            if type_of_passability_check == 1 or type_of_passability_check == 2: # default value, do nothing. this can happen when corridor is defined for the first time and we have not yet set type_of_passability_check to 1 for corridor passability check. in this case we can skip passability check and wait for next iteration when type_of_passability_check is set to 1. because at first time corridor definition, we may not have the data needed for passability check, so we can skip passability check at that iteration and wait for next iteration when we have the data needed for passability check.
                
                if type_of_passability_check == 1: # corridor passability check. request could come from traversal node or by default.
                    rospy.loginfo("Performing corridor passability check.")
                    # Define Z extents relative to door
                    z_min = self.minimum_depth_for_back_projection
                    z_max = self.corridor_middle_point_depth_dynamic_along_corridor_axis + self.maximum_depth_beyond_corridor_center_point_for_back_projection
                    #reference_depth_for_visualization = self.corridor_middle_point_depth_dynamic_along_corridor_axis - 0.5 * self.robot_length  # we want to check the passability at the point which is half of robot length before the corridor center because that is the point where we want the robot to be when it is passing through the door. if we check passability at corridor center, it may be too late for the robot to react and adjust its trajectory to pass through the door. by checking passability at a point before the corridor center, we can give the robot more time to react and adjust its trajectory to successfully pass through the door. so that is why we use corridor_middle_point_depth_dynamic_along_corridor_axis - 0.5*robot_length as reference depth for back projection in corridor passability check. but in local passability check, we can use corridor_middle_point_depth_dynamic_along_corridor_axis as reference depth because local passability check is more focused on checking the immediate area around the robot for obstacles, so using the current position of the robot as reference depth is more appropriate for local passability check.
                    if self.enable_visualization:
                        left_line_start_robot_frame = Pr_now_3d - self.half_width * corridor_lateral_3d_unit_vector_robot_frame_now
                        right_line_start_robot_frame = Pr_now_3d + self.half_width * corridor_lateral_3d_unit_vector_robot_frame_now 
                        left_line_end_robot_frame = left_line_start_robot_frame + z_max * corridor_forward_3d_unit_vector_robot_frame_now
                        right_line_end_robot_frame = right_line_start_robot_frame + z_max * corridor_forward_3d_unit_vector_robot_frame_now
                        #Transform to Camera Frame
                        left_line_start_camera_frame = self.R_rc.T @ (left_line_start_robot_frame - self.t_rc)
                        right_line_start_camera_frame = self.R_rc.T @ (right_line_start_robot_frame - self.t_rc)
                        left_line_end_camera_frame = self.R_rc.T @ (left_line_end_robot_frame - self.t_rc)
                        right_line_end_camera_frame = self.R_rc.T @ (right_line_end_robot_frame - self.t_rc)

                elif type_of_passability_check == 2: # local passability check
                    #Local passability answers:Can the robot move forward safely in its CURRENT heading
                    #This is purely robot-centric. It does NOT depend on corridor axis.It does NOT depend on door orientation.So the correct frame for local passability is: current robot frame
                    rospy.loginfo("Performing local passability check.")
                    z_min = self.minimum_depth_for_back_projection_local_passability_check  # closer range for local passability
                    z_max = self.maximum_depth_for_back_projection_local_passability_check  # only up to door
                    #reference_depth_for_visualization = z_min + (z_max - z_min) / 2.0  # mid depth for local passability
                    if self.enable_visualization:
                        #For local passability, the region is aligned with the robot’s current frame, not the corridor axis.
                        # So:Forward axis = robot X,  Lateral axis = robot Y, Up axis = robot Z . No rotated basis needed.
                        # Near boundary (z_min)
                        left_line_start_robot_frame  = np.array([z_min,  self.half_width, 0.0])
                        right_line_start_robot_frame = np.array([z_min, -self.half_width, 0.0])

                        # Far boundary (z_max)
                        left_line_end_robot_frame   = np.array([z_max,  self.half_width, 0.0])
                        right_line_end_robot_frame  = np.array([z_max, -self.half_width, 0.0])
                        #Transform to Camera Frame
                        left_line_start_camera_frame  = self.R_rc.T @ (left_line_start_robot_frame - self.t_rc)
                        right_line_start_camera_frame = self.R_rc.T @ (right_line_start_robot_frame - self.t_rc)
                        left_line_end_camera_frame    = self.R_rc.T @ (left_line_end_robot_frame - self.t_rc)
                        right_line_end_camera_frame   = self.R_rc.T @ (right_line_end_robot_frame - self.t_rc)


                # Backproject once, then crop in robot/corridor coordinates. An
                # image rectangle would be wrong when the doorway is angled.
                valid_points_3d_camera_frame, uv, _ = backproject_depth_to_points(
                    self.depth_image, self.fx, self.fy, self.cx, self.cy,
                    max_depth=z_max, min_depth=z_min,
                    subsample=self.subsample, roi_polygon=None
                )

                rospy.loginfo(f"z_min is {z_min} and z_max is {z_max} for back projection in passability check. number of valid depth points for passability check is {len(valid_points_3d_camera_frame)}")
                if len(valid_points_3d_camera_frame) == 0: 
                    rospy.logwarn("No valid depth points for passability check.")
                    front_clearance = z_max
                    passability_view = self.color_image

                else:
                    # Equivalent to (R @ p.T).T + t, but without the two transposes
                    # and the float64 promotion of the whole cloud.
                    valid_points_3d_robot_frame = valid_points_3d_camera_frame @ self.R_rc_f32.T + self.t_rc_f32
                    # Keep only ground-plane components
                    valid_points_2d_robot_frame = valid_points_3d_robot_frame[:, :2]  # Extract 2D points (x, y)
                    
                    if type_of_passability_check == 1: # corridor passability check, we need to consider the points within the corridor defined by the corridor axis and robot width. so we need to calculate the lateral and forward distance of the points with respect to the corridor axis and keep only the points within the corridor.
                        # -------------------------------------------------
                        # Express relative to corridor center
                        # -------------------------------------------------

                        delta = valid_points_2d_robot_frame - Pr_now   # (N,2)
                        # -------------------------------------------------
                        # Project onto corridor axes
                        # -------------------------------------------------

                        lateral_values = delta @ corridor_lateral_2d_unit_vector_robot_frame_now
                        forward_values = delta @ corridor_forward_2d_unit_vector_robot_frame_now
                    
                    elif type_of_passability_check == 2: # local passability check, we need to consider the points within the area in front of the robot defined by the robot width and the z_max for local passability check. so we need to calculate the lateral and forward distance of the points with respect to the robot frame and keep only the points within the area in front of the robot.
                        # for local passability, we use robot forward axis as reference, so we can directly use the x and y values in robot frame to calculate the lateral and forward distance of the points with respect to the robot frame. and we do not need to express the points relative to corridor center because local passability is purely robot-centric and does not depend on corridor definition. so we can directly calculate the lateral and forward distance of the points with respect to the robot frame without considering the corridor center.
                        lateral_values = valid_points_2d_robot_frame[:, 1]  # y values in robot frame represent lateral distance (left is positive, right is negative)
                        forward_values = valid_points_2d_robot_frame[:, 0]  # x values
              
                    camera_height_in_robot_base = self.t_rc[2]  # z component of camera translation in robot base frame represents the camera height from the ground
                    mask_below_robot_eye_level = (
                            valid_points_3d_robot_frame[:, 2] < camera_height_in_robot_base + self.robot_height_above_camera_level
                        )
                    
                    # Limit obstacles to the robot's swept width, forward range,
                    # and body height. Overhead points cannot block traversal.
                    corridor_mask_robot_frame = (
                        (np.abs(lateral_values) < self.half_width) &
                        (forward_values >= 0) &  # only consider points in front of the robot
                        (forward_values <= z_max) & # only consider points within the max depth for passability check
                        mask_below_robot_eye_level #
                    )

                    corridor_pts_camera_frame = valid_points_3d_camera_frame[corridor_mask_robot_frame]
                    corridor_uv_camera_frame = uv[corridor_mask_robot_frame]
                    
                    # not using final points here as the corridor is already defined
                    front_clearance, passability_view, _ = self.passability_checker.run(
                        self.depth_image,
                        self.color_image,
                        corridor_pts_camera_frame,
                        corridor_uv_camera_frame,
                        z_max
                    )

            #to trigger the traversal node based on corridor passability result
            # Latch permission once. Clearance can shrink naturally as the robot
            # enters the corridor and should not restart the controller handshake.
            if self.trigger_traversal_node is False and type_of_passability_check == 1 and front_clearance is not None: # if traversal node is not yet triggered, we can do passability check and trigger traversal node based on the result. but once traversal node is triggered, we should not reset the trigger even if passability goes false in later iterations. because it only makes sense to trigger traversal node once when we detect passability for the first time. and during traversal passability may go false but we do not want to retrigger traversal node.
                #executes only once
                corridor_Passability_status = front_clearance > self.corridor_middle_point_depth_dynamic_along_corridor_axis + self.min_corridor_clearance_beyond_door_to_trigger_traversal_node
                if corridor_Passability_status:
                    # trigger traversal node if it is passable for the first time.
                    #but in later if instantanous passability goes false, do not reset the trigger
                    #trigger traverse node once we detect passability. but during the traversal passability
                    # may be false, we do not want to retrigger the traversal node.
                    self.trigger_traversal_node = True

                #in local passability check we do not trigger traversal node, because it is already triggered in corridor passability check which will execute first

                        


            self.trigger_traversal_node_pub.publish(Bool(data=self.trigger_traversal_node))

            msg = Robot_passability()
            msg.front_clearance = float(front_clearance) if front_clearance is not None else float('nan')
            msg.x_corridor_center_start_yaw_frame = float(x_corridor_center_start_yaw_frame) if x_corridor_center_start_yaw_frame is not None else float('nan')
            msg.type_of_passability_check = type_of_passability_check # 1 for corridor , 2 for local
            msg.x_corridor_center_cam = float(x_corridor_center_cam) if x_corridor_center_cam is not None else float('nan')
            msg.corridor_middle_point_depth = float(corridor_middle_point_depth) if corridor_middle_point_depth is not None else float('nan')
            msg.clearance_needed_beyond_mid_corridor = float(clearance_needed_beyond_mid_corridor) if clearance_needed_beyond_mid_corridor is not None else float('nan')
            msg.heading_error = float(heading_error) if heading_error is not None else float('nan')

            #publish results
            self.Passability_pub.publish(msg)

            # Publish visualizations
            try:
                if self.enable_visualization and passability_view is not None:
                    # Convert boundaries to pixel coordinates for visualization    
                    #x_left_boundary_uv = int((x_left_limit * self.fx) / reference_depth_for_visualization + self.cx)
                    #x_right_boundary_uv = int((x_right_limit * self.fx) / reference_depth_for_visualization + self.cx)
                    #cv2.line(passability_view, (x_left_boundary_uv, 0), (x_left_boundary_uv, passability_view.shape[0]), (0, 165, 255), 4)  # orange
                    #cv2.line(passability_view, (x_right_boundary_uv, 0), (x_right_boundary_uv, passability_view.shape[0]), (0, 0, 255), 4)  # red
                    p1 = self.project_to_pixel(left_line_start_camera_frame)
                    p2 = self.project_to_pixel(left_line_end_camera_frame)
                    p3 = self.project_to_pixel(right_line_start_camera_frame)
                    p4 = self.project_to_pixel(right_line_end_camera_frame)

                    # Draw lines on the passability view
                    cv2.line(passability_view, p1, p2, (0, 255, 0), 2)  # green
                    cv2.line(passability_view, p3, p4, (0, 0, 255), 2)  # red
                    
                    self.passability_view_pub.publish(self.bridge.cv2_to_imgmsg(passability_view, encoding="bgr8"))

            except Exception as e:
                rospy.logdebug(f"Viz publish exception: {e}")


            self.rate.sleep()

    

def main():
    """
    Start the ROS corridor-passability node.

    The function initializes ROS, creates :class:`PassabilityCheckerNode`, and
    enters its blocking processing loop.

    Raises:
        rospy.ROSInterruptException:
            If ROS interrupts the node while it is running.
    """
    rospy.init_node("check_if_corridor_is_passable")
    node = PassabilityCheckerNode()
    rospy.loginfo("Check if corridor is passable node started.")
    node.run()


if __name__ == "__main__":
    main()
