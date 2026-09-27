#!/usr/bin/env python3
"""Republish a Gazebo model pose and twist as standard ROS odometry.

In bag mode there is no simulator, so a fixed pose at the origin is published
instead.
"""

import rospy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class GazeboOdomBridge:
    """
    Bridge one Gazebo model from model states to standard ROS odometry.

    Attributes:
        model_name:
            Gazebo model selected from each ``ModelStates`` message.

        input_mode:
            ``"gazebo"`` to bridge model states, or ``"bag"`` to publish a
            fixed pose at the origin.

        odom_pub:
            Publisher for the synthesized ``/odom_bridge_output`` messages.
    """

    def __init__(self):
        """
        Initialize the odometry publisher and its source.

        Notes:
            The model defaults to ``"go1_gazebo"`` and can be changed with the
            private ``~model_name`` parameter. With ``~input_mode`` set to
            ``"bag"`` no Gazebo subscription is made; a static pose is
            published at ``~static_pose_rate_hz`` instead.
        """
        self.model_name = rospy.get_param("~model_name", "go1_gazebo")
        self.input_mode = rospy.get_param("~input_mode", "gazebo")
        self.odom_pub = rospy.Publisher("/odom_bridge_output", Odometry, queue_size=10)

        if self.input_mode == "bag":
            # A recorded camera stream carries no robot motion and nothing
            # publishes /gazebo/model_states, yet the passability and traversal
            # nodes wait for a pose before they run. Hold the robot at the
            # origin. The pose is repeated rather than sent once because those
            # nodes subscribe only when a door cycle starts.
            rate_hz = rospy.get_param("~static_pose_rate_hz", 10.0)
            self.static_pose_timer = rospy.Timer(rospy.Duration(1.0 / rate_hz), self.publish_static_pose)
            rospy.loginfo("odom_bridge: bag mode, publishing a fixed pose at the origin at %.1f Hz", rate_hz)
        else:
            # /gazebo/model_states is high rate and only the latest sample is used.
            rospy.Subscriber("/gazebo/model_states", ModelStates, self.cb, queue_size=1)

    def make_odom(self):
        """
        Create an odometry message with the bridge's stamp and frames.

        Returns:
            ``Odometry`` stamped now, in ``odom`` with child frame ``trunk``.
            The pose is at the origin with an identity orientation.
        """
        odom = Odometry()
        # Gazebo does not attach a per-model timestamp here, so stamp the bridge output.
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "trunk"
        # The message default is an all-zero quaternion, which is not a valid
        # rotation and gives a meaningless yaw downstream.
        odom.pose.pose.orientation.w = 1.0
        return odom

    def publish_static_pose(self, _event=None):
        """
        Publish the fixed bag-mode pose: origin, zero yaw and zero velocity.
        """
        self.odom_pub.publish(self.make_odom())

    def cb(self, msg):
        """
        Publish the configured model's latest Gazebo state as odometry.

        Args:
            msg:
                ``gazebo_msgs/ModelStates`` containing names, poses, and
                twists for all models.

        Notes:
            Messages that do not contain ``model_name`` are ignored.
        """
        # ModelStates is a single array for the whole simulation; indexes in
        # name, pose, and twist refer to the same model.
        if self.model_name not in msg.name:
            return

        i = msg.name.index(self.model_name)

        odom = self.make_odom()
        odom.pose.pose = msg.pose[i]
        odom.twist.twist = msg.twist[i]

        self.odom_pub.publish(odom)

if __name__ == "__main__":
    # spin() is enough because all bridge work happens in the subscriber or
    # timer callback.
    rospy.init_node("gazebo_odom_bridge")
    GazeboOdomBridge()
    rospy.spin()
