#!/usr/bin/env python3
"""Republish a Gazebo model pose and twist as standard ROS odometry."""

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

        odom_pub:
            Publisher for the synthesized ``/odom_bridge_output`` messages.
    """

    def __init__(self):
        """
        Initialize the Gazebo subscriber and odometry publisher.

        Notes:
            The model defaults to ``"go1_gazebo"`` and can be changed with the
            private ``~model_name`` parameter.
        """
        self.model_name = rospy.get_param("~model_name", "go1_gazebo")
        self.odom_pub = rospy.Publisher("/odom_bridge_output", Odometry, queue_size=10)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.cb)

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
        if self.model_name not in msg.name:
            return

        i = msg.name.index(self.model_name)

        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "trunk"

        odom.pose.pose = msg.pose[i]
        odom.twist.twist = msg.twist[i]

        self.odom_pub.publish(odom)

if __name__ == "__main__":
    rospy.init_node("gazebo_odom_bridge")
    GazeboOdomBridge()
    rospy.spin()
