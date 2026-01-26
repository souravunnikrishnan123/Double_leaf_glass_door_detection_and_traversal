#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class GazeboOdomBridge:
    def __init__(self):
        self.model_name = rospy.get_param("~model_name", "go1_gazebo")
        self.odom_pub = rospy.Publisher("/odom_bridge_output", Odometry, queue_size=10)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.cb)

    def cb(self, msg):
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
